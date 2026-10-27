#!/usr/bin/env python3
"""Extract and normalize licensed LSEG Fed Funds futures SETTLE data.

The original LSEG workbooks are read-only inputs and are never modified. This
preparer accepts the wide Excel layout produced by LSEG Workspace formulas as
well as conventional long-form CSV exports. Only the ``SETTLE`` field is read
from Excel. Processing has an explicit CSV boundary:

1. cached Excel arrays are extracted to ``ffr_futures_lseg_extracted.csv``;
2. that CSV is read back, validated, deduplicated, and written to the stable
   benchmark input ``ffr_futures.csv``.

The benchmark and all later stages read only the stable CSV schema:

    date,contract_month,settle

``settle`` must be the futures price, so the contract-implied average EFFR is
``100 - settle``. Trade dates on or after 2026-07-01 are excluded. Contract
months may extend into July because the July contract is required to forecast
the June 2026 FOMC decision.

No network requests are made by this file.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "ffr_futures.csv"
DEFAULT_EXTRACTED_OUTPUT = BASE_DIR / "ffr_futures_lseg_extracted.csv"
DEFAULT_LSEG_SETTLE_XLSX = (
    BASE_DIR / "0_lseg" / "ffr" /
    "FFR Futures 2025-2026 plus July (SETTLE).xlsx"
)
DEFAULT_LSEG_SETTLE_INPUTS = (
    BASE_DIR / "0_lseg" / "ffr" / "FFR Futures 2022-2025Jan.xlsx",
    DEFAULT_LSEG_SETTLE_XLSX,
)
SAMPLE_END_EXCLUSIVE = pd.Timestamp("2026-07-01")

SPREADSHEET_NS = (
    "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
DOCUMENT_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
PACKAGE_REL_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships")

MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

COLUMN_ALIASES = {
    "date": ("date", "tradedate", "businessdate", "bizdt",
             "settlementpricedate", "trsettlementpricedate"),
    "contract": ("contractmonth", "expirationmonth", "maturitymonth",
                 "mmy", "matdt", "maturitydate", "expirationdate",
                 "contract", "instrument", "instrumentric", "ric",
                 "trric", "id", "symbol", "sym"),
    "settle": ("settle", "settlement", "settlementprice", "settlprice",
               "trsettlementprice", "setlprice", "pxsettle"),
}


@dataclass(frozen=True)
class LsegQuery:
    sheet_name: str
    sheet_path: str
    formula_cell: str
    row_index: int
    column_index: int
    ric: str
    field: str


def normalized_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(frame, role):
    lookup = {normalized_name(column): column for column in frame.columns}
    for candidate in COLUMN_ALIASES[role]:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"Could not identify the {role!r} column. Available columns: "
        f"{list(frame.columns)}")


def symbol_period(token, trade_date):
    match = re.search(r"(?:ZQ|FF)?([FGHJKMNQUVXZ])(\d{1,4})(?:\^\d+)?$",
                      str(token).strip().upper())
    if not match:
        return pd.NaT
    month = MONTH_CODES[match.group(1)]
    raw_year = match.group(2)
    if len(raw_year) == 4:
        year = int(raw_year)
    elif len(raw_year) == 2:
        year = 2000 + int(raw_year)
    else:
        # One-digit exchange years are assigned to the nearest non-past
        # decade relative to the observation date.
        base = trade_date.year if pd.notna(trade_date) else 2020
        decade = (base // 10) * 10
        candidates = [decade - 10 + int(raw_year),
                      decade + int(raw_year),
                      decade + 10 + int(raw_year)]
        year = min(candidates, key=lambda candidate: abs(candidate - base))
    return pd.Period(year=year, month=month, freq="M")


def parse_contract_period(token, trade_date):
    if pd.isna(token):
        return pd.NaT
    text = re.sub(r"\.0$", "", str(token).strip())
    compact = re.sub(r"\D", "", text)
    if len(compact) == 6 and 1990 <= int(compact[:4]) <= 2100:
        return pd.Period(year=int(compact[:4]), month=int(compact[4:]),
                         freq="M")
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed.to_period("M")
    return symbol_period(text, trade_date)


def load_csv_export(path):
    raw = pd.read_csv(path)
    date_column = find_column(raw, "date")
    contract_column = find_column(raw, "contract")
    settle_column = find_column(raw, "settle")

    frame = pd.DataFrame({
        "date": pd.to_datetime(raw[date_column], errors="coerce"),
        "contract_raw": raw[contract_column],
        "settle": pd.to_numeric(raw[settle_column], errors="coerce"),
    })
    frame["contract_month"] = [
        parse_contract_period(token, trade_date)
        for token, trade_date in zip(frame["contract_raw"], frame["date"])
    ]
    frame["source_file"] = Path(path).name
    return frame


def excel_column_index(cell_reference):
    """Return a zero-based column index from an Excel cell reference."""
    match = re.fullmatch(r"([A-Z]+)(\d+)", str(cell_reference).upper())
    if not match:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference!r}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def excel_row_index(cell_reference):
    match = re.fullmatch(r"[A-Z]+(\d+)", str(cell_reference).upper())
    if not match:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference!r}")
    return int(match.group(1)) - 1


def workbook_sheet_paths(archive):
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships.findall(
            f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    result = {}
    sheets = workbook.find(f"{{{SPREADSHEET_NS}}}sheets")
    if sheets is None:
        return result
    for sheet in sheets:
        name = sheet.attrib["name"]
        relationship_id = sheet.attrib[f"{{{DOCUMENT_REL_NS}}}id"]
        target = targets[relationship_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        result[name] = target
    return result


def lseg_queries(path):
    """Read LSEG HistoricalPricing query metadata without recalculating Excel."""
    query_pattern = re.compile(
        r'HistoricalPricing\(\s*"([^"]+)"\s*,\s*"([^"]+)"',
        flags=re.IGNORECASE,
    )
    queries = []
    with ZipFile(path) as archive:
        for sheet_name, sheet_path in workbook_sheet_paths(archive).items():
            sheet = ElementTree.fromstring(archive.read(sheet_path))
            for cell in sheet.iter(f"{{{SPREADSHEET_NS}}}c"):
                formula = cell.find(f"{{{SPREADSHEET_NS}}}f")
                if formula is None or not formula.text:
                    continue
                match = query_pattern.search(formula.text)
                if not match:
                    continue
                cell_reference = cell.attrib["r"]
                queries.append(LsegQuery(
                    sheet_name=sheet_name,
                    sheet_path=sheet_path,
                    formula_cell=cell_reference,
                    row_index=excel_row_index(cell_reference),
                    column_index=excel_column_index(cell_reference),
                    ric=match.group(1).strip(),
                    field=match.group(2).strip().upper(),
                ))
    return queries


def load_lseg_xlsx(path):
    """Extract only cached SETTLE arrays from an LSEG Workspace workbook."""
    queries = lseg_queries(path)
    settle_queries = [query for query in queries if query.field == "SETTLE"]
    if not settle_queries:
        fields = sorted({query.field for query in queries})
        raise ValueError(
            f"{path} has no LSEG HistoricalPricing SETTLE query. "
            f"Detected fields: {fields or 'none'}")

    sheet_names = sorted({query.sheet_name for query in settle_queries})
    sheets = pd.read_excel(
        path, sheet_name=sheet_names, header=None, engine="openpyxl")
    rows = []
    skipped = []
    for query in settle_queries:
        sheet = sheets[query.sheet_name]
        date_column = query.column_index
        settle_column = query.column_index + 1
        if settle_column >= sheet.shape[1]:
            skipped.append(
                f"{query.sheet_name}!{query.formula_cell} ({query.ric})")
            continue

        header_date = normalized_name(
            sheet.iloc[query.row_index, date_column])
        header_settle = normalized_name(
            sheet.iloc[query.row_index, settle_column])
        if header_date != "timestamp" or header_settle != "settle":
            # Some legacy workbooks retain failed LSEG formulas alongside
            # valid cached arrays (for example, "The universe is not found").
            # They contain no observations to extract, so report and skip
            # them while preserving the strict Timestamp/SETTLE requirement.
            skipped.append(
                f"{query.sheet_name}!{query.formula_cell} ({query.ric})")
            continue

        values = sheet.iloc[query.row_index + 1:, [
            date_column, settle_column]].copy()
        values.columns = ["date", "settle"]
        values["date"] = pd.to_datetime(values["date"], errors="coerce")
        values["settle"] = pd.to_numeric(values["settle"], errors="coerce")
        values["contract_raw"] = query.ric
        values["source_file"] = Path(path).name
        values["source_sheet"] = query.sheet_name
        values["source_cell"] = query.formula_cell
        rows.append(values)

    if skipped:
        print(f"Skipped {len(skipped)} SETTLE quer{'y' if len(skipped) == 1 else 'ies'} "
              f"without a cached Timestamp/SETTLE table in {Path(path).name}: "
              f"{', '.join(skipped)}.")
    if not rows:
        raise ValueError(
            f"{path} contains SETTLE formulas but no cached "
            "Timestamp/SETTLE observations.")

    frame = pd.concat(rows, ignore_index=True)
    frame["contract_month"] = [
        parse_contract_period(token, trade_date)
        for token, trade_date in zip(frame["contract_raw"], frame["date"])
    ]
    return frame


def load_export(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv_export(path)
    if suffix in {".xlsx", ".xlsm"}:
        return load_lseg_xlsx(path)
    raise ValueError(
        f"Unsupported input format {suffix!r} for {path}; use CSV or XLSX.")


def extract_to_csv(paths, extracted_output=DEFAULT_EXTRACTED_OUTPUT):
    """Write one auditable long-form CSV from the cached Excel arrays."""
    if not paths:
        paths = list(DEFAULT_LSEG_SETTLE_INPUTS)
    combined = pd.concat([load_export(path) for path in paths],
                         ignore_index=True)
    combined = combined.dropna(subset=["date", "contract_month", "settle"])

    columns = [
        "date", "contract_month", "settle", "contract_raw",
        "source_file", "source_sheet", "source_cell",
    ]
    for column in columns:
        if column not in combined:
            combined[column] = ""
    extracted = combined[columns].copy()
    extracted["date"] = extracted["date"].dt.strftime("%Y-%m-%d")
    extracted["contract_month"] = extracted["contract_month"].astype(str)
    extracted = extracted.sort_values(
        ["date", "contract_month", "source_file"])

    extracted_output = Path(extracted_output)
    extracted_output.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_csv(extracted_output, index=False)
    print(f"Extracted {len(extracted):,} cached SETTLE observations from "
          f"{len(paths)} source file(s) to {extracted_output}.")
    return extracted_output


def prepare_from_csv(extracted_csv, output=DEFAULT_OUTPUT):
    """Validate the extracted CSV and write the canonical benchmark CSV."""
    combined = load_csv_export(extracted_csv)
    combined = combined.dropna(subset=["date", "contract_month", "settle"])
    combined = combined[combined["date"] < SAMPLE_END_EXCLUSIVE]

    bad_prices = combined[~combined["settle"].between(90.0, 101.0)]
    if len(bad_prices):
        examples = bad_prices[["source_file", "settle"]].head().to_dict(
            "records")
        raise ValueError(
            "Settle values must be futures prices (normally near 100), "
            f"not interest rates. Examples: {examples}")

    output_frame = (combined.sort_values(["date", "contract_month"])
                    .drop_duplicates(["date", "contract_month"], keep="last")
                    [["date", "contract_month", "settle"]])
    if output_frame.empty:
        raise ValueError("No valid SETTLE observations were found.")
    output_frame["date"] = output_frame["date"].dt.strftime("%Y-%m-%d")
    output_frame["contract_month"] = output_frame["contract_month"].astype(str)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)
    print(f"Wrote {output}: {len(output_frame):,} daily settlements; "
          f"trade dates {output_frame['date'].min()} .. "
          f"{output_frame['date'].max()}.")


def prepare(paths, extracted_output=DEFAULT_EXTRACTED_OUTPUT,
            output=DEFAULT_OUTPUT):
    extracted_csv = extract_to_csv(paths, extracted_output)
    prepare_from_csv(extracted_csv, output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="*", type=Path,
        help=("Licensed futures SETTLE CSV/XLSX export(s). Defaults to the "
              "two SETTLE workbooks in 0_lseg/ffr."))
    parser.add_argument(
        "--extracted-output", type=Path, default=DEFAULT_EXTRACTED_OUTPUT,
        help="Auditable long-form CSV extracted directly from the workbooks.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    prepare(options.inputs, options.extracted_output, options.output)
