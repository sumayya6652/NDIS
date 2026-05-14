

"""
Fixed NDIS data preparation pipeline.

This script reads the four public NDIS CSV/Excel files, cleans the key columns,
normalises support-class names, avoids duplicate ALL rows, and creates:
    data/processed/ndis_model_data.csv

Put the raw files in data/raw/ using either the original file names or the short
names shown in FILE_CANDIDATES below, then run:
    python fixed_ndis_pipeline.py
"""

from pathlib import Path
from functools import reduce
import re

import numpy as np
import pandas as pd

# Works whether you run from project root with: python src/prepare_data.py
# or run this file directly.
PROJECT_DIR = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path.cwd()

RAW_DIRS = [
    PROJECT_DIR / "data" / "raw",
    PROJECT_DIR / "raw",
    Path.cwd() / "data" / "raw",
    Path.cwd() / "raw",
    Path.cwd(),
]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

VALID_STATES = {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"}
KEYS = ["state", "age_group", "disability_type", "support_class"]

FILE_CANDIDATES = {
    "participants": [
        "participant_budgets.csv",
        "participant_budget.csv",
        "participant_budget*.csv",
        "Participant numbers and plan budgets data December 2025.csv",
        "Participant numbers and plan budgets data December 2025.xlsx",
    ],
    "utilisation": [
        "utilisation.csv",
        "Utilisation of Plan budgets data December 2025.csv",
        "Utilisation of Plan budgets data December 2025.xlsx",
    ],
    "providers": [
        "active_providers.csv",
        "Active providers data as at 31 December 2025.csv",
        "Active providers data as at 31 December 2025.xlsx",
    ],
    "payments": [
        "payments.csv",
        "Payments data December 2025.csv",
        "Payments data December 2025.xlsx",
    ],
}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column matching reliable across NDIS files."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percentage", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def find_existing_file(candidates: list[str]) -> Path:
    searched = []
    for directory in RAW_DIRS:
        if not directory.exists():
            searched.append(str(directory))
            continue
        for name in candidates:
            # Exact filename match
            path = directory / name
            if path.exists():
                return path

            # Wildcard match, e.g. participant_budget*.csv
            matches = sorted(directory.glob(name))
            if matches:
                return matches[0]
        searched.append(str(directory))

    raise FileNotFoundError(
        "Could not find the required raw file.\n"
        "Folders checked:\n"
        + "\n".join(f"  - {folder}" for folder in searched)
        + "\n\nCandidate names checked:\n"
        + "\n".join(f"  - {name}" for name in candidates)
    )


def load_file(kind: str) -> pd.DataFrame:
    path = find_existing_file(FILE_CANDIDATES[kind])
    print(f"Loaded {kind}: {path}")

    if path.suffix.lower() == ".csv":
        return clean_columns(pd.read_csv(path, dtype=str, low_memory=False))

    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        best_sheet, best_df = max(sheets.items(), key=lambda item: len(item[1]))
        print(f"  Using Excel sheet: {best_sheet}")
        return clean_columns(best_df)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_column(df: pd.DataFrame, possible_names: list[str], required: bool = True) -> str | None:
    """Find a column by exact cleaned name first, then by contains matching."""
    cols = list(df.columns)
    options = [name.lower().strip() for name in possible_names]

    for name in options:
        if name in cols:
            return name

    compact_cols = {re.sub(r"[^a-z0-9]", "", c): c for c in cols}
    for name in options:
        compact_name = re.sub(r"[^a-z0-9]", "", name)
        if compact_name in compact_cols:
            return compact_cols[compact_name]

    for name in options:
        compact_name = re.sub(r"[^a-z0-9]", "", name)
        for compact_col, original_col in compact_cols.items():
            if compact_name in compact_col or compact_col in compact_name:
                return original_col

    if required:
        raise KeyError(f"Could not find required column. Tried: {possible_names}. Available: {cols}")
    return None


def parse_number(series: pd.Series, suppressed_strategy: str = "upper_bound") -> pd.Series:
    """
    Convert strings like '23,000.00', '$1,000', '67%', '<11', '<38' to numeric.

    Public NDIS files suppress small cells using values such as '<11'. The exact
    value is not recoverable. upper_bound converts '<11' to 10 and '<38' to 37,
    which avoids fake constant 0/1 values while staying inside the published range.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
    )

    def convert_suppressed(value: str) -> str:
        match = re.match(r"^<\s*(\d+(?:\.\d+)?)$", value)
        if not match:
            return value
        limit = float(match.group(1))
        if suppressed_strategy == "midpoint":
            return str(limit / 2)
        return str(max(limit - 1, 0))

    cleaned = cleaned.map(convert_suppressed)
    cleaned = cleaned.replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
        "none": np.nan,
        "NULL": np.nan,
        "null": np.nan,
        "n/a": np.nan,
        "N/A": np.nan,
        "ALL": np.nan,
        "All": np.nan,
        "all": np.nan,
        "Missing": np.nan,
        "missing": np.nan,
        "SuppClass_Missing": np.nan,
        "suppclass_missing": np.nan,
        "State_Missing": np.nan,
        "state_missing": np.nan,
        "-": np.nan,
    })
    return pd.to_numeric(cleaned, errors="coerce")


def standardise_state(value) -> str | float:
    text = str(value).strip().upper()
    long_names = {
        "NEW SOUTH WALES": "NSW",
        "VICTORIA": "VIC",
        "QUEENSLAND": "QLD",
        "SOUTH AUSTRALIA": "SA",
        "WESTERN AUSTRALIA": "WA",
        "TASMANIA": "TAS",
        "AUSTRALIAN CAPITAL TERRITORY": "ACT",
        "NORTHERN TERRITORY": "NT",
    }
    text = long_names.get(text, text)
    return text if text in VALID_STATES else np.nan


def standardise_age(value) -> str | float:
    text = str(value).strip()
    invalid = {"", "nan", "none", "all", "missing", "agebnd_missing", "age_band_missing"}
    return np.nan if text.lower() in invalid else text


def standardise_disability(value) -> str | float:
    text = str(value).strip()
    low = text.lower()
    invalid = {"", "nan", "none", "all", "missing", "dsbltygrp_missing", "not stated"}
    if low in invalid:
        return np.nan
    if low == "abi":
        return "ABI"
    return text.title().replace("And", "and")


def standardise_support_class(value) -> str | float:
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)
    mapping = {
        "capacitybuilding": "Capacity Building",
        "capital": "Capital",
        "core": "Core",
    }
    return mapping.get(compact, np.nan)


def is_all(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().eq("ALL")


def weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.any():
        return np.average(values[mask], weights=weights[mask])
    return values.mean()


def print_mapping(name: str, mapping: dict[str, str | None]) -> None:
    print(f"\nColumn matches for {name}:")
    for target, source in mapping.items():
        print(f"  {target:28} <- {source}")


def clean_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["state"] = df["state"].map(standardise_state)
    df["age_group"] = df["age_group"].map(standardise_age)
    df["disability_type"] = df["disability_type"].map(standardise_disability)
    df["support_class"] = df["support_class"].map(standardise_support_class)
    df = df.dropna(subset=KEYS)
    return df


def load_participants() -> pd.DataFrame:
    raw = load_file("participants")
    mapping = {
        "state": find_column(raw, ["statecd", "state", "participant_state", "prtcptstatnm"]),
        "service_district": find_column(raw, ["srvcdstrctnm", "service_district", "rsdsinsrvcdstrctnm"], required=False),
        "disability_type": find_column(raw, ["dsbltygrpnm", "ndisdsbltygrpnm", "disability_type", "disability_group"]),
        "age_group": find_column(raw, ["agebnd", "ndiaagebnd", "age_group", "age_band"]),
        "support_class": find_column(raw, ["suppclass", "support_class", "support_type"]),
        "average_support_budget": find_column(raw, ["avganlsdcmtdsuppbdgt", "average_support_budget", "avg_budget"]),
        "participant_count": find_column(raw, ["actvprtcpnt", "actvprtcptnt", "countofparticipants", "participant_count"]),
    }
    print_mapping("participants", mapping)

    df = pd.DataFrame({
        "state": raw[mapping["state"]],
        "service_district": raw[mapping["service_district"]] if mapping["service_district"] else "ALL",
        "disability_type": raw[mapping["disability_type"]],
        "age_group": raw[mapping["age_group"]],
        "support_class": raw[mapping["support_class"]],
        "participant_count": parse_number(raw[mapping["participant_count"]]),
        "average_support_budget": parse_number(raw[mapping["average_support_budget"]]),
    })
    df = clean_key_columns(df)

    # These files contain both state-level rows (service_district = ALL) and district rows.
    # Use state-level rows to avoid double-counting districts.
    all_geo = df[is_all(df["service_district"])]
    if not all_geo.empty:
        df = all_geo.copy()

    grouped = (
        df.groupby(KEYS, dropna=False)
        .apply(lambda g: pd.Series({
            "participant_count": g["participant_count"].sum(min_count=1),
            "average_support_budget": weighted_mean(g, "average_support_budget", "participant_count"),
        }), include_groups=False)
        .reset_index()
    )
    return grouped


def load_utilisation() -> pd.DataFrame:
    raw = load_file("utilisation")
    mapping = {
        "state": find_column(raw, ["statecd", "state", "participant_state", "prtcptstatnm"]),
        "service_district": find_column(raw, ["srvcdstrctnm", "service_district", "rsdsinsrvcdstrctnm"], required=False),
        "disability_type": find_column(raw, ["dsbltygrpnm", "ndisdsbltygrpnm", "disability_type", "disability_group"]),
        "age_group": find_column(raw, ["agebnd", "ndiaagebnd", "age_group", "age_band"]),
        "silor_sda": find_column(raw, ["silorsda", "sil_or_sda"], required=False),
        "support_class": find_column(raw, ["suppclass", "support_class", "support_type"]),
        "utilisation_rate": find_column(raw, ["utlstn", "utlsn", "utilisation_rate", "utilisation"]),
    }
    print_mapping("utilisation", mapping)

    df = pd.DataFrame({
        "state": raw[mapping["state"]],
        "service_district": raw[mapping["service_district"]] if mapping["service_district"] else "ALL",
        "disability_type": raw[mapping["disability_type"]],
        "age_group": raw[mapping["age_group"]],
        "silor_sda": raw[mapping["silor_sda"]] if mapping["silor_sda"] else "ALL",
        "support_class": raw[mapping["support_class"]],
        "utilisation_rate": parse_number(raw[mapping["utilisation_rate"]]),
    })
    df = clean_key_columns(df)

    # Keep total SIL/SDA rows only; otherwise Yes + No + ALL creates duplicates.
    df = df[is_all(df["silor_sda"])]

    # Prefer state-level rows to avoid double-counting service districts.
    all_geo = df[is_all(df["service_district"])]
    if not all_geo.empty:
        df = all_geo.copy()

    grouped = (
        df.groupby(KEYS, dropna=False, as_index=False)
        .agg(utilisation_rate=("utilisation_rate", "mean"))
    )
    return grouped


def load_providers() -> pd.DataFrame:
    raw = load_file("providers")
    mapping = {
        "state": find_column(raw, ["statecd", "state", "provider_state", "prvdrstatnm"]),
        "service_district": find_column(raw, ["srvcdstrctnm", "service_district"], required=False),
        "disability_type": find_column(raw, ["dsbltygrpnm", "ndisdsbltygrpnm", "disability_type", "disability_group"]),
        "age_group": find_column(raw, ["agebnd", "ndiaagebnd", "age_group", "age_band"]),
        "support_class": find_column(raw, ["suppclass", "support_class", "support_type"]),
        "active_provider_count": find_column(raw, ["prvdrcnt", "active_provider_count", "active_providers", "provider_count"]),
    }
    print_mapping("providers", mapping)

    df = pd.DataFrame({
        "state": raw[mapping["state"]],
        "service_district": raw[mapping["service_district"]] if mapping["service_district"] else "ALL",
        "disability_type": raw[mapping["disability_type"]],
        "age_group": raw[mapping["age_group"]],
        "support_class": raw[mapping["support_class"]],
        "active_provider_count": parse_number(raw[mapping["active_provider_count"]]),
    })
    df = clean_key_columns(df)

    # Prefer state-level rows to avoid double-counting districts.
    all_geo = df[is_all(df["service_district"])]
    if not all_geo.empty:
        df = all_geo.copy()

    grouped = (
        df.groupby(KEYS, dropna=False, as_index=False)
        .agg(active_provider_count=("active_provider_count", lambda s: s.sum(min_count=1)))
    )
    return grouped


def load_payments() -> pd.DataFrame:
    raw = load_file("payments")
    mapping = {
        "state": find_column(raw, ["rsdsinstatecd", "statecd", "state", "participant_state", "prtcptstatnm"]),
        "service_district": find_column(raw, ["rsdsinsrvcdstrctnm", "srvcdstrctnm", "service_district"], required=False),
        "disability_type": find_column(raw, ["ndisdsbltygrpnm", "dsbltygrpnm", "disability_type", "disability_group"]),
        "age_group": find_column(raw, ["ndiaagebnd", "agebnd", "age_group", "age_band"]),
        "support_class": find_column(raw, ["suppclass", "support_class", "support_type"]),
        "support_category": find_column(raw, ["suppcatnm", "support_category"], required=False),
        "support_item_number": find_column(raw, ["suppitemnmbr", "support_item_number"], required=False),
        "support_item_desc": find_column(raw, ["suppitemdesc", "support_item_desc"], required=False),
        "payment_amount": find_column(raw, ["pmtamt", "payment_amount", "amount_paid", "total_payments"]),
        "payment_participant_count": find_column(raw, ["countofparticipants", "participant_count", "participants"], required=False),
    }
    print_mapping("payments", mapping)

    df = pd.DataFrame({
        "state": raw[mapping["state"]],
        "service_district": raw[mapping["service_district"]] if mapping["service_district"] else "",
        "disability_type": raw[mapping["disability_type"]],
        "age_group": raw[mapping["age_group"]],
        "support_class": raw[mapping["support_class"]],
        "support_category": raw[mapping["support_category"]] if mapping["support_category"] else "ALL",
        "support_item_number": raw[mapping["support_item_number"]] if mapping["support_item_number"] else "ALL",
        "support_item_desc": raw[mapping["support_item_desc"]] if mapping["support_item_desc"] else "ALL",
        "payment_amount": parse_number(raw[mapping["payment_amount"]]),
        "payment_participant_count": parse_number(raw[mapping["payment_participant_count"]]) if mapping["payment_participant_count"] else np.nan,
    })
    df = clean_key_columns(df)

    # The payment file also has support-category and support-item breakdowns.
    # Keep only the ALL/ALL/ALL payment rows so item/category rows are not added again.
    df = df[
        is_all(df["support_category"])
        & is_all(df["support_item_number"])
        & is_all(df["support_item_desc"])
    ]

    grouped = (
        df.groupby(KEYS, dropna=False, as_index=False)
        .agg(
            payment_amount=("payment_amount", lambda s: s.sum(min_count=1)),
            payment_participant_count=("payment_participant_count", lambda s: s.sum(min_count=1)),
        )
    )
    return grouped


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "payment_amount",
    ]:
        df[f"{col}_was_missing"] = df[col].isna()
    return df


def fill_groupwise(df: pd.DataFrame, column: str, groups: list[list[str]], default: float) -> pd.Series:
    """Fill missing numeric values using progressively broader group medians."""
    result = pd.to_numeric(df[column], errors="coerce")
    for group_cols in groups:
        medians = df.groupby(group_cols)[column].transform("median")
        result = result.fillna(medians)
    return result.fillna(default)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "payment_amount",
        "payment_participant_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_missingness_flags(df)

    # Do not fill everything with fake constants like 1 or 50.
    # Counts/payments stay based on source data, with limited fallback only where needed.
    df["participant_count"] = df["participant_count"].fillna(df.get("payment_participant_count"))
    df["participant_count"] = fill_groupwise(
        df,
        "participant_count",
        groups=[["state", "age_group", "disability_type"], ["state", "age_group"], ["state"]],
        default=0,
    ).round(0)

    df["active_provider_count"] = fill_groupwise(
        df,
        "active_provider_count",
        groups=[["state", "age_group", "disability_type"], ["state", "support_class"], ["state"]],
        default=0,
    ).round(0)

    df["average_support_budget"] = fill_groupwise(
        df,
        "average_support_budget",
        groups=[["state", "age_group", "disability_type"], ["state", "support_class"], ["state"]],
        default=0,
    )

    df["utilisation_rate"] = fill_groupwise(
        df,
        "utilisation_rate",
        groups=[["state", "age_group", "disability_type"], ["state", "support_class"], ["state"]],
        default=df["utilisation_rate"].median() if df["utilisation_rate"].notna().any() else 0,
    ).clip(0, 100)

    # Payment Amount should not be median-imputed. If no payment is published for a row,
    # use 0 for the numeric model feature and keep the *_was_missing flag for transparency.
    df["payment_amount"] = df["payment_amount"].fillna(0)

    safe_provider_denominator = df["active_provider_count"].replace(0, np.nan)
    df["participant_to_provider_ratio"] = df["participant_count"] / safe_provider_denominator

    # If there are no providers in the published row, use participant_count as a high-pressure ratio
    # rather than dividing by 1 and hiding that provider_count was zero/missing.
    df["participant_to_provider_ratio"] = df["participant_to_provider_ratio"].fillna(df["participant_count"])

    df["utilisation_gap"] = 100 - df["utilisation_rate"]

    safe_participant_denominator = df["participant_count"].replace(0, np.nan)
    df["payment_per_participant"] = df["payment_amount"] / safe_participant_denominator
    df["payment_per_participant"] = df["payment_per_participant"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Normalised score: higher means more access pressure.
    df["access_gap_score"] = (
        df["participant_to_provider_ratio"].rank(pct=True) * 0.45
        + df["utilisation_gap"].rank(pct=True) * 0.35
        + df["average_support_budget"].rank(pct=True) * 0.20
    )

    try:
        df["access_risk"] = pd.qcut(
            df["access_gap_score"],
            q=3,
            labels=["Low Risk", "Medium Risk", "High Risk"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        df["access_risk"] = "Medium Risk"

    # Cleaner display order for Streamlit data explorer.
    display_cols = [
        "state",
        "age_group",
        "disability_type",
        "support_class",
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "payment_amount",
        "participant_to_provider_ratio",
        "utilisation_gap",
        "payment_per_participant",
        "access_gap_score",
        "access_risk",
    ]
    # Hide internal missingness/debug columns from the final CSV/Data Explorer.
    flag_cols = [c for c in df.columns if c.endswith("_was_missing")]
    extra_cols = [c for c in df.columns if c not in display_cols + flag_cols]
    df = df[display_cols + extra_cols]

    # Friendly formatting while keeping numeric columns numeric.
    for col in [
        "participant_count",
        "active_provider_count",
        "average_support_budget",
        "payment_amount",
        "payment_per_participant",
    ]:
        df[col] = df[col].round(0).astype("Int64")

    df["utilisation_rate"] = df["utilisation_rate"].round(2)
    df["participant_to_provider_ratio"] = df["participant_to_provider_ratio"].round(4)
    df["utilisation_gap"] = df["utilisation_gap"].round(2)
    df["access_gap_score"] = df["access_gap_score"].round(4)

    return df.sort_values(KEYS).reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    print("\nLoading and cleaning source datasets...")
    participants = load_participants()
    utilisation = load_utilisation()
    providers = load_providers()
    payments = load_payments()

    print("\nCleaned source shapes:")
    print(f"  participants: {participants.shape}")
    print(f"  utilisation:  {utilisation.shape}")
    print(f"  providers:    {providers.shape}")
    print(f"  payments:     {payments.shape}")

    dataframes = [participants, utilisation, providers, payments]
    df = reduce(lambda left, right: left.merge(right, on=KEYS, how="outer", validate="one_to_one"), dataframes)
    df = create_features(df)
    return df


def main() -> None:
    df = build_dataset()
    output_path = PROCESSED_DIR / "ndis_model_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Final shape: {df.shape}")
    print("\nQuick QA:")
    print("  states:", sorted(df["state"].dropna().unique().tolist()))
    print("  support classes:", sorted(df["support_class"].dropna().unique().tolist()))
    print("  participant_count unique values:", df["participant_count"].nunique())
    print("  active_provider_count unique values:", df["active_provider_count"].nunique())
    print("  payment_amount unique values:", df["payment_amount"].nunique())
    print("  utilisation_rate unique values:", df["utilisation_rate"].nunique())
    print("\nSample rows:")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
