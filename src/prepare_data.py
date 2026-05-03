import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percentage", regex=False)
        .str.replace("__", "_")
    )
    return df


def to_number(series):
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    cleaned = cleaned.replace({
        "nan": np.nan,
        "None": np.nan,
        "none": np.nan,
        "": np.nan,
        "-": np.nan,
        "suppressed": np.nan,
        "Suppressed": np.nan,
        "n/a": np.nan,
        "N/A": np.nan
    })

    return pd.to_numeric(cleaned, errors="coerce")


def load_any_file(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        print(f"Loaded CSV: {path.name}")
        return clean_columns(pd.read_csv(path))

    if path.suffix.lower() in [".xlsx", ".xls"]:
        print(f"Loaded Excel: {path.name}")
        sheets = pd.read_excel(path, sheet_name=None)

        best_sheet = None
        best_df = None
        best_rows = 0

        for sheet_name, sheet_df in sheets.items():
            if sheet_df is not None and len(sheet_df) > best_rows:
                best_sheet = sheet_name
                best_df = sheet_df
                best_rows = len(sheet_df)

        print(f"Using sheet: {best_sheet}")
        return clean_columns(best_df)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_column(df, possible_names):
    cols = list(df.columns)
    possible_names = [name.lower() for name in possible_names]

    for name in possible_names:
        for col in cols:
            if name == col:
                return col

    for name in possible_names:
        for col in cols:
            if name in col:
                return col

    return None


def print_column_matches(dataset_name, mapping):
    print(f"\nColumn matches for {dataset_name}:")
    for new_col, old_col in mapping.items():
        print(f"{new_col:30} <- {old_col}")


def remove_total_rows(df):
    df = df.copy()

    bad_values = [
        "total",
        "all",
        "australia",
        "grand total",
        "unknown total",
        "not stated total"
    ]

    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df = df[
            ~df[col].astype(str).str.strip().str.lower().isin(bad_values)
        ]

    return df


def disability_column_names():
    return [
        "dsbltygrpnm",
        "dsblty_grp_nm",
        "disability_group_name",
        "disability_group",
        "primary_disability",
        "primary_disability_group",
        "disability_type",
        "disability"
    ]


def standardise_participants(df):
    mapping = {
        "state": find_column(df, [
            "state",
            "state_territory",
            "jurisdiction",
            "participant_state",
            "prtcptstatnm"
        ]),
        "age_group": find_column(df, [
            "age_group",
            "participant_age",
            "participant_age_band",
            "age_band",
            "age",
            "agebandnm",
            "age_bnd_nm"
        ]),
        "disability_type": find_column(df, disability_column_names()),
        "support_class": find_column(df, [
            "support_class",
            "support_category",
            "support_type",
            "support",
            "sprtclsnm",
            "support_class_name"
        ]),
        "participant_count": find_column(df, [
            "participant_count",
            "active_participants",
            "participants",
            "number_of_participants",
            "participant",
            "prtcptcnt",
            "participant_cnt"
        ]),
        "average_support_budget": find_column(df, [
            "average_support_budget",
            "average_annualised_committed_support",
            "average_committed",
            "committed_support",
            "budget",
            "average_budget",
            "avg_budget",
            "avganlzdcmtdsprt",
            "avg_anlzd_cmtd_sprt"
        ]),
    }

    print_column_matches("participant_budgets.csv", mapping)

    selected = {}
    for new_col, old_col in mapping.items():
        if old_col is not None:
            selected[new_col] = df[old_col]

    out = pd.DataFrame(selected)

    for col in ["state", "age_group", "disability_type", "support_class"]:
        if col not in out.columns:
            out[col] = "Unknown"

    if "participant_count" not in out.columns:
        out["participant_count"] = 1

    if "average_support_budget" not in out.columns:
        out["average_support_budget"] = np.nan

    out["participant_count"] = to_number(out["participant_count"])
    out["average_support_budget"] = to_number(out["average_support_budget"])

    return out


def standardise_utilisation(df):
    mapping = {
        "state": find_column(df, [
            "state",
            "state_territory",
            "jurisdiction",
            "participant_state",
            "prtcptstatnm"
        ]),
        "age_group": find_column(df, [
            "age_group",
            "participant_age",
            "participant_age_band",
            "age_band",
            "age",
            "agebandnm",
            "age_bnd_nm"
        ]),
        "disability_type": find_column(df, disability_column_names()),
        "support_class": find_column(df, [
            "support_class",
            "support_category",
            "support_type",
            "support",
            "sprtclsnm",
            "support_class_name"
        ]),
        "utilisation_rate": find_column(df, [
            "utilisation_rate",
            "utilisation_percentage",
            "utilisation",
            "utilisation_of_plan_budgets",
            "percentage_utilised",
            "utlsnrt",
            "utlsn_rate",
            "utilisation_rate_percentage"
        ]),
    }

    print_column_matches("utilisation.csv", mapping)

    selected = {}
    for new_col, old_col in mapping.items():
        if old_col is not None:
            selected[new_col] = df[old_col]

    out = pd.DataFrame(selected)

    for col in ["state", "age_group", "disability_type", "support_class"]:
        if col not in out.columns:
            out[col] = "Unknown"

    if "utilisation_rate" not in out.columns:
        out["utilisation_rate"] = np.nan

    out["utilisation_rate"] = to_number(out["utilisation_rate"])

    return out


def standardise_providers(df):
    mapping = {
        "state": find_column(df, [
            "state",
            "state_territory",
            "jurisdiction",
            "provider_state",
            "prvdrstatnm"
        ]),
        "age_group": find_column(df, [
            "age_group",
            "participant_age",
            "participant_age_band",
            "age_band",
            "age",
            "agebandnm",
            "age_bnd_nm"
        ]),
        "disability_type": find_column(df, disability_column_names()),
        "support_class": find_column(df, [
            "support_class",
            "support_category",
            "support_type",
            "support",
            "sprtclsnm",
            "support_class_name"
        ]),
        "active_provider_count": find_column(df, [
            "active_provider_count",
            "active_providers",
            "providers",
            "provider_count",
            "number_of_providers",
            "prvdrcnt",
            "provider_cnt"
        ]),
    }

    print_column_matches("active_providers.csv", mapping)

    selected = {}
    for new_col, old_col in mapping.items():
        if old_col is not None:
            selected[new_col] = df[old_col]

    out = pd.DataFrame(selected)

    for col in ["state", "age_group", "disability_type", "support_class"]:
        if col not in out.columns:
            out[col] = "Unknown"

    if "active_provider_count" not in out.columns:
        out["active_provider_count"] = 1

    out["active_provider_count"] = to_number(out["active_provider_count"])

    return out


def standardise_payments(df):
    mapping = {
        "state": find_column(df, [
            "state",
            "state_territory",
            "jurisdiction",
            "participant_state",
            "prtcptstatnm"
        ]),
        "age_group": find_column(df, [
            "age_group",
            "participant_age",
            "participant_age_band",
            "age_band",
            "age",
            "agebandnm",
            "age_bnd_nm"
        ]),
        "disability_type": find_column(df, disability_column_names()),
        "support_class": find_column(df, [
            "support_class",
            "support_category",
            "support_type",
            "support",
            "sprtclsnm",
            "support_class_name"
        ]),
        "payment_amount": find_column(df, [
            "payment_amount",
            "total_amount_paid",
            "total_payments",
            "amount_paid",
            "payments",
            "total_paid",
            "pymtamt",
            "payment_amt"
        ]),
    }

    print_column_matches("payments.csv", mapping)

    selected = {}
    for new_col, old_col in mapping.items():
        if old_col is not None:
            selected[new_col] = df[old_col]

    out = pd.DataFrame(selected)

    for col in ["state", "age_group", "disability_type", "support_class"]:
        if col not in out.columns:
            out[col] = "Unknown"

    if "payment_amount" not in out.columns:
        out["payment_amount"] = np.nan

    out["payment_amount"] = to_number(out["payment_amount"])

    return out


def aggregate_participants(df):
    keys = ["state", "age_group", "disability_type", "support_class"]
    df = remove_total_rows(df)

    return df.groupby(keys, as_index=False).agg({
        "participant_count": "sum",
        "average_support_budget": "mean"
    })


def aggregate_utilisation(df):
    keys = ["state", "age_group", "disability_type", "support_class"]
    df = remove_total_rows(df)

    return df.groupby(keys, as_index=False).agg({
        "utilisation_rate": "mean"
    })


def aggregate_providers(df):
    keys = ["state", "age_group", "disability_type", "support_class"]
    df = remove_total_rows(df)

    return df.groupby(keys, as_index=False).agg({
        "active_provider_count": "sum"
    })


def aggregate_payments(df):
    keys = ["state", "age_group", "disability_type", "support_class"]
    df = remove_total_rows(df)

    return df.groupby(keys, as_index=False).agg({
        "payment_amount": "sum"
    })


def create_features(df):
    df = df.copy()

    numeric_cols = [
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "payment_amount"
    ]

    for col in numeric_cols:
        df[col] = to_number(df[col])

    df["participant_count"] = df["participant_count"].fillna(0)
    df["active_provider_count"] = df["active_provider_count"].fillna(1)
    df["active_provider_count"] = df["active_provider_count"].replace(0, 1)

    if df["average_support_budget"].notna().sum() == 0:
        df["average_support_budget"] = 0
    else:
        df["average_support_budget"] = df["average_support_budget"].fillna(
            df["average_support_budget"].median()
        )

    if df["utilisation_rate"].notna().sum() == 0:
        df["utilisation_rate"] = 50
    else:
        df["utilisation_rate"] = df["utilisation_rate"].fillna(
            df["utilisation_rate"].median()
        )

    if df["payment_amount"].notna().sum() == 0:
        df["payment_amount"] = 0
    else:
        df["payment_amount"] = df["payment_amount"].fillna(
            df["payment_amount"].median()
        )

    df["participant_to_provider_ratio"] = (
        df["participant_count"] / df["active_provider_count"]
    )

    df["utilisation_gap"] = 100 - df["utilisation_rate"]

    df["payment_per_participant"] = (
        df["payment_amount"] / df["participant_count"].replace(0, 1)
    )

    df["payment_per_participant"] = df["payment_per_participant"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    if df["payment_per_participant"].notna().sum() == 0:
        df["payment_per_participant"] = 0
    else:
        df["payment_per_participant"] = df["payment_per_participant"].fillna(
            df["payment_per_participant"].median()
        )

    df["access_gap_score"] = (
        (df["participant_to_provider_ratio"].rank(pct=True) * 0.45)
        + (df["utilisation_gap"].rank(pct=True) * 0.35)
        + (df["average_support_budget"].rank(pct=True) * 0.20)
    )

    df["access_risk"] = pd.qcut(
        df["access_gap_score"],
        q=3,
        labels=["Low Risk", "Medium Risk", "High Risk"],
        duplicates="drop"
    )

    df["access_risk"] = df["access_risk"].astype(str)

    return df


def main():
    participant_path = RAW_DIR / "participant_budgets.csv"
    utilisation_path = RAW_DIR / "utilisation.csv"
    provider_path = RAW_DIR / "active_providers.csv"
    payments_path = RAW_DIR / "payments.csv"

    print("\nLoading raw datasets...")

    participants_raw = load_any_file(participant_path)
    utilisation_raw = load_any_file(utilisation_path)
    providers_raw = load_any_file(provider_path)
    payments_raw = load_any_file(payments_path)

    print("\nStandardising datasets...")

    participants = standardise_participants(participants_raw)
    utilisation = standardise_utilisation(utilisation_raw)
    providers = standardise_providers(providers_raw)
    payments = standardise_payments(payments_raw)

    print("\nBefore aggregation:")
    print("Participants:", participants.shape)
    print("Utilisation:", utilisation.shape)
    print("Providers:", providers.shape)
    print("Payments:", payments.shape)

    participants = aggregate_participants(participants)
    utilisation = aggregate_utilisation(utilisation)
    providers = aggregate_providers(providers)
    payments = aggregate_payments(payments)

    print("\nAfter aggregation:")
    print("Participants:", participants.shape)
    print("Utilisation:", utilisation.shape)
    print("Providers:", providers.shape)
    print("Payments:", payments.shape)

    merge_keys = ["state", "age_group", "disability_type", "support_class"]

    print("\nMerging datasets...")

    df = participants.merge(
        utilisation,
        on=merge_keys,
        how="left",
        validate="one_to_one"
    )

    df = df.merge(
        providers,
        on=merge_keys,
        how="left",
        validate="one_to_one"
    )

    df = df.merge(
        payments,
        on=merge_keys,
        how="left",
        validate="one_to_one"
    )

    print("Merged shape:", df.shape)

    print("\nCreating features...")

    df = create_features(df)
    df = df.drop_duplicates()

    output_path = PROCESSED_DIR / "ndis_model_data.csv"
    df.to_csv(output_path, index=False)

    print("\nProcessed dataset saved successfully.")
    print("Output file:", output_path)
    print("Final shape:", df.shape)

    print("\nPreview:")
    print(df.head())

    print("\nDisability type distribution:")
    print(df["disability_type"].value_counts().head(20))

    print("\nAccess risk distribution:")
    print(df["access_risk"].value_counts())

    print("\nColumns in final processed dataset:")
    for col in df.columns:
        print(" -", col)


if __name__ == "__main__":
    main()