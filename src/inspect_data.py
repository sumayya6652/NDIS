import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

files = [
    "participant_budgets.csv",
    "utilisation.csv",
    "active_providers.csv",
    "payments.csv",
    "seifa_lga.xlsx",
    "remoteness_areas_2021.xlsx",
]

def load_any_file(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)

for file in files:
    path = RAW_DIR / file

    if not path.exists():
        print(f"\nMissing file: {path}")
        continue

    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    try:
        df = load_any_file(path)
        print("Shape:", df.shape)
        print("Columns:")
        for col in df.columns:
            print(" -", col)
        print("\nFirst rows:")
        print(df.head())
    except Exception as e:
        print("Error reading file:", e)