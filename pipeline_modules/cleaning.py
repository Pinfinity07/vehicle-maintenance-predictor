import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path


def get_data_path():
    """Get the path to raw dataset, handling different directory structures."""
    possible_paths = [
        Path(__file__).parent.parent / "data" / "raw_dataset.csv",
        Path.cwd() / "data" / "raw_dataset.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(f"Could not find raw_dataset.csv. Tried: {possible_paths}")


def load_and_preprocess_data():
    """Load and preprocess vehicle maintenance data."""
    data_path = get_data_path()
    df = pd.read_csv(data_path)

    # Remove duplicates and null values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Date engineering
    REFERENCE_DATE = pd.Timestamp("2026-02-20")
    for col in ["Last_Service_Date", "Warranty_Expiry_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col + "_days"] = (REFERENCE_DATE - df[col]).dt.days

    # Drop original date columns
    date_cols = [col for col in ["Last_Service_Date", "Warranty_Expiry_Date"] if col in df.columns]
    df.drop(columns=date_cols, inplace=True)

    # IQR-based outlier clipping
    numerical_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Need_Maintenance" in numerical_cols_raw:
        numerical_cols_raw.remove("Need_Maintenance")

    for col in numerical_cols_raw:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


if __name__ == "__main__":
    df = load_and_preprocess_data()
    print(f"✓ Data loaded and preprocessed")
    print(f"  Shape: {df.shape}")
