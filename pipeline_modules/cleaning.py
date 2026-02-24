import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

def load_and_preprocess_data():
    df = pd.read_csv('./data/raw_dataset.csv')

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    REFERENCE_DATE = pd.Timestamp("2026-02-20")

    for col in ["Last_Service_Date", "Warranty_Expiry_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col + "_days"] = (REFERENCE_DATE - df[col]).dt.days

    df.drop(columns=["Last_Service_Date", "Warranty_Expiry_Date"], inplace=True)

    numerical_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols_raw.remove("Need_Maintenance")

    for col in numerical_cols_raw:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df
