import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

from pipeline_modules.cleaning import load_and_preprocess_data


def encode_and_split():
    df = load_and_preprocess_data()


    # Mutual Information Feature Selection

    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include="object").columns.tolist()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for c in cat_cols:
        df_encoded[c] = le.fit_transform(df_encoded[c].astype(str))

    X_tmp = df_encoded.drop(columns=["Need_Maintenance"])
    y_tmp = df_encoded["Need_Maintenance"]

    mi_scores = mutual_info_classif(
        X_tmp, y_tmp, discrete_features="auto", random_state=42
    )

    mi_series = pd.Series(mi_scores, index=X_tmp.columns)
    zero_mi_features = mi_series[mi_series == 0].index.tolist()

    if zero_mi_features:
        df.drop(columns=zero_mi_features, inplace=True)


    # Feature Categorization

    ordinal_features = {
        "Maintenance_History": ["Poor", "Average", "Good"],
        "Tire_Condition": ["Worn Out", "Good", "New"],
        "Brake_Condition": ["Worn Out", "Good", "New"],
        "Battery_Status": ["Weak", "Good", "Strong"],
    }

    ordinal_features = {
        k: v for k, v in ordinal_features.items() if k in df.columns
    }

    nominal_features = [
        c for c in ["Vehicle_Model", "Fuel_Type",
                    "Transmission_Type", "Owner_Type"]
        if c in df.columns
    ]

    numerical_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != "Need_Maintenance"
    ]


    # Train-Test Split

    X = df.drop(columns=["Need_Maintenance"])
    y = df["Need_Maintenance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Column Transformer

    ordinal_transformer = OrdinalEncoder(
        categories=[ordinal_features[k] for k in ordinal_features],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    nominal_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    numerical_transformer = RobustScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", ordinal_transformer, list(ordinal_features.keys())),
            ("nom", nominal_transformer, nominal_features),
            ("num", numerical_transformer, numerical_features),
        ],
        remainder="drop",
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)


    # SMOTE

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

    return X_train_sm, X_test_proc, y_train_sm, y_test

if __name__ == "__main__":
    X_train_sm, X_test_proc, y_train_sm, y_test = encode_and_split()
    print("Encoding successful.")
    print("Train shape:", X_train_sm.shape)