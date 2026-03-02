"""
Train the Vehicle Maintenance Predictor and save artifacts.

Usage:
    python scripts/train.py

Saves to models/:
    model.joblib, preprocessor.joblib, feature_order.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
from pathlib import Path

# Add project root to path so pipeline/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

from pipeline.cleaning import load_and_preprocess_data

# ── Paths ────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR.mkdir(exist_ok=True)


def train_and_save():
    """Full training pipeline — saves model, preprocessor, and feature order."""
    print("=" * 60)
    print("  VEHICLE MAINTENANCE PREDICTOR — TRAINING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print(f"  ✓ Dataset shape: {df.shape}")

    # 2. Mutual Information feature selection
    print("\n[2/4] Feature selection (Mutual Information)...")
    df_encoded = df.copy()
    le = LabelEncoder()
    for c in df_encoded.select_dtypes(include="object").columns:
        df_encoded[c] = le.fit_transform(df_encoded[c].astype(str))

    X_tmp = df_encoded.drop(columns=["Need_Maintenance"])
    y_tmp = df_encoded["Need_Maintenance"]

    mi_scores = mutual_info_classif(X_tmp, y_tmp, discrete_features="auto", random_state=42)
    mi_series = pd.Series(mi_scores, index=X_tmp.columns)
    zero_mi = mi_series[mi_series == 0].index.tolist()

    if zero_mi:
        print(f"  Dropping zero-MI features: {zero_mi}")
        df.drop(columns=zero_mi, inplace=True)

    # Feature groups
    ordinal_features = {
        "Maintenance_History": ["Poor", "Average", "Good"],
        "Tire_Condition": ["Worn Out", "Good", "New"],
        "Brake_Condition": ["Worn Out", "Good", "New"],
        "Battery_Status": ["Weak", "Good", "Strong"],
    }
    ordinal_features = {k: v for k, v in ordinal_features.items() if k in df.columns}

    nominal_features = [
        c for c in ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]
        if c in df.columns
    ]
    numerical_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != "Need_Maintenance"
    ]

    feature_order = list(ordinal_features.keys()) + nominal_features + numerical_features
    print(f"  ✓ Features selected: {len(feature_order)}")

    # 3. Encode, split, SMOTE, train
    print("\n[3/4] Training model (GridSearchCV + 5-fold CV)...")
    X = df.drop(columns=["Need_Maintenance"])
    y = df["Need_Maintenance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(
                categories=[ordinal_features[k] for k in ordinal_features],
                handle_unknown="use_encoded_value", unknown_value=-1,
            ), list(ordinal_features.keys())),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
            ("num", RobustScaler(), numerical_features),
        ],
        remainder="drop",
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_proc, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid={
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [1, 5, 10, 20],
            "criterion": ["gini", "entropy"],
        },
        cv=cv, scoring="f1", n_jobs=-1, verbose=0,
    )
    dt_grid.fit(X_train_sm, y_train_sm)
    model = dt_grid.best_estimator_

    y_pred = model.predict(X_test_proc)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  ✓ Best params: {dt_grid.best_params_}")
    print(f"  ✓ Test accuracy: {test_acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, digits=4)}")

    # 4. Save artifacts
    print("[4/4] Saving artifacts...")
    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    with open(OUTPUT_DIR / "feature_order.pkl", "wb") as f:
        pickle.dump(feature_order, f)

    print(f"  ✓ model.joblib")
    print(f"  ✓ preprocessor.joblib")
    print(f"  ✓ feature_order.pkl")
    print(f"\n{'=' * 60}")
    print(f"  DONE — artifacts saved to {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_and_save()
