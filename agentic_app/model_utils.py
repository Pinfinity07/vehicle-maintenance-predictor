"""ML model utilities — trains the same Decision Tree pipeline from Milestone 1."""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "columns.joblib")

ORDINAL_FEATURES = {
    "Maintenance_History": ["Poor", "Average", "Good"],
    "Tire_Condition": ["Worn Out", "Good", "New"],
    "Brake_Condition": ["Worn Out", "Good", "New"],
    "Battery_Status": ["Weak", "Good", "Strong"],
}

NOMINAL_FEATURES = ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]

NUMERICAL_FEATURES = [
    "Mileage", "Reported_Issues", "Vehicle_Age", "Engine_Size",
    "Odometer_Reading", "Insurance_Premium", "Service_History",
    "Accident_History", "Fuel_Efficiency",
    "Last_Service_Date_days", "Warranty_Expiry_Date_days",
]

VEHICLE_MODELS = ["Truck", "Van", "Bus", "SUV", "Sedan"]
FUEL_TYPES = ["Electric", "Diesel", "Petrol", "Hybrid"]
TRANSMISSION_TYPES = ["Automatic", "Manual"]
OWNER_TYPES = ["First", "Second", "Third"]


# ---------------------------------------------------------------------------
# Synthetic data generation (mirrors the Kaggle dataset schema)
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 8000) -> pd.DataFrame:
    """Create a realistic synthetic dataset matching the vehicle maintenance schema."""
    rng = np.random.RandomState(42)

    data = {
        "Vehicle_Model": rng.choice(VEHICLE_MODELS, n),
        "Mileage": rng.randint(5000, 150000, n),
        "Maintenance_History": rng.choice(["Poor", "Average", "Good"], n, p=[0.25, 0.40, 0.35]),
        "Reported_Issues": rng.randint(0, 6, n),
        "Vehicle_Age": rng.randint(1, 20, n),
        "Fuel_Type": rng.choice(FUEL_TYPES, n),
        "Transmission_Type": rng.choice(TRANSMISSION_TYPES, n),
        "Engine_Size": rng.choice([1000, 1500, 2000, 2500, 3000, 3500], n),
        "Odometer_Reading": rng.randint(5000, 250000, n),
        "Owner_Type": rng.choice(OWNER_TYPES, n, p=[0.5, 0.35, 0.15]),
        "Insurance_Premium": rng.randint(8000, 35000, n),
        "Service_History": rng.randint(0, 15, n),
        "Accident_History": rng.randint(0, 5, n),
        "Fuel_Efficiency": rng.uniform(8.0, 22.0, n),
        "Tire_Condition": rng.choice(["Worn Out", "Good", "New"], n, p=[0.25, 0.45, 0.30]),
        "Brake_Condition": rng.choice(["Worn Out", "Good", "New"], n, p=[0.20, 0.45, 0.35]),
        "Battery_Status": rng.choice(["Weak", "Good", "Strong"], n, p=[0.25, 0.40, 0.35]),
        "Last_Service_Date_days": rng.randint(30, 900, n),
        "Warranty_Expiry_Date_days": rng.randint(-400, 800, n),
    }

    df = pd.DataFrame(data)

    # Deterministic target: vehicles in bad shape need maintenance
    risk_score = np.zeros(n, dtype=float)
    risk_score += (df["Maintenance_History"] == "Poor").astype(float) * 2.0
    risk_score += (df["Tire_Condition"] == "Worn Out").astype(float) * 1.5
    risk_score += (df["Brake_Condition"] == "Worn Out").astype(float) * 1.5
    risk_score += (df["Battery_Status"] == "Weak").astype(float) * 1.0
    risk_score += (df["Reported_Issues"] / 5.0) * 1.5
    risk_score += (df["Vehicle_Age"] / 20.0) * 1.0
    risk_score += (df["Accident_History"] / 5.0) * 1.0
    risk_score += (df["Last_Service_Date_days"] / 900.0) * 1.5
    risk_score += (df["Mileage"] / 150000.0) * 1.0
    risk_score -= (df["Service_History"] / 15.0) * 0.5

    threshold = np.percentile(risk_score, 55)
    noise = rng.normal(0, 0.3, n)
    df["Need_Maintenance"] = ((risk_score + noise) >= threshold).astype(int)

    return df


# ---------------------------------------------------------------------------
# Preprocessing & training
# ---------------------------------------------------------------------------

def build_preprocessor():
    ordinal_transformer = OrdinalEncoder(
        categories=[ORDINAL_FEATURES[k] for k in ORDINAL_FEATURES],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    nominal_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer = RobustScaler()

    return ColumnTransformer(
        transformers=[
            ("ord", ordinal_transformer, list(ORDINAL_FEATURES.keys())),
            ("nom", nominal_transformer, NOMINAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_model(df: pd.DataFrame | None = None):
    """Train Decision Tree with GridSearchCV + SMOTE (same pipeline as Milestone 1)."""
    if df is None:
        df = generate_synthetic_data()

    X = df.drop(columns=["Need_Maintenance"])
    y = df["Need_Maintenance"]
    feature_columns = X.columns.tolist()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "max_depth": [5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train_sm, y_train_sm)
    best_model = grid.best_estimator_

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(feature_columns, COLUMNS_PATH)

    return best_model, preprocessor, feature_columns


def load_model():
    """Load saved model artifacts, or train from scratch if missing."""
    if all(os.path.exists(p) for p in [MODEL_PATH, PREPROCESSOR_PATH, COLUMNS_PATH]):
        return (
            joblib.load(MODEL_PATH),
            joblib.load(PREPROCESSOR_PATH),
            joblib.load(COLUMNS_PATH),
        )
    return train_model()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

_model, _preprocessor, _feature_columns = None, None, None


def _ensure_loaded():
    global _model, _preprocessor, _feature_columns
    if _model is None:
        _model, _preprocessor, _feature_columns = load_model()


def predict(vehicle_data: dict) -> dict:
    """Run maintenance prediction on a single vehicle.

    Returns dict with: prediction, probability, risk_level, feature_importances.
    """
    _ensure_loaded()

    df = pd.DataFrame([vehicle_data])

    # Ensure all expected columns exist
    for col in _feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[_feature_columns]
    X_proc = _preprocessor.transform(df)

    pred = int(_model.predict(X_proc)[0])
    proba = float(_model.predict_proba(X_proc)[0][1])

    if proba >= 0.75:
        risk_level = "CRITICAL"
    elif proba >= 0.50:
        risk_level = "HIGH"
    elif proba >= 0.30:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Top contributing features
    importances = _model.feature_importances_
    proc_feature_names = _preprocessor.get_feature_names_out()
    top_indices = np.argsort(importances)[::-1][:5]
    top_features = [
        {"feature": proc_feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in top_indices
    ]

    return {
        "needs_maintenance": pred,
        "probability": round(proba, 4),
        "risk_level": risk_level,
        "top_features": top_features,
    }


# Sample vehicles for the UI
SAMPLE_VEHICLES = {
    "Truck-001: High Mileage Fleet Truck": {
        "Vehicle_Model": "Truck", "Mileage": 128000, "Maintenance_History": "Poor",
        "Reported_Issues": 4, "Vehicle_Age": 12, "Fuel_Type": "Diesel",
        "Transmission_Type": "Manual", "Engine_Size": 3500, "Odometer_Reading": 210000,
        "Owner_Type": "Second", "Insurance_Premium": 28000, "Service_History": 3,
        "Accident_History": 2, "Fuel_Efficiency": 9.5,
        "Tire_Condition": "Worn Out", "Brake_Condition": "Worn Out",
        "Battery_Status": "Weak",
        "Last_Service_Date_days": 420, "Warranty_Expiry_Date_days": -180,
    },
    "Van-002: Recently Serviced Delivery Van": {
        "Vehicle_Model": "Van", "Mileage": 45000, "Maintenance_History": "Good",
        "Reported_Issues": 0, "Vehicle_Age": 3, "Fuel_Type": "Petrol",
        "Transmission_Type": "Automatic", "Engine_Size": 2000, "Odometer_Reading": 52000,
        "Owner_Type": "First", "Insurance_Premium": 15000, "Service_History": 8,
        "Accident_History": 0, "Fuel_Efficiency": 15.2,
        "Tire_Condition": "New", "Brake_Condition": "New",
        "Battery_Status": "Strong",
        "Last_Service_Date_days": 45, "Warranty_Expiry_Date_days": 500,
    },
    "Bus-003: Aging City Bus": {
        "Vehicle_Model": "Bus", "Mileage": 95000, "Maintenance_History": "Average",
        "Reported_Issues": 2, "Vehicle_Age": 8, "Fuel_Type": "Diesel",
        "Transmission_Type": "Automatic", "Engine_Size": 3000, "Odometer_Reading": 180000,
        "Owner_Type": "Third", "Insurance_Premium": 22000, "Service_History": 6,
        "Accident_History": 1, "Fuel_Efficiency": 11.0,
        "Tire_Condition": "Good", "Brake_Condition": "Worn Out",
        "Battery_Status": "Good",
        "Last_Service_Date_days": 200, "Warranty_Expiry_Date_days": -90,
    },
    "SUV-004: Low Risk Personal Vehicle": {
        "Vehicle_Model": "SUV", "Mileage": 22000, "Maintenance_History": "Good",
        "Reported_Issues": 0, "Vehicle_Age": 2, "Fuel_Type": "Hybrid",
        "Transmission_Type": "Automatic", "Engine_Size": 2000, "Odometer_Reading": 24000,
        "Owner_Type": "First", "Insurance_Premium": 18000, "Service_History": 10,
        "Accident_History": 0, "Fuel_Efficiency": 18.5,
        "Tire_Condition": "New", "Brake_Condition": "New",
        "Battery_Status": "Strong",
        "Last_Service_Date_days": 60, "Warranty_Expiry_Date_days": 700,
    },
}
