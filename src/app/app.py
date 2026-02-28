"""Gradio app for Vehicle Maintenance Predictor"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import gradio as gr

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.cleaning import load_and_preprocess_data, get_data_path


def initialize_model():
    """Initialize and train the model."""
    print("Initializing model...")
    
    # Load and clean data
    df = load_and_preprocess_data()
    
    # Mutual Information Feature Selection
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
    
    # Train-test split
    X = df.drop(columns=["Need_Maintenance"])
    y = df["Need_Maintenance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing
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
    
    # SMOTE
    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_proc, y_train)
    
    # Train model
    print("Training model… (this takes ~30 s)")
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
    print(f"Best params: {dt_grid.best_params_}")
    
    feature_order = list(ordinal_features.keys()) + nominal_features + numerical_features
    
    return model, preprocessor, df, feature_order


# Initialize model
print("=" * 60)
print("Vehicle Maintenance Predictor - Initializing...")
print("=" * 60)
model, preprocessor, df, feature_order = initialize_model()
print("✓ Model initialized and ready for predictions\n")


def predict(
    vehicle_model, maintenance_history, fuel_type, transmission_type, owner_type,
    tire_condition, brake_condition, battery_status,
    mileage, reported_issues, vehicle_age, engine_size,
    odometer_reading, service_history, accident_history, fuel_efficiency,
):
    """Make prediction based on input parameters."""
    row = pd.DataFrame([{
        "Vehicle_Model": vehicle_model,
        "Maintenance_History": maintenance_history,
        "Fuel_Type": fuel_type,
        "Transmission_Type": transmission_type,
        "Owner_Type": owner_type,
        "Tire_Condition": tire_condition,
        "Brake_Condition": brake_condition,
        "Battery_Status": battery_status,
        "Mileage": mileage,
        "Reported_Issues": reported_issues,
        "Vehicle_Age": vehicle_age,
        "Engine_Size": engine_size,
        "Odometer_Reading": odometer_reading,
        "Service_History": service_history,
        "Accident_History": accident_history,
        "Fuel_Efficiency": fuel_efficiency,
    }])
    
    available = [c for c in feature_order if c in df.columns or c in row.columns]
    row = row[[c for c in available if c in row.columns]]
    
    X_input = preprocessor.transform(row)
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    
    label = "MAINTENANCE NEEDED" if pred == 1 else "NO MAINTENANCE NEEDED"
    confidence = f"{max(proba) * 100:.1f}%"
    breakdown = {
        "No Maintenance": float(round(proba[0], 4)),
        "Maintenance Needed": float(round(proba[1], 4)),
    }
    return label, confidence, breakdown


# CSS Styling
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

* { font-family: 'JetBrains Mono', monospace !important; }

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

#header {
    border-bottom: 1px solid #333;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}

#header h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0 0 0.25rem 0 !important;
}

#header p { font-size: 0.8rem !important; opacity: 0.55; margin: 0 !important; }

.section-label {
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    opacity: 0.45;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.4rem;
}

#predict-btn {
    margin-top: 1.5rem;
    border-radius: 2px !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

#result-label textarea, #result-confidence textarea {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
}

label span { font-size: 0.7rem !important; letter-spacing: 0.05em !important; opacity: 0.7; }
"""


def build_app():
    """Build the Gradio interface."""
    with gr.Blocks(title="Vehicle Maintenance Predictor", theme=gr.themes.Monochrome(), css=CSS) as demo:
        with gr.Column(elem_id="header"):
            gr.Markdown("# Vehicle Maintenance Predictor")
            gr.Markdown("Fill in the vehicle details and click Predict.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Vehicle Profile", elem_classes="section-label")
                vehicle_model = gr.Dropdown(["Bus", "Car", "Motorcycle", "SUV", "Truck", "Van"],
                                            label="Vehicle Model", value="Car")
                fuel_type = gr.Dropdown(["Diesel", "Electric", "Petrol"],
                                        label="Fuel Type", value="Petrol")
                transmission_type = gr.Dropdown(["Automatic", "Manual"],
                                                label="Transmission Type", value="Automatic")
                owner_type = gr.Dropdown(["First", "Second", "Third"],
                                         label="Owner Type", value="First")
                engine_size = gr.Dropdown([800, 1000, 1500, 2000, 2500],
                                          label="Engine Size (cc)", value=1500)

                gr.Markdown("Condition", elem_classes="section-label")
                maintenance_history = gr.Dropdown(["Poor", "Average", "Good"],
                                                  label="Maintenance History", value="Good")
                tire_condition = gr.Dropdown(["Worn Out", "Good", "New"],
                                             label="Tire Condition", value="Good")
                brake_condition = gr.Dropdown(["Worn Out", "Good", "New"],
                                              label="Brake Condition", value="Good")
                battery_status = gr.Dropdown(["Weak", "Good", "New"],
                                             label="Battery Status", value="Good")

            with gr.Column():
                gr.Markdown("Usage & History", elem_classes="section-label")
                mileage = gr.Slider(30001, 80000, step=1000, value=55000,
                                    label="Mileage (km)")
                vehicle_age = gr.Slider(1, 10, step=1, value=5,
                                        label="Vehicle Age (years)")
                odometer_reading = gr.Slider(1001, 149999, step=1000, value=75000,
                                             label="Odometer Reading (km)")
                fuel_efficiency = gr.Slider(10.0, 20.0, step=0.1, value=15.0,
                                            label="Fuel Efficiency (km/l)")
                reported_issues = gr.Slider(0, 5, step=1, value=1,
                                            label="Reported Issues")
                service_history = gr.Slider(1, 10, step=1, value=5,
                                            label="Service History (count)")
                accident_history = gr.Slider(0, 3, step=1, value=0,
                                             label="Accident History (count)")

        predict_btn = gr.Button("Predict", variant="primary", size="lg", elem_id="predict-btn")

        gr.Markdown("Result", elem_classes="section-label")
        with gr.Row():
            result_label = gr.Textbox(label="Prediction", interactive=False, elem_id="result-label")
            result_confidence = gr.Textbox(label="Confidence", interactive=False, elem_id="result-confidence")
        result_proba = gr.Label(label="Class Probabilities")

        predict_btn.click(
            fn=predict,
            inputs=[
                vehicle_model, maintenance_history, fuel_type, transmission_type, owner_type,
                tire_condition, brake_condition, battery_status,
                mileage, reported_issues, vehicle_age, engine_size,
                odometer_reading, service_history, accident_history, fuel_efficiency,
            ],
            outputs=[result_label, result_confidence, result_proba],
        )
    
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
