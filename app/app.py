"""
Gradio app for Vehicle Maintenance Predictor.
Loads pre-trained model artifacts from models/ — no training on startup.

Usage:
    python app/app.py
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import gradio as gr


# ── Load pre-trained artifacts ───────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models"

print("=" * 60)
print("Vehicle Maintenance Predictor - Loading model...")
print("=" * 60)

model = joblib.load(MODEL_DIR / "model.joblib")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
with open(MODEL_DIR / "feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

print("✓ Model loaded and ready for predictions\n")


# ── Prediction ───────────────────────────────────────────────
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

    row = row[[c for c in feature_order if c in row.columns]]

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


# ── CSS ──────────────────────────────────────────────────────
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


# ── Gradio UI ────────────────────────────────────────────────
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
