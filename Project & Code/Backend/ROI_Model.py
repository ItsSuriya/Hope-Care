# File: roi_model.py
import pandas as pd
import joblib
import os
import sys

# --- Paths ---
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "proactive_cost_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

# --- Check files ---
missing_files = []
if not os.path.exists(MODEL_PATH):
    missing_files.append("proactive_cost_model.pkl")
if not os.path.exists(COLUMNS_PATH):
    missing_files.append("model_columns.pkl")

# --- Load model & columns ---
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

def generate_roi_report(stratification_results: list):
    """
    stratification_results: JSON output from /stratify endpoint
    Returns processed ROI report
    """
    reports = []

    for patient in stratification_results:
        age = patient.get("age", 0)
        patient_id = patient.get("patientId", "UNKNOWN")

        roi_entries = []

        for outcome in patient.get("predictedOutcomes", []):
            condition = outcome.get("condition", "UNKNOWN")

            # --- Prepare features ---
            features = pd.DataFrame([{"age": age, "condition": condition}])
            features = pd.get_dummies(features, columns=['condition'])   
            features = features.reindex(columns=model_columns, fill_value=0)

            # --- Predict proactive cost ---
            proactive_cost = float(model.predict(features)[0])

            # --- Calculate reactive cost & savings ---
            reactive_cost = proactive_cost * 8.5  # assumption
            savings = reactive_cost - proactive_cost

            roi_entries.append({
                "condition": condition,
                "predicted_proactive_cost": round(proactive_cost, 2),
                "predicted_reactive_cost": round(reactive_cost, 2),
                "potential_savings": round(savings, 2)
            })

        reports.append({
            "patientId": patient_id,
            "age_used_for_prediction": age,
            "predictedCosts": roi_entries
        })

    return reports
