import pandas as pd
import joblib
import shap
import numpy as np
import json
import os
from ROI_Model import calculate_roi

# --- 1. Load All Necessary Artifacts ---
print("--- Loading Model and Data ---")
BASE_DIR = os.path.dirname(__file__)  # folder of current script
MODEL_FILE = os.path.join(BASE_DIR, 'proactive_cost_model.pkl')
DATASET_FILE = 'prognos_health_multilabel_dataset.parquet'

model = joblib.load(MODEL_FILE)
final_df = pd.read_parquet(DATASET_FILE)

# --- Prepare column lists ---
cols_to_drop = [
    'DESYNPUF_ID', 'BENE_BIRTH_DT', 'BENE_DEATH_DT', 'SP_STATE_CODE',
    'BENE_COUNTY_CD', 'was_hospitalized_in_2010'
]
target_cols = [col for col in final_df.columns if col.startswith('HAD_')]
X = final_df.drop(columns=cols_to_drop + target_cols)
TRAINING_COLUMNS = X.columns.tolist()
CONDITIONS = target_cols

# --- 2. Create SHAP Explainers ---
print("Creating SHAP explainers...")
explainers = [shap.TreeExplainer(estimator) for estimator in model.estimators_]

# --- 3. Define the Master Prediction Function ---
def predict_patient_risk(patient_data: dict):
    patient_df = pd.DataFrame([patient_data]).reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    predicted_outcomes = []
    risk_scores = []
    x_transformed = patient_df.copy() 
    
    for i, condition in enumerate(CONDITIONS):
        estimator = model.estimators_[i]
        prob_array = estimator.predict_proba(x_transformed)
        risk_score = float(prob_array.flatten()[-1])
        risk_scores.append(risk_score)

        shap_values = explainers[i](x_transformed)
        abs_shap = np.abs(shap_values.values)
        feature_indices = np.argsort(abs_shap)[0, ::-1][:3]
        feature_names_array = np.array(shap_values.feature_names)
        key_risk_factors = feature_names_array[feature_indices].tolist()
        
        if risk_score >= 0.75: risk_tier = "Tier 4: High Risk"
        elif risk_score >= 0.50: risk_tier = "Tier 3: Moderate Risk"
        elif risk_score >= 0.25: risk_tier = "Tier 2: Low Risk"
        else: risk_tier = "Tier 1: Minimal Risk"

        predicted_outcomes.append({
            "condition": condition.replace('HAD_', '').replace('IN_2010', ' ').replace('', ' ').strip().replace('ACUTE', 'SEVERE'),
            "riskScore": round(risk_score, 3), "riskTier": risk_tier, "keyRiskFactors": key_risk_factors
        })
        
        if i < len(CONDITIONS) - 1:
            x_transformed[condition] = prob_array[:, 1]

    overall_risk_score = float(max(risk_scores))
    primary_condition_index = risk_scores.index(overall_risk_score)
    primary_condition = predicted_outcomes[primary_condition_index]['condition']
    
    prediction_result = {
        "patientId": patient_data.get("DESYNPUF_ID", "UNKNOWN"),
        "age": patient_data.get("Age"),
        "overallRiskScore": round(overall_risk_score, 3),
        "presentRiskCondition": primary_condition,
        "predictedOutcomes": [predicted_outcomes[primary_condition_index]]
    }

    roi_analysis = calculate_roi(prediction_result)
    prediction_result.update(roi_analysis)
    
    print(json.dumps(prediction_result, indent=2))
    
    return prediction_result

# --- 4. Process Patients from a CSV Input File ---

# Define the path to your input CSV file
#CSV_INPUT_FILE = r"D:\Projects\CTS project\Model\test_csv_for _model.csv"
'''
print(f"\n--- Loading patients from {CSV_INPUT_FILE} ---")

# Load the CSV into a pandas DataFrame
try:
    input_df = pd.read_csv(CSV_INPUT_FILE)
except FileNotFoundError:
    print(f"Error: The file '{CSV_INPUT_FILE}' was not found. Please create it in the same directory.")
    exit()

# Convert the DataFrame into a list of dictionaries (one for each patient)
patients_to_predict = input_df.to_dict(orient='records')

# Initialize a list to store predictions for all patients
all_predictions = []

# Loop through each patient record, make a prediction, and collect the result
print(f"--- Predicting risk for {len(patients_to_predict)} patients ---")
for patient_data in patients_to_predict:
    final_prediction = predict_patient_risk(patient_data)
    all_predictions.append(final_prediction)

# Print the final list of predictions as a single JSON array
print("\n--- Final JSON Output for all Patients ---")
print(json.dumps(all_predictions, indent=2))'''