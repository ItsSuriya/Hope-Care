import pandas as pd
import joblib
import shap
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from Backend.ROI_Model import generate_roi_report



MODEL_FILE = "Backend/classifier_chain_gpu_model_v2.joblib"
DATASET_FILE = "Backend/prognos_health_multilabel_dataset.parquet"

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

explainers = [shap.TreeExplainer(estimator) for estimator in model.estimators_]

# --- 3. Define the Master Prediction Function ---
risk_factor_mapping = {
    "inpatient_visit_count_08_09": "Inpatient Visit Count ",
    "outpatient_visit_count_08_09": "Outpatient Visit Count ",
    "unique_drug_count_08_09": " Total number of Unique Drug Count is high ",
    "rx_claim_count_08_09": "Prescription (Rx) Claim count is High",
    "BENRES_IP": "Beneficiary Responsibility InPatient",
    "BENRES_OP": "Beneficiary Responsibility OutPatient",
    "BENRES_CAR": "Beneficiary Responsibility Carrier",
    "stratification_risk_score": "Stratification Risk Score",
    "Length_of_Stay": "Length of Stay",
    "total_inpatient_days_08_09": "Total Inpatient Days in the recent years is high",
    "MEDREIMB_IP": "Medicare Reimbursement InPatient",
    "MEDREIMB_OP": "Medicare Reimbursement OutPatient",
    "MEDREIMB_CAR": "Medicare Reimbursement Carrier",
    "total_inpatient_cost_08_09": "Total Inpatient Cost is high",
    "total_outpatient_cost_08_09": "Total Outpatient Cost high",
    "total_drug_cost_08_09": "Total Drug Cost is high in the recent ",
    "SP_CHF": "Special Profile flag for Congestive Heart Failure",
    "SP_CHRNKIDN": "Special Profile flag for Chronic Kidney Disease",
    "SP_COPD": "Special Profile flag for Chronic Obstructive Pulmonary Disease",
    "SP_DIABETES": "Special Profile flag for Diabetes",
    "HAD_ACUTE_HEART_FAILURE_IN_2010": "Had Acute Heart Failure ",
    "PLAN_CVRG_MOS_NUM": "Plan Coverage Months Number"
}
def predict_patient_risk(patient_data: dict):
    patient_df = pd.DataFrame([patient_data]).reindex(columns=TRAINING_COLUMNS, fill_value=0)

    predicted_outcomes = []
    risk_scores = []
    x_transformed = patient_df.copy()

    for i, condition in enumerate(CONDITIONS):
        estimator = model.estimators_[i]
        prob_array = estimator.predict_proba(x_transformed)
        risk_score = float(prob_array.flatten()[-1])
        if not np.isfinite(risk_score):
            risk_score = 0.0
        risk_scores.append(risk_score)

        shap_values = explainers[i](x_transformed)
        abs_shap = np.abs(shap_values.values)
        feature_indices = np.argsort(abs_shap)[0, ::-1][:3]
        feature_names_array = np.array(shap_values.feature_names)
        key_risk_factors = feature_names_array[feature_indices].tolist()
        key_risk_factors = [risk_factor_mapping.get(factor, factor) for factor in key_risk_factors]

        if risk_score >= 0.75:
            risk_tier = "Tier 4: High Risk"
        elif risk_score >= 0.50:
            risk_tier = "Tier 3: Moderate Risk"
        elif risk_score >= 0.25:
            risk_tier = "Tier 2: Low Risk"
        else:
            risk_tier = "Tier 1: Minimal Risk"

        predicted_outcomes.append({
            "condition": condition.replace('HAD_', '').replace('IN_2010', ' ')
                                  .replace('_', ' ').strip()
                                  .replace('ACUTE', 'SEVERE'),
            "riskScore": round(risk_score, 3),
            "riskTier": risk_tier,
            "keyRiskFactors": key_risk_factors
        })

        if i < len(CONDITIONS) - 1:
            x_transformed[condition] = prob_array[:, 1]

    # Find the primary risk condition
    overall_risk_score = float(max(risk_scores)) if risk_scores else 0.0
    primary_condition_index = risk_scores.index(overall_risk_score) if overall_risk_score in risk_scores else 0
    primary_condition_data = predicted_outcomes[primary_condition_index] if predicted_outcomes else {}

    return {
        "patientId": patient_data.get("DESYNPUF_ID", "UNKNOWN"),
        "age": patient_data.get("Age"),
        "location": patient_data.get("location", "UNKNOWN"),
        "income": patient_data.get("income", "UNKNOWN"),
        "employment": patient_data.get("employee", "UNKNOWN"),
        "hospital_visit": patient_data.get("inpatient_visit_count_08_09", 0),
        "primary_condition": primary_condition_data.get("condition"),
        "risk_score": primary_condition_data.get("riskScore"),
        "risk_tier": primary_condition_data.get("riskTier"),
        "key_risk_factors": primary_condition_data.get("keyRiskFactors"),
        "predictedOutcomes": [primary_condition_data],
    }

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoint for Prediction ---
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        print("❌ Error reading CSV file:", str(e))
        return {"status": "error", "message": str(e)}

    all_predictions = []
    for patient_data in df.to_dict(orient="records"):
        prediction = predict_patient_risk(patient_data)
        #print(prediction)
        all_predictions.append(prediction)

    roi_report = generate_roi_report(all_predictions)

    return {"status": "success", "predictions": all_predictions, "roiReport": roi_report}

# --- Serve React Frontend ---
frontend_path = "CTS-Project/dist"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

# Run with: uvicorn main:app --reload
