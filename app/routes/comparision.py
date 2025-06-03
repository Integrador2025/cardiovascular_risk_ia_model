# app/routes/comparison.py
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
import load # Assumes load.py is in your project root
from typing import Optional, List
from pydantic import BaseModel
from app.model_training.data_processing import preprocess_data
from core.model_loader import load_model_and_features # Load the individual model
from app.routes.socio_environmental import categorizar_riesgo # Reuse the risk categorization function

router = APIRouter(prefix="/v1/comparison", tags=["Comparison Analysis"])

# Define the patient input model (similar to the individual prediction)
class PatientInputForComparison(BaseModel):
    age: int
    sex: str
    marital_status: str
    education_level: int
    socioeconomic_status: int
    occupation: str
    ethnicity: str
    bmi: float
    heart_rate: int
    total_cholesterol: float
    glucose: float
    is_smoker: int
    bpm_meds: int
    diabetes: int
    family_history: int
    diagnosis_year: int
    diagnosis_month: int
    rural_area: int
    has_electricity: int
    has_water_supply: int
    has_sewage: int
    has_gas: int
    has_internet: int
    department: str
    municipality: str
    climate_classification: str

@router.post("/patient-to-demographic-benchmark")
async def compare_patient_to_demographic_benchmark(
    patient_data: PatientInputForComparison,
    compare_by_sex: bool = Query(True, description="Include sex in demographic comparison."),
    compare_by_age_group: bool = Query(True, description="Include age group (e.g., 10-year bins) in demographic comparison."),
    compare_by_socioeconomic_status: bool = Query(True, description="Include socioeconomic status in demographic comparison."),
    min_patients_in_group: int = Query(10, description="Minimum number of patients required in the benchmark group.")
):
    """
    Compares a specific patient's risk score against the average risk of a similar demographic group.
    The demographic group can be defined by sex, age group, and/or socioeconomic status.
    """
    try:
        # Load the full dataset for comparison data
        df_full, _, _ = load.load_dataset()
        df_full.columns = df_full.columns.str.lower()
        
        # Filter relevant non-null data for comparison
        df_full = df_full[df_full["risk_score"].notnull()]
        
        if df_full.empty:
            raise HTTPException(status_code=404, detail="No data available for demographic benchmarking after cleaning.")

        # Load the individual model and preprocessors
        model, scaler, encoder, feature_names = load_model_and_features()
        if model is None or scaler is None or encoder is None or feature_names is None:
            raise HTTPException(
                status_code=404,
                detail="Individual model or preprocessors not found. Please train the individual model first."
            )

        # 1. Predict the given patient's risk
        patient_df = pd.DataFrame([patient_data.dict()])
        
        # Ensure `diagnosis_date` exists for `preprocess_data` if needed
        # If not in PatientInput, add a placeholder or handle in preprocess_data
        if 'diagnosis_date' not in patient_df.columns:
            patient_df['diagnosis_date'] = '2023-01-01' # Or current date for days_since_diagnosis calculation
            
        X_patient, _, _, _, _ = preprocess_data(patient_df, scaler, encoder, feature_names, is_inference=True)
        patient_risk = model.predict(X_patient).flatten()[0]

        # 2. Define the comparison group
        benchmark_query = []
        group_description_parts = []

        # Sex
        if compare_by_sex:
            benchmark_query.append(f"sex == {1 if patient_data.sex.lower() == 'masculino' else 0}")
            group_description_parts.append(patient_data.sex)

        # Age Group (10-year bins)
        if compare_by_age_group:
            age_bin_start = (patient_data.age // 10) * 10
            age_bin_end = age_bin_start + 9
            benchmark_query.append(f"age >= {age_bin_start} and age <= {age_bin_end}")
            group_description_parts.append(f"edad {age_bin_start}-{age_bin_end}")

        # Socioeconomic Status
        if compare_by_socioeconomic_status:
            benchmark_query.append(f"socioeconomic_status == {patient_data.socioeconomic_status}")
            group_description_parts.append(f"estrato {patient_data.socioeconomic_status}")

        # Build and apply the filter
        query_str = " and ".join(benchmark_query)
        benchmark_group_df = df_full.query(query_str).copy() if query_str else df_full.copy()
        
        benchmark_group_description = ", ".join(group_description_parts) if group_description_parts else "all patients"

        if benchmark_group_df.empty or len(benchmark_group_df) < min_patients_in_group:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough patients found ({len(benchmark_group_df)} found, {min_patients_in_group} required) for the comparison group: {benchmark_group_description}. Try relaxing the comparison criteria."
            )

        # 3. Calculate the average risk for the group
        benchmark_avg_risk = benchmark_group_df["risk_score"].mean()

        # 4. Prepare the response
        return {
            "patient_risk_score": round(float(patient_risk), 4),
            "patient_risk_category": categorizar_riesgo(patient_risk),
            "benchmark_group_description": benchmark_group_description.capitalize(),
            "benchmark_average_risk": round(float(benchmark_avg_risk), 4),
            "benchmark_risk_category": categorizar_riesgo(benchmark_avg_risk),
            "total_patients_in_benchmark_group": int(len(benchmark_group_df)),
            "patient_risk_vs_benchmark_difference": round(float(patient_risk - benchmark_avg_risk), 4),
            "note": "A positive value means the patient's risk is higher than the group average."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in demographic comparison analysis: {str(e)}")