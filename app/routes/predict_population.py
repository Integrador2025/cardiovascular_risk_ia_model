from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from app.model_population.data_processing_population import convert_precipitation
from app.model_config import PACIENTES_PATH, MUNICIPIOS_PATH
from app.model_population.utils_population import load_population_model

router = APIRouter(prefix="/v1/population", tags=["Population Prediction"])

@router.get("/riesgo-poblacional/{municipio}")
async def predecir_riesgo_poblacional(municipio: str):
    """
    Predict population risk for a given municipality by demographic groups
    
    Returns:
    - Average risk score
    - Risk by demographic groups
    - Number of groups analyzed
    """
    try:
        # 1. Verify model files exist (removed columns.pkl check)
        required_files = {
            "model": "population_model.keras",
            "scaler": "population_scaler.pkl",
            "encoder": "population_encoder.pkl"
        }
        
        missing_files = []
        for file in required_files.values():
            if not os.path.exists(f"model/{file}"):
                missing_files.append(file)
        
        if missing_files:
            raise HTTPException(
                status_code=400,
                detail=f"Model files missing: {', '.join(missing_files)}. Please train the model first."
            )

        # 2. Load and merge data
        try:
            df_pac = pd.read_csv(PACIENTES_PATH)
            df_mun = pd.read_csv(MUNICIPIOS_PATH)
            
            # Normalize column names to lowercase
            df_pac.columns = df_pac.columns.str.lower()
            df_mun.columns = df_mun.columns.str.lower()
            
            df = df_pac.merge(df_mun, on=["department", "municipality"], how="left")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error loading data: {str(e)}"
            )

        # 3. Process precipitation data
        if "annual_precipitation" in df.columns:
            df["annual_precipitation"] = df["annual_precipitation"].astype(str).apply(convert_precipitation)

        # 4. Filter by municipality
        df_mpio = df[df["municipality"].str.strip().str.upper() == municipio.strip().upper()]
        if df_mpio.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for municipality: {municipio}"
            )

        # 5. Group by demographic characteristics
        grupos = df_mpio.groupby([
            "rural_area", "socioeconomic_status", "ethnicity", 
            "occupation", "education_level", "climate_classification"
        ], as_index=False).agg({
            "age": "mean",
            "bmi": "mean",
            "total_cholesterol": "mean",
            "is_smoker": "mean",
            "family_history": "mean",
            "has_electricity": "mean",
            "has_water_supply": "mean",
            "has_sewage": "mean",
            "has_gas": "mean",
            "has_internet": "mean",
            "latitude": "first",
            "longitude": "first",
            "average_altitude": "first",
            "average_temperature": "first",
            "annual_precipitation": "first",
            "estimated_population": "first"
        })

        # 6. Load model and preprocessors (without columns file)
        try:
            model = tf.keras.models.load_model("model/population_model.keras")
            scaler = joblib.load("model/population_scaler.pkl")
            encoder = joblib.load("model/population_encoder.pkl")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(e)}"
            )

        # 7. Prepare features - define columns dynamically
        categorical_cols = [
            "rural_area", "socioeconomic_status", "ethnicity",
            "occupation", "education_level", "climate_classification"
        ]
        numerical_cols = [
            col for col in grupos.columns 
            if col not in categorical_cols + ["estimated_population"]
        ]

        try:
            # Process categorical features
            encoded_cat = encoder.transform(grupos[categorical_cols])
            
            # Process numerical features
            scaled_num = scaler.transform(grupos[numerical_cols])
            
            # Combine features
            X = np.concatenate([scaled_num, encoded_cat], axis=1)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error preprocessing data: {str(e)}"
            )

        # 8. Make predictions
        try:
            predicciones = model.predict(X).flatten()
            grupos["risk_estimate"] = predicciones.tolist()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )

        # 9. Prepare response
        response_data = {
            "municipio": municipio,
            "n_grupos": len(grupos),
            "riesgo_promedio_total": round(float(np.mean(predicciones)), 4),
            "grupos": grupos[categorical_cols + ["risk_estimate"]].to_dict(orient="records")
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )