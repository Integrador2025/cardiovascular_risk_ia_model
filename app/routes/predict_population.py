from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
from app.model_population.data_processing_population import convert_precipitation, preprocess_population_data
from app.model_config import PACIENTES_PATH, MUNICIPIOS_PATH
from app.model_population.load_population_model import load_population_model # Importar la nueva función de carga
from app.model_population.data_processing_population import load_and_group_population_data # Importar la función de carga y agrupación

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
        # 1. Load model and preprocessors
        model, scaler, encoder, feature_names_population = load_population_model()
        if model is None or scaler is None or encoder is None or feature_names_population is None:
            raise HTTPException(
                status_code=404,
                detail="Modelo poblacional o preprocesadores no encontrados. Por favor, entrene el modelo poblacional primero."
            )

        # 2. Load and merge data
        try:
            df = load_and_group_population_data(PACIENTES_PATH, MUNICIPIOS_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error cargando datos: {str(e)}"
            )

        # 3. Filter by municipality
        df_mpio = df[df["municipality"].str.strip().str.upper() == municipio.strip().upper()].copy() # Usar .copy() para evitar SettingWithCopyWarning
        if df_mpio.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron datos para el municipio: {municipio}"
            )

        # 4. Preprocess the filtered data using the loaded preprocessors
        # Asegurarse de que df_mpio tenga la columna 'risk_score' (aunque sea un placeholder) para preprocess_population_data
        if 'risk_score' not in df_mpio.columns:
            df_mpio['risk_score'] = 0.0 # Placeholder para que preprocess_population_data no falle
            
        X_processed_mpio, _, _, _, _ = preprocess_population_data(
            df_mpio, 
            scaler_obj=scaler, 
            encoder_obj=encoder, 
            feature_names_obj=feature_names_population # Pasar los nombres de las características para alineación
        )

        # 5. Make predictions
        try:
            predicciones = model.predict(X_processed_mpio).flatten()
            df_mpio["risk_estimate"] = predicciones.tolist()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Fallo en la predicción: {str(e)}"
            )

        # 6. Prepare response
        # Agrupar de nuevo para el output, si es necesario, o usar los grupos originales con la estimación
        # Si df_mpio ya está agrupado por las características demográficas relevantes, usarlo directamente
        # Si no, agrupar para el output final como se hacía antes
        
        # Usar las columnas categóricas que se usaron para la agrupación original en load_and_group_population_data
        categorical_cols_for_output = [
            "rural_area", "socioeconomic_status", "ethnicity", 
            "occupation", "education_level", "climate_classification"
        ]
        
        # Asegurarse de que solo las columnas existentes se incluyan en el output
        final_output_cols = [col for col in categorical_cols_for_output if col in df_mpio.columns] + ["risk_estimate"]

        response_data = {
            "municipio": municipio,
            "n_grupos": len(df_mpio), # Ahora df_mpio ya contiene los grupos
            "riesgo_promedio_total": round(float(np.mean(predicciones)), 4),
            "grupos": df_mpio[final_output_cols].to_dict(orient="records")
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado: {str(e)}"
        )