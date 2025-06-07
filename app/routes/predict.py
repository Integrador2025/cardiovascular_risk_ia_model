# app/routes/predict.py

from fastapi.responses import PlainTextResponse
import json
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data
from app.model_training.schemas import PatientInput
import tensorflow as tf

router = APIRouter(prefix="/v1/modelo", tags=["Modelo"])

def categorizar_riesgo(puntaje: float) -> str:
    if puntaje < 0.2:
        return "Muy Bajo"
    elif puntaje < 0.4:
        return "Bajo"
    elif puntaje < 0.6:
        return "Moderado"
    elif puntaje < 0.8:
        return "Alto"
    else:
        return "Muy Alto"

@router.post("/predict")
async def predecir_riesgo(paciente: PatientInput):
    try:
        # Cargar modelo, scaler y feature_names
        model, scaler, feature_names = load_model_and_features()
        if model is None or scaler is None or feature_names is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no cargados. Entrene el modelo primero.")

        # Convertir input a DataFrame
        df_input = pd.DataFrame([paciente.dict()])

        # Asegurar columnas necesarias
        if 'risk_score' not in df_input.columns:
            df_input["risk_score"] = 0.0
        if 'cardiovascular_risk' not in df_input.columns:
            df_input["cardiovascular_risk"] = 0
        if 'diagnosis_date' not in df_input.columns:
            df_input["diagnosis_date"] = pd.Timestamp("2024-01-01")

        # Preprocesar
        X_input, _, _, _ = preprocess_data(df_input, scaler_obj=scaler, feature_names_obj=feature_names)

        # Predecir
        pred = float(model.predict(X_input)[0][0])
        categoria = categorizar_riesgo(pred)

        # Capa densa y pesos
        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró capa densa en el modelo para calcular importancia.")

        pesos = first_dense.get_weights()[0]
        importancia_ponderada = np.abs(pesos[:, 0]) * X_input[0]
        top_indices = np.argsort(importancia_ponderada)[::-1][:5]
        top_features = [(feature_names[i], round(float(importancia_ponderada[i]), 4)) for i in top_indices]

        # Construir resultado y serializar
        resultado = {
            "puntaje_riesgo": round(pred, 4),
            "categoria": categoria,
            "factores_relevantes": top_features,
            "nota": "Factores relevantes calculados como activación * peso en la primera capa densa"
        }

        json_data = json.dumps(resultado)
        escaped_string = json.dumps(json_data)

        return PlainTextResponse(content=escaped_string, media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")