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
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(status_code=404, detail="Modelo no cargado")

        # Convertir input a DataFrame
        df_input = pd.DataFrame([paciente.dict()])

        # Columnas dummy para que preprocess_data no falle (actualizadas a nombres en inglés)
        df_input["risk_score"] = 0  # Antes PUNTAJE_RIESGO
        df_input["cardiovascular_risk"] = 0  # Antes RIESGO_CARDIOVASCULAR
        df_input["diagnosis_date"] = pd.Timestamp("2024-01-01")  # Antes FECHA_DIAGNOSTICO

        # Preprocesar datos del paciente
        X_input, _, _, features_input = preprocess_data(df_input)

        # Alinear con el orden de features esperadas por el modelo
        X_alineado = np.zeros((1, len(feature_names)), dtype=np.float32)
        for i, name in enumerate(features_input):
            if name in feature_names:
                idx = feature_names.index(name)
                X_alineado[0, idx] = X_input[0, i]

        # Hacer predicción
        pred = float(model.predict(X_alineado)[0][0])
        categoria = categorizar_riesgo(pred)

        # Calcular importancia de features para esta predicción
        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró capa densa en el modelo")
            
        pesos = first_dense.get_weights()[0]
        importancia = np.abs(pesos[:, 0])
        activaciones = X_alineado[0]
        ponderadas = importancia * activaciones

        top_indices = np.argsort(ponderadas)[::-1][:5]
        top_features = [(feature_names[i], round(float(ponderadas[i]), 4)) for i in top_indices]

        return {
            "puntaje_riesgo": round(pred, 4),
            "categoria": categoria,
            "factores_relevantes": top_features,
            "nota": "Factores relevantes calculados como activación * peso en la primera capa densa"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")