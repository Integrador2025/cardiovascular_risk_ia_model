from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data
from app.model_training.schemas import PacienteInput  # Asegúrate de tener este modelo
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
async def predecir_riesgo(paciente: PacienteInput):
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(status_code=404, detail="Modelo no cargado")

        # Convertir input a DataFrame
        df_input = pd.DataFrame([paciente.dict()])

        # Columnas dummy para que preprocess_data no falle
        df_input["PUNTAJE_RIESGO"] = 0
        df_input["RIESGO_CARDIOVASCULAR"] = 0  # ← necesaria para evitar error
        df_input["FECHA_DIAGNOSTICO"] = pd.Timestamp("2024-01-01")  

        # Preprocesar datos del paciente (sin augment ni filtrado)
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
        pesos = first_dense.get_weights()[0]  # (features, neuronas)
        importancia = np.abs(pesos[:, 0])     # suma o magnitud de cada feature
        activaciones = X_alineado[0]
        ponderadas = importancia * activaciones

        top_indices = np.argsort(ponderadas)[::-1][:5]
        top_features = [(feature_names[i], round(float(ponderadas[i]), 4)) for i in top_indices]

        return {
            "puntaje_riesgo": round(pred, 4),
            "categoria": categoria,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")