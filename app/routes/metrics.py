# app/routes/metrics.py

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd # Necesario para crear DataFrames dummy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data
import load # Asume que load.py está en la raíz del proyecto

router = APIRouter()

router = APIRouter(prefix="/v1/metrics", tags=["Metrics"])

@router.get("/metricas/")
async def get_regression_metrics():
    try:
        # Cargar modelo, scaler y feature_names
        model, scaler, feature_names = load_model_and_features()
        if model is None or scaler is None or feature_names is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df_full, _, _ = load.load_dataset()
        
        # Preprocesar el dataset completo para obtener X y Y_reg
        # Usamos el scaler y feature_names cargados para asegurar consistencia
        X_full, Y_reg_full, _, _ = preprocess_data(df_full, augment=False, scaler_obj=scaler, feature_names_obj=feature_names)

        # Dividir los datos preprocesados en conjuntos de entrenamiento y validación
        # No es necesario estratificar aquí si solo estamos evaluando métricas
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, Y_reg_full, test_size=0.2, random_state=42
        )

        y_pred = model.predict(X_val).ravel()

        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        return {
            "mean_squared_error": round(float(mse), 4),
            "mean_absolute_error": round(float(mae), 4),
            "r2_score": round(float(r2), 4),
            "nota": "Evaluación realizada sobre 20% del conjunto de datos de validación"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener métricas: {str(e)}")