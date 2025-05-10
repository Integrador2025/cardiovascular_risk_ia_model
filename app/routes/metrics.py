from fastapi import APIRouter, HTTPException
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data
import load

router = APIRouter()

router = APIRouter(prefix="/v1/metrics", tags=["Metrics"])

@router.get("/metricas/")
async def get_regression_metrics():
    try:
        model, _ = load_model_and_features()
        if model is None:
            raise HTTPException(status_code=404, detail="Modelo no encontrado.")

        df, _, _ = load.load_dataset()
        X, Y_reg, _, _ = preprocess_data(df, augment=False)

        X_train, X_val, y_train, y_val = train_test_split(
            X, Y_reg, test_size=0.2, random_state=42
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))