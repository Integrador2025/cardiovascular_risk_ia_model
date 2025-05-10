from fastapi import APIRouter, HTTPException
import os
import tensorflow as tf
import load
import pandas as pd
from app.model_training.data_processing import preprocess_data
from app.model_training.training import train_with_stratified_kfold
from core.model_loader import load_model_and_features, MODEL_PATH
from app.model_training.utils import plot_metrics, save_feature_importances
from app.model_population.training_population import train_population_model

router = APIRouter()

router = APIRouter(prefix="/v1/training", tags=["Training"])

@router.post("/train-model/")
async def train_model():
    try:
        df_pacientes, df_municipios, _ = load.load_dataset()
        X, Y_reg, _, feature_names = preprocess_data(df_pacientes)
        model, history = train_with_stratified_kfold(X, Y_reg, feature_names)
        model.save(MODEL_PATH)
        load_model_and_features.cache_clear()

        # Verificar o crear carpeta "model"
        model_dir = "model"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Carpeta creada: {model_dir}")
        else:
            print(f"Carpeta ya existe: {model_dir}")

        # plot_metrics(history, "loss", os.path.join(model_dir, "loss_final_regresion.png"), branch='')
        # save_feature_importances(model, feature_names, os.path.join(model_dir, "feature_importances_final.csv"))

        plot_path = os.path.join(model_dir, "loss_final_regresion.png")
        plot_metrics(history, "loss", plot_path, branch='')
        print(f"Verificando si se generó: {os.path.exists(plot_path)}")  # ← Agregado

        feat_path = os.path.join(model_dir, "feature_importances_final.csv")
        save_feature_importances(model, feature_names, feat_path)
        print(f"Verificando si se generó: {os.path.exists(feat_path)}")


        return {
            "status": "success",
            "message": "Modelo entrenado y guardado",
            "path": MODEL_PATH,
            "feature_count": len(feature_names)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/entrenar-modelo-poblacional")
async def entrenar_modelo_poblacional():
    try:
        # Verificar disponibilidad de datos primero
        try:
            df_pac, df_mun, _ = load.load_dataset()
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=f"Archivo no encontrado: {str(e)}")

        # Verificar columnas requeridas
        required_columns = ["department", "municipality"]  # Ajustar según tus nombres reales
        for col in required_columns:
            if col not in df_pac.columns or col not in df_mun.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Columna requerida '{col}' no encontrada en los datos"
                )

        model, history = train_population_model()
        return {
            "mensaje": "Modelo poblacional complejo entrenado exitosamente.",
            "epochs": len(history.history["loss"]),
            "loss_final": round(history.history["loss"][-1], 4),
            "mae_final": round(history.history["mae"][-1], 4)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento: {str(e)}"
        )
