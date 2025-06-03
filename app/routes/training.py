# app/routes/training.py

from fastapi import APIRouter, HTTPException
import load # Asume que load.py está en la raíz del proyecto
from app.model_training.training import train_individual_model # Importar la función de entrenamiento individual
from app.model_population.training_population import train_population_model # Importar la función de entrenamiento poblacional
from core.model_loader import load_model_and_features # Para limpiar la caché del cargador de modelos

router = APIRouter(prefix="/v1/training", tags=["Training"])

@router.post("/train-model/")
async def train_model_endpoint(): # Renombrado a train_model_endpoint para mayor claridad
    """
    Triggers the training process for the individual cardiovascular risk model.
    """
    try:
        df_pacientes, _, _ = load.load_dataset()
        
        # Llamar a la función de entrenamiento del modelo individual
        # Esta función ahora maneja el preprocesamiento, entrenamiento y guardado de artefactos.
        _, _, final_feature_names, history = train_individual_model(df_pacientes)
        
        # Limpiar la caché del cargador de modelos para que cargue los nuevos artefactos.
        load_model_and_features.cache_clear()

        return {
            "status": "success",
            "message": "Modelo de riesgo individual entrenado y guardado exitosamente.",
            "feature_count": len(final_feature_names),
            "final_val_loss": round(history.history['val_loss'][-1], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el entrenamiento del modelo individual: {str(e)}")

@router.post("/entrenar-modelo-poblacional")
async def entrenar_modelo_poblacional_endpoint(): # Renombrado para mayor claridad
    """
    Triggers the training process for the population risk model.
    """
    try:
        # Verificar disponibilidad de datos primero
        try:
            df_pac, df_mun, _ = load.load_dataset()
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=f"Archivo no encontrado: {str(e)}")

        # Verificar columnas requeridas
        # Asegurarse de que los nombres de las columnas coincidan con los de los archivos CSV
        required_columns = ["department", "municipality"] 
        # Convertir columnas a minúsculas para la verificación
        df_pac.columns = df_pac.columns.str.lower()
        df_mun.columns = df_mun.columns.str.lower()

        for col in required_columns:
            if col not in df_pac.columns or col not in df_mun.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Columna requerida '{col}' no encontrada en los datos. Asegúrese de que los nombres de las columnas sean correctos y estén en minúsculas."
                )

        # Llamar a la función de entrenamiento del modelo poblacional
        model, history = train_population_model()
        
        # Limpiar la caché del cargador de modelos por si el modelo poblacional también se carga en algún otro lugar
        load_model_and_features.cache_clear() # Aunque load_population_model es diferente, es buena práctica si se usa lru_cache allí también.

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
