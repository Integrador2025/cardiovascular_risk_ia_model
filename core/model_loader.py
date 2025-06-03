# core/model_loader.py

import os
import tensorflow as tf
from functools import lru_cache
import joblib # Importar joblib para cargar los preprocesadores
import load # Asume que load.py está en la raíz del proyecto
from app.model_training.data_processing import preprocess_data # Necesario para la función, aunque no se usa en load_model_and_features
from app.model_config import INDIVIDUAL_MODEL_PATH, INDIVIDUAL_SCALER_PATH, INDIVIDUAL_FEATURE_NAMES_PATH

MODEL_DIR = "model"
# Asegurarse de que el directorio del modelo exista
os.makedirs(MODEL_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def load_model_and_features():
    """
    Loads the trained individual risk model, its scaler, and feature names.
    Uses lru_cache to optimize loading by caching the results.
    
    Returns:
        tuple: (tf.keras.Model, RobustScaler, list) or (None, None, None) if loading fails.
    """
    try:
        # Verificar que todos los archivos necesarios existan
        if not os.path.exists(INDIVIDUAL_MODEL_PATH) or \
           not os.path.exists(INDIVIDUAL_SCALER_PATH) or \
           not os.path.exists(INDIVIDUAL_FEATURE_NAMES_PATH):
            print("Advertencia: Uno o más archivos del modelo individual (modelo, scaler, nombres de características) no encontrados. Entrene el modelo primero.")
            return None, None, None
            
        model = tf.keras.models.load_model(INDIVIDUAL_MODEL_PATH)
        scaler = joblib.load(INDIVIDUAL_SCALER_PATH)
        feature_names = joblib.load(INDIVIDUAL_FEATURE_NAMES_PATH)
        
        print("✅ Modelo individual, scaler y nombres de características cargados exitosamente.")
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error cargando modelo individual o preprocesadores: {e}")
        return None, None, None