import tensorflow as tf
import joblib
import os
from app.model_config import POPULATION_MODEL_PATH, POPULATION_SCALER_PATH, POPULATION_ENCODER_PATH

# Definir la ruta para los nombres de las características del modelo poblacional
POPULATION_FEATURE_NAMES_PATH = POPULATION_MODEL_PATH.replace(".keras", "_feature_names.pkl")

def load_population_model():
    """
    Loads the trained population model, its scaler, encoder, and feature names.
    
    Returns:
        tuple: (tf.keras.Model, StandardScaler, OneHotEncoder, list) or (None, None, None, None) if loading fails.
    """
    try:
        # Verificar que todos los archivos necesarios existan
        if not os.path.exists(POPULATION_MODEL_PATH) or \
           not os.path.exists(POPULATION_SCALER_PATH) or \
           not os.path.exists(POPULATION_ENCODER_PATH) or \
           not os.path.exists(POPULATION_FEATURE_NAMES_PATH): # Verificar también el archivo de nombres de características
            print("Advertencia: Uno o más archivos del modelo poblacional (modelo, scaler, encoder, nombres de características) no encontrados. Entrene el modelo poblacional primero.")
            return None, None, None, None
            
        model = tf.keras.models.load_model(POPULATION_MODEL_PATH)
        scaler = joblib.load(POPULATION_SCALER_PATH)
        encoder = joblib.load(POPULATION_ENCODER_PATH)
        feature_names = joblib.load(POPULATION_FEATURE_NAMES_PATH) # Cargar los nombres de las características

        print("✅ Modelo poblacional, scaler, encoder y nombres de características cargados exitosamente.")
        return model, scaler, encoder, feature_names
    except Exception as e:
        print(f"Error cargando modelo poblacional o preprocesadores: {e}")
        return None, None, None, None