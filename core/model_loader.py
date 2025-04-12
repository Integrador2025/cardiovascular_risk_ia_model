import os
import tensorflow as tf
from functools import lru_cache
import load
from app.model_training.data_processing import preprocess_data

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_regresion_final.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def load_model_and_features():
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None
        model = tf.keras.models.load_model(MODEL_PATH)
        df, _, _ = load.load_dataset()
        _, _, _, feature_names = preprocess_data(df, augment=True, noise_std=0.02, augment_factor=2)
        return model, feature_names
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None
