import tensorflow as tf
import joblib

MODEL_PATH = "model/population_model.keras"
SCALER_PATH = "model/pop_scaler.pkl"
ENCODER_PATH = "model/pop_encoder.pkl"
COLUMNS_PATH = "model/pop_columns.pkl"

def load_population_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, encoder, columns