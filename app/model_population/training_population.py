from app.model_population.data_processing_population import (
    load_and_group_population_data, 
    preprocess_population_data
)
from app.model_population.model_architecture_population import build_population_model
from app.model_config import PACIENTES_PATH, MUNICIPIOS_PATH, POPULATION_MODEL_PATH, POPULATION_SCALER_PATH, POPULATION_ENCODER_PATH
import tensorflow as tf
import joblib # Necesario para guardar scaler y encoder
import os # Necesario para crear directorios si no existen
from sklearn.model_selection import train_test_split # Importar train_test_split

def train_population_model():
    """
    Trains the population risk model, saves the model and its preprocessors.
    
    Returns:
        tuple: (model, history) of the trained model and its training history.
    """
    print("🔍 Cargando y procesando datos poblacionales...")
    df = load_and_group_population_data(PACIENTES_PATH, MUNICIPIOS_PATH)
    
    # Preprocesar datos para el modelo poblacional
    # Ahora preprocess_population_data devuelve el scaler, encoder y feature_names
    X_processed, y, fitted_scaler, fitted_encoder, final_feature_names = preprocess_population_data(df)

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    print("🧠 Construyendo modelo poblacional...")
    model = build_population_model(X_train.shape[1])

    # Definir callbacks para el entrenamiento
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30, # Aumentar paciencia para permitir más epochs
        restore_best_weights=True,
        mode='min'
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15, # Aumentar paciencia
        min_lr=1e-7,
        verbose=1
    )

    print("🚀 Entrenando modelo poblacional...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500, # Aumentar epochs máximos
        batch_size=64, # Ajustar batch size
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )

    # Asegurarse de que el directorio del modelo exista
    model_dir = os.path.dirname(POPULATION_MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("💾 Guardando modelo poblacional y preprocesadores...")
    model.save(POPULATION_MODEL_PATH)
    joblib.dump(fitted_scaler, POPULATION_SCALER_PATH)
    joblib.dump(fitted_encoder, POPULATION_ENCODER_PATH)
    # Opcional: guardar los nombres de las características finales si se necesitan para la inferencia
    joblib.dump(final_feature_names, POPULATION_MODEL_PATH.replace(".keras", "_feature_names.pkl"))

    print("✅ Entrenamiento poblacional completo.")
    return model, history