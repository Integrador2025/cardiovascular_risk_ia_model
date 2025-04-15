from app.model_population.data_processing_population import load_and_group_population_data, preprocess_population_data
from app.model_population.model_architecture_population import build_population_model
from app.model_config import PACIENTES_PATH, MUNICIPIOS_PATH
import tensorflow as tf

MODEL_PATH = "model/population_model.keras"


def train_population_model():
    print("🔍 Cargando y procesando datos...")
    df = load_and_group_population_data(PACIENTES_PATH, MUNICIPIOS_PATH)
    X_train, X_test, y_train, y_test = preprocess_population_data(df)

    print("🧠 Construyendo modelo...")
    model = build_population_model(X_train.shape[1])

    print("🚀 Entrenando modelo...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=300,
        batch_size=32,
        verbose=1
    )

    print("💾 Guardando modelo en formato .keras...")
    model.save(MODEL_PATH)

    print("✅ Entrenamiento completo.")
    return model, history
