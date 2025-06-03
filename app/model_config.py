# app/model_config.py

# Rutas y configuración centralizada para el modelo poblacional
POPULATION_MODEL_PATH = "model/population_model.keras"
POPULATION_SCALER_PATH = "model/population_scaler.pkl"
POPULATION_ENCODER_PATH = "model/population_encoder.pkl"
# POPULATION_COLUMNS_PATH no es necesario si las columnas se reconstruyen dinámicamente

# Rutas y configuración centralizada para el modelo de riesgo individual
INDIVIDUAL_MODEL_PATH = "model/modelo_regresion_final.keras"
INDIVIDUAL_SCALER_PATH = "model/individual_scaler.pkl"
INDIVIDUAL_FEATURE_NAMES_PATH = "model/individual_feature_names.pkl"

# Rutas de los datasets
PACIENTES_PATH = "datasets/pacient_dataset.csv"
MUNICIPIOS_PATH = "datasets/municipios_colombia.csv"
