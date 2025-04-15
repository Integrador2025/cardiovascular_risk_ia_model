import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def preprocess_data(df, augment=False, noise_std=0.01, augment_factor=1):
    """Preprocesa los datos para el entrenamiento del modelo, con opción de data augmentation."""

    # Asegurar que trabajamos sobre una copia del DataFrame original
    df = df.copy()

    # Convertir fecha a datetime
    df["FECHA_DIAGNOSTICO"] = pd.to_datetime(df["FECHA_DIAGNOSTICO"], errors='coerce')

    # Extraer componentes temporales útiles
    df["AÑO_DIAGNOSTICO"] = df["FECHA_DIAGNOSTICO"].dt.year
    df["MES_DIAGNOSTICO"] = df["FECHA_DIAGNOSTICO"].dt.month
    df["TRIMESTRE"] = "T" + df["FECHA_DIAGNOSTICO"].dt.quarter.astype(str)
    df["SEMANA_DEL_AÑO"] = df["FECHA_DIAGNOSTICO"].dt.isocalendar().week.astype(int)
    df["DIAS_DESDE_DIAGNOSTICO"] = (pd.Timestamp.today() - df["FECHA_DIAGNOSTICO"]).dt.days
    df["ES_PANDEMIA"] = df["AÑO_DIAGNOSTICO"].apply(lambda y: 1 if y in [2020, 2021, 2022] else 0)

    # Variables objetivo
    Y_reg = df["PUNTAJE_RIESGO"].astype(float)

    # Columnas numéricas y categóricas
    numeric_features = [
        'EDAD', 'IMC', 'SEXO', 'COLESTEROL',
        'ACCESO_ELECTRICO', 'ACUEDUCTO', 'ALCANTARILLADO', 'GAS_NATURAL',
        'ANTECEDENTES_FAMILIARES', 'FUMADOR',
        'AÑO_DIAGNOSTICO', 'MES_DIAGNOSTICO', 'SEMANA_DEL_AÑO',
        'DIAS_DESDE_DIAGNOSTICO', 'ES_PANDEMIA'
    ]

    categorical_features = [
        'DEPARTAMENTO', 'MUNICIPIO', 'ESTADO_CIVIL', 'AREA', 'ESTRATO',
        'NIVEL_EDUCATIVO', 'INTERNET', 'ETNIA', 'OCUPACION',
        'TRIMESTRE'
    ]

    # Separar características (X) y eliminar columnas objetivo
    X = df.drop(columns=["RIESGO_CARDIOVASCULAR", "PUNTAJE_RIESGO", "FECHA_DIAGNOSTICO"])

    # Convertir columnas numéricas a float
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Convertir columnas categóricas a categoría
    for col in categorical_features:
        X[col] = X[col].astype("category")

    # Eliminar filas con valores faltantes
    X.dropna(inplace=True)
    Y_reg = Y_reg.loc[X.index]  # Alinear Y con X

    # Escalado robusto de variables numéricas
    scaler_numeric = RobustScaler()
    X[numeric_features] = scaler_numeric.fit_transform(X[numeric_features])

    # One-hot encoding para las categóricas
    X = pd.get_dummies(X, columns=categorical_features)

    # Convertir a numpy arrays
    X_array = X.values.astype(np.float32)
    Y_reg_array = Y_reg.values.astype(np.float32)
    feature_names = X.columns.tolist()

    # Data augmentation (si se solicita)
    if augment:
        augmented_X = []
        augmented_Y_reg = []
        for _ in range(augment_factor):
            noise = np.random.normal(0, noise_std, X_array[:, :len(numeric_features)].shape)
            X_aug = X_array.copy()
            X_aug[:, :len(numeric_features)] += noise
            augmented_X.append(X_aug)
            augmented_Y_reg.append(Y_reg_array)

        X_array = np.vstack([X_array] + augmented_X)
        Y_reg_array = np.concatenate([Y_reg_array] + augmented_Y_reg)

    return X_array, Y_reg_array, scaler_numeric, feature_names
