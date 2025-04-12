import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

def preprocess_data(df, augment=False, noise_std=0.01, augment_factor=1):
    """Preprocesa los datos para el entrenamiento del modelo, con opción de data augmentation."""

    # Variables objetivo
    Y_reg = df['PUNTAJE_RIESGO'].astype(float)

    # Definir columnas numéricas y categóricas
    numeric_features = [
        'EDAD', 'IMC', 'SEXO', 'COLESTEROL',
        'ACCESO_ELECTRICO', 'ACUEDUCTO', 'ALCANTARILLADO', 'GAS_NATURAL',
        'ANTECEDENTES_FAMILIARES', 'FUMADOR'
    ]

    categorical_features = [
        'DEPARTAMENTO', 'MUNICIPIO', 'ESTADO_CIVIL', 'AREA', 'ESTRATO', 'NIVEL_EDUCATIVO',
        'INTERNET', 'ETNIA', 'OCUPACION'
    ]

    # Eliminar columnas objetivo de las características
    X = df.drop(['RIESGO_CARDIOVASCULAR', 'PUNTAJE_RIESGO'], axis=1)

    # Asegurar que las columnas numéricas sean tipo float
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Asegurar que las columnas categóricas sean tipo 'category'
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Eliminar filas con NaN
    X.dropna(inplace=True)
    Y_reg = Y_reg[X.index]

    # Escalado de características numéricas
    scaler_numeric = RobustScaler()
    X[numeric_features] = scaler_numeric.fit_transform(X[numeric_features])

    # One-hot encoding para variables categóricas (mantiene nombres reales)
    X = pd.get_dummies(X, columns=categorical_features)

    # Convertir a arrays de numpy
    X_array = X.values.astype(np.float32)
    Y_reg_array = Y_reg.values.astype(np.float32)

    feature_names = X.columns.tolist()

    if augment:
        augmented_X = []
        augmented_Y_reg = []
        for _ in range(augment_factor):
            noise = np.random.normal(0, noise_std, X_array[:, :len(numeric_features)].shape)
            X_aug = X_array.copy()
            X_aug[:, :len(numeric_features)] += noise  # Solo agregar ruido a columnas numéricas
            augmented_X.append(X_aug)
            augmented_Y_reg.append(Y_reg_array)

        X_array = np.vstack([X_array] + augmented_X)
        Y_reg_array = np.concatenate([Y_reg_array] + augmented_Y_reg)

    return X_array, Y_reg_array, scaler_numeric, feature_names