import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def preprocess_data(df, augment=False, noise_std=0.01, augment_factor=1):
    """Preprocess data for model training, with optional data augmentation."""
    
    df = df.copy()
    
    # Rename target columns
    df = df.rename(columns={
        'risk_score': 'target_score',
        'cardiovascular_risk': 'target_risk_category'
    })

    # Convert date to datetime
    df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')

    # Extract temporal features - mantener trimestre como categórica
    df["diagnosis_year"] = df["diagnosis_date"].dt.year
    df["diagnosis_month"] = df["diagnosis_date"].dt.month
    df["diagnosis_trimester"] = "T" + df["diagnosis_date"].dt.quarter.astype(str)  # Cambiado a trimester
    df["days_since_diagnosis"] = (pd.Timestamp.today() - df["diagnosis_date"]).dt.days
    df["pandemic_period"] = df["diagnosis_year"].apply(lambda y: 1 if y in [2020, 2021, 2022] else 0)

    # Target variables
    Y_reg = df["target_score"].astype(float)

    # Definir features - diagnosis_trimester va en categóricas
    numeric_features = [
        'age', 'bmi', 'heart_rate', 'total_cholesterol', 'glucose',
        'is_smoker', 'bpm_meds', 'diabetes', 'rural_area',
        'has_electricity', 'has_water_supply', 'has_gas', 
        'has_internet', 'family_history',
        'diagnosis_year', 'diagnosis_month', 'days_since_diagnosis', 'pandemic_period'
    ]

    categorical_features = [
        'department', 'municipality', 'sex', 'marital_status',
        'education_level', 'socioeconomic_status', 'occupation', 'ethnicity',
        'diagnosis_trimester'  # Incluida como categórica
    ]

    # Separar características
    X = df.drop(columns=["target_risk_category", "target_score", "diagnosis_date"])

    # Convertir y limpiar datos
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    for col in categorical_features:
        X[col] = X[col].astype("category")

    # Verificación adicional
    non_numeric = X.select_dtypes(exclude=['number', 'category']).columns
    if not non_numeric.empty:
        raise ValueError(f"Columnas no procesadas correctamente: {list(non_numeric)}")

    X.dropna(inplace=True)
    Y_reg = Y_reg.loc[X.index]

    # Escalado y encoding
    scaler_numeric = RobustScaler()
    X[numeric_features] = scaler_numeric.fit_transform(X[numeric_features])
    X = pd.get_dummies(X, columns=categorical_features)

    # Convertir a arrays
    X_array = X.values.astype(np.float32)
    Y_reg_array = Y_reg.values.astype(np.float32)
    feature_names = X.columns.tolist()

    # Data augmentation (opcional)
    if augment:
        augmented_X = []
        augmented_Y_reg = []
        for _ in range(augment_factor):
            # Solo aplicar ruido a features numéricas originales
            noise = np.random.normal(0, noise_std, X_array[:, :len(numeric_features)].shape)
            X_aug = X_array.copy()
            X_aug[:, :len(numeric_features)] += noise
            augmented_X.append(X_aug)
            augmented_Y_reg.append(Y_reg_array)

        X_array = np.vstack([X_array] + augmented_X)
        Y_reg_array = np.concatenate([Y_reg_array] + augmented_Y_reg)

    return X_array, Y_reg_array, scaler_numeric, feature_names