import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib # Necesario para guardar y cargar el scaler y feature_names

def preprocess_data(df, augment=False, noise_std=0.01, augment_factor=1, 
                    scaler_obj=None, feature_names_obj=None):
    """
    Preprocesses data for model training or inference, with optional data augmentation.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing raw patient data.
        augment (bool): Whether to apply data augmentation. Defaults to False.
        noise_std (float): Standard deviation of Gaussian noise for augmentation. Defaults to 0.01.
        augment_factor (int): Factor by which to augment data. Defaults to 1.
        scaler_obj (RobustScaler, optional): Pre-fitted RobustScaler object for transforming numerical features.
                                             If None, a new scaler will be fitted.
        feature_names_obj (list, optional): List of feature names (including one-hot encoded ones) 
                                            from training. If None, new feature names will be generated.
                                            Crucial for aligning columns during inference.
                                            
    Returns:
        tuple: (X_array, Y_reg_array, fitted_scaler, final_feature_names)
               - X_array (np.array): Preprocessed feature matrix.
               - Y_reg_array (np.array): Target variable array.
               - fitted_scaler (RobustScaler): The fitted (or provided) RobustScaler.
               - final_feature_names (list): List of all feature names after preprocessing.
    """
    
    df = df.copy()
    
    # Renombrar columnas objetivo (asegurando minúsculas para consistencia)
    df.columns = df.columns.str.lower() # Convertir todas las columnas a minúsculas
    df = df.rename(columns={
        'risk_score': 'target_score',
        'cardiovascular_risk': 'target_risk_category'
    })

    # Convertir fecha a datetime
    df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')

    # Extraer características temporales
    df["diagnosis_year"] = df["diagnosis_date"].dt.year
    df["diagnosis_month"] = df["diagnosis_date"].dt.month
    df["diagnosis_trimester"] = "T" + df["diagnosis_date"].dt.quarter.astype(str)
    df["days_since_diagnosis"] = (pd.Timestamp.today() - df["diagnosis_date"]).dt.days
    df["pandemic_period"] = df["diagnosis_year"].apply(lambda y: 1 if y in [2020, 2021, 2022] else 0)

    # Variables objetivo
    # Asegurarse de que 'target_score' exista antes de intentar acceder a ella
    if 'target_score' in df.columns:
        Y_reg = df["target_score"].astype(float)
    else:
        # Si 'target_score' no está presente (ej. en datos de inferencia), crear un placeholder
        Y_reg = pd.Series([np.nan] * len(df), index=df.index) 


    # Definir features (nombres en minúsculas para consistencia)
    numeric_features = [
        'age', 'bmi', 'heart_rate', 'total_cholesterol', 'glucose',
        'is_smoker', 'bpm_meds', 'diabetes',
        'has_electricity', 'has_water_supply', 'has_gas', 
        'has_internet', 'family_history',
        'diagnosis_year', 'diagnosis_month', 'days_since_diagnosis', 'pandemic_period'
    ]

    categorical_features = [
        'department', 'municipality', 'sex', 'marital_status', 'rural_area',
        'education_level', 'socioeconomic_status', 'occupation', 'ethnicity',
        'climate_classification', # Añadir climate_classification aquí si está en df_mun y se fusiona
        'diagnosis_trimester'
    ]
    
    # Asegurarse de que 'climate_classification' solo se incluya si está presente en el DataFrame
    # (puede venir de la fusión con municipios_colombia.csv)
    if 'climate_classification' not in df.columns:
        categorical_features.remove('climate_classification')

    # Separar características, excluyendo las columnas objetivo y de fecha original
    columns_to_drop = ["target_risk_category", "target_score", "diagnosis_date"]
    # Filtrar solo las columnas que realmente existen en el DataFrame
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Convertir y limpiar datos
    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        else:
            # Añadir columna si no existe, rellenar con NaN
            X[col] = np.nan 
    
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype("category")
        else:
            # Añadir columna si no existe, rellenar con un valor que OneHotEncoder pueda manejar
            X[col] = 'missing' # O np.nan, dependiendo de cómo se maneje en el encoder


    # Rellenar valores numéricos faltantes con la media (o estrategia adecuada)
    # Esto es importante antes de escalar para evitar NaNs en el array final
    for col in numeric_features:
        if col in X.columns and X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())

    # Alinear Y_reg con X después de cualquier filtrado o eliminación de NaNs
    original_indices = X.index
    X.dropna(inplace=True) # Eliminar filas con NaNs restantes
    Y_reg = Y_reg.loc[X.index] # Alinear Y_reg con las filas restantes de X

    # Escalado de características numéricas
    if scaler_obj:
        # Usar scaler pre-entrenado para transformar
        scaler = scaler_obj
        X[numeric_features] = scaler.transform(X[numeric_features])
    else:
        # Entrenar y transformar un nuevo scaler
        scaler = RobustScaler()
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Codificación One-Hot de características categóricas
    # pd.get_dummies maneja nuevas categorías como nuevas columnas, pero para inferencia
    # es crucial alinear las columnas con las del entrenamiento.
    X_processed = pd.get_dummies(X, columns=categorical_features, dummy_na=False)

    # Alinear columnas si se proporcionan feature_names_obj (para inferencia)
    if feature_names_obj is not None:
        # Reindexar X_processed para que tenga las mismas columnas en el mismo orden
        # Rellenar con 0 si una columna no existe en los datos de entrada (para categorías no vistas)
        # Eliminar columnas que no estaban en el entrenamiento
        missing_cols = set(feature_names_obj) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0
        X_processed = X_processed[feature_names_obj] # Asegurar el orden correcto

    final_feature_names = X_processed.columns.tolist()

    # Convertir a arrays de NumPy
    X_array = X_processed.values.astype(np.float32)
    Y_reg_array = Y_reg.values.astype(np.float32)

    # Data augmentation (opcional)
    if augment and scaler_obj is None: # Solo aumentar si estamos en fase de entrenamiento (no se pasó un scaler)
        augmented_X = []
        augmented_Y_reg = []
        for _ in range(augment_factor):
            # Solo aplicar ruido a features numéricas originales (ya escaladas)
            # Asegurarse de que la forma del ruido coincida con las columnas numéricas escaladas
            noise = np.random.normal(0, noise_std, X_array[:, :len(numeric_features)].shape)
            X_aug = X_array.copy()
            X_aug[:, :len(numeric_features)] += noise
            augmented_X.append(X_aug)
            augmented_Y_reg.append(Y_reg_array)

        X_array = np.vstack([X_array] + augmented_X)
        Y_reg_array = np.concatenate([Y_reg_array] + augmented_Y_reg)

    return X_array, Y_reg_array, scaler, final_feature_names