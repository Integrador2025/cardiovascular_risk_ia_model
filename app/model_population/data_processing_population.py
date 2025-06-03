import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

def convert_precipitation(value):
    """
    Convert precipitation values from various formats to numerical values.
    Handles:
    - Ranges (e.g., '1000-1500' → 1250)
    - Single values (e.g., '1200' → 1200.0)
    - Missing/empty values (→ np.nan)
    
    Args:
        value: Input precipitation value (str, float, or None)
        
    Returns:
        float or np.nan: Converted numerical value
    """
    # Handle missing/empty values
    if pd.isna(value) or value == '':
        return np.nan
    
    # Convert to string if not already
    str_value = str(value).strip()
    
    try:
        # Handle range format (e.g., "1000-1500")
        if '-' in str_value:
            parts = [float(p.strip()) for p in str_value.split('-')]
            return sum(parts) / len(parts)  # Return average of range
        
        # Handle single value
        return float(str_value)
        
    except (ValueError, TypeError):
        # Handle any conversion errors
        return np.nan

def load_and_group_population_data(pacientes_path, municipios_path):
    """Load and process patient and municipality data"""
    # Load data keeping original English column names
    df_pac = pd.read_csv(pacientes_path)
    df_mun = pd.read_csv(municipios_path)
    
    # Convert column names to lowercase for consistency
    df_pac.columns = df_pac.columns.str.lower()
    df_mun.columns = df_mun.columns.str.lower()

    # Merge datasets using English column names
    df = df_pac.merge(df_mun, on=["department", "municipality"], how="left")
    
    # Process precipitation (using original English name)
    if "annual_precipitation" in df.columns:
        df["annual_precipitation"] = df["annual_precipitation"].astype(str).apply(convert_precipitation)
    
    # Grouping with English column names
    grouped = df.groupby([
        "department", "municipality", "rural_area", "socioeconomic_status", "ethnicity",
        "occupation", "education_level", "climate_classification" # Añadir climate_classification aquí
    ], as_index=False).agg({
        "age": "mean",
        "bmi": "mean",
        "total_cholesterol": "mean",
        "is_smoker": "mean",
        "family_history": "mean",
        "has_electricity": "mean",
        "has_water_supply": "mean",
        "has_sewage": "mean",
        "has_gas": "mean",
        "has_internet": "mean",
        "risk_score": "mean",
        "latitude": "first",
        "longitude": "first",
        "average_altitude": "first", # Se mantiene si ya está en el CSV o se añade
        "average_temperature": "first",
        "annual_precipitation": "first",
        "estimated_population": "first",
        "literacy_rate": "first",  # Nueva columna
        "poverty_rate": "first"    # Nueva columna
    })
    
    return grouped

def preprocess_population_data(df, scaler_obj=None, encoder_obj=None, feature_names_obj=None, augment=False, noise_std=0.01):
    """
    Preprocessing for the population model. Can fit new preprocessors or use existing ones.
    
    Args:
        df (pd.DataFrame): Input DataFrame (grouped population data).
        scaler_obj (StandardScaler, optional): Pre-fitted StandardScaler. If None, a new one is fitted.
        encoder_obj (OneHotEncoder, optional): Pre-fitted OneHotEncoder. If None, a new one is fitted.
        feature_names_obj (list, optional): List of feature names from training. Used for alignment during inference.
        augment (bool): Whether to apply data augmentation (add noise to numerical features). Defaults to False.
        noise_std (float): Standard deviation of Gaussian noise for augmentation. Defaults to 0.01.
        
    Returns:
        tuple: (X_processed, y, fitted_scaler, fitted_encoder, final_feature_names)
               - X_processed (np.array): Preprocessed feature matrix.
               - y (np.array): Target variable array.
               - fitted_scaler (StandardScaler): The fitted (or provided) StandardScaler.
               - fitted_encoder (OneHotEncoder): The fitted (or provided) OneHotEncoder.
               - final_feature_names (list): List of all feature names after preprocessing.
    """
    y = df["risk_score"].values
    
    # Remove unused columns (English names)
    # Asegurarse de que 'department' y 'municipality' existan antes de intentar eliminarlas
    columns_to_drop = ["risk_score"]
    if "department" in df.columns:
        columns_to_drop.append("department")
    if "municipality" in df.columns:
        columns_to_drop.append("municipality")

    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # Categorical and numerical columns (English names)
    categorical_cols = [
        "rural_area", "socioeconomic_status", "ethnicity", 
        "occupation", "education_level", "climate_classification"
    ]
    
    numerical_cols = [
        "age", "bmi", "total_cholesterol", "is_smoker", "family_history",
        "has_electricity", "has_water_supply", "has_sewage", "has_gas", "has_internet",
        "latitude", "longitude", "average_altitude", # average_altitude se mantiene si está en el CSV
        "average_temperature", "annual_precipitation",
        "estimated_population", "literacy_rate", "poverty_rate" # Incluir estimated_population, literacy_rate, poverty_rate como numéricas
    ]

    # Filtrar columnas numéricas y categóricas para asegurar que solo se procesen las que existen en X
    numerical_cols = [col for col in numerical_cols if col in X.columns]
    categorical_cols = [col for col in categorical_cols if col in X.columns]

    # Rellenar valores numéricos faltantes con la media
    for col in numerical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())

    # Rellenar valores categóricos faltantes con un placeholder
    for col in categorical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna('Missing') # O el valor que consideres más adecuado

    # Procesamiento de características numéricas
    if scaler_obj:
        scaler = scaler_obj
        X_num = scaler.transform(X[numerical_cols])
    else:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X[numerical_cols])
    
    # Procesamiento de características categóricas
    if encoder_obj:
        encoder = encoder_obj
        # handle_unknown='ignore' es crucial para inferencia con categorías no vistas
        X_cat = encoder.transform(X[categorical_cols])
    else:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(X[categorical_cols])
    
    X_processed_df = pd.DataFrame(X_num, columns=numerical_cols)
    # Obtener los nombres de las columnas del encoder para las características categóricas
    if categorical_cols: # Solo si hay columnas categóricas
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
        X_processed_df = pd.concat([X_processed_df, pd.DataFrame(X_cat, columns=encoded_feature_names)], axis=1)

    # Alinear columnas si se proporcionan feature_names_obj (para inferencia)
    if feature_names_obj is not None:
        # Asegurarse de que todas las columnas esperadas estén presentes, rellenando con 0 si faltan
        missing_cols = set(feature_names_obj) - set(X_processed_df.columns)
        for c in missing_cols:
            X_processed_df[c] = 0
        # Asegurar que el orden de las columnas sea el mismo que durante el entrenamiento
        X_processed_df = X_processed_df[feature_names_obj]
    
    final_feature_names = X_processed_df.columns.tolist()
    X_processed = X_processed_df.values

    # Data augmentation (añadir ruido a características numéricas)
    if augment and scaler_obj is None: # Solo aplicar aumento si estamos en fase de entrenamiento (no se pasó un scaler)
        # Identificar las columnas numéricas en el array X_processed
        # Esto asume que las columnas numéricas están al principio después de la concatenación
        # Una forma más robusta sería guardar los índices de las columnas numéricas
        num_features_count = len(numerical_cols)
        noise = np.random.normal(0, noise_std, X_processed[:, :num_features_count].shape)
        X_processed[:, :num_features_count] += noise
        
    return X_processed, y, scaler, encoder, final_feature_names