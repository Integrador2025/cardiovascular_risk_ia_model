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
    
    # Merge datasets using English column names
    df = df_pac.merge(df_mun, on=["department", "municipality"], how="left")
    
    # Process precipitation (using original English name)
    if "annual_precipitation" in df.columns:
        df["annual_precipitation"] = df["annual_precipitation"].astype(str).apply(
            lambda x: (float(x.split('-')[0]) + float(x.split('-')[1]))/2 if '-' in x else float(x) if x else np.nan
        )
    
    # Grouping with English column names
    grouped = df.groupby([
        "department", "municipality", "rural_area", "socioeconomic_status", "ethnicity",
        "occupation", "education_level"
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
        "average_altitude": "first",
        "average_temperature": "first",
        "annual_precipitation": "first",
        "climate_classification": "first",
        "estimated_population": "first"
    })
    
    return grouped

def preprocess_population_data(df):
    """Preprocessing for the model"""
    y = df["risk_score"].values
    
    # Remove unused columns (English names)
    X = df.drop(columns=["risk_score", "department", "municipality"])
    
    # Categorical and numerical columns (English names)
    categorical_cols = [
        "rural_area", "socioeconomic_status", "ethnicity", 
        "occupation", "education_level", "climate_classification"
    ]
    
    numerical_cols = [
        "age", "bmi", "total_cholesterol", "is_smoker", "family_history",
        "has_electricity", "has_water_supply", "has_sewage", "has_gas", "has_internet",
        "latitude", "longitude", "average_altitude", "average_temperature", "annual_precipitation"
    ]
    
    # Processing
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()
    
    X_cat = encoder.fit_transform(X[categorical_cols])
    X_num = scaler.fit_transform(X[numerical_cols])
    X_processed = np.concatenate([X_num, X_cat], axis=1)
    
    # Save preprocessors
    joblib.dump(encoder, "model/population_encoder.pkl")
    joblib.dump(scaler, "model/population_scaler.pkl")
    
    return train_test_split(X_processed, y, test_size=0.2, random_state=42)