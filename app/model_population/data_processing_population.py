import pandas as pd
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

def normalize_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    return df

def convertir_precipitacion(valor):
    try:
        partes = valor.split('-')
        return (float(partes[0]) + float(partes[1])) / 2
    except:
        return None

def load_and_group_population_data(pacientes_path, municipios_path):
    df_pac = pd.read_csv(pacientes_path)
    df_mun = pd.read_csv(municipios_path)

    df_pac = normalize_column_names(df_pac)
    df_mun = normalize_column_names(df_mun)

    df = df_pac.merge(df_mun, on=["DEPARTAMENTO", "MUNICIPIO"], how="left")

    if "PRECIPITACION ANUAL" in df.columns:
        df["PRECIPITACION ANUAL"] = df["PRECIPITACION ANUAL"].astype(str).apply(convertir_precipitacion)

    grouped = df.groupby([
        "DEPARTAMENTO", "MUNICIPIO", "AREA", "ESTRATO", "ETNIA",
        "OCUPACION", "NIVEL_EDUCATIVO"
    ], as_index=False).agg({
        "EDAD": "mean",
        "IMC": "mean",
        "COLESTEROL": "mean",
        "FUMADOR": "mean",
        "ANTECEDENTES_FAMILIARES": "mean",
        "ACCESO_ELECTRICO": "mean",
        "ACUEDUCTO": "mean",
        "ALCANTARILLADO": "mean",
        "GAS_NATURAL": "mean",
        "INTERNET": "mean",
        "PUNTAJE_RIESGO": "mean",
        "LATITUD": "first",
        "LONGITUD": "first",
        "ALTITUD MEDIA": "first",
        "TEMPERATURA PROMEDIO": "first",
        "PRECIPITACION ANUAL": "first",
        "CLASIFICACION CLIMATICA": "first",
        "POBLACION ESTIMADA": "first"
    })

    return grouped

def preprocess_population_data(df):
    y = df["PUNTAJE_RIESGO"].values

    X = df.drop(columns=["PUNTAJE_RIESGO", "DEPARTAMENTO", "MUNICIPIO"])

    # Identificar columnas categóricas y numéricas
    categorical_cols = ["AREA", "ESTRATO", "ETNIA", "OCUPACION", "NIVEL_EDUCATIVO", "CLASIFICACION CLIMATICA"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Codificación y escalado
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_cat = encoder.fit_transform(X[categorical_cols])

    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(X[numerical_cols])

    import numpy as np
    X_processed = np.concatenate([scaled_num, encoded_cat], axis=1)

    # Guardar preprocesadores
    joblib.dump(encoder, "model/pop_encoder.pkl")
    joblib.dump(scaler, "model/pop_scaler.pkl")
    joblib.dump(X.columns.tolist(), "model/pop_columns.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
