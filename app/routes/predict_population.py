from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from app.model_population.data_processing_population import normalize_column_names, convertir_precipitacion
from app.model_config import PACIENTES_PATH, MUNICIPIOS_PATH
from app.model_population.utils_population import load_population_model

router = APIRouter()

@router.get("/riesgo-poblacional/{municipio}")
async def predecir_riesgo_poblacional(municipio: str):
    try:
        df_pac = pd.read_csv(PACIENTES_PATH)
        df_mun = pd.read_csv(MUNICIPIOS_PATH)

        df_pac = normalize_column_names(df_pac)
        df_mun = normalize_column_names(df_mun)

        df = df_pac.merge(df_mun, on=["DEPARTAMENTO", "MUNICIPIO"], how="left")

        if "PRECIPITACION ANUAL" in df.columns:
            df["PRECIPITACION ANUAL"] = df["PRECIPITACION ANUAL"].astype(str).apply(convertir_precipitacion)

        df_mpio = df[df["MUNICIPIO"].str.strip().str.upper() == municipio.strip().upper()]
        if df_mpio.empty:
            raise HTTPException(status_code=404, detail="Municipio no encontrado")

        grupos = df_mpio.groupby([
            "AREA", "ESTRATO", "ETNIA", "OCUPACION", "NIVEL_EDUCATIVO",
            "CLASIFICACION CLIMATICA"
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
            "LATITUD": "first",
            "LONGITUD": "first",
            "ALTITUD MEDIA": "first",
            "TEMPERATURA PROMEDIO": "first",
            "PRECIPITACION ANUAL": "first",
            "POBLACION ESTIMADA": "first"
        })

        model, scaler, encoder, feature_names = load_population_model()

        categorical_cols = ["AREA", "ESTRATO", "ETNIA", "OCUPACION", "NIVEL_EDUCATIVO", "CLASIFICACION CLIMATICA"]
        numerical_cols = [col for col in grupos.columns if col not in categorical_cols]

        encoded_cat = encoder.transform(grupos[categorical_cols])
        scaled_num = scaler.transform(grupos[numerical_cols])
        X = np.concatenate([scaled_num, encoded_cat], axis=1)

        predicciones = model.predict(X).flatten().tolist()

        grupos["RIESGO_ESTIMADO"] = predicciones

        return {
            "municipio": municipio,
            "n_grupos": len(grupos),
            "riesgo_promedio_total": round(np.mean(predicciones), 4),
            "grupos": grupos[categorical_cols + ["RIESGO_ESTIMADO"]].to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la prediccion: {str(e)}")
