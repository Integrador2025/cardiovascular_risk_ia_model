from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data
import load

router = APIRouter(
    prefix="/v1/summary",
    tags=["Resumen por Departamento"]
)

def categorizar_riesgo(promedio: float) -> str:
    if promedio < 0.2:
        return "Muy Bajo"
    elif promedio < 0.4:
        return "Bajo"
    elif promedio < 0.6:
        return "Moderado"
    elif promedio < 0.8:
        return "Alto"
    else:
        return "Muy Alto"


@router.get("/por-departamento")
async def resumen_por_departamento(top_n: int = 5):
    """
    Retorna resumen poblacional por departamento:
    - Número de pacientes
    - Puntaje promedio
    - Clasificación de riesgo
    - Principales factores según activación ponderada
    """
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")

        df, _, _ = load.load_dataset()

        if 'department' not in df.columns or 'risk_score' not in df.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes para agrupar por departamento")

        departamentos = df['department'].unique()
        resumen = []

        for depto in departamentos:
            df_depto = df[df['department'] == depto]

            if df_depto.empty or len(df_depto) < 10:
                continue

            X_depto, Y_depto, _, feature_names_local = preprocess_data(df_depto, augment=False)
            
            first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
            if first_dense is None:
                continue

            pesos = first_dense.get_weights()[0]
            importancia = dict(zip(feature_names, np.mean(np.abs(pesos), axis=1)))

            activacion_media = np.mean(X_depto, axis=0)
            importancia_ponderada = [
                (name, float(activacion * importancia.get(name, 0)))
                for name, activacion in zip(feature_names_local, activacion_media)
                if not name.startswith("department_") and not name.startswith("municipality_")
            ]
            importancia_ponderada.sort(key=lambda x: x[1], reverse=True)

            resumen.append({
                "departamento": depto,
                "pacientes": int(len(df_depto)),
                "promedio_riesgo": round(float(Y_depto.mean()), 4),
                "categoria": categorizar_riesgo(float(Y_depto.mean())),
                "top_factores": importancia_ponderada[:top_n]
            })

        return {"departamentos": resumen}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar resumen: {str(e)}")

@router.get("/por-municipio")
async def resumen_por_municipio(top_n: int = 5, min_pacientes: int = 10):
    """
    Resumen poblacional por municipio:
    - Promedio de puntaje de riesgo
    - Clasificación (bajo, medio, alto)
    - Principales factores según activación ponderada
    """
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")

        df, _, _ = load.load_dataset()

        if 'municipality' not in df.columns or 'risk_score' not in df.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes para agrupar por municipio")

        municipios = df['municipality'].unique()
        resumen = []

        for mpio in municipios:
            df_mpio = df[df['municipality'] == mpio]

            if df_mpio.empty or len(df_mpio) < min_pacientes:
                continue

            X_mpio, Y_mpio, _, feature_names_local = preprocess_data(df_mpio, augment=False)

            first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
            if first_dense is None:
                continue

            pesos = first_dense.get_weights()[0]
            importancia = dict(zip(feature_names, np.mean(np.abs(pesos), axis=1)))

            activacion_media = np.mean(X_mpio, axis=0)
            importancia_ponderada = [
                (name, float(activacion * importancia.get(name, 0)))
                for name, activacion in zip(feature_names_local, activacion_media)
                if not name.startswith("department_") and not name.startswith("municipality_")
            ]
            importancia_ponderada.sort(key=lambda x: x[1], reverse=True)

            resumen.append({
                "municipio": mpio,
                "pacientes": int(len(df_mpio)),
                "promedio_riesgo": round(float(Y_mpio.mean()), 4),
                "categoria": categorizar_riesgo(float(Y_mpio.mean())),
                "top_factores": importancia_ponderada[:top_n]
            })

        return {"municipios": resumen}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar resumen por municipio: {str(e)}")

@router.get("/por-ocupacion")
async def analisis_por_ocupacion(min_pacientes: int = 10):
    try:
        df, _, _ = load.load_dataset()

        if "occupation" not in df.columns or "risk_score" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias")

        agrupado = df.groupby("occupation").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por ocupación: {str(e)}")

@router.get("/por-edad")
async def analisis_por_edad():
    try:
        df, _, _ = load.load_dataset()

        df = df[df["age"].notnull() & df["risk_score"].notnull()]

        bins = [0, 18, 30, 45, 60, 75, 120]
        labels = ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"]

        df["grupo_edad"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

        agrupado = df.groupby("grupo_edad", observed=False).agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado.dropna(subset=["promedio_riesgo"])
        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        agrupado["grupo_edad"] = agrupado["grupo_edad"].astype(str)
        agrupado["promedio_riesgo"] = agrupado["promedio_riesgo"].astype(float)
        agrupado["total_pacientes"] = agrupado["total_pacientes"].astype(int)
        agrupado["categoria"] = agrupado["categoria"].astype(str)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por edad: {str(e)}")

@router.get("/por-nivel-educativo")
async def analisis_por_nivel_educativo(min_pacientes: int = 10):
    try:
        df, _, _ = load.load_dataset()

        if "education_level" not in df.columns or "risk_score" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias")

        df = df[df["risk_score"].notnull() & df["education_level"].notnull()]

        agrupado = df.groupby("education_level").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        agrupado["education_level"] = agrupado["education_level"].astype(str)
        agrupado["promedio_riesgo"] = agrupado["promedio_riesgo"].astype(float)
        agrupado["total_pacientes"] = agrupado["total_pacientes"].astype(int)
        agrupado["categoria"] = agrupado["categoria"].astype(str)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por nivel educativo: {str(e)}")

@router.get("/por-estrato")
async def analisis_por_estrato(min_pacientes: int = 10):
    try:
        df, _, _ = load.load_dataset()

        if "socioeconomic_status" not in df.columns or "risk_score" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias")

        df = df[df["risk_score"].notnull() & df["socioeconomic_status"].notnull()]

        agrupado = df.groupby("socioeconomic_status").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        agrupado["socioeconomic_status"] = agrupado["socioeconomic_status"].astype(str)
        agrupado["promedio_riesgo"] = agrupado["promedio_riesgo"].astype(float)
        agrupado["total_pacientes"] = agrupado["total_pacientes"].astype(int)
        agrupado["categoria"] = agrupado["categoria"].astype(str)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por estrato: {str(e)}")

@router.get("/por-año")
async def analisis_por_año():
    try:
        df, _, _ = load.load_dataset()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["risk_score"].notnull() & df["diagnosis_date"].notnull()]

        df["diagnosis_year"] = df["diagnosis_date"].dt.year
        agrupado = df.groupby("diagnosis_year").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["diagnosis_year"] = agrupado["diagnosis_year"].astype(int)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año: {str(e)}")

@router.get("/por-trimestre")
async def analisis_por_trimestre():
    try:
        df, _, _ = load.load_dataset()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["risk_score"].notnull() & df["diagnosis_date"].notnull()]

        df["diagnosis_quarter"] = "T" + df["diagnosis_date"].dt.quarter.astype(str)
        agrupado = df.groupby("diagnosis_quarter", observed=False).agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["diagnosis_quarter"] = agrupado["diagnosis_quarter"].astype(str)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por trimestre: {str(e)}")