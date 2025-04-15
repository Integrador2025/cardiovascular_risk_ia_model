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

        if 'DEPARTAMENTO' not in df.columns or 'PUNTAJE_RIESGO' not in df.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes para agrupar por departamento")

        departamentos = df['DEPARTAMENTO'].unique()
        resumen = []

        for depto in departamentos:
            df_depto = df[df['DEPARTAMENTO'] == depto]

            if df_depto.empty or len(df_depto) < 10:  # omitir grupos muy pequeños
                continue

            # Preprocesar solo este subconjunto
            X_depto, Y_depto, _, feature_names_local = preprocess_data(df_depto, augment=False)
            
            # Obtener pesos del modelo
            first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
            if first_dense is None:
                continue

            pesos = first_dense.get_weights()[0]
            importancia = dict(zip(feature_names, np.mean(np.abs(pesos), axis=1)))

            activacion_media = np.mean(X_depto, axis=0)
            importancia_ponderada = [
                (name, float(activacion * importancia.get(name, 0)))
                for name, activacion in zip(feature_names_local, activacion_media)
                if not name.startswith("DEPARTAMENTO_") and not name.startswith("MUNICIPIO_")
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

        if 'MUNICIPIO' not in df.columns or 'PUNTAJE_RIESGO' not in df.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes para agrupar por municipio")

        municipios = df['MUNICIPIO'].unique()
        resumen = []

        for mpio in municipios:
            df_mpio = df[df['MUNICIPIO'] == mpio]

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
                if not name.startswith("DEPARTAMENTO_") and not name.startswith("MUNICIPIO_")
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

        if "OCUPACION" not in df.columns or "PUNTAJE_RIESGO" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias")

        agrupado = df.groupby("OCUPACION").agg(
            promedio_riesgo=("PUNTAJE_RIESGO", "mean"),
            total_pacientes=("PUNTAJE_RIESGO", "count")
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

        # Eliminar filas con datos faltantes
        df = df[df["EDAD"].notnull() & df["PUNTAJE_RIESGO"].notnull()]

        bins = [0, 18, 30, 45, 60, 75, 120]
        labels = ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"]

        df["grupo_edad"] = pd.cut(df["EDAD"], bins=bins, labels=labels, right=False)

        agrupado = df.groupby("grupo_edad", observed=False).agg(
            promedio_riesgo=("PUNTAJE_RIESGO", "mean"),
            total_pacientes=("PUNTAJE_RIESGO", "count")
        ).reset_index()

        agrupado = agrupado.dropna(subset=["promedio_riesgo"])
        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        # Asegurar serialización compatible con JSON
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

        if "NIVEL_EDUCATIVO" not in df.columns or "PUNTAJE_RIESGO" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias")

        df = df[df["PUNTAJE_RIESGO"].notnull() & df["NIVEL_EDUCATIVO"].notnull()]

        agrupado = df.groupby("NIVEL_EDUCATIVO").agg(
            promedio_riesgo=("PUNTAJE_RIESGO", "mean"),
            total_pacientes=("PUNTAJE_RIESGO", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        agrupado["NIVEL_EDUCATIVO"] = agrupado["NIVEL_EDUCATIVO"].astype(str)
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

        if "ESTRATO" not in df.columns or "PUNTAJE_RIESGO" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias")

        df = df[df["PUNTAJE_RIESGO"].notnull() & df["ESTRATO"].notnull()]

        agrupado = df.groupby("ESTRATO").agg(
            promedio_riesgo=("PUNTAJE_RIESGO", "mean"),
            total_pacientes=("PUNTAJE_RIESGO", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        agrupado["ESTRATO"] = agrupado["ESTRATO"].astype(str)
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
        df["FECHA_DIAGNOSTICO"] = pd.to_datetime(df["FECHA_DIAGNOSTICO"], errors='coerce')
        df = df[df["PUNTAJE_RIESGO"].notnull() & df["FECHA_DIAGNOSTICO"].notnull()]

        df["AÑO_DIAGNOSTICO"] = df["FECHA_DIAGNOSTICO"].dt.year
        agrupado = df.groupby("AÑO_DIAGNOSTICO").agg(
            promedio_riesgo=("PUNTAJE_RIESGO", "mean"),
            total_pacientes=("PUNTAJE_RIESGO", "count")
        ).reset_index()

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["AÑO_DIAGNOSTICO"] = agrupado["AÑO_DIAGNOSTICO"].astype(int)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año: {str(e)}")

@router.get("/por-trimestre")
async def analisis_por_trimestre():
    try:
        df, _, _ = load.load_dataset()
        df["FECHA_DIAGNOSTICO"] = pd.to_datetime(df["FECHA_DIAGNOSTICO"], errors='coerce')
        df = df[df["PUNTAJE_RIESGO"].notnull() & df["FECHA_DIAGNOSTICO"].notnull()]

        df["TRIMESTRE"] = "T" + df["FECHA_DIAGNOSTICO"].dt.quarter.astype(str)
        agrupado = df.groupby("TRIMESTRE", observed=False).agg(
            promedio_riesgo=("PUNTAJE_RIESGO", "mean"),
            total_pacientes=("PUNTAJE_RIESGO", "count")
        ).reset_index()

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["TRIMESTRE"] = agrupado["TRIMESTRE"].astype(str)

        return agrupado.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por trimestre: {str(e)}")
