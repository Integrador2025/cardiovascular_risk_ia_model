# app/routes/summary.py

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data
import load

router = APIRouter(
    prefix="/v1/summary",
    tags=["Resumen por Departamento"] # Cambiado el tag, ahora incluye más que solo departamento
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
async def resumen_por_departamento(top_n: int = Query(5, description="Number of top factors to return per department."),
                                   min_patients: int = Query(10, description="Minimum number of patients required for a department to be included.")):
    """
    Returns population summary by department:
    - Number of patients
    - Average risk score
    - Risk classification
    - Top factors based on weighted activation
    """
    try:
        # Cargar modelo, scaler y feature_names
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df_full, _, _ = load.load_dataset()
        
        # Asegurarse de que los nombres de las columnas estén en minúsculas
        df_full.columns = df_full.columns.str.lower()

        if 'department' not in df_full.columns or 'risk_score' not in df_full.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes para agrupar por departamento: 'department' o 'risk_score' no encontrados.")

        departamentos = df_full['department'].unique()
        resumen = []

        # Obtener los pesos del modelo una sola vez
        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo.")
        pesos_global = first_dense.get_weights()[0]
        importancia_global = dict(zip(feature_names_global, np.mean(np.abs(pesos_global), axis=1)))


        for depto in departamentos:
            df_depto = df_full[df_full['department'] == depto].copy() # Usar .copy() para evitar SettingWithCopyWarning

            if df_depto.empty or len(df_depto) < min_patients:
                continue

            # Asegurarse de que df_depto tenga las columnas objetivo dummy si es necesario para preprocess_data
            if 'risk_score' not in df_depto.columns:
                df_depto['risk_score'] = 0.0 # Placeholder
            if 'cardiovascular_risk' not in df_depto.columns:
                df_depto['cardiovascular_risk'] = 0 # Placeholder
            if 'diagnosis_date' not in df_depto.columns:
                df_depto['diagnosis_date'] = pd.Timestamp("2024-01-01") # Placeholder
            
            # Preprocesar datos del departamento usando el scaler y feature_names_global
            X_depto, Y_depto_original, _, _ = preprocess_data(df_depto, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            
            # Calcular la activación media para este departamento
            activacion_media = np.mean(X_depto, axis=0)
            
            importancia_ponderada = []
            for name, activacion in zip(feature_names_global, activacion_media):
                imp = importancia_global.get(name, 0.0)
                # Excluir características geográficas específicas si son parte del nombre
                if not (name.startswith("department_") or name.startswith("municipality_")):
                    importancia_ponderada.append((name, float(activacion * imp)))
            
            importancia_ponderada.sort(key=lambda x: x[1], reverse=True)

            resumen.append({
                "departamento": depto,
                "pacientes": int(len(df_depto)),
                "promedio_riesgo": round(float(Y_depto_original.mean()), 4), # Usar el riesgo original del dataset
                "categoria": categorizar_riesgo(float(Y_depto_original.mean())),
                "top_factores": importancia_ponderada[:top_n]
            })

        return {"departamentos": resumen}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar resumen por departamento: {str(e)}")

@router.get("/por-municipio")
async def resumen_por_municipio(top_n: int = Query(5, description="Number of top factors to return per municipality."), 
                                min_pacientes: int = Query(10, description="Minimum number of patients required for a municipality to be included.")):
    """
    Population summary by municipality:
    - Average risk score
    - Classification (low, medium, high)
    - Top factors based on weighted activation
    """
    try:
        # Cargar modelo, scaler y feature_names
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df_full, _, _ = load.load_dataset()
        
        # Asegurarse de que los nombres de las columnas estén en minúsculas
        df_full.columns = df_full.columns.str.lower()

        if 'municipality' not in df_full.columns or 'risk_score' not in df_full.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes para agrupar por municipio: 'municipality' o 'risk_score' no encontrados.")

        municipios = df_full['municipality'].unique()
        resumen = []

        # Obtener los pesos del modelo una sola vez
        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo.")
        pesos_global = first_dense.get_weights()[0]
        importancia_global = dict(zip(feature_names_global, np.mean(np.abs(pesos_global), axis=1)))


        for mpio in municipios:
            df_mpio = df_full[df_full['municipality'] == mpio].copy() # Usar .copy()

            if df_mpio.empty or len(df_mpio) < min_pacientes:
                continue

            # Asegurarse de que df_mpio tenga las columnas objetivo dummy si es necesario para preprocess_data
            if 'risk_score' not in df_mpio.columns:
                df_mpio['risk_score'] = 0.0 # Placeholder
            if 'cardiovascular_risk' not in df_mpio.columns:
                df_mpio['cardiovascular_risk'] = 0 # Placeholder
            if 'diagnosis_date' not in df_mpio.columns:
                df_mpio['diagnosis_date'] = pd.Timestamp("2024-01-01") # Placeholder

            # Preprocesar datos del municipio usando el scaler y feature_names_global
            X_mpio, Y_mpio_original, _, _ = preprocess_data(df_mpio, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)

            activacion_media = np.mean(X_mpio, axis=0)
            importancia_ponderada = [
                (name, float(activacion * importancia_global.get(name, 0)))
                for name, activacion in zip(feature_names_global, activacion_media)
                if not name.startswith("department_") and not name.startswith("municipality_")
            ]
            importancia_ponderada.sort(key=lambda x: x[1], reverse=True)

            resumen.append({
                "municipio": mpio,
                "pacientes": int(len(df_mpio)),
                "promedio_riesgo": round(float(Y_mpio_original.mean()), 4), # Usar el riesgo original del dataset
                "categoria": categorizar_riesgo(float(Y_mpio_original.mean())),
                "top_factores": importancia_ponderada[:top_n]
            })

        return {"municipios": resumen}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar resumen por municipio: {str(e)}")

@router.get("/por-ocupacion")
async def analisis_por_ocupacion(min_pacientes: int = Query(10, description="Minimum number of patients required for an occupation group to be included.")):
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        if "occupation" not in df.columns or "risk_score" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias: 'occupation' o 'risk_score'.")

        df = df[df["risk_score"].notnull() & df["occupation"].notnull()] # Filtrar nulos

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        agrupado = df.groupby("occupation").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        
        if agrupado.empty:
            raise HTTPException(status_code=404, detail=f"No occupation groups found with at least {min_pacientes} patients.")

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        # Convertir a tipos nativos de Python para JSON
        results = agrupado.to_dict(orient="records")
        for item in results:
            item["promedio_riesgo"] = round(float(item["promedio_riesgo"]), 4)
            item["total_pacientes"] = int(item["total_pacientes"])
            item["occupation"] = str(item["occupation"]) # Asegurar que sea string

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por ocupación: {str(e)}")

@router.get("/por-edad")
async def analisis_por_edad():
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        df = df[df["age"].notnull() & df["risk_score"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        bins = [0, 18, 30, 45, 60, 75, 120]
        labels = ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"]

        df["grupo_edad"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

        agrupado = df.groupby("grupo_edad", observed=False).agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado.dropna(subset=["promedio_riesgo"]) # Eliminar grupos con promedio NaN
        
        if agrupado.empty:
            raise HTTPException(status_code=404, detail="No age groups found with valid data.")

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        # Convertir a tipos nativos de Python para JSON
        agrupado["grupo_edad"] = agrupado["grupo_edad"].astype(str)
        agrupado["promedio_riesgo"] = agrupado["promedio_riesgo"].astype(float)
        agrupado["total_pacientes"] = agrupado["total_pacientes"].astype(int)
        agrupado["categoria"] = agrupado["categoria"].astype(str)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por edad: {str(e)}")

@router.get("/por-nivel-educativo")
async def analisis_por_nivel_educativo(min_pacientes: int = Query(10, description="Minimum number of patients required for an education level group to be included.")):
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        if "education_level" not in df.columns or "risk_score" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias: 'education_level' o 'risk_score'.")

        df = df[df["risk_score"].notnull() & df["education_level"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        agrupado = df.groupby("education_level").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        
        if agrupado.empty:
            raise HTTPException(status_code=404, detail=f"No education level groups found with at least {min_pacientes} patients.")

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        # Convertir a tipos nativos de Python para JSON
        agrupado["education_level"] = agrupado["education_level"].astype(str)
        agrupado["promedio_riesgo"] = agrupado["promedio_riesgo"].astype(float)
        agrupado["total_pacientes"] = agrupado["total_pacientes"].astype(int)
        agrupado["categoria"] = agrupado["categoria"].astype(str)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por nivel educativo: {str(e)}")

@router.get("/por-estrato")
async def analisis_por_estrato(min_pacientes: int = Query(10, description="Minimum number of patients required for a socioeconomic status group to be included.")):
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        if "socioeconomic_status" not in df.columns or "risk_score" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias: 'socioeconomic_status' o 'risk_score'.")

        df = df[df["risk_score"].notnull() & df["socioeconomic_status"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        agrupado = df.groupby("socioeconomic_status").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado = agrupado[agrupado["total_pacientes"] >= min_pacientes]
        
        if agrupado.empty:
            raise HTTPException(status_code=404, detail=f"No socioeconomic status groups found with at least {min_pacientes} patients.")

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)

        # Convertir a tipos nativos de Python para JSON
        agrupado["socioeconomic_status"] = agrupado["socioeconomic_status"].astype(str)
        agrupado["promedio_riesgo"] = agrupado["promedio_riesgo"].astype(float)
        agrupado["total_pacientes"] = agrupado["total_pacientes"].astype(int)
        agrupado["categoria"] = agrupado["categoria"].astype(str)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por estrato: {str(e)}")

@router.get("/por-año")
async def analisis_por_año():
    """
    Returns annual cardiovascular risk analysis globally.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["risk_score"].notnull() & df["diagnosis_date"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        df["diagnosis_year"] = df["diagnosis_date"].dt.year
        agrupado = df.groupby("diagnosis_year").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["diagnosis_year"] = agrupado["diagnosis_year"].astype(int)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año: {str(e)}")


@router.get("/por-año-departamento/{departamento}")
async def analisis_por_año_departamento(departamento: str):
    """
    Returns annual cardiovascular risk analysis for a specific department.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        # Convert diagnosis_date to datetime FIRST
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')

        # Check if the department actually exists in the original dataset before any cleaning
        if departamento.upper() not in df['department'].str.upper().unique():
            raise HTTPException(status_code=404, detail=f"Department '{departamento}' not found in the dataset. Please check the spelling.")

        # Filter by department first
        df_filtrado = df[df["department"].str.upper() == departamento.upper()].copy() # Use .copy()

        # Now apply cleaning filters on the filtered DataFrame
        df_filtrado = df_filtrado[
            df_filtrado["risk_score"].notnull() & 
            df_filtrado["diagnosis_date"].notnull()
        ] # department.notnull() is implicitly true now

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No valid patient data (missing risk score or diagnosis date) found for department '{departamento}' after cleaning.")

        df_filtrado["diagnosis_year"] = df_filtrado["diagnosis_date"].dt.year
        agrupado = df_filtrado.groupby("diagnosis_year").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        if agrupado.empty:
            raise HTTPException(status_code=404, detail=f"No annual data found for department '{departamento}'.")

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["diagnosis_year"] = agrupado["diagnosis_year"].astype(int)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año para departamento '{departamento}': {str(e)}")


@router.get("/por-año-municipio/{municipio}")
async def analisis_por_año_municipio(municipio: str):
    """
    Returns annual cardiovascular risk analysis for a specific municipality.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        # Convert diagnosis_date to datetime FIRST
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')

        # Check if the municipality actually exists in the original dataset before any cleaning
        if municipio.upper() not in df['municipality'].str.upper().unique():
            raise HTTPException(status_code=404, detail=f"Municipality '{municipio}' not found in the dataset. Please check the spelling.")

        # Filter by municipality first
        df_filtrado = df[df["municipality"].str.upper() == municipio.upper()].copy() # Use .copy()

        # Now apply cleaning filters on the filtered DataFrame
        df_filtrado = df_filtrado[
            df_filtrado["risk_score"].notnull() & 
            df_filtrado["diagnosis_date"].notnull()
        ] # municipality.notnull() is implicitly true now

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No valid patient data (missing risk score or diagnosis date) found for municipality '{municipio}' after cleaning.")

        df_filtrado["diagnosis_year"] = df_filtrado["diagnosis_date"].dt.year
        agrupado = df_filtrado.groupby("diagnosis_year").agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        if agrupado.empty:
            raise HTTPException(status_code=404, detail=f"No annual data found for municipality '{municipio}'.")

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["diagnosis_year"] = agrupado["diagnosis_year"].astype(int)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año para municipio '{municipio}': {str(e)}")


@router.get("/total-pacientes-por-año")
async def total_pacientes_por_año():
    """
    Returns the total number of patients per year globally.
    Useful for trend analysis of data volume over time.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Ensure lowercase column names

        # Filter out rows where diagnosis_date or risk_score are null,
        # as these are essential for determining a "patient record" for this context.
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["diagnosis_date"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        df["diagnosis_year"] = df["diagnosis_date"].dt.year
        
        # Group by year and count the number of patients
        agrupado = df.groupby("diagnosis_year").agg(
            total_pacientes=("diagnosis_year", "count") # Count occurrences of the year, effectively counting patients
        ).reset_index()

        if agrupado.empty:
            raise HTTPException(status_code=404, detail="No patient data found per year.")

        agrupado["diagnosis_year"] = agrupado["diagnosis_year"].astype(int)
        agrupado["total_pacientes"] = agrupado["total_pacientes"].astype(int)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la consulta de total de pacientes por año: {str(e)}")
    
@router.get("/distribucion-categoria-por-año/{nombre_caracteristica}")
async def distribucion_categoria_por_año(nombre_caracteristica: str):
    """
    Returns the distribution (percentages) of a specified categorical feature per year.
    Useful for visualizing trends in categorical data over time (e.g., sex distribution over years).
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        
        # Validar que la característica exista y no sea nula
        if nombre_caracteristica not in df.columns:
            raise HTTPException(status_code=404, detail=f"Feature '{nombre_caracteristica}' not found in the dataset.")
        
        df = df[df[nombre_caracteristica].notnull() & df["diagnosis_date"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for feature '{nombre_caracteristica}' or diagnosis date after cleaning.")

        df["diagnosis_year"] = df["diagnosis_date"].dt.year
        
        # Calcular el total de pacientes por año para normalización
        total_por_año = df.groupby("diagnosis_year").size().reset_index(name='total_anual')
        
        # Agrupar por año y la característica, y contar ocurrencias
        agrupado = df.groupby(["diagnosis_year", nombre_caracteristica]).size().reset_index(name='count')
        
        # Unir con el total anual para calcular porcentajes
        agrupado = pd.merge(agrupado, total_por_año, on="diagnosis_year")
        agrupado["percentage"] = (agrupado["count"] / agrupado["total_anual"] * 100).round(2)

        # Formatear la salida para el dashboard
        results = []
        for year in sorted(df["diagnosis_year"].unique()):
            year_data = agrupado[agrupado["diagnosis_year"] == year]
            if not year_data.empty:
                category_percentages = {}
                for _, row in year_data.iterrows():
                    category_percentages[str(row[nombre_caracteristica])] = float(row["percentage"])
                
                results.append({
                    "year": int(year),
                    "category_percentages": category_percentages
                })
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No data found for feature '{nombre_caracteristica}' per year.")

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la consulta de distribución de categoría por año para '{nombre_caracteristica}': {str(e)}")

@router.get("/por-trimestre")
async def analisis_por_trimestre():
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["risk_score"].notnull() & df["diagnosis_date"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        df["diagnosis_quarter"] = "T" + df["diagnosis_date"].dt.quarter.astype(str)
        agrupado = df.groupby("diagnosis_quarter", observed=False).agg(
            promedio_riesgo=("risk_score", "mean"),
            total_pacientes=("risk_score", "count")
        ).reset_index()

        agrupado["categoria"] = agrupado["promedio_riesgo"].apply(categorizar_riesgo)
        agrupado["diagnosis_quarter"] = agrupado["diagnosis_quarter"].astype(str)

        return agrupado.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por trimestre: {str(e)}")