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
    tags=["Resumen por Departamento y Otros Criterios"]
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
    - Average risk score (predicted by model)
    - Risk classification
    - Top factors based on weighted activation
    """
    try:
        # Cargar modelo, scaler y feature_names
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df_full, _, _ = load.load_dataset()
        df_full.columns = df_full.columns.str.lower()

        if 'department' not in df_full.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes: 'department' no encontrado.")

        departamentos = df_full['department'].unique()
        resumen = []

        # Obtener los pesos del modelo una sola vez
        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo.")
        pesos_global = first_dense.get_weights()[0]
        importancia_global = dict(zip(feature_names_global, np.mean(np.abs(pesos_global), axis=1)))

        for depto in departamentos:
            df_depto = df_full[df_full['department'] == depto].copy()
            if df_depto.empty or len(df_depto) < min_patients:
                continue

            # Asegurarse de que df_depto tenga las columnas necesarias para preprocess_data
            if 'risk_score' not in df_depto.columns:
                df_depto['risk_score'] = 0.0  # Placeholder
            if 'cardiovascular_risk' not in df_depto.columns:
                df_depto['cardiovascular_risk'] = 0  # Placeholder
            if 'diagnosis_date' not in df_depto.columns:
                df_depto['diagnosis_date'] = pd.Timestamp("2024-01-01")  # Placeholder

            # Preprocesar datos para predicción
            X_depto, _, _, _ = preprocess_data(df_depto, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            
            # Predecir puntajes de riesgo con el modelo
            risk_predictions = model.predict(X_depto, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            # Calcular importancia de características
            activacion_media = np.mean(X_depto, axis=0)
            importancia_ponderada = []
            for name, activacion in zip(feature_names_global, activacion_media):
                imp = importancia_global.get(name, 0.0)
                if not (name.startswith("department_") or name.startswith("municipality_")):
                    importancia_ponderada.append((name, float(activacion * imp)))
            importancia_ponderada.sort(key=lambda x: x[1], reverse=True)

            resumen.append({
                "departamento": depto,
                "pacientes": int(len(df_depto)),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "categoria": categorizar_riesgo(promedio_riesgo),
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
    - Average risk score (predicted by model)
    - Classification
    - Top factors based on weighted activation
    """
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df_full, _, _ = load.load_dataset()
        df_full.columns = df_full.columns.str.lower()

        if 'municipality' not in df_full.columns:
            raise HTTPException(status_code=400, detail="Datos insuficientes: 'municipality' no encontrado.")

        municipios = df_full['municipality'].unique()
        resumen = []

        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo.")
        pesos_global = first_dense.get_weights()[0]
        importancia_global = dict(zip(feature_names_global, np.mean(np.abs(pesos_global), axis=1)))

        for mpio in municipios:
            df_mpio = df_full[df_full['municipality'] == mpio].copy()
            if df_mpio.empty or len(df_mpio) < min_pacientes:
                continue

            if 'risk_score' not in df_mpio.columns:
                df_mpio['risk_score'] = 0.0
            if 'cardiovascular_risk' not in df_mpio.columns:
                df_mpio['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in df_mpio.columns:
                df_mpio['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_mpio, _, _, _ = preprocess_data(df_mpio, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_mpio, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

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
                "promedio_riesgo": round(promedio_riesgo, 4),
                "categoria": categorizar_riesgo(promedio_riesgo),
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
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["occupation"].notnull()]

        if "occupation" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias: 'occupation'.")

        agrupado = df.groupby("occupation")
        results = []

        for occupation, group in agrupado:
            if len(group) < min_pacientes:
                continue

            # Asegurarse de que el grupo tenga las columnas necesarias
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            # Preprocesar y predecir
            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "occupation": str(occupation),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por ocupación: {str(e)}")

@router.get("/por-edad")
async def analisis_por_edad():
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["age"].notnull()]

        bins = [0, 18, 30, 45, 60, 75, 120]
        labels = ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"]
        df["grupo_edad"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

        agrupado = df.groupby("grupo_edad", observed=False)
        results = []

        for grupo_edad, group in agrupado:
            if group["age"].isna().all():
                continue

            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "grupo_edad": str(grupo_edad),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por edad: {str(e)}")

@router.get("/por-nivel-educativo")
async def analisis_por_nivel_educativo(min_pacientes: int = Query(10, description="Minimum number of patients required for an education level group to be included.")):
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["education_level"].notnull()]

        if "education_level" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias.")

        agrupado = df.groupby("education_level")
        results = []

        for education_level, group in agrupado:
            if len(group) < min_pacientes:
                continue

            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "education_level": str(education_level),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por nivel educativo: {str(e)}")

@router.get("/por-estrato")
async def analisis_por_estrato(min_pacientes: int = Query(10, description="Minimum number of patients required for a socioeconomic status group to be included.")):
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["socioeconomic_status"].notnull()]

        if "socioeconomic_status" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias.")

        agrupado = df.groupby("socioeconomic_status")
        results = []

        for status, group in agrupado:
            if len(group) < min_pacientes:
                continue

            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "socioeconomic_status": str(status),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por estrato: {str(e)}")

@router.get("/por-año")
async def analisis_por_año():
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["diagnosis_date"].notnull()]

        df["diagnosis_year"] = df["diagnosis_date"].dt.year
        agrupado = df.groupby("diagnosis_year")
        results = []

        for year, group in agrupado:
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "diagnosis_year": int(year),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año: {str(e)}")

@router.get("/por-año-departamento/{departamento}")
async def analisis_por_año_departamento(departamento: str):
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')

        if departamento.upper() not in df['department'].str.upper().unique():
            raise HTTPException(status_code=404, detail=f"Department '{departamento}' not found in the dataset.")

        df_filtrado = df[df["department"].str.upper() == departamento.upper()].copy()
        df_filtrado = df_filtrado[df_filtrado["diagnosis_date"].notnull()]

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No valid patient data found for department '{departamento}'.")

        df_filtrado["diagnosis_year"] = df_filtrado["diagnosis_date"].dt.year
        agrupado = df_filtrado.groupby("diagnosis_year")
        results = []

        for year, group in agrupado:
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "diagnosis_year": int(year),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año para departamento '{departamento}': {str(e)}")

@router.get("/por-año-municipio/{municipio}")
async def analisis_por_año_municipio(municipio: str):
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')

        if municipio.upper() not in df['municipality'].str.upper().unique():
            raise HTTPException(status_code=404, detail=f"Municipality '{municipio}' not found in the dataset.")

        df_filtrado = df[df["municipality"].str.upper() == municipio.upper()].copy()
        df_filtrado = df_filtrado[df_filtrado["diagnosis_date"].notnull()]

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No valid patient data found for municipality '{municipio}'.")

        df_filtrado["diagnosis_year"] = df_filtrado["diagnosis_date"].dt.year
        agrupado = df_filtrado.groupby("diagnosis_year")
        results = []

        for year, group in agrupado:
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "diagnosis_year": int(year),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por año para municipio '{municipio}': {str(e)}")

@router.get("/total-pacientes-por-año")
async def total_pacientes_por_año():
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["diagnosis_date"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        df["diagnosis_year"] = df["diagnosis_date"].dt.year
        agrupado = df.groupby("diagnosis_year").agg(
            total_pacientes=("diagnosis_year", "count")
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
    try:
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df[nombre_caracteristica].notnull() & df["diagnosis_date"].notnull()]

        if nombre_caracteristica not in df.columns:
            raise HTTPException(status_code=404, detail=f"Feature '{nombre_caracteristica}' not found in the dataset.")

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for feature '{nombre_caracteristica}'.")

        df["diagnosis_year"] = df["diagnosis_date"].dt.year
        
        # Preprocesar todos los datos para predicción
        if 'risk_score' not in df.columns:
            df['risk_score'] = 0.0
        if 'cardiovascular_risk' not in df.columns:
            df['cardiovascular_risk'] = 0
        if 'diagnosis_date' not in df.columns:
            df['diagnosis_date'] = pd.Timestamp("2024-01-01")

        X_df, _, _, _ = preprocess_data(df, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
        risk_predictions = model.predict(X_df, verbose=0).flatten()
        df["predicted_risk"] = risk_predictions

        total_por_año = df.groupby("diagnosis_year").size().reset_index(name='total_anual')
        agrupado = df.groupby(["diagnosis_year", nombre_caracteristica]).size().reset_index(name='count')
        agrupado = pd.merge(agrupado, total_por_año, on="diagnosis_year")
        agrupado["percentage"] = (agrupado["count"] / agrupado["total_anual"] * 100).round(2)

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
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors='coerce')
        df = df[df["diagnosis_date"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        df["diagnosis_quarter"] = "T" + df["diagnosis_date"].dt.quarter.astype(str)
        agrupado = df.groupby("diagnosis_quarter", observed=False)
        results = []

        for quarter, group in agrupado:
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "diagnosis_quarter": str(quarter),
                "promedio_riesgo": round(promedio_riesgo, 4),
                "total_pacientes": int(len(group)),
                "categoria": categorizar_riesgo(promedio_riesgo)
            })

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por trimestre: {str(e)}")