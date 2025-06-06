# app/routes/socio_environmental.py

from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
import load # Asume que load.py está en la raíz de tu proyecto
from app.model_training.data_processing import preprocess_data
from core.model_loader import load_model_and_features
import os # Para la importación de joblib, aunque no se usa directamente aquí
import joblib # Para cargar scaler y encoder si fuera necesario en el futuro
from collections import defaultdict
from typing import List, Dict, Any, Optional

router = APIRouter(
    prefix="/v1/socio-environmental-analysis",
    tags=["Socio-Environmental Analysis"]
)

def categorizar_riesgo(puntaje: float) -> str:
    """Categoriza un puntaje de riesgo en niveles (Muy Bajo, Bajo, Moderado, Alto, Muy Alto)."""
    if puntaje < 0.2:
        return "Muy Bajo"
    elif puntaje < 0.4:
        return "Bajo"
    elif puntaje < 0.6:
        return "Moderado"
    elif puntaje < 0.8:
        return "Alto"
    else:
        return "Muy Alto"

# --- Endpoints de Factores Sociales ---

@router.get("/by-socioeconomic-status")
async def get_analysis_by_socioeconomic_status(
    min_patients: int = Query(10, description="Minimum number of patients required for a socioeconomic status group to be included.")
):
    """
    Returns the average cardiovascular risk and related metrics by socioeconomic status.
    """
    try:
        # Cargar modelo, scaler y nombres de características
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["socioeconomic_status"].notnull()]

        if "socioeconomic_status" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias: 'socioeconomic_status'.")

        grouped = df.groupby("socioeconomic_status")
        results = []

        for status, group in grouped:
            if len(group) < min_patients:
                continue

            # Asegurarse de que el grupo tenga las columnas necesarias para preprocess_data
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0  # Placeholder
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0  # Placeholder
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")  # Placeholder

            # Preprocesar datos para predicción
            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            
            # Predecir puntajes de riesgo con el modelo
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "socioeconomic_status": int(status),
                "average_risk": round(promedio_riesgo, 4),
                "risk_category": categorizar_riesgo(promedio_riesgo),
                "total_patients": int(len(group)),
                "avg_bmi": round(float(group["bmi"].mean()), 2) if pd.notna(group["bmi"].mean()) else None,
                "avg_age": round(float(group["age"].mean()), 2) if pd.notna(group["age"].mean()) else None,
                "smoking_percentage": round(float(group["is_smoker"].mean() * 100), 2) if pd.notna(group["is_smoker"].mean()) else None,
                "diabetes_percentage": round(float(group["diabetes"].mean() * 100), 2) if pd.notna(group["diabetes"].mean()) else None
            })

        if not results:
            raise HTTPException(status_code=404, detail="No data available for analysis after applying minimum patient threshold.")

        return {"socioeconomic_status_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por estado socioeconómico: {str(e)}") from e

@router.get("/by-education-level")
async def get_analysis_by_education_level(
    min_patients: int = Query(10, description="Minimum number of patients required for an education level group to be included.")
):
    """
    Returns the average cardiovascular risk and related metrics by education level.
    """
    try:
        # Cargar modelo, scaler y nombres de características
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["education_level"].notnull()]

        if "education_level" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias: 'education_level'.")

        education_mapping = {
            1: "Ninguno",
            2: "Preescolar",
            3: "Primaria",
            4: "Secundaria",
            5: "Media",
            6: "Normal",
            7: "Técnico",
            8: "Técnologo",
            9: "Universitario",
            10: "Especialización"
        }

        grouped = df.groupby("education_level")
        results = []

        for education_level, group in grouped:
            if len(group) < min_patients:
                continue

            # Asegurarse de que el grupo tenga las columnas necesarias para preprocess_data
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0  # Placeholder
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0  # Placeholder
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")  # Placeholder

            # Preprocesar datos para predicción
            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            
            # Predecir puntajes de riesgo con el modelo
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            education_label = education_mapping.get(int(education_level), "Desconocido")

            results.append({
                "education_level": education_label,
                "education_level_code": int(education_level),
                "average_risk": round(promedio_riesgo, 4),
                "risk_category": categorizar_riesgo(promedio_riesgo),
                "total_patients": int(len(group)),
                "avg_bmi": round(float(group["bmi"].mean()), 2) if pd.notna(group["bmi"].mean()) else None,
                "avg_age": round(float(group["age"].mean()), 2) if pd.notna(group["age"].mean()) else None,
                "smoking_percentage": round(float(group["is_smoker"].mean() * 100), 2) if pd.notna(group["is_smoker"].mean()) else None,
                "diabetes_percentage": round(float(group["diabetes"].mean() * 100), 2) if pd.notna(group["diabetes"].mean()) else None
            })

        if not results:
            raise HTTPException(status_code=404, detail="No data available for analysis after applying minimum patient threshold.")

        return {"education_level_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por nivel educativo: {str(e)}") from e

@router.get("/infrastructure-access")
async def get_analysis_by_infrastructure_access(
    min_patients: int = Query(10, description="Minimum number of patients required for an infrastructure access group to be included.")
):
    """
    Returns the average cardiovascular risk and related metrics for each infrastructure access type (electricity, water supply, gas, internet).
    """
    try:
        # Cargar modelo, scaler y nombres de características
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()

        infrastructure_columns = ["has_electricity", "has_water_supply", "has_gas", "has_internet"]
        missing_columns = [col for col in infrastructure_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Faltan columnas necesarias: {', '.join(missing_columns)}")

        results = {}

        for infra_col in infrastructure_columns:
            df_infra = df[df[infra_col].notnull()]
            grouped = df_infra.groupby(infra_col)
            infra_results = []

            for access_value, group in grouped:
                if len(group) < min_patients:
                    continue

                # Asegurarse de que el grupo tenga las columnas necesarias para preprocess_data
                if 'risk_score' not in group.columns:
                    group['risk_score'] = 0.0  # Placeholder
                if 'cardiovascular_risk' not in group.columns:
                    group['cardiovascular_risk'] = 0  # Placeholder
                if 'diagnosis_date' not in group.columns:
                    group['diagnosis_date'] = pd.Timestamp("2024-01-01")  # Placeholder

                # Preprocesar datos para predicción
                X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
                
                # Predecir puntajes de riesgo con el modelo
                risk_predictions = model.predict(X_group, verbose=0).flatten()
                promedio_riesgo = float(np.mean(risk_predictions))

                access_label = "Con acceso" if int(access_value) == 1 else "Sin acceso"

                infra_results.append({
                    "access_value": access_label,
                    "access_code": int(access_value),
                    "average_risk": round(promedio_riesgo, 4),
                    "risk_category": categorizar_riesgo(promedio_riesgo),
                    "total_patients": int(len(group)),
                    "avg_bmi": round(float(group["bmi"].mean()), 2) if pd.notna(group["bmi"].mean()) else None,
                    "avg_age": round(float(group["age"].mean()), 2) if pd.notna(group["age"].mean()) else None,
                    "smoking_percentage": round(float(group["is_smoker"].mean() * 100), 2) if pd.notna(group["is_smoker"].mean()) else None,
                    "diabetes_percentage": round(float(group["diabetes"].mean() * 100), 2) if pd.notna(group["diabetes"].mean()) else None
                })

            if infra_results:
                results[infra_col] = infra_results
            else:
                results[infra_col] = f"No data available for '{infra_col}' after applying minimum patient threshold."

        if not any(isinstance(v, list) for v in results.values()):
            raise HTTPException(status_code=404, detail="No data available for any infrastructure access type after applying minimum patient threshold.")

        return {"infrastructure_access_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por acceso a infraestructura: {str(e)}") from e

@router.get("/rural-vs-urban")
async def get_analysis_rural_vs_urban(
    min_patients: int = Query(10, description="Minimum number of patients required for a rural/urban group to be included.")
):
    """
    Returns the average cardiovascular risk and related metrics for rural vs urban areas.
    """
    try:
        # Cargar modelo, scaler y nombres de características
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["rural_area"].notnull()]

        if "rural_area" not in df.columns:
            raise HTTPException(status_code=400, detail="Faltan columnas necesarias: 'rural_area'.")

        grouped = df.groupby("rural_area")
        results = []

        for area_type, group in grouped:
            if len(group) < min_patients:
                continue

            # Asegurarse de que el grupo tenga las columnas necesarias para preprocess_data
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0  # Placeholder
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0  # Placeholder
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")  # Placeholder

            # Preprocesar datos para predicción
            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            
            # Predecir puntajes de riesgo con el modelo
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            area_label = "Urbano" if int(area_type) == 0 else "Rural"

            results.append({
                "area_type": area_label,
                "area_code": int(area_type),
                "average_risk": round(promedio_riesgo, 4),
                "risk_category": categorizar_riesgo(promedio_riesgo),
                "total_patients": int(len(group)),
                "avg_bmi": round(float(group["bmi"].mean()), 2) if pd.notna(group["bmi"].mean()) else None,
                "avg_age": round(float(group["age"].mean()), 2) if pd.notna(group["age"].mean()) else None,
                "smoking_percentage": round(float(group["is_smoker"].mean() * 100), 2) if pd.notna(group["is_smoker"].mean()) else None,
                "diabetes_percentage": round(float(group["diabetes"].mean() * 100), 2) if pd.notna(group["diabetes"].mean()) else None
            })

        if not results:
            raise HTTPException(status_code=404, detail="No data available for analysis after applying minimum patient threshold.")

        return {"rural_urban_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis rural vs urbano: {str(e)}") from e


@router.get("/by-temperature-range")
async def get_analysis_by_temperature_range(
    bins: List[float] = Query([10, 15, 20, 25, 30], description="Temperature range boundaries in Celsius."),
    min_patients: int = Query(10, description="Minimum number of patients required for a temperature range group to be included.")
):
    """
    Returns the average cardiovascular risk and related metrics grouped by temperature ranges.
    """
    try:
        # Cargar modelo, scaler y nombres de características
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        # Cargar datasets de pacientes y municipios
        df_pacientes, df_municipios, _ = load.load_dataset()
        df_pacientes.columns = df_pacientes.columns.str.lower()
        df_municipios.columns = df_municipios.columns.str.lower()

        # Fusionar datasets en department y municipality
        df_merged = df_pacientes.merge(df_municipios, on=["department", "municipality"], how="left")

        # Filtrar datos con temperature disponible
        df_merged = df_merged[df_merged["average_temperature"].notnull()]

        if df_merged.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after merging and cleaning.")

        # Definir rangos de temperatura
        bins = sorted(bins)
        labels = [f"{bins[i]}-{bins[i+1]}°C" for i in range(len(bins)-1)]
        df_merged["temperature_range"] = pd.cut(df_merged["average_temperature"], bins=bins, labels=labels, include_lowest=True)

        print(df_merged)  # Debugging line

        # Agrupar por rangos de temperatura
        grouped = df_merged.groupby("temperature_range", observed=False)
        results = []

        for temp_range, group in grouped:
            if len(group) < min_patients:
                continue

            # Asegurarse de que el grupo tenga las columnas necesarias para preprocess_data
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0  # Placeholder
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0  # Placeholder
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")  # Placeholder

            # Preprocesar datos para predicción
            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            
            # Predecir puntajes de riesgo con el modelo
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            results.append({
                "temperature_range": str(temp_range),
                "average_risk": round(promedio_riesgo, 4),
                "risk_category": categorizar_riesgo(promedio_riesgo),
                "total_patients": int(len(group)),
                "avg_bmi": round(float(group["bmi"].mean()), 2) if pd.notna(group["bmi"].mean()) else None,
                "avg_age": round(float(group["age"].mean()), 2) if pd.notna(group["age"].mean()) else None,
                "smoking_percentage": round(float(group["is_smoker"].mean() * 100), 2) if pd.notna(group["is_smoker"].mean()) else None,
                "diabetes_percentage": round(float(group["diabetes"].mean() * 100), 2) if pd.notna(group["diabetes"].mean()) else None
            })

        if not results:
            raise HTTPException(status_code=404, detail="No data available for analysis after applying minimum patient threshold.")

        return {"temperature_range_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por rangos de temperatura: {str(e)}") from e


@router.get("/by-precipitation-range")
async def get_analysis_by_precipitation_range(
    bins: List[float] = Query([500, 1000, 1500, 2000, 2500], description="Precipitation range boundaries in mm/year."),
    min_patients: int = Query(10, description="Minimum number of patients required for a precipitation range group to be included.")
):
    try:
        # Cargar modelo y preprocesadores
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados.")

        # Cargar y preparar datos
        df_pacientes, df_municipios, _ = load.load_dataset()
        df_pacientes.columns = df_pacientes.columns.str.lower()
        df_municipios.columns = df_municipios.columns.str.lower()

        # Fusionar datasets
        df_merged = df_pacientes.merge(df_municipios, on=["department", "municipality"], how="left")
        df_merged = df_merged[df_merged["annual_precipitation"].notnull()]

        if df_merged.empty:
            raise HTTPException(status_code=404, detail="No data available after merging and cleaning.")

        # Convertir precipitación a valores numéricos
        def convert_precipitation(value):
            if pd.isna(value) or value == '':
                return np.nan
            str_value = str(value).strip()
            if '-' in str_value:
                parts = [float(p.strip()) for p in str_value.split('-')]
                return sum(parts) / len(parts)
            try:
                return float(str_value)
            except ValueError:
                return np.nan

        df_merged["annual_precipitation"] = df_merged["annual_precipitation"].apply(convert_precipitation)
        print(f"Precipitation: {df_merged['annual_precipitation']}")  # Debugging line
        df_merged = df_merged[df_merged["annual_precipitation"].notnull()]


        # Definir rangos de precipitación
        bins = sorted(bins)
        labels = [f"{bins[i]}-{bins[i+1]}mm" for i in range(len(bins)-1)]
        df_merged["precipitation_range"] = pd.cut(df_merged["annual_precipitation"], bins=bins, labels=labels, include_lowest=True)

        # Agrupar por rangos
        grouped = df_merged.groupby("precipitation_range", observed=False)
        results = []

        # Procesar cada grupo
        for precip_range, group in grouped:
            if len(group) < min_patients:
                continue

            # Añadir columnas placeholder si faltan (requeridas por preprocess_data)
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            # Preprocesar datos y predecir riesgos
            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            # Construir resultado
            results.append({
                "precipitation_range": str(precip_range),
                "average_risk": round(promedio_riesgo, 4),
                "risk_category": categorizar_riesgo(promedio_riesgo),
                "total_patients": int(len(group)),
                "avg_temperature": round(float(group["average_temperature"].mean()), 2) if pd.notna(group["average_temperature"].mean()) else None,
                "avg_altitude": round(float(group["average_altitude"].mean()), 2) if pd.notna(group["average_altitude"].mean()) else None
            })

        if not results:
            raise HTTPException(status_code=404, detail="No data available after applying minimum patient threshold.")

        return {"precipitation_range_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis por rangos de precipitación: {str(e)}")

# --- Endpoint de Análisis Combinado de Factores de Riesgo (Ejemplo) ---

@router.get("/smoker-diabetes-correlation")
async def get_smoker_diabetes_correlation(
    min_patients: int = Query(10, description="Minimum number of patients required for a smoker-diabetes group to be included.")
):
    try:
        # Cargar modelo y preprocesadores
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados.")

        # Cargar y limpiar datos
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()
        df = df[df["is_smoker"].notnull() & df["diabetes"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available after cleaning.")

        # Crear grupos basados en tabaquismo y diabetes
        def get_smoker_diabetes_group(row):
            smoker_status = "Fumador" if row["is_smoker"] == 1 else "No Fumador"
            diabetes_status = "Diabético" if row["diabetes"] == 1 else "No Diabético"
            return f"{smoker_status} / {diabetes_status}"

        df["smoker_diabetes_group"] = df.apply(get_smoker_diabetes_group, axis=1)
        grouped = df.groupby("smoker_diabetes_group")
        results = []

        # Procesar cada grupo
        for group_name, group in grouped:
            if len(group) < min_patients:
                continue

            # Añadir columnas placeholder si faltan
            if 'risk_score' not in group.columns:
                group['risk_score'] = 0.0
            if 'cardiovascular_risk' not in group.columns:
                group['cardiovascular_risk'] = 0
            if 'diagnosis_date' not in group.columns:
                group['diagnosis_date'] = pd.Timestamp("2024-01-01")

            # Preprocesar datos y predecir riesgos
            X_group, _, _, _ = preprocess_data(group, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
            risk_predictions = model.predict(X_group, verbose=0).flatten()
            promedio_riesgo = float(np.mean(risk_predictions))

            # Construir resultado
            results.append({
                "group": group_name,
                "average_risk": round(promedio_riesgo, 4),
                "risk_category": categorizar_riesgo(promedio_riesgo),
                "total_patients": int(len(group)),
                "avg_bmi": round(float(group["bmi"].mean()), 2) if pd.notna(group["bmi"].mean()) else None,
                "avg_age": round(float(group["age"].mean()), 2) if pd.notna(group["age"].mean()) else None
            })

        if not results:
            raise HTTPException(status_code=404, detail="No data available after applying minimum patient threshold.")

        return {"smoker_diabetes_correlation": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis de correlación fumador-diabetes: {str(e)}")

# --- Endpoint de Distribución de Riesgo (Ejemplo) ---

@router.get("/risk-distribution")
async def get_risk_distribution(
    bins: List[float] = Query([0.2, 0.4, 0.6, 0.8], description="Risk score bin boundaries."),
    by_department: Optional[str] = Query(None, description="Optional department to filter the distribution by.")
):
    try:
        # Cargar modelo y preprocesadores
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados.")

        # Cargar y filtrar datos
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower()

        if by_department:
            df = df[df["department"].str.upper() == by_department.upper()]
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No data found for department: {by_department}")

        # Añadir columnas placeholder si faltan
        if 'risk_score' not in df.columns:
            df['risk_score'] = 0.0
        if 'cardiovascular_risk' not in df.columns:
            df['cardiovascular_risk'] = 0
        if 'diagnosis_date' not in df.columns:
            df['diagnosis_date'] = pd.Timestamp("2024-01-01")

        # Preprocesar datos y predecir riesgos
        X_df, _, _, _ = preprocess_data(df, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
        risk_predictions = model.predict(X_df, verbose=0).flatten()

        # Definir rangos y etiquetas
        bins = sorted(bins)
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        df["risk_group"] = pd.cut(risk_predictions, bins=bins, labels=labels, include_lowest=True)

        # Calcular distribución
        distribution = df["risk_group"].value_counts(normalize=True).sort_index()
        counts = df["risk_group"].value_counts().sort_index()

        results = []
        for label in labels:
            proportion = distribution.get(label, 0)
            count = counts.get(label, 0)
            results.append({
                "risk_range": label,
                "percentage": round(float(proportion * 100), 2),
                "count": int(count)
            })

        return {
            "department": by_department if by_department else "Global",
            "risk_distribution": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en distribución de riesgo: {str(e)}")