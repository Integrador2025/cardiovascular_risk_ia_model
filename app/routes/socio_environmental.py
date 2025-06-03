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
async def get_analysis_by_socioeconomic_status(min_patients: int = Query(10, description="Minimum number of patients required for a socioeconomic status group to be included.")):
    """
    Analyzes cardiovascular risk and patient profiles grouped by socioeconomic status.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas
        
        # Ensure risk_score and socioeconomic_status columns exist and are not null
        df = df[df["risk_score"].notnull() & df["socioeconomic_status"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        grouped = df.groupby("socioeconomic_status").agg(
            average_risk=("risk_score", "mean"),
            total_patients=("risk_score", "count"),
            avg_bmi=("bmi", "mean"),
            avg_total_cholesterol=("total_cholesterol", "mean"),
            smoker_percentage=("is_smoker", lambda x: x.mean() * 100),
            has_electricity_percentage=("has_electricity", lambda x: x.mean() * 100),
            has_water_supply_percentage=("has_water_supply", lambda x: x.mean() * 100),
            has_internet_percentage=("has_internet", lambda x: x.mean() * 100)
        ).reset_index()

        # Filter out groups with fewer than min_patients
        grouped = grouped[grouped["total_patients"] >= min_patients]

        if grouped.empty:
            raise HTTPException(status_code=404, detail=f"No socioeconomic status groups found with at least {min_patients} patients.")

        # Add risk category and format output
        grouped["risk_category"] = grouped["average_risk"].apply(categorizar_riesgo)
        
        # Convert specific columns to native Python types for JSON serialization
        results = grouped.to_dict(orient="records")
        for item in results:
            item["socioeconomic_status"] = int(item["socioeconomic_status"])
            item["average_risk"] = round(float(item["average_risk"]), 4)
            item["total_patients"] = int(item["total_patients"])
            item["avg_bmi"] = round(float(item["avg_bmi"]), 2) if pd.notna(item["avg_bmi"]) else None
            item["avg_total_cholesterol"] = round(float(item["avg_total_cholesterol"]), 2) if pd.notna(item["avg_total_cholesterol"]) else None
            item["smoker_percentage"] = round(float(item["smoker_percentage"]), 2) if pd.notna(item["smoker_percentage"]) else None
            item["has_electricity_percentage"] = round(float(item["has_electricity_percentage"]), 2) if pd.notna(item["has_electricity_percentage"]) else None
            item["has_water_supply_percentage"] = round(float(item["has_water_supply_percentage"]), 2) if pd.notna(item["has_water_supply_percentage"]) else None
            item["has_internet_percentage"] = round(float(item["has_internet_percentage"]), 2) if pd.notna(item["has_internet_percentage"]) else None
            
        return {"socioeconomic_status_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing by socioeconomic status: {str(e)}")

@router.get("/by-education-level")
async def get_analysis_by_education_level(min_patients: int = Query(10, description="Minimum number of patients required for an education level group to be included.")):
    """
    Analyzes cardiovascular risk and patient profiles grouped by education level.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas
        
        # Ensure risk_score and education_level columns exist and are not null
        df = df[df["risk_score"].notnull() & df["education_level"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        grouped = df.groupby("education_level").agg(
            average_risk=("risk_score", "mean"),
            total_patients=("risk_score", "count"),
            avg_age=("age", "mean"),
            avg_bmi=("bmi", "mean"),
            smoker_percentage=("is_smoker", lambda x: x.mean() * 100)
        ).reset_index()

        # Filter out groups with fewer than min_patients
        grouped = grouped[grouped["total_patients"] >= min_patients]

        if grouped.empty:
            raise HTTPException(status_code=404, detail=f"No education level groups found with at least {min_patients} patients.")

        # Add risk category and format output
        grouped["risk_category"] = grouped["average_risk"].apply(categorizar_riesgo)
        
        # Map education levels to descriptive names
        education_mapping = {
            1: "Ninguno", 2: "Preescolar", 3: "Primaria", 4: "Secundaria",
            5: "Media", 6: "Técnico", 7: "Tecnólogo", 8: "Universitario",
            9: "Especialización", 10: "Posgrado" # Añadir Posgrado si es posible
        }
        # Usar .get para manejar valores que no estén en el mapeo
        grouped["education_level_description"] = grouped["education_level"].apply(lambda x: education_mapping.get(x, f"Nivel {x} Desconocido"))
        
        # Convert specific columns to native Python types for JSON serialization
        results = grouped.to_dict(orient="records")
        for item in results:
            item["education_level"] = int(item["education_level"])
            item["average_risk"] = round(float(item["average_risk"]), 4)
            item["total_patients"] = int(item["total_patients"])
            item["avg_age"] = round(float(item["avg_age"]), 1) if pd.notna(item["avg_age"]) else None
            item["avg_bmi"] = round(float(item["avg_bmi"]), 2) if pd.notna(item["avg_bmi"]) else None
            item["smoker_percentage"] = round(float(item["smoker_percentage"]), 2) if pd.notna(item["smoker_percentage"]) else None

        return {"education_level_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing by education level: {str(e)}")

@router.get("/infrastructure-access")
async def get_analysis_by_infrastructure_access(min_patients: int = Query(10, description="Minimum number of patients required for a group to be included.")):
    """
    Compares cardiovascular risk and patient profiles based on access to various infrastructure services.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas
        
        infrastructure_cols = ['has_electricity', 'has_water_supply', 'has_sewage', 'has_gas', 'has_internet']
        
        # Ensure risk_score and infrastructure columns exist and are not null
        for col in infrastructure_cols:
            df = df[df[col].notnull()]
        df = df[df["risk_score"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        results = {}
        for col in infrastructure_cols:
            # Group by access (0 or 1)
            grouped_by_access = df.groupby(col).agg(
                average_risk=("risk_score", "mean"),
                total_patients=("risk_score", "count"),
                avg_age=("age", "mean"),
                avg_bmi=("bmi", "mean")
            ).reset_index()

            # Filter groups with fewer than min_patients
            grouped_by_access = grouped_by_access[grouped_by_access["total_patients"] >= min_patients]

            if not grouped_by_access.empty:
                access_data = []
                for _, row in grouped_by_access.iterrows():
                    access_status = "Yes" if row[col] == 1 else "No"
                    access_data.append({
                        "access": access_status,
                        "average_risk": round(float(row["average_risk"]), 4),
                        "risk_category": categorizar_riesgo(float(row["average_risk"])),
                        "total_patients": int(row["total_patients"]),
                        "avg_age": round(float(row["avg_age"]), 1) if pd.notna(row["avg_age"]) else None,
                        "avg_bmi": round(float(row["avg_bmi"]), 2) if pd.notna(row["avg_bmi"]) else None
                    })
                results[col] = access_data
            else:
                results[col] = f"No data for '{col}' with sufficient patients after filtering."
                
        if not results:
             raise HTTPException(status_code=404, detail="No infrastructure access data found with sufficient patients after filtering.")

        return {"infrastructure_access_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing by infrastructure access: {str(e)}")

# --- Endpoints de Factores Ambientales/Geográficos ---

@router.get("/rural-vs-urban")
async def get_analysis_rural_vs_urban(min_patients: int = Query(10, description="Minimum number of patients required for a rural/urban group to be included.")):
    """
    Compares cardiovascular risk and patient profiles between rural and urban areas.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas
        
        # Ensure risk_score and rural_area columns exist and are not null
        df = df[df["risk_score"].notnull() & df["rural_area"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        grouped = df.groupby("rural_area").agg(
            average_risk=("risk_score", "mean"),
            total_patients=("risk_score", "count"),
            avg_age=("age", "mean"),
            avg_bmi=("bmi", "mean"),
            smoker_percentage=("is_smoker", lambda x: x.mean() * 100),
            avg_total_cholesterol=("total_cholesterol", "mean")
        ).reset_index()

        # Filter out groups with fewer than min_patients
        grouped = grouped[grouped["total_patients"] >= min_patients]

        if grouped.empty:
            raise HTTPException(status_code=404, detail=f"No rural/urban groups found with at least {min_patients} patients.")

        # Add risk category and format output
        grouped["risk_category"] = grouped["average_risk"].apply(categorizar_riesgo)
        
        # Convert specific columns to native Python types for JSON serialization
        results = []
        for _, row in grouped.iterrows(): # Iterar sobre 'row' en lugar de 'item'
            area_type = "Rural" if row["rural_area"] == 1 else "Urban"
            results.append({
                "area_type": area_type,
                "average_risk": round(float(row["average_risk"]), 4),
                "risk_category": row["risk_category"],
                "total_patients": int(row["total_patients"]),
                "avg_age": round(float(row["avg_age"]), 1) if pd.notna(row["avg_age"]) else None,
                "avg_bmi": round(float(row["avg_bmi"]), 2) if pd.notna(row["avg_bmi"]) else None,
                "smoker_percentage": round(float(row["smoker_percentage"]), 2) if pd.notna(row["smoker_percentage"]) else None, # Corregido 'item' a 'row'
                "avg_total_cholesterol": round(float(row["avg_total_cholesterol"]), 2) if pd.notna(row["avg_total_cholesterol"]) else None
            })
            
        return {"rural_urban_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing rural vs urban areas: {str(e)}")


@router.get("/by-temperature-range")
async def get_analysis_by_temperature_range(
    bins: List[float] = Query([10, 15, 20, 25, 30, 35], description="Temperature range boundaries in Celsius. Example: [10, 20, 30] creates ranges <10, 10-20, 20-30, >30."),
    min_patients: int = Query(10, description="Minimum number of patients required for a temperature group to be included.")
):
    """
    Analyzes cardiovascular risk grouped by ranges of average temperature in the municipality.
    """
    try:
        df_pac, df_mun, _ = load.load_dataset()
        
        # Merge df_pacientes with df_municipios to get temperature
        df_pac.columns = df_pac.columns.str.lower() # Asegurar minúsculas
        df_mun.columns = df_mun.columns.str.lower() # Asegurar minúsculas

        df_merged = df_pac.merge(df_mun, on=["department", "municipality"], how="left")
        
        # Ensure necessary columns exist and are not null
        df_merged = df_merged[df_merged["risk_score"].notnull() & df_merged["average_temperature"].notnull()]

        if df_merged.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")
        
        # Create labels for bins (ajustado para que coincida con pd.cut)
        # pd.cut crea intervalos como (a, b], por lo que los labels deben reflejar eso.
        # Para rangos abiertos, se pueden usar -np.inf y np.inf
        full_bins = [-np.inf] + sorted(bins) + [np.inf]
        labels = [f"({full_bins[i]}, {full_bins[i+1]}]°C" for i in range(len(full_bins)-1)]
        
        df_merged["temperature_group"] = pd.cut(df_merged["average_temperature"], bins=full_bins, right=True, include_lowest=True, labels=labels)

        grouped = df_merged.groupby("temperature_group", observed=False).agg(
            average_risk=("risk_score", "mean"),
            total_patients=("risk_score", "count"),
            avg_age=("age", "mean"),
            avg_bmi=("bmi", "mean")
        ).reset_index()

        # Filter out groups with fewer than min_patients
        grouped = grouped[grouped["total_patients"] >= min_patients]

        if grouped.empty:
            raise HTTPException(status_code=404, detail=f"No temperature groups found with at least {min_patients} patients.")

        # Add risk category and format output
        grouped["risk_category"] = grouped["average_risk"].apply(categorizar_riesgo)
        
        results = []
        for _, row in grouped.iterrows():
            if pd.notna(row["temperature_group"]): # Ensure the interval is not NaN
                results.append({
                    "temperature_range": str(row["temperature_group"]),
                    "average_risk": round(float(row["average_risk"]), 4),
                    "risk_category": row["risk_category"],
                    "total_patients": int(row["total_patients"]),
                    "avg_age": round(float(row["avg_age"]), 1) if pd.notna(row["avg_age"]) else None,
                    "avg_bmi": round(float(row["avg_bmi"]), 2) if pd.notna(row["avg_bmi"]) else None
                })
        
        return {"temperature_range_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing by temperature range: {str(e)}")


@router.get("/by-precipitation-range")
async def get_analysis_by_precipitation_range(
    bins: List[float] = Query([500, 1000, 1500, 2000, 2500, 3000], description="Precipitation range boundaries in mm/year. Example: [500, 1500, 2500] creates ranges <500, 500-1500, 1500-2500, >2500."),
    min_patients: int = Query(10, description="Minimum number of patients required for a precipitation group to be included.")
):
    """
    Analyzes cardiovascular risk grouped by ranges of annual precipitation in the municipality.
    """
    try:
        df_pac, df_mun, _ = load.load_dataset()
        
        # Standardize column names to lowercase for merging
        df_pac.columns = df_pac.columns.str.lower()
        df_mun.columns = df_mun.columns.str.lower()

        df_merged = df_pac.merge(df_mun, on=["department", "municipality"], how="left")
        
        # Ensure necessary columns exist and are not null
        df_merged = df_merged[df_merged["risk_score"].notnull() & df_merged["annual_precipitation"].notnull()]

        if df_merged.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")
        
        # Apply the convert_precipitation logic from data_processing_population.py
        # To avoid circular imports or direct dependency, copy/paste it or make it a common utility
        # For simplicity here, I'll put a placeholder for its logic, ideally it's imported
        def convert_precipitation_local(value):
            if pd.isna(value) or value == '':
                return np.nan
            str_value = str(value).strip()
            # MODIFICACIÓN: Eliminar " mm" del final si está presente
            if str_value.endswith(" mm"):
                str_value = str_value[:-3].strip() # Eliminar " mm" y espacios extra
            try:
                if '-' in str_value:
                    parts = [float(p.strip()) for p in str_value.split('-')]
                    return sum(parts) / len(parts)
                return float(str_value)
            except (ValueError, TypeError):
                return np.nan

        df_merged["annual_precipitation_numeric"] = df_merged["annual_precipitation"].astype(str).apply(convert_precipitation_local)
        df_merged = df_merged[df_merged["annual_precipitation_numeric"].notnull()] # Filter out NaNs after conversion

        if df_merged.empty:
            raise HTTPException(status_code=404, detail="No valid precipitation data available after conversion.")

        # Create labels for bins (ajustado para que coincida con pd.cut)
        full_bins = [-np.inf] + sorted(bins) + [np.inf]
        labels = [f"({full_bins[i]}, {full_bins[i+1]}]mm" for i in range(len(full_bins)-1)]

        df_merged["precipitation_group"] = pd.cut(df_merged["annual_precipitation_numeric"], bins=full_bins, right=True, include_lowest=True, labels=labels)

        grouped = df_merged.groupby("precipitation_group", observed=False).agg(
            average_risk=("risk_score", "mean"),
            total_patients=("risk_score", "count"),
            avg_temperature=("average_temperature", "mean"),
            avg_altitude=("average_altitude", "mean")
        ).reset_index()

        # Filter out groups with fewer than min_patients
        grouped = grouped[grouped["total_patients"] >= min_patients]

        if grouped.empty:
            raise HTTPException(status_code=404, detail=f"No precipitation groups found with at least {min_patients} patients.")

        # Add risk category and format output
        grouped["risk_category"] = grouped["average_risk"].apply(categorizar_riesgo)
        
        results = []
        for _, row in grouped.iterrows():
            if pd.notna(row["precipitation_group"]): # Ensure the interval is not NaN
                results.append({
                    "precipitation_range": str(row["precipitation_group"]),
                    "average_risk": round(float(row["average_risk"]), 4),
                    "risk_category": row["risk_category"],
                    "total_patients": int(row["total_patients"]),
                    "avg_temperature": round(float(row["avg_temperature"]), 1) if pd.notna(row["avg_temperature"]) else None,
                    "avg_altitude": round(float(row["avg_altitude"]), 1) if pd.notna(row["avg_altitude"]) else None
                })
        
        return {"precipitation_range_analysis": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing by precipitation range: {str(e)}")

# --- Endpoint de Análisis Combinado de Factores de Riesgo (Ejemplo) ---

@router.get("/smoker-diabetes-correlation")
async def get_smoker_diabetes_correlation(min_patients: int = Query(10, description="Minimum number of patients required for a group to be included.")):
    """
    Analyzes cardiovascular risk for combinations of smoking status and diabetes presence.
    """
    try:
        df, _, _ = load.load_dataset()
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        # Ensure necessary columns exist and are not null
        df = df[df["risk_score"].notnull() & df["is_smoker"].notnull() & df["diabetes"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis after cleaning.")

        # Create a combined 'group' column for easy grouping
        def get_smoker_diabetes_group(row):
            smoker_status = "Smoker" if row["is_smoker"] == 1 else "Non-Smoker"
            diabetes_status = "Diabetes" if row["diabetes"] == 1 else "No-Diabetes"
            return f"{smoker_status} / {diabetes_status}"

        df["smoker_diabetes_group"] = df.apply(get_smoker_diabetes_group, axis=1)

        grouped = df.groupby("smoker_diabetes_group").agg(
            average_risk=("risk_score", "mean"),
            total_patients=("risk_score", "count"),
            avg_age=("age", "mean"),
            avg_bmi=("bmi", "mean")
        ).reset_index()

        # Filter out groups with fewer than min_patients
        grouped = grouped[grouped["total_patients"] >= min_patients]

        if grouped.empty:
            raise HTTPException(status_code=404, detail=f"No smoker-diabetes groups found with at least {min_patients} patients.")

        # Add risk category and format output
        grouped["risk_category"] = grouped["average_risk"].apply(categorizar_riesgo)
        
        results = grouped.to_dict(orient="records")
        for item in results:
            item["average_risk"] = round(float(item["average_risk"]), 4)
            item["total_patients"] = int(item["total_patients"])
            item["avg_age"] = round(float(item["avg_age"]), 1) if pd.notna(item["avg_age"]) else None
            item["avg_bmi"] = round(float(item["avg_bmi"]), 2) if pd.notna(item["avg_bmi"]) else None

        return {"smoker_diabetes_correlation": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing smoker-diabetes correlation: {str(e)}")

# --- Endpoint de Distribución de Riesgo (Ejemplo) ---

@router.get("/risk-distribution")
async def get_risk_distribution(
    bins: List[float] = Query([0.2, 0.4, 0.6, 0.8], description="Risk score bin boundaries. Example: [0.2, 0.4, 0.6] creates bins <0.2, 0.2-0.4, 0.4-0.6, >0.6."),
    by_department: Optional[str] = Query(None, description="Optional department to filter the distribution by.")
):
    """
    Returns the distribution of predicted risk scores in specified bins, globally or filtered by department.
    """
    try:
        df_pac, _, _ = load.load_dataset()
        df = df_pac.copy() # Use the patient dataset directly for risk_score
        df.columns = df.columns.str.lower() # Asegurar minúsculas

        # Ensure risk_score column exists and is not null
        df = df[df["risk_score"].notnull()]

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for risk distribution analysis after cleaning.")

        if by_department:
            df = df[df["department"].str.strip().str.upper() == by_department.strip().upper()]
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No data found for department: {by_department}")

        # Definir los bins y etiquetas para las categorías de riesgo
        risk_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Asegurarse de que el rango sea de 0 a 1
        risk_labels = ["Muy Bajo", "Bajo", "Moderado", "Alto", "Muy Alto"]

        df["risk_group"] = pd.cut(df["risk_score"], bins=risk_bins, right=False, include_lowest=True, labels=risk_labels)


        distribution = df["risk_group"].value_counts(normalize=True).sort_index()
        counts = df["risk_group"].value_counts().sort_index()

        results = []
        for label, proportion in distribution.items():
            if pd.notna(label): # Ensure label is not NaN
                results.append({
                    "risk_category": str(label),
                    "percentage": round(float(proportion * 100), 2),
                    "count": int(counts.get(label, 0)) # Use .get with default 0 in case of missing label
                })
        
        # Sort results based on the original category order
        category_order = ["Muy Bajo", "Bajo", "Moderado", "Alto", "Muy Alto"]
        results.sort(key=lambda x: category_order.index(x["risk_category"]))

        return {
            "department": by_department if by_department else "Global",
            "risk_distribution": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting risk distribution: {str(e)}")