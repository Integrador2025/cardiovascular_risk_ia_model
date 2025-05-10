from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import numpy as np
from typing import Dict, Optional
from pydantic import BaseModel
from core.model_loader import load_model_and_features
import tensorflow as tf
from collections import defaultdict

router = APIRouter(
    prefix="/v1/features",
    tags=["Features"],
    responses={404: {"description": "Not found"}}
)

# Mapeo de features a categorías
# CATEGORIAS = {
#     "Factores Médicos": ['age', 'bmi', 'total_cholesterol', 'is_smoker', 'family_history'],
#     "Condiciones Sociales": ['marital_status', 'occupation', 'education_level', 'socioeconomic_status'],
#     "Infraestructura": ['has_electricity', 'has_water_supply', 'has_sewage', 'has_gas', 'has_internet'],
#     "Ambientales/Geográficas": ['department', 'municipality'],
#     "Otros Demográficos": ['sex', 'area', 'ethnicity']
# }

CATEGORIAS = {
    "Factores Médicos": [
        'age',               # Edad (antes EDAD)
        'bmi',               # IMC (antes IMC)
        'total_cholesterol', # Colesterol total (antes COLESTEROL)
        'is_smoker',         # Fumador (antes FUMADOR)
        'family_history',    # Antecedentes familiares (antes ANTECEDENTES_FAMILIARES)
        'heart_rate',        # Frecuencia cardíaca (nueva)
        'glucose',           # Glucosa (nueva)
        'diabetes',          # Diabetes (nueva)
        'bpm_meds'           # Medicación para presión (antes bpm_meds)
    ],
    "Condiciones Sociales": [
        'marital_status',    # Estado civil (antes ESTADO_CIVIL)
        'occupation',        # Ocupación (antes OCUPACION)
        'education_level',   # Nivel educativo (antes NIVEL_EDUCATIVO)
        'socioeconomic_status' # Estrato (antes ESTRATO)
    ],
    "Infraestructura": [
        'has_electricity',   # Acceso eléctrico (antes ACCESO_ELECTRICO)
        'has_water_supply',  # Acueducto (antes ACUEDUCTO)
        'has_sewage',        # Alcantarillado (antes ALCANTARILLADO)
        'has_gas',           # Gas natural (antes GAS_NATURAL)
        'has_internet',      # Internet (antes INTERNET)
        'rural_area'         # Área rural (antes AREA)
    ],
    "Ambientales/Geográficas": [
        'department',        # Departamento (antes DEPARTAMENTO)
        'municipality',      # Municipio (antes MUNICIPIO)
        'latitude',          # Latitud (nueva)
        'longitude',         # Longitud (nueva)
        'average_altitude',  # Altitud media (nueva)
        'climate_classification' # Clasificación climática (nueva)
    ],
    "Otros Demográficos": [
        'sex',               # Sexo (antes SEXO)
        'ethnicity',         # Etnia (antes ETNIA)
        'diagnosis_year',    # Año diagnóstico (nueva)
        'diagnosis_month',   # Mes diagnóstico (nueva)
        'pandemic_period'    # Periodo pandemia (nueva)
    ]
}

class FeatureConfig(BaseModel):
    top_n: Optional[int] = None
    threshold: Optional[float] = None

def get_feature_weights(model) -> tuple:
    """Centralized feature weight calculation"""
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    
    if not dense_layers:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Model architecture incompatible - no dense layers found"
        )
    
    first_dense = dense_layers[0]
    weights = first_dense.get_weights()[0]
    return np.mean(np.abs(weights), axis=1), weights.shape[0]

@router.get(
    "/importance",
    summary="Get global feature importance",
    response_description="Feature importance scores"
)
async def get_feature_importance(
    config: FeatureConfig = Depends(),
    exclude_geo: bool = True
):
    """
    Get feature importance scores based on model weights with options to:
    - Filter top N features
    - Apply importance threshold
    - Exclude geographical features
    """
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found. Train a model first."
            )

        importance, num_features = get_feature_weights(model)

        # Validate feature names match
        if len(feature_names) != num_features:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Feature count mismatch between model and dataset"
            )

        # Create importance dictionary - updated column name prefixes
        features = [
            (name, float(imp)) 
            for name, imp in zip(feature_names, importance)
            if not (exclude_geo and ("municipality_" in name or "department_" in name))
        ]

        # Apply filters
        if config.top_n:
            features = sorted(features, key=lambda x: x[1], reverse=True)[:config.top_n]
            
        if config.threshold:
            features = [f for f in features if f[1] >= config.threshold]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "count": len(features),
                "features": dict(features),
                "stats": {
                    "max": max(imp for _, imp in features),
                    "min": min(imp for _, imp in features),
                    "mean": np.mean([imp for _, imp in features])
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature calculation failed: {str(e)}"
        )

@router.get(
    "/geography/municipalities",
    summary="Get municipality importance",
    response_description="Municipality impact scores"
)
async def get_municipality_importance(top: int = 10):
    """Get top municipalities by model feature importance"""
    return await _get_geo_importance("municipality_", top)  # Updated prefix

@router.get(
    "/geography/departments",
    summary="Get department importance", 
    response_description="Department impact scores"
)
async def get_department_importance(top: int = 10):
    """Get top departments by model feature importance""" 
    return await _get_geo_importance("department_", top)  # Updated prefix

async def _get_geo_importance(prefix: str, top: int):
    """Shared logic for geographical features"""
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found. Train a model first."
            )

        importance, _ = get_feature_weights(model)
        
        geo_features = [
            (name.replace(prefix, ""), float(imp))
            for name, imp in zip(feature_names, importance)
            if name.startswith(prefix)
        ]

        geo_features.sort(key=lambda x: x[1], reverse=True)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "top": top,
                "features": dict(geo_features[:top]),
                "total": len(geo_features)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Geography analysis failed: {str(e)}"
        )

@router.get(
    "/statistics",
    summary="Get detailed statistics of feature importance",
    response_description="Statistics of feature importance scores"
)
async def get_feature_statistics(exclude_geo: bool = True):
    """
    Get detailed statistics of feature importance, including:
    - Min, max, mean, median, percentiles
    - Total number of features
    """
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found. Train a model first."
            )

        importance, num_features = get_feature_weights(model)

        # Filter features if exclude_geo is True - updated column name prefixes
        filtered_importance = [
            imp for name, imp in zip(feature_names, importance)
            if not (exclude_geo and ("municipality_" in name or "department_" in name))
        ]

        if not filtered_importance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No features found after applying filters."
            )

        # Calculate statistics
        stats = {
            "min": float(np.min(filtered_importance)),
            "max": float(np.max(filtered_importance)),
            "mean": float(np.mean(filtered_importance)),
            "median": float(np.median(filtered_importance)),
            "percentile_25": float(np.percentile(filtered_importance, 25)),
            "percentile_75": float(np.percentile(filtered_importance, 75)),
            "total_features": len(filtered_importance)
        }

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"statistics": stats}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics calculation failed: {str(e)}"
        )

@router.get(
    "/least_important",
    summary="Get least important features",
    response_description="List of least important features"
)
async def get_least_important_features(
    top: int = 10,
    exclude_geo: bool = True
):
    """
    Get the least important features based on model weights.
    """
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found. Train a model first."
            )

        importance, num_features = get_feature_weights(model)

        # Filter features if exclude_geo is True - updated column name prefixes
        features = [
            (name, float(imp))
            for name, imp in zip(feature_names, importance)
            if not (exclude_geo and ("municipality_" in name or "department_" in name))
        ]

        # Sort by importance (ascending) and get the least important
        features.sort(key=lambda x: x[1])
        least_important = features[:top]

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "count": len(least_important),
                "features": dict(least_important)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Least important feature calculation failed: {str(e)}"
        )
    
@router.get("/por-categoria")
async def importancia_por_categoria(incluir_geo: bool = False):
    try:
        model, feature_names = load_model_and_features()
        if model is None:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")

        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos")

        pesos = first_dense.get_weights()[0]
        importancias = dict(zip(feature_names, np.mean(np.abs(pesos), axis=1)))

        resumen = defaultdict(float)

        for feat, valor in importancias.items():
            asignado = False
            for categoria, base_names in CATEGORIAS.items():
                if any(feat.startswith(base) for base in base_names):
                    if not incluir_geo and categoria == "Ambientales/Geográficas":
                        continue
                    resumen[categoria] += float(valor)
                    asignado = True
                    break
            if not asignado:
                resumen["Otras"] += float(valor)

        total = sum(resumen.values())
        porcentajes = {cat: round(float(val / total), 4) for cat, val in resumen.items()}

        return {
            "totales": resumen,
            "porcentajes": porcentajes,
            "nota": "Basado en la primera capa densa del modelo. Se agruparon variables por tipo semántico."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al calcular importancias por categoría: {str(e)}")