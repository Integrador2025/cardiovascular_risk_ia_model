# app/routes/features.py

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import numpy as np
from typing import Dict, Optional, List
from pydantic import BaseModel
from core.model_loader import load_model_and_features
import tensorflow as tf
from collections import defaultdict

router = APIRouter(
    prefix="/v1/features",
    tags=["Features"],
    responses={404: {"description": "Not found"}}
)

# Mapeo de features a categorías (mantener en inglés para consistencia con el código, pero los comentarios pueden ser en español)
CATEGORIAS = {
    "Medical Factors": [
        'age', 'bmi', 'heart_rate', 'total_cholesterol', 'glucose',
        'is_smoker', 'family_history', 'diabetes', 'bpm_meds'
    ],
    "Social Conditions": [
        'marital_status', 'occupation', 'education_level', 'socioeconomic_status'
    ],
    "Infrastructure": [
        'has_electricity', 'has_water_supply', 'has_sewage', 'has_gas', 
        'has_internet', 'rural_area'
    ],
    "Environmental/Geographical": [
        'department', 'municipality', 'latitude', 'longitude', 
        'average_altitude', 'average_temperature', 'annual_precipitation', 'climate_classification'
    ],
    "Other Demographic": [
        'sex', 'ethnicity', 'diagnosis_year', 'diagnosis_month', 'days_since_diagnosis', 'pandemic_period'
    ]
}

class FeatureConfig(BaseModel):
    top_n: Optional[int] = None
    threshold: Optional[float] = None

def get_feature_weights(model) -> tuple:
    """Calculates and returns the absolute mean weights from the first dense layer of the model."""
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    
    if not dense_layers:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Model architecture incompatible - no dense layers found"
        )
    
    first_dense = dense_layers[0]
    weights = first_dense.get_weights()[0]
    # Retorna la importancia (promedio absoluto de los pesos) y el número de características de entrada
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
        # Cargar modelo, scaler y feature_names
        model, _, feature_names = load_model_and_features()
        if model is None or feature_names is None: # Scaler no es necesario aquí
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model or feature names not found. Train a model first."
            )

        importance, num_features = get_feature_weights(model)

        # Validate feature names match
        if len(feature_names) != num_features:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Feature count mismatch between model and loaded feature names."
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

        if not features:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No features found after applying filters."
            )

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
    return await _get_geo_importance("municipality_", top)

@router.get(
    "/geography/departments",
    summary="Get department importance", 
    response_description="Department impact scores"
)
async def get_department_importance(top: int = 10):
    """Get top departments by model feature importance""" 
    return await _get_geo_importance("department_", top)

async def _get_geo_importance(prefix: str, top: int):
    """Shared logic for geographical features"""
    try:
        # Cargar modelo, scaler y feature_names
        model, _, feature_names = load_model_and_features()
        if model is None or feature_names is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model or feature names not found. Train a model first."
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
        # Cargar modelo, scaler y feature_names
        model, _, feature_names = load_model_and_features()
        if model is None or feature_names is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model or feature names not found. Train a model first."
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
        # Cargar modelo, scaler y feature_names
        model, _, feature_names = load_model_and_features()
        if model is None or feature_names is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model or feature names not found. Train a model first."
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
        # Cargar modelo, scaler y feature_names
        model, _, feature_names = load_model_and_features()
        if model is None or feature_names is None:
            raise HTTPException(status_code=404, detail="Modelo o nombres de características no encontrados.")

        first_dense = next((layer for layer in model.layers if hasattr(layer, 'kernel')), None)
        if first_dense is None:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo.")

        pesos = first_dense.get_weights()[0]
        importancias = dict(zip(feature_names, np.mean(np.abs(pesos), axis=1)))

        resumen = defaultdict(float)

        for feat, valor in importancias.items():
            asignado = False
            for categoria, base_names in CATEGORIAS.items():
                # Comprobar si la característica comienza con alguno de los nombres base
                if any(feat.startswith(base) for base in base_names):
                    # Excluir categorías geográficas si se especifica
                    if not incluir_geo and categoria == "Environmental/Geographical":
                        continue
                    resumen[categoria] += float(valor)
                    asignado = True
                    break
            if not asignado:
                resumen["Others"] += float(valor) # Cambiado a "Others" para consistencia

        total = sum(resumen.values())
        if total == 0:
            raise HTTPException(status_code=500, detail="El total de importancia de características es cero, no se pueden calcular porcentajes.")

        porcentajes = {cat: round(float(val / total), 4) for cat, val in resumen.items()}

        return {
            "totals": resumen,
            "percentages": porcentajes,
            "note": "Based on the first dense layer of the model. Variables grouped by semantic type."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating importance by category: {str(e)}")