# app/routes/analisis.py

from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
import load
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data

router = APIRouter(prefix="/v1/analysis", tags=["Analysis"])

@router.get("/importancia-departamento/{departamento}")
async def importancia_por_departamento(departamento: str, num_factores: int = 10):
    try:
        # Cargar modelo, scaler y feature_names
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df['department'] = df['department'].astype(str)
        df_filtrado = df[df['department'].str.upper() == departamento.upper()]

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el departamento '{departamento}'")

        # Preprocesar datos del departamento usando el scaler y feature_names_global
        # Asegurarse de que df_filtrado tenga las columnas objetivo dummy si es necesario para preprocess_data
        if 'risk_score' not in df_filtrado.columns:
            df_filtrado['risk_score'] = 0.0 # Placeholder
        if 'cardiovascular_risk' not in df_filtrado.columns:
            df_filtrado['cardiovascular_risk'] = 0 # Placeholder
        if 'diagnosis_date' not in df_filtrado.columns:
            df_filtrado['diagnosis_date'] = pd.Timestamp("2024-01-01") # Placeholder
            
        X_mpio, _, _, _ = preprocess_data(df_filtrado, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
        
        dense_layers = [layer for layer in model.layers if hasattr(layer, 'kernel')]
        if not dense_layers:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo")

        weights = dense_layers[0].get_weights()[0]
        
        # Calcular la importancia global de las características
        # La importancia se calcula sobre los pesos de la primera capa densa.
        # np.mean(np.abs(weights), axis=1) es una forma común de obtener la importancia de los pesos.
        global_importance = np.mean(np.abs(weights), axis=1)
        importance_dict = dict(zip(feature_names_global, global_importance))

        # Calcular la activación promedio para el departamento
        # Si X_mpio tiene múltiples filas (varios pacientes en el departamanento), promediar las activaciones
        activacion_promedio = np.mean(X_mpio, axis=0)

        importancia_ponderada = []
        # Iterar sobre las características globales y sus activaciones promedio en el departamento
        for name, activacion in zip(feature_names_global, activacion_promedio):
            imp = importance_dict.get(name, 0.0) # Obtener la importancia global
            importancia_ponderada.append((name, float(activacion * imp))) # Ponderar por la activación

        # Filtrar características geográficas y ordenar
        resultado = [
            (name, imp) for name, imp in importancia_ponderada
            if not (name.startswith("municipality_") or name.startswith("department_"))
        ]

        resultado.sort(key=lambda x: x[1], reverse=True)

        return {
            "departamento": departamento.upper(),
            "top_factores": resultado[:num_factores], # Aplicar el límite de num_factores
            "nota": f"Top {num_factores} factores no geográficos ponderados por activación media en este departamento"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el análisis por departamento: {str(e)}") # Corregido el mensaje de error

@router.get("/importancia-municipio/{municipio}")
async def importancia_por_municipio(municipio: str, num_factores: int = 10):
    try:
        # Cargar modelo, scaler y feature_names
        model, scaler, feature_names_global = load_model_and_features()
        if model is None or scaler is None or feature_names_global is None:
            raise HTTPException(status_code=404, detail="Modelo o preprocesadores no encontrados. Entrene el modelo primero.")

        df, _, _ = load.load_dataset()
        df['municipality'] = df['municipality'].astype(str)
        df_filtrado = df[df['municipality'].str.upper() == municipio.upper()]

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el municipio '{municipio}'")

        # Preprocesar datos del municipio usando el scaler y feature_names_global
        # Asegurarse de que df_filtrado tenga las columnas objetivo dummy si es necesario para preprocess_data
        if 'risk_score' not in df_filtrado.columns:
            df_filtrado['risk_score'] = 0.0 # Placeholder
        if 'cardiovascular_risk' not in df_filtrado.columns:
            df_filtrado['cardiovascular_risk'] = 0 # Placeholder
        if 'diagnosis_date' not in df_filtrado.columns:
            df_filtrado['diagnosis_date'] = pd.Timestamp("2024-01-01") # Placeholder
            
        X_mpio, _, _, _ = preprocess_data(df_filtrado, augment=False, scaler_obj=scaler, feature_names_obj=feature_names_global)
        
        dense_layers = [layer for layer in model.layers if hasattr(layer, 'kernel')]
        if not dense_layers:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo")

        weights = dense_layers[0].get_weights()[0]
        
        # Calcular la importancia global de las características
        # La importancia se calcula sobre los pesos de la primera capa densa.
        # np.mean(np.abs(weights), axis=1) es una forma común de obtener la importancia de los pesos.
        global_importance = np.mean(np.abs(weights), axis=1)
        importance_dict = dict(zip(feature_names_global, global_importance))

        # Calcular la activación promedio para el municipio
        # Si X_mpio tiene múltiples filas (varios pacientes en el municipio), promediar las activaciones
        activacion_promedio = np.mean(X_mpio, axis=0)

        importancia_ponderada = []
        # Iterar sobre las características globales y sus activaciones promedio en el municipio
        for name, activacion in zip(feature_names_global, activacion_promedio):
            imp = importance_dict.get(name, 0.0) # Obtener la importancia global
            importancia_ponderada.append((name, float(activacion * imp))) # Ponderar por la activación

        # Filtrar características geográficas y ordenar
        resultado = [
            (name, imp) for name, imp in importancia_ponderada
            if not (name.startswith("municipality_") or name.startswith("department_"))
        ]

        resultado.sort(key=lambda x: x[1], reverse=True)

        return {
            "municipio": municipio.upper(),
            "top_factores": resultado[:num_factores], # Aplicar el límite de num_factores
            "nota": f"Top {num_factores} factores no geográficos ponderados por activación media en este municipio"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el análisis por municipio: {str(e)}")