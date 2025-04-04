from fastapi import FastAPI, HTTPException
from typing import Optional
import numpy as np
import tensorflow as tf
import os
import load
from app.model_training.data_processing import preprocess_data
from app.model_training.training import train_with_stratified_kfold
from functools import lru_cache

app = FastAPI()

# Configuración de rutas
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_final.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def load_model_and_features():
    """Carga el modelo y los nombres de características con cache"""
    try:
        if not os.path.exists(MODEL_PATH):
            return None, None
        
        # Cargar modelo
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Cargar feature names (requiere cargar datos)
        df_pacientes, _, _ = load.load_dataset()
        _, _, _, feature_names = preprocess_data(df_pacientes)
        
        return model, feature_names
    except Exception as e:
        print(f"Error cargando modelo: {str(e)}")
        return None, None

@app.get("/")
def read_root():
    model_loaded = "SI" if os.path.exists(MODEL_PATH) else "NO"
    return {
        "message": "Bienvenido al API del modelo de riesgo cardiovascular",
        "model_cargado": model_loaded,
        "endpoints": {
            "train": "POST /train-model/",
            "features": "GET /feature-importance/"
        }
    }

@app.post("/train-model/")
async def train_model():
    """Endpoint para entrenar y guardar el modelo"""
    try:
        df_pacientes, _, _ = load.load_dataset()
        X, Y, _, feature_names = preprocess_data(df_pacientes)
        
        model = train_with_stratified_kfold(X, Y, feature_names)
        
        # Guardar modelo
        model.save(MODEL_PATH)
        load_model_and_features.cache_clear()  # Limpiar cache para recargar
        
        return {
            "status": "success",
            "message": "Modelo entrenado y guardado",
            "path": MODEL_PATH,
            "feature_count": len(feature_names)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance/")
async def get_feature_importance():
    """
    Retorna las características y sus importancias calculadas
    """
    try:
        model, feature_names = load_model_and_features()
        
        if model is None:
            raise HTTPException(
                status_code=404,
                detail="Modelo no encontrado. Entrene primero con POST /train-model/"
            )

        # Identificar capas densas disponibles
        dense_layers = [i for i, layer in enumerate(model.layers) 
                      if hasattr(layer, 'kernel')]
        
        if not dense_layers:
            raise HTTPException(
                status_code=500,
                detail="El modelo no contiene capas densas"
            )
        
        # Usar la primera capa densa por defecto
        layer_idx = dense_layers[0]
        
        # Calcular importancia
        weights = model.layers[layer_idx].get_weights()[0]
        importance = np.mean(np.abs(weights), axis=1)
        
        # Procesar resultados
        features_importance = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Retornar solo el diccionario con las características e importancias
        return {
            name: float(imp) 
            for name, imp in features_importance
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)