from fastapi import APIRouter, HTTPException
import numpy as np
import load
from core.model_loader import load_model_and_features
from app.model_training.data_processing import preprocess_data

router = APIRouter()

@router.get("/importancia-municipio/{municipio}")
async def importancia_por_municipio(municipio: str):
    try:
        model, _ = load_model_and_features()
        if model is None:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")

        df, _, _ = load.load_dataset()
        df['MUNICIPIO'] = df['MUNICIPIO'].astype(str)
        df_filtrado = df[df['MUNICIPIO'].str.upper() == municipio.upper()]

        if df_filtrado.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para el municipio '{municipio}'")

        X_mpio, _, _, feature_names_mpio = preprocess_data(df_filtrado, augment=False)
        _, _, _, feature_names_global = preprocess_data(df, augment=False)

        dense_layers = [layer for layer in model.layers if hasattr(layer, 'kernel')]
        if not dense_layers:
            raise HTTPException(status_code=500, detail="No se encontró una capa densa con pesos en el modelo")

        weights = dense_layers[0].get_weights()[0]
        global_importance = np.mean(np.abs(weights), axis=1)
        importance_dict = dict(zip(feature_names_global, global_importance))

        activacion_promedio = np.mean(X_mpio, axis=0)

        importancia_ponderada = []
        for name, activacion in zip(feature_names_mpio, activacion_promedio):
            imp = importance_dict.get(name, 0.0)
            importancia_ponderada.append((name, float(activacion * imp)))

        resultado = [
            (name, imp) for name, imp in importancia_ponderada
            if not (name.startswith("MUNICIPIO_") or name.startswith("DEPARTAMENTO_"))
        ]

        resultado.sort(key=lambda x: x[1], reverse=True)

        return {
            "municipio": municipio.upper(),
            "top_factores": resultado[:10],
            "nota": "Factores no geográficos ponderados por activación media en este municipio"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
