from fastapi import FastAPI
from app.routes import training, features, metrics, analisis

app = FastAPI(
    title="API de Riesgo Cardiovascular",
    description="Servicio para entrenamiento y análisis de modelos de predicción de riesgo cardiovascular",
    version="1.0"
)

app.include_router(training.router)
app.include_router(features.router)
app.include_router(metrics.router)
app.include_router(analisis.router)

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido al API del modelo de riesgo cardiovascular",
        "endpoints": [
            "/train-model/",
            "/feature-importance/",
            "/municipios/",
            "/departamentos/",
            "/metricas/",
            "/importancia-municipio/{municipio}"
        ]
    }
