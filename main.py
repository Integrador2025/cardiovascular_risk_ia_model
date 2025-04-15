from fastapi import FastAPI
from app.routes import training, features, metrics, analisis, summary, predict,predict_population

app = FastAPI(
    title="API de Riesgo Cardiovascular",
    description="Servicio para entrenamiento y análisis de modelos de predicción de riesgo cardiovascular",
    version="1.0"
)

app.include_router(training.router)
app.include_router(features.router)
app.include_router(metrics.router)
app.include_router(analisis.router)
app.include_router(summary.router)
app.include_router(predict.router)
app.include_router(predict_population.router)

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido al API del modelo de riesgo cardiovascular",
    }
