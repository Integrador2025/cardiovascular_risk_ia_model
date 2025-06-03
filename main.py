# main.py

from fastapi import FastAPI
# import py_eureka_client.eureka_client as eureka_client
# import asyncio
from app.routes import training, features, metrics, analisis, summary, predict, predict_population
from app.routes import socio_environmental, comparision

app = FastAPI(
    title="API de Riesgo Cardiovascular",
    description="Servicio para entrenamiento y análisis de modelos de predicción de riesgo cardiovascular",
    version="1.0"
)

# @app.on_event("startup")
# async def startup_event():
#     await eureka_client.init_async(
#     eureka_server="http://localhost:8761/eureka",
#     app_name="CR-PYTHON_AI",
#     instance_host="localhost",  
#     instance_port=8000
#     )

#     print("Servicio esta en Eureka")

# @app.on_event("shutdown")
# async def shutdown_event():
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(None, eureka_client.stop)
#     print("Servicio no esta en Eureka")

app.include_router(training.router)
app.include_router(features.router)
app.include_router(metrics.router)
app.include_router(analisis.router)
app.include_router(summary.router)
app.include_router(predict.router)
app.include_router(predict_population.router)
app.include_router(socio_environmental.router)
app.include_router(comparision.router)

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido al API del modelo de riesgo cardiovascular",
    }