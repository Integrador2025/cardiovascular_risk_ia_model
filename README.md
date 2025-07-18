# Medical IA

## Descripción general
Medical IA es una API REST desarrollada con FastAPI para analizar datos relacionados con el riesgo cardiovascular y datos poblacionales. Proporciona endpoints para calcular métricas de riesgo y la importancia de factores en niveles de departamento y municipio, utilizando modelos predictivos. La aplicación está diseñada para ser escalable y fácil de usar, con scripts auxiliares para procesar datos.

## Requisitos previos
- **Python 3.9+**
- **Git** para clonar el repositorio
- **pip** para instalar dependencias de Python
- Un entorno virtual (recomendado)

## Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/Integrador2025/cardiovascular_risk_ia_model
cd cardiovascular_risk_ia_model
```

### 2. Crear un entorno virtual
```bash
python -m venv venv
source venv/bin/activate
**En Windows:** venv\Scripts\activate
```

### 3. Instalar dependencias
Instala las dependencias de Python necesarias:
```bash
pip install -r requirements.txt
```
Dependencias principales:
- `fastapi`
- `uvicorn`
- `pandas`
- `numpy`
- `python-dotenv`

### 4. Configuración inicial
Por ahora, la aplicación no requiere conexión a una base de datos. Asegúrate de que los modelos predictivos y los datos necesarios (por ejemplo, archivos CSV o datasets locales) estén disponibles en el directorio correspondiente (ver la sección de estructura del proyecto).

## Ejecutar la API
1. Inicia el servidor FastAPI:
   ```bash
   uvicorn app.main:app --reload
   ```
2. Accede a la API en `http://localhost:8000`.
3. Explora los endpoints disponibles en la documentación interactiva: `http://localhost:8000/docs`.

### Ejemplos de endpoints
- `GET /v1/analysis/importancia-departamento/{departamento}`: Devuelve la importancia de los factores que influyen en el riesgo cardiovascular para un departamento.
- `GET /v1/analysis/importancia-municipio/{municipio}`: Devuelve la importancia de los factores para un municipio.

## Uso de scripts
El proyecto incluye scripts para procesar datos localmente:
1. Para analizar un dataset (por ejemplo, `municipios_colombia.csv` y `pacient_dataset.csv`):
   ```bash
   python scripts/process_data.py
   ```
2. Edita el script para ajustar los nombres de los archivos o parámetros según tus necesidades.

## Estructura del proyecto
```
├── app/
│   ├── routes/          # Endpoints de la API (ej. analisis.py)
│   ├── main.py          # Punto de entrada de la aplicación FastAPI
├── scripts/             # Scripts utilitarios (ej. process_data.py)
├── data/                # Datasets locales (ej. municipalities.csv)
├── models/              # Modelos predictivos preentrenados
├── requirements.txt     # Dependencias de Python
├── README.md            # Documentación del proyecto
```
