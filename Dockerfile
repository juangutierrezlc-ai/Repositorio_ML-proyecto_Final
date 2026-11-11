# Imagen de la aplicación: API de predicción (FastAPI + Uvicorn) y app de monitoreo (Streamlit)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Dependencias (capa separada para aprovechar la caché de Docker)
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

# 2. Código fuente, modelo entrenado y datos de referencia para el monitoreo
COPY mlops_pipeline/ ./mlops_pipeline/
COPY modelos/ ./modelos/
COPY Base_de_datos.csv .

# 3. Usuario sin privilegios (buena práctica de seguridad)
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p monitoreo \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Por defecto se levanta la API. Para la app de monitoreo:
#   docker run -p 8501:8501 credito-ml streamlit run mlops_pipeline/app_monitoreo.py --server.address=0.0.0.0
CMD ["uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline", "--host", "0.0.0.0", "--port", "8000"]