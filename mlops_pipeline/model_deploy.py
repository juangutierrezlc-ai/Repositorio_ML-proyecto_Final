"""
model_deploy.py
===============
Núcleo del despliegue: expone el mejor modelo como un servicio REST con FastAPI.

Responsabilidades:
    - Carga del modelo entrenado (modelos/mejor_modelo.joblib) y de su metadata.
    - Lógica de predicción: recibe los datos crudos del crédito, el Pipeline del modelo
      aplica las mismas transformaciones de ft_eng.py y retorna el resultado.
    - Endpoint /predict que acepta JSON y /predict/csv que acepta un archivo CSV.
    - Predicción por lotes: ambos endpoints reciben muchos registros en una sola solicitud.
    - Cada predicción se guarda en la tabla de monitoreo (ver model_monitoring.py).

Ejecutar localmente (desde la raíz del repositorio):
    uvicorn model_deploy:app --app-dir mlops_pipeline --host 0.0.0.0 --port 8000
    o bien: python mlops_pipeline/model_deploy.py

Documentación interactiva: http://localhost:8000/docs
"""

import io
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = Path(os.getenv("MODELO_PATH", RAIZ / "modelos" / "mejor_modelo.joblib"))
RUTA_METADATA = Path(os.getenv("METADATA_PATH", RAIZ / "modelos" / "metadata_mejor_modelo.json"))
GUARDAR_PREDICCIONES = os.getenv("GUARDAR_PREDICCIONES", "true").lower() == "true"
MAX_REGISTROS = int(os.getenv("MAX_REGISTROS", "20000"))
MAX_TAMANO_CSV_MB = 10
MARGEN_RIESGO_MEDIO = 0.10

COLUMNAS_REQUERIDAS = [
    "tipo_credito", "fecha_prestamo", "capital_prestado", "plazo_meses", "edad_cliente",
    "tipo_laboral", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta", "saldo_mora",
    "saldo_total", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal", "promedio_ingresos_datacredito", "tendencia_ingresos",
]

logger = logging.getLogger("model_deploy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Esquemas de entrada y salida
# ---------------------------------------------------------------------------
class Credito(BaseModel):
    """Datos de una solicitud de crédito (los mismos campos de Base_de_datos.csv)."""
    tipo_credito: int = Field(..., examples=[4])
    fecha_prestamo: str | None = Field(None, examples=["2025-01-07 14:40:36"])
    capital_prestado: float = Field(..., ge=0, examples=[1852560.0])
    plazo_meses: int = Field(..., ge=1, examples=[12])
    edad_cliente: int | None = Field(None, examples=[32])
    tipo_laboral: str | None = Field(None, examples=["Empleado"])
    salario_cliente: float | None = Field(None, examples=[3500000])
    total_otros_prestamos: float | None = Field(None, examples=[1000000])
    cuota_pactada: float = Field(..., ge=0, examples=[128650])
    puntaje_datacredito: float | None = Field(None, examples=[795.0])
    cant_creditosvigentes: int = Field(0, ge=0, examples=[0])
    huella_consulta: int = Field(0, ge=0, examples=[2])
    saldo_mora: float | None = Field(0.0, examples=[0.0])
    saldo_total: float | None = Field(None, examples=[0.0])
    creditos_sectorFinanciero: int = Field(0, ge=0, examples=[0])
    creditos_sectorCooperativo: int = Field(0, ge=0, examples=[0])
    creditos_sectorReal: int = Field(0, ge=0, examples=[0])
    promedio_ingresos_datacredito: float | None = Field(None, examples=[916148.0])
    tendencia_ingresos: str | None = Field(None, examples=["Creciente"])


class SolicitudLote(BaseModel):
    registros: list[Credito] = Field(..., min_length=1)


class Prediccion(BaseModel):
    indice: int
    prob_no_pago: float
    prob_pago: float
    prediccion: int
    resultado: str
    nivel_riesgo: str


class RespuestaPrediccion(BaseModel):
    modelo: str
    umbral_decision: float
    n_registros: int
    predicciones: list[Prediccion]


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def cargar_modelo():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {RUTA_MODELO}. Ejecuta model_training_evaluation.py"
        )
    logger.info("Cargando modelo desde %s", RUTA_MODELO)
    return joblib.load(RUTA_MODELO)


@lru_cache(maxsize=1)
def cargar_metadata() -> dict:
    if RUTA_METADATA.exists():
        return json.loads(RUTA_METADATA.read_text(encoding="utf-8"))
    return {"modelo": type(cargar_modelo()).__name__}


# ---------------------------------------------------------------------------
# Lógica de predicción
# ---------------------------------------------------------------------------
def _umbral_no_pago(modelo) -> float:
    """Probabilidad de no pago a partir de la cual el modelo predice 'no paga'."""
    umbral_pago = getattr(modelo, "best_threshold_", 0.5)
    return 1 - float(umbral_pago)


def predecir(datos: pd.DataFrame) -> pd.DataFrame:
    """Aplica el modelo (que incluye todas las transformaciones) a un lote de registros."""
    faltantes = [c for c in ("tipo_credito", "capital_prestado", "plazo_meses", "cuota_pactada")
                 if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
    datos = datos.reindex(columns=list(dict.fromkeys(COLUMNAS_REQUERIDAS + list(datos.columns))))

    modelo = cargar_modelo()
    prob_no_pago = modelo.predict_proba(datos)[:, list(modelo.classes_).index(0)]
    prediccion = modelo.predict(datos).astype(int)
    corte = _umbral_no_pago(modelo)

    resultado = datos.copy()
    resultado["prob_no_pago"] = prob_no_pago.round(4)
    resultado["prob_pago"] = (1 - prob_no_pago).round(4)
    resultado["prediccion"] = prediccion
    resultado["resultado"] = ["Paga a tiempo" if p == 1 else "Riesgo de no pago" for p in prediccion]
    resultado["nivel_riesgo"] = [
        "Alto" if p == 0 else "Medio" if prob >= corte - MARGEN_RIESGO_MEDIO else "Bajo"
        for p, prob in zip(prediccion, prob_no_pago)
    ]
    return resultado


def _registrar_en_monitoreo(resultado: pd.DataFrame) -> None:
    """Guarda datos + pronóstico en la tabla de monitoreo sin afectar la respuesta."""
    if not GUARDAR_PREDICCIONES:
        return
    try:
        from model_monitoring import guardar_predicciones
        guardar_predicciones(resultado, origen="api")
    except Exception:  # el monitoreo nunca debe tumbar el servicio
        logger.exception("No se pudieron guardar las predicciones en la tabla de monitoreo")


def _armar_respuesta(resultado: pd.DataFrame) -> RespuestaPrediccion:
    columnas = ["prob_no_pago", "prob_pago", "prediccion", "resultado", "nivel_riesgo"]
    predicciones = [Prediccion(indice=i, **fila) for i, fila in
                    enumerate(resultado[columnas].to_dict(orient="records"))]
    return RespuestaPrediccion(
        modelo=cargar_metadata().get("modelo", "desconocido"),
        umbral_decision=round(_umbral_no_pago(cargar_modelo()), 4),
        n_registros=len(predicciones),
        predicciones=predicciones,
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API - Predicción de pago a tiempo de créditos",
    description="Proyecto final de Machine Learning. Predice si un cliente pagará su crédito a tiempo.",
    version="1.0.0",
)


@app.get("/")
def inicio():
    return {"servicio": "Predicción de pago a tiempo", "documentacion": "/docs",
            "endpoints": ["/health", "/modelo", "/predict", "/predict/csv"]}


@app.get("/health")
def health():
    try:
        cargar_modelo()
        return {"estado": "ok", "modelo_cargado": True}
    except FileNotFoundError:
        return {"estado": "sin modelo", "modelo_cargado": False}


@app.get("/modelo")
def info_modelo():
    return cargar_metadata()


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(solicitud: SolicitudLote):
    """Predicción por lotes en formato JSON: {"registros": [{...}, {...}]}."""
    if len(solicitud.registros) > MAX_REGISTROS:
        raise HTTPException(413, f"Máximo {MAX_REGISTROS} registros por solicitud")
    datos = pd.DataFrame([r.model_dump() for r in solicitud.registros])
    try:
        resultado = predecir(datos)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    _registrar_en_monitoreo(resultado)
    return _armar_respuesta(resultado)


@app.post("/predict/csv")
async def predict_csv(archivo: UploadFile = File(...),
                      formato: str = Query("json", pattern="^(json|csv)$")):
    """Predicción por lotes desde un CSV. Con formato=csv devuelve el CSV con las predicciones."""
    if not (archivo.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "El archivo debe ser .csv")
    contenido = await archivo.read()
    if len(contenido) > MAX_TAMANO_CSV_MB * 1024 * 1024:
        raise HTTPException(413, f"El archivo supera {MAX_TAMANO_CSV_MB} MB")
    try:
        datos = pd.read_csv(io.BytesIO(contenido))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise HTTPException(400, "No se pudo leer el CSV") from error
    if len(datos) > MAX_REGISTROS:
        raise HTTPException(413, f"Máximo {MAX_REGISTROS} registros por solicitud")
    try:
        resultado = predecir(datos)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    _registrar_en_monitoreo(resultado)

    if formato == "csv":
        salida = io.StringIO()
        resultado.to_csv(salida, index=False)
        return StreamingResponse(iter([salida.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=predicciones.csv"})
    return _armar_respuesta(resultado)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))