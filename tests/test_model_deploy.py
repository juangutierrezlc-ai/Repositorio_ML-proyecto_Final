"""Pruebas de integración de la API (FastAPI)."""

import io

import pytest
from fastapi.testclient import TestClient

import model_deploy

REGISTRO = {
    "tipo_credito": 4, "fecha_prestamo": "2025-01-07 14:40:36", "capital_prestado": 1852560,
    "plazo_meses": 12, "edad_cliente": 32, "tipo_laboral": "Empleado", "salario_cliente": 3500000,
    "total_otros_prestamos": 1000000, "cuota_pactada": 128650, "puntaje_datacredito": 795,
    "cant_creditosvigentes": 0, "huella_consulta": 2, "saldo_mora": 0, "saldo_total": 0,
    "promedio_ingresos_datacredito": 916148, "tendencia_ingresos": "Creciente",
}


@pytest.fixture()
def cliente(monkeypatch, modelo_pequeno):
    monkeypatch.setattr(model_deploy, "cargar_modelo", lambda: modelo_pequeno)
    monkeypatch.setattr(model_deploy, "cargar_metadata", lambda: {"modelo": "Prueba"})
    monkeypatch.setattr(model_deploy, "GUARDAR_PREDICCIONES", False)
    return TestClient(model_deploy.app)


def test_inicio_y_health(cliente):
    assert cliente.get("/").status_code == 200
    assert cliente.get("/health").json() == {"estado": "ok", "modelo_cargado": True}
    assert cliente.get("/modelo").json()["modelo"] == "Prueba"


def test_health_sin_modelo(monkeypatch):
    def sin_modelo():
        raise FileNotFoundError

    monkeypatch.setattr(model_deploy, "cargar_modelo", sin_modelo)
    respuesta = TestClient(model_deploy.app).get("/health")
    assert respuesta.json()["modelo_cargado"] is False


def test_predict_json_por_lotes(cliente):
    riesgoso = {**REGISTRO, "saldo_mora": 800000, "huella_consulta": 12, "puntaje_datacredito": 450}
    respuesta = cliente.post("/predict", json={"registros": [REGISTRO, riesgoso, REGISTRO]})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["n_registros"] == 3 and cuerpo["modelo"] == "Prueba"
    primera = cuerpo["predicciones"][0]
    assert primera["prob_no_pago"] + primera["prob_pago"] == pytest.approx(1, abs=1e-3)
    assert primera["prediccion"] in (0, 1)
    assert cuerpo["predicciones"][1]["prob_no_pago"] > primera["prob_no_pago"]


def test_predict_valida_campos(cliente):
    assert cliente.post("/predict", json={"registros": [{"tipo_credito": 4}]}).status_code == 422
    assert cliente.post("/predict", json={"registros": []}).status_code == 422


def test_predict_csv_json_y_csv(cliente, datos):
    csv = datos.head(25).to_csv(index=False).encode("utf-8")
    respuesta = cliente.post("/predict/csv", files={"archivo": ("lote.csv", io.BytesIO(csv), "text/csv")})
    assert respuesta.status_code == 200 and respuesta.json()["n_registros"] == 25

    respuesta = cliente.post("/predict/csv?formato=csv",
                             files={"archivo": ("lote.csv", io.BytesIO(csv), "text/csv")})
    assert respuesta.status_code == 200
    lineas = respuesta.text.strip().splitlines()
    assert len(lineas) == 26 and "prob_no_pago" in lineas[0]


def test_predict_csv_errores(cliente):
    no_csv = cliente.post("/predict/csv", files={"archivo": ("lote.txt", io.BytesIO(b"a,b"), "text/plain")})
    assert no_csv.status_code == 400
    sin_columnas = cliente.post("/predict/csv", files={"archivo": ("lote.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")})
    assert sin_columnas.status_code == 422
    vacio = cliente.post("/predict/csv", files={"archivo": ("lote.csv", io.BytesIO(b""), "text/csv")})
    assert vacio.status_code == 400


def test_registro_en_monitoreo_no_rompe_el_servicio(monkeypatch, modelo_pequeno, datos):
    import model_monitoring

    def falla(*args, **kwargs):
        raise RuntimeError("base no disponible")

    monkeypatch.setattr(model_deploy, "GUARDAR_PREDICCIONES", True)
    monkeypatch.setattr(model_monitoring, "guardar_predicciones", falla)
    monkeypatch.setattr(model_deploy, "cargar_modelo", lambda: modelo_pequeno)
    resultado = model_deploy.predecir(datos.head(3))
    model_deploy._registrar_en_monitoreo(resultado)  # no debe lanzar excepción
    assert set(resultado["nivel_riesgo"]) <= {"Alto", "Medio", "Bajo"}
