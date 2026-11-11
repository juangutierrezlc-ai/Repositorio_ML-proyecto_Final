"""Prueba de la app de Streamlit con el framework de pruebas de Streamlit (AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import model_monitoring as mon

APP = str(Path(__file__).resolve().parent.parent / "mlops_pipeline" / "app_monitoreo.py")


@pytest.fixture()
def base_monitoreo(tmp_path, monkeypatch, modelo_pequeno):
    monkeypatch.setattr(mon, "DIR_MONITOREO", tmp_path)
    monkeypatch.setattr(mon, "RUTA_DB", tmp_path / "monitoreo.db")
    monkeypatch.setattr(mon, "cargar_modelo", lambda *a, **k: modelo_pequeno)
    return tmp_path


def test_app_sin_resultados_muestra_instrucciones(base_monitoreo):
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("Ejecutar monitoreo" in info.value for info in app.info)


def test_app_con_resultados(base_monitoreo):
    mon.ejecutar_monitoreo("M", 300, incluir_simulacion=True)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value.startswith("📊 Monitoreo")
    assert len(app.tabs) == 6
    assert len(app.dataframe) >= 3
    assert len(app.error) + len(app.warning) > 0  # hay alertas con el drift simulado

    # cambiar de periodo y de variable sigue funcionando
    next(s for s in app.sidebar.selectbox if s.label == "Periodo a analizar").set_value("2025-03").run()
    assert not app.exception
    variable = next(s for s in app.main.selectbox if s.label == "Variable")
    variable.set_value("tipo_laboral").run()
    assert not app.exception


def test_app_boton_ejecuta_monitoreo(base_monitoreo):
    app = AppTest.from_file(APP, default_timeout=90).run()
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert (base_monitoreo / "monitoreo.db").exists()
