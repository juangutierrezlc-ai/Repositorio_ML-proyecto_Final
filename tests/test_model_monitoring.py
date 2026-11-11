"""Pruebas del monitoreo y de las métricas de data drift."""

import numpy as np
import pandas as pd
import pytest

import model_monitoring as mon

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_psi_cero_para_distribuciones_iguales_y_alto_si_cambian():
    referencia = RNG.normal(0, 1, 5000)
    p_ref, p_igual = mon.distribuciones_numericas(referencia, RNG.normal(0, 1, 5000))
    _, p_desplazada = mon.distribuciones_numericas(referencia, RNG.normal(1.5, 1, 5000))
    assert mon.psi(p_ref, p_igual) < 0.02
    assert mon.psi(p_ref, p_desplazada) > mon.UMBRALES["psi"][1]


def test_jensen_shannon_limites():
    assert mon.jensen_shannon([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert mon.jensen_shannon([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_kolmogorov_smirnov_detecta_cambio():
    estadistico, p_valor = mon.kolmogorov_smirnov(RNG.normal(0, 1, 2000), RNG.normal(1, 1, 2000))
    assert estadistico > 0.3
    assert p_valor < 0.001


def test_chi_cuadrado():
    ref = pd.Series(["A"] * 500 + ["B"] * 500)
    _, p_igual, v_igual = mon.chi_cuadrado(ref, pd.Series(["A"] * 250 + ["B"] * 250))
    _, p_distinta, v_distinta = mon.chi_cuadrado(ref, pd.Series(["A"] * 450 + ["B"] * 50))
    assert p_igual > 0.5 and v_igual < 0.01
    assert p_distinta < 0.001 and v_distinta > 0.3
    assert mon.chi_cuadrado(pd.Series(["A"] * 10), pd.Series(["A"] * 10)) == (0.0, 1.0, 0.0)


def test_variable_casi_constante_no_falla():
    referencia = np.r_[np.zeros(990), np.ones(10) * 500]
    p_ref, p_act = mon.distribuciones_numericas(referencia, np.r_[np.zeros(400), np.ones(100) * 500])
    assert len(p_ref) == len(p_act) == 2
    assert mon.psi(p_ref, p_act) > 0


def test_drift_variable_semaforo():
    referencia = RNG.normal(0, 1, 3000)
    estable = mon.drift_variable(referencia, RNG.normal(0, 1, 500), "numerica")
    cambio = mon.drift_variable(referencia, RNG.normal(2, 1, 500), "numerica")
    categorica = mon.drift_variable(pd.Series(["A", "B"] * 500), pd.Series(["A"] * 450 + ["B"] * 50),
                                    "categorica")
    assert estable["estado"] == "Verde"
    assert cambio["estado"] == "Rojo"
    assert categorica["estado"] == "Rojo" and not np.isnan(categorica["chi2_p_valor"])


def test_calcular_drift_cubre_todas_las_variables(datos):
    referencia = datos.assign(prob_no_pago=RNG.uniform(size=len(datos)))
    actual = referencia.sample(400, random_state=1)
    tabla = mon.calcular_drift(referencia, actual)
    esperadas = mon.VARIABLES_NUMERICAS + [mon.VARIABLE_PREDICCION] + mon.VARIABLES_CATEGORICAS
    assert tabla["variable"].tolist() == esperadas
    assert set(tabla["estado"]) <= {"Verde", "Amarillo", "Rojo"}


# ---------------------------------------------------------------------------
# Muestreo, tabla y análisis temporal
# ---------------------------------------------------------------------------
def test_muestreo_periodico(datos):
    muestras = mon.muestreo_periodico(datos, "M", tamano=200)
    assert "2025-01" in muestras
    assert all(len(m) <= 200 for m in muestras.values())
    assert len(muestras["2026-04"]) == 11  # periodo con pocos registros: se toman todos


def test_guardar_y_leer_predicciones(tmp_path, datos, modelo_pequeno):
    ruta_db = tmp_path / "prueba.db"
    puntuados = mon.puntuar(datos.head(30).drop(columns=["Pago_atiempo"]), modelo_pequeno)
    assert puntuados["prob_no_pago"].between(0, 1).all()
    assert mon.guardar_predicciones(puntuados, origen="api", ruta_db=ruta_db) == 30
    mon.guardar_predicciones(puntuados, origen="api", ruta_db=ruta_db, reemplazar=True)
    tabla = mon.leer_tabla("predicciones", ruta_db)
    assert len(tabla) == 30
    assert list(tabla.columns) == mon.COLUMNAS_TABLA
    assert mon.leer_tabla("no_existe", ruta_db).empty
    assert mon.leer_tabla("predicciones", tmp_path / "otra.db").empty


def test_desempeno_con_y_sin_resultado_real():
    muestra = pd.DataFrame({"prediccion": [0, 1] * 50, "prob_no_pago": np.linspace(0, 1, 100),
                            "Pago_atiempo": [0] * 30 + [1] * 70})
    con_real = mon.desempeno(muestra)
    sin_real = mon.desempeno(muestra.assign(Pago_atiempo=np.nan))
    assert con_real["casos_no_pago"] == 30 and not np.isnan(con_real["roc_auc"])
    assert np.isnan(sin_real["roc_auc"]) and sin_real["tasa_no_pago_predicha"] == 0.5


def _metricas_sinteticas():
    periodos = [f"2025-{m:02d}" for m in range(1, 7)]
    filas = []
    for i, periodo in enumerate(periodos):
        filas.append({"periodo": periodo, "variable": "creciente", "psi": 0.02 + 0.05 * i,
                      "nivel": 2 if i >= 5 else 0, "muestra_suficiente": True})
        filas.append({"periodo": periodo, "variable": "abrupta", "psi": 0.6 if i == 3 else 0.02,
                      "nivel": 2 if i == 3 else 0, "muestra_suficiente": True})
        filas.append({"periodo": periodo, "variable": "prob_no_pago", "psi": 0.3 if i == 5 else 0.01,
                      "nivel": 0, "muestra_suficiente": True})
    return pd.DataFrame(filas)


def test_detectar_tendencias_y_cambios_abruptos():
    tendencias = mon.detectar_tendencias(_metricas_sinteticas()).set_index("variable")
    assert tendencias.loc["creciente", "tendencia"] == "Creciente"
    assert tendencias.loc["abrupta", "cambios_abruptos"] == "2025-04"


def test_generar_alertas_recomienda_reentrenar():
    metricas = _metricas_sinteticas()
    tendencias = mon.detectar_tendencias(metricas)
    resumen = pd.DataFrame({
        "periodo": ["2025-04", "2025-06"], "muestra_suficiente": [True, True],
        "porcentaje_variables_rojo": [0.5, 0.5], "psi_prediccion": [0.01, 0.3],
        "roc_auc": [0.70, 0.50],
    })
    alertas = mon.generar_alertas(resumen, metricas, tendencias, auc_referencia=0.70)
    criticas = alertas[alertas["nivel"] == "Crítica"]
    assert "2025-06" in criticas["periodo"].tolist()
    assert criticas["recomendacion"].str.contains("Reentrenar").any()
    assert (alertas["mensaje"].str.contains("ROC-AUC")).any()


def test_ejecutar_monitoreo_completo(tmp_path, monkeypatch, modelo_pequeno):
    monkeypatch.setattr(mon, "DIR_MONITOREO", tmp_path)
    monkeypatch.setattr(mon, "cargar_modelo", lambda *a, **k: modelo_pequeno)
    ruta_db = tmp_path / "monitoreo.db"
    resultado = mon.ejecutar_monitoreo("Q", tamano_muestra=300, incluir_simulacion=True, ruta_db=ruta_db)
    resumen = resultado["resumen"]
    assert resumen["origen"].str.contains("simulado").any()
    assert (resumen.loc[resumen["origen"] == "simulado", "estado_global"] == "Rojo").any()
    assert not resultado["alertas"].empty
    assert (tmp_path / "metricas_drift.csv").exists()
    assert not mon.leer_tabla("configuracion", ruta_db).empty
    # una segunda ejecución sin simulación elimina los periodos simulados
    resultado = mon.ejecutar_monitoreo("Q", 300, incluir_simulacion=False, ruta_db=ruta_db)
    assert not resultado["resumen"]["origen"].str.contains("simulado").any()


def test_cargar_modelo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        mon.cargar_modelo(tmp_path / "no_existe.joblib")
