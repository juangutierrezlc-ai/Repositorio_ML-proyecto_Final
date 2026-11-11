"""Pruebas de la componente de feature engineering."""

import numpy as np
import pandas as pd
import pytest

import ft_eng


def test_obtener_ruta_datos_encuentra_csv():
    assert ft_eng.obtener_ruta_datos().name == "Base_de_datos.csv"


def test_obtener_ruta_datos_falla_si_no_existe():
    with pytest.raises(FileNotFoundError):
        ft_eng.obtener_ruta_datos("no_existe.csv")


def test_cargar_datos_dimensiones(datos):
    assert datos.shape == (10763, 23)
    assert pd.api.types.is_datetime64_any_dtype(datos["fecha_prestamo"])


def _registro_base(**cambios):
    base = {
        "tipo_credito": 4, "fecha_prestamo": pd.Timestamp("2025-01-11 10:00"),  # sábado
        "capital_prestado": 1_000_000.0, "plazo_meses": 12, "edad_cliente": 30,
        "tipo_laboral": "Empleado", "salario_cliente": 2_000_000, "total_otros_prestamos": 500_000,
        "cuota_pactada": 100_000, "puntaje": 90.0, "puntaje_datacredito": 700.0,
        "cant_creditosvigentes": 2, "huella_consulta": 1, "saldo_mora": 0.0, "saldo_total": 0.0,
        "saldo_principal": 0.0, "saldo_mora_codeudor": 0.0, "creditos_sectorFinanciero": 1,
        "creditos_sectorCooperativo": 0, "creditos_sectorReal": 1,
        "promedio_ingresos_datacredito": 1_000_000.0, "tendencia_ingresos": "Creciente",
        "Pago_atiempo": 1,
    }
    base.update(cambios)
    return pd.DataFrame([base])


def test_limpieza_convierte_valores_imposibles_a_nan():
    df = _registro_base(edad_cliente=122, salario_cliente=0, puntaje_datacredito=999,
                        tendencia_ingresos="8315")
    salida = ft_eng.LimpiezaYAtributos().fit_transform(df)
    assert np.isnan(salida.loc[0, "edad_cliente"])
    assert salida.loc[0, "edad_desconocida"] == 1
    assert np.isnan(salida.loc[0, "salario_cliente"])
    assert np.isnan(salida.loc[0, "puntaje_datacredito"])
    assert pd.isna(salida.loc[0, "tendencia_ingresos"])


def test_limpieza_agrupa_tipos_de_credito_raros():
    salida = ft_eng.LimpiezaYAtributos().transform(_registro_base(tipo_credito=68))
    assert salida.loc[0, "tipo_credito"] == "Otros"


def test_limpieza_crea_atributos_y_elimina_variables():
    salida = ft_eng.LimpiezaYAtributos().transform(_registro_base(saldo_mora=5000.0))
    assert salida.loc[0, "tiene_mora"] == 1
    assert salida.loc[0, "fin_de_semana"] == 1
    assert salida.loc[0, "cuota_salario"] == pytest.approx(0.05)
    assert salida.loc[0, "total_creditos_sectores"] == 2
    assert salida.loc[0, "capital_prestado"] == pytest.approx(np.log1p(1_000_000))
    for eliminada in ft_eng.VARIABLES_ELIMINAR + [ft_eng.VARIABLE_OBJETIVO]:
        assert eliminada not in salida.columns
    assert list(salida.columns) == list(ft_eng.LimpiezaYAtributos().get_feature_names_out())


def test_winsorizador_limita_extremos_y_respeta_binarias():
    rng = np.random.default_rng(0)
    X = np.column_stack([
        np.r_[rng.normal(size=999), 1000.0],          # continua con un valor extremo
        np.r_[np.zeros(995), np.ones(5)],             # binaria poco frecuente
        np.r_[np.zeros(995), rng.uniform(1, 9, 5)],   # casi constante
    ])
    w = ft_eng.Winsorizador().fit(X)
    salida = w.transform(X)
    assert salida[:, 0].max() < 1000
    np.testing.assert_array_equal(salida[:, 1], X[:, 1])
    np.testing.assert_array_equal(salida[:, 2], X[:, 2])


def test_ft_engineering_particion_estratificada(particion):
    X_train, X_test, y_train, y_test, preprocesador = particion
    assert len(X_train) + len(X_test) == 10763
    assert len(X_test) == pytest.approx(0.2 * 10763, abs=1)
    assert ft_eng.VARIABLE_OBJETIVO not in X_train.columns
    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert not hasattr(preprocesador, "n_features_in_")  # se entrega sin ajustar


def test_preprocesador_sin_nulos_y_columnas_esperadas(particion):
    X_train_t, X_test_t, _, _, _ = ft_eng.obtener_datos_transformados()
    assert X_train_t.isna().sum().sum() == 0
    assert X_test_t.shape[1] == X_train_t.shape[1] == 35
    assert {"tiene_mora", "tipo_credito_Otros", "tendencia_ingresos"} <= set(X_train_t.columns)
