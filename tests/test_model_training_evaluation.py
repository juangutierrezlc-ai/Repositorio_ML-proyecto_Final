"""Pruebas de las funciones reutilizables de entrenamiento y evaluación."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import model_training_evaluation as mte


def test_summarize_classification_valores_conocidos():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.6, 0.9, 0.8, 0.7, 0.4])
    metricas = mte.summarize_classification(y_true, y_pred, y_proba)
    assert metricas["recall_no_pago"] == pytest.approx(0.5)
    assert metricas["precision_no_pago"] == pytest.approx(0.5)
    assert metricas["accuracy"] == pytest.approx(4 / 6)
    assert metricas["roc_auc"] == pytest.approx(7 / 8)
    assert set(metricas) == {"roc_auc", "pr_auc_no_pago", "balanced_accuracy", "accuracy",
                             "recall_no_pago", "precision_no_pago", "f1_no_pago"}


def test_normalizar():
    serie = pd.Series([1.0, 2.0, 3.0])
    assert mte._normalizar(serie).tolist() == [0.0, 0.5, 1.0]
    assert mte._normalizar(serie, mayor_es_mejor=False).tolist() == [1.0, 0.5, 0.0]
    assert mte._normalizar(pd.Series([2.0, 2.0])).tolist() == [1.0, 1.0]


def _tabla_modelos():
    return pd.DataFrame({
        "test_roc_auc": [0.70, 0.72, 0.60], "test_balanced_accuracy": [0.65, 0.66, 0.55],
        "test_pr_auc_no_pago": [0.18, 0.20, 0.10], "cv_roc_auc_std": [0.01, 0.03, 0.02],
        "sobreajuste_auc": [0.02, 0.30, 0.10], "tiempo_entrenamiento_s": [1.0, 10.0, 2.0],
        "tiempo_prediccion_ms_x1000": [10.0, 50.0, 20.0], "tamano_modelo_mb": [0.1, 20.0, 0.5],
    }, index=["Logística", "Random Forest", "KNN"])


def test_seleccionar_mejor_modelo_pondera_los_tres_criterios():
    tabla = mte.seleccionar_mejor_modelo(_tabla_modelos())
    assert tabla.index[0] == "Logística"
    assert tabla["score_final"].between(0, 1).all()
    assert tabla["score_final"].is_monotonic_decreasing


def test_build_model_y_evaluaciones(particion):
    X_train, X_test, y_train, _, _ = particion
    muestra = X_train.sample(1500, random_state=0)
    y_muestra = y_train.loc[muestra.index]
    resultado = mte.build_model("Logística", LogisticRegression(class_weight="balanced", max_iter=500),
                                {"modelo__C": [0.1, 1]}, muestra, y_muestra)
    modelo = resultado["modelo"]
    assert resultado["mejores_parametros"]["C"] in (0.1, 1)
    assert 0 < modelo.best_threshold_ < 1

    consistencia = mte.evaluar_consistencia(modelo, muestra, y_muestra)
    assert len(consistencia["cv_roc_auc_folds"]) == mte.N_FOLDS
    escalabilidad = mte.evaluar_escalabilidad(modelo, X_test.head(200))
    assert escalabilidad["tamano_modelo_mb"] > 0


def test_graficos_se_guardan(tmp_path, particion, modelo_pequeno):
    _, X_test, _, y_test, _ = particion
    tabla = mte.seleccionar_mejor_modelo(_tabla_modelos())
    for col in ["test_recall_no_pago", "test_precision_no_pago", "test_f1_no_pago"]:
        tabla[col] = 0.5
    mte.graficar_comparacion_metricas(tabla, tmp_path / "metricas.png")
    mte.graficar_escalabilidad(tabla, tmp_path / "escalabilidad.png")
    mte.graficar_puntajes(tabla, tmp_path / "puntajes.png")
    resultados = {"A": {"modelo": modelo_pequeno, "cv_roc_auc_folds": np.array([0.6, 0.7, 0.65])},
                  "B": {"modelo": modelo_pequeno, "cv_roc_auc_folds": np.array([0.62, 0.66, 0.64])}}
    mte.graficar_curvas(resultados, X_test, y_test, tmp_path / "curvas.png")
    mte.graficar_matrices_confusion(resultados, X_test, y_test, tmp_path / "matrices.png")
    mte.graficar_consistencia(resultados, tmp_path / "consistencia.png")
    assert len(list(tmp_path.glob("*.png"))) == 6
