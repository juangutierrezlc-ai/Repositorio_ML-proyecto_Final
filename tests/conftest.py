"""Fixtures compartidas por las pruebas."""

import sys
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "mlops_pipeline"))

from ft_eng import cargar_datos, construir_preprocesador, ft_engineering  # noqa: E402


@pytest.fixture(scope="session")
def datos():
    """Dataset completo (se lee una sola vez)."""
    return cargar_datos()


@pytest.fixture(scope="session")
def particion():
    return ft_engineering()


@pytest.fixture(scope="session")
def modelo_pequeno(particion):
    """
    Modelo rápido con la misma estructura del modelo real (preprocesamiento + regresión
    logística + umbral ajustado). Así las pruebas no dependen de modelos/mejor_modelo.joblib.
    """
    X_train, _, y_train, _, _ = particion
    pipeline = Pipeline([
        ("preprocesamiento", construir_preprocesador()),
        ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    modelo = TunedThresholdClassifierCV(pipeline, scoring="balanced_accuracy", cv=3)
    return modelo.fit(X_train, y_train)
