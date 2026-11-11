"""
model_training_evaluation.py
============================
Segunda componente del flujo: entrenar y evaluar varios modelos supervisados de
clasificacion y escoger el mejor segun tres criterios: consistency, performance y scalibility.


 - Performance  : que tan bien predice en datos que no vio (conjunto de evaluacion).
    - Consistency  : que tan estable es entre particiones de validación cruzada y
                     cuanto sobreajusta (diferencia entre entrenamiento y validacion).
    - Scalability  : cuanto tarda en entrenar y en predecir, y cuánto pesa el modelo.

Modelos evaluados:
    1. Regresion logistica   (equivalente de la regresion lineal para un objetivo 0/1)
    2. Arbol de decision
    3. Random Forest
    4. KNN
    5. Red neuronal (MLP)
"""

import io
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # guarda los gráficos en archivos sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV, cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ft_eng import RANDOM_STATE, construir_preprocesador, ft_engineering

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# configuracion
# -----------------------

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"

N_FOLDS = 5
CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Clase de interes: 0 = no pago a tiempo (el riesgo que queremos detectar)
CLASE_RIESGO = 0

# Pesos de cada criterio en el puntaje final
PESOS = {"performance": 0.5, "consistency": 0.3, "scalability": 0.2}

# Modelos y sus de hiperparametros
MODELOS = {
    "Regresión logística": (
        LogisticRegression(class_weight="balanced", max_iter=2000),
        {"modelo__C": [0.01, 0.1, 1, 10]},
    ),
    "Árbol de decisión": (
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"modelo__max_depth": [3, 5, 8], "modelo__min_samples_leaf": [20, 50, 100]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=RANDOM_STATE),
        {"modelo__max_depth": [6, 10, None], "modelo__min_samples_leaf": [5, 20]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"modelo__n_neighbors": [25, 51, 101], "modelo__weights": ["uniform", "distance"]},
    ),
    "Red neuronal (MLP)": (
        MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
        {"modelo__hidden_layer_sizes": [(16,), (32, 16)], "modelo__alpha": [1e-3, 1e-1]},
    ),
}

# ----------------------
# funciones reutilizables
# ----------------------


def summarize_classification(y_true, y_pred, y_proba_pago) -> dict:
    """
    Resume el desempeño de un clasificador.

    y_proba_pago es la probabilidad de la clase 1 (pago a tiempo). Las metricas de
    precision / recall / F1 se calculan para la clase 0 (no pago), que es la de riesgo.
    """
    y_true = np.asarray(y_true)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba_pago),
        "pr_auc_no_pago": average_precision_score(y_true == CLASE_RIESGO, 1 - y_proba_pago),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_no_pago": recall_score(y_true, y_pred, pos_label=CLASE_RIESGO, zero_division=0),
        "precision_no_pago": precision_score(y_true, y_pred, pos_label=CLASE_RIESGO, zero_division=0),
        "f1_no_pago": f1_score(y_true, y_pred, pos_label=CLASE_RIESGO, zero_division=0),
    }


def build_model(nombre: str, estimador, grilla: dict, X_train, y_train) -> dict:
    """
    Construye y entrena un modelo completo:

    1. Pipeline = preprocesamiento de ft_eng + estimador.
    2. GridSearchCV elige los hiperparámetros con ROC-AUC (no depende del umbral).
    3. TunedThresholdClassifierCV ajusta el umbral de decisión para maximizar la
       balanced accuracy. Con clases 95/5 el umbral de 0,5 casi nunca predice
       "no paga" (sobre todo en KNN y MLP, que no tienen class_weight).

    Devuelve el modelo final y los datos de tiempo y busqueda.
    """
    pipeline = Pipeline([("preprocesamiento", construir_preprocesador()), ("modelo", estimador)])

    busqueda = GridSearchCV(pipeline, grilla, scoring="roc_auc", cv=CV, n_jobs=-1)
    busqueda.fit(X_train, y_train)

    modelo = TunedThresholdClassifierCV(
        busqueda.best_estimator_, scoring="balanced_accuracy", cv=CV, n_jobs=-1
    )
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    return {
        "nombre": nombre,
        "modelo": modelo,
        "mejores_parametros": {k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()},
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
    }


def evaluar_consistencia(modelo, X_train, y_train) -> dict:
    """Validación cruzada del modelo final: media, desviación y sobreajuste."""
    cv = cross_validate(clone(modelo), X_train, y_train, cv=CV,
                        scoring=["roc_auc", "balanced_accuracy"],
                        return_train_score=True, n_jobs=-1)
    return {
        "cv_roc_auc_media": cv["test_roc_auc"].mean(),
        "cv_roc_auc_std": cv["test_roc_auc"].std(),
        "cv_bal_acc_media": cv["test_balanced_accuracy"].mean(),
        "cv_bal_acc_std": cv["test_balanced_accuracy"].std(),
        "sobreajuste_auc": cv["train_roc_auc"].mean() - cv["test_roc_auc"].mean(),
        "cv_roc_auc_folds": cv["test_roc_auc"],
    }


def evaluar_escalabilidad(modelo, X_test) -> dict:
    """Tiempo de predicción por cada 1.000 registros y tamaño del modelo serializado."""
    repeticiones = 5
    inicio = time.perf_counter()
    for _ in range(repeticiones):
        modelo.predict_proba(X_test)
    tiempo = (time.perf_counter() - inicio) / repeticiones
    buffer = io.BytesIO()
    joblib.dump(modelo, buffer)
    return {
        "tiempo_prediccion_ms_x1000": tiempo / len(X_test) * 1000 * 1000,
        "tamano_modelo_mb": buffer.getbuffer().nbytes / 1024 ** 2,
    }


def _normalizar(serie: pd.Series, mayor_es_mejor: bool = True) -> pd.Series:
    """Escala a 0–1 entre los modelos (1 = el mejor)."""
    rango = serie.max() - serie.min()
    if rango == 0:
        return pd.Series(1.0, index=serie.index)
    norm = (serie - serie.min()) / rango
    return norm if mayor_es_mejor else 1 - norm


def seleccionar_mejor_modelo(tabla: pd.DataFrame) -> pd.DataFrame:
    """Calcula un puntaje 0–1 por criterio y un puntaje final ponderado."""
    tabla = tabla.copy()
    tabla["score_performance"] = (
        _normalizar(tabla["test_roc_auc"]) + _normalizar(tabla["test_balanced_accuracy"])
        + _normalizar(tabla["test_pr_auc_no_pago"])
    ) / 3
    tabla["score_consistency"] = (
        _normalizar(tabla["cv_roc_auc_std"], mayor_es_mejor=False)
        + _normalizar(tabla["sobreajuste_auc"], mayor_es_mejor=False)
    ) / 2
    tabla["score_scalability"] = (
        _normalizar(np.log10(tabla["tiempo_entrenamiento_s"]), mayor_es_mejor=False)
        + _normalizar(np.log10(tabla["tiempo_prediccion_ms_x1000"]), mayor_es_mejor=False)
        + _normalizar(np.log10(tabla["tamano_modelo_mb"]), mayor_es_mejor=False)
    ) / 3
    tabla["score_final"] = sum(PESOS[c] * tabla[f"score_{c}"] for c in PESOS)
    return tabla.sort_values("score_final", ascending=False)


# ----------------------
# graficos comparativos
# ----------------------

def graficar_comparacion_metricas(tabla: pd.DataFrame, ruta: Path):
    metricas = ["test_roc_auc", "test_pr_auc_no_pago", "test_balanced_accuracy",
                "test_recall_no_pago", "test_precision_no_pago", "test_f1_no_pago"]
    datos = tabla[metricas].rename(columns=lambda c: c.replace("test_", ""))
    ax = datos.T.plot(kind="bar", figsize=(13, 5.5), width=0.8, colormap="tab10")
    ax.set_title("Métricas en el conjunto de evaluación por modelo")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Modelo", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def graficar_curvas(resultados: dict, X_test, y_test, ruta: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for nombre, r in resultados.items():
        proba_no_pago = r["modelo"].predict_proba(X_test)[:, 0]
        RocCurveDisplay.from_predictions(y_test == CLASE_RIESGO, proba_no_pago, name=nombre, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test == CLASE_RIESGO, proba_no_pago, name=nombre, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
    axes[0].set_title("Curva ROC (clase de riesgo: no pago)")
    tasa_base = (y_test == CLASE_RIESGO).mean()
    axes[1].axhline(tasa_base, color="black", linestyle="--", linewidth=1, label=f"Azar ({tasa_base:.1%})")
    axes[1].set_title("Curva Precision-Recall (clase: no pago)")
    axes[0].set_xlabel("Tasa de falsos positivos")
    axes[0].set_ylabel("Tasa de verdaderos positivos (recall no pago)")
    axes[1].set_xlabel("Recall (no pago)")
    axes[1].set_ylabel("Precision (no pago)")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def graficar_matrices_confusion(resultados: dict, X_test, y_test, ruta: Path):
    fig, axes = plt.subplots(1, len(resultados), figsize=(4.2 * len(resultados), 4))
    for ax, (nombre, r) in zip(axes, resultados.items()):
        cm = confusion_matrix(y_test, r["modelo"].predict(X_test), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["No paga", "Paga"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(nombre, fontsize=10)
    plt.suptitle("Matrices de confusión en el conjunto de evaluación")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def graficar_consistencia(resultados: dict, ruta: Path):
    datos = pd.DataFrame({n: r["cv_roc_auc_folds"] for n, r in resultados.items()})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(datos.values, tick_labels=datos.columns)
    for i, col in enumerate(datos.columns, start=1):
        ax.scatter(np.full(len(datos), i), datos[col], color="firebrick", alpha=0.7, zorder=3)
    ax.set_title(f"Consistencia: ROC-AUC en las {N_FOLDS} particiones de validación cruzada")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def graficar_escalabilidad(tabla: pd.DataFrame, ruta: Path):
    columnas = [("tiempo_entrenamiento_s", "Tiempo de entrenamiento (s)"),
                ("tiempo_prediccion_ms_x1000", "Predicción (ms por 1.000 registros)"),
                ("tamano_modelo_mb", "Tamaño del modelo (MB)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (col, titulo) in zip(axes, columnas):
        datos = tabla[col].sort_values()
        ax.barh(datos.index, datos.values, color="steelblue")
        ax.set_xscale("log")
        ax.set_title(titulo)
    plt.suptitle("Escalabilidad (escala logarítmica, menor es mejor)")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()


def graficar_puntajes(tabla: pd.DataFrame, ruta: Path):
    columnas = ["score_performance", "score_consistency", "score_scalability", "score_final"]
    ax = tabla[columnas].plot(kind="barh", figsize=(11, 5), width=0.8)
    ax.set_title("Puntaje por criterio (0 = peor, 1 = mejor entre los modelos)")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.legend(["Performance", "Consistency", "Scalability", "Final ponderado"],
              bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(ruta, dpi=120)
    plt.close()

# ----------------------
# flujo principal
# ----------------------


def entrenar_y_evaluar():
    DIR_MODELOS.mkdir(exist_ok=True)
    DIR_RESULTADOS.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, _ = ft_engineering()
    print(f"Entrenamiento: {X_train.shape} | Evaluación: {X_test.shape}\n")

    resultados, filas = {}, []
    for nombre, (estimador, grilla) in MODELOS.items():
        print(f"Entrenando {nombre}...")
        r = build_model(nombre, estimador, grilla, X_train, y_train)
        modelo = r["modelo"]

        # Performance en entrenamiento y evaluación
        m_train = summarize_classification(y_train, modelo.predict(X_train), modelo.predict_proba(X_train)[:, 1])
        m_test = summarize_classification(y_test, modelo.predict(X_test), modelo.predict_proba(X_test)[:, 1])
        r.update(evaluar_consistencia(modelo, X_train, y_train))
        r.update(evaluar_escalabilidad(modelo, X_test))
        resultados[nombre] = r

        filas.append({
            "modelo": nombre,
            **{f"test_{k}": v for k, v in m_test.items()},
            "train_roc_auc": m_train["roc_auc"],
            **{k: v for k, v in r.items() if k.startswith(("cv_", "sobreajuste", "tiempo_", "tamano"))
               and k != "cv_roc_auc_folds"},
            "umbral": float(modelo.best_threshold_),
            "mejores_parametros": str(r["mejores_parametros"]),
        })
        print(f"  ROC-AUC test = {m_test['roc_auc']:.3f} | recall no pago = {m_test['recall_no_pago']:.3f}")

    tabla = seleccionar_mejor_modelo(pd.DataFrame(filas).set_index("modelo"))
    mejor_nombre = tabla.index[0]
    mejor_modelo = resultados[mejor_nombre]["modelo"]

    # tabla resumen de resultados
    tabla.round(4).to_csv(DIR_RESULTADOS / "tabla_resumen.csv", encoding="utf-8-sig")
    columnas_resumen = ["test_roc_auc", "test_pr_auc_no_pago", "test_balanced_accuracy",
                        "test_recall_no_pago", "test_precision_no_pago", "cv_roc_auc_media",
                        "cv_roc_auc_std", "sobreajuste_auc", "tiempo_entrenamiento_s",
                        "tiempo_prediccion_ms_x1000", "score_performance", "score_consistency",
                        "score_scalability", "score_final"]

    # graficos comparativos
    graficar_comparacion_metricas(tabla, DIR_RESULTADOS / "comparacion_metricas.png")
    graficar_curvas(resultados, X_test, y_test, DIR_RESULTADOS / "curvas_roc_pr.png")
    graficar_matrices_confusion(resultados, X_test, y_test, DIR_RESULTADOS / "matrices_confusion.png")
    graficar_consistencia(resultados, DIR_RESULTADOS / "consistencia_cv.png")
    graficar_escalabilidad(tabla, DIR_RESULTADOS / "escalabilidad.png")
    graficar_puntajes(tabla, DIR_RESULTADOS / "puntajes_seleccion.png")

    # guardar el mejor modelo y su metadata (los usan model_deploy.py y model_monitoring.py)
    joblib.dump(mejor_modelo, DIR_MODELOS / "mejor_modelo.joblib")
    metadata = {
        "modelo": mejor_nombre,
        "mejores_parametros": resultados[mejor_nombre]["mejores_parametros"],
        "umbral_decision": float(mejor_modelo.best_threshold_),
        "metricas_test": {k.replace("test_", ""): round(float(v), 4)
                          for k, v in tabla.loc[mejor_nombre].items() if k.startswith("test_")},
        "pesos_seleccion": PESOS,
        "fecha_entrenamiento": pd.Timestamp.now().isoformat(timespec="seconds"),
        "registros_entrenamiento": len(X_train),
    }
    (DIR_MODELOS / "metadata_mejor_modelo.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Tabla resumen de evaluacion ===")
    print(tabla[columnas_resumen].round(3).to_string())
    print(f"\nMejor modelo: {mejor_nombre} (score final = {tabla.loc[mejor_nombre, 'score_final']:.3f})")
    print(f"Guardado en: {DIR_MODELOS / 'mejor_modelo.joblib'}")
    print(f"Graficos y tabla en: {DIR_RESULTADOS}")
    return mejor_modelo, tabla


if __name__ == "__main__":
    entrenar_y_evaluar()
