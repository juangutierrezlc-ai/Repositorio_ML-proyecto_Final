"""
model_monitoring.py
===================
Componente de monitoreo del modelo en producción y medición del data drift.

Flujo:
    1. Tabla de monitoreo: los datos de cada crédito junto con el pronóstico entregado
       por el modelo se guardan en la tabla `predicciones` (SQLite, monitoreo/monitoreo.db).
       La API (model_deploy.py) agrega allí cada predicción que hace.
    2. Muestreo periódico: con la periodicidad definida (mensual por defecto) se toma una
       muestra de los registros de cada periodo.
    3. Métricas de data drift por variable, comparando cada periodo con la población de
       entrenamiento (referencia histórica):
         - Kolmogorov-Smirnov (KS)       -> variables numéricas
         - Population Stability Index    -> numéricas y categóricas
         - Divergencia de Jensen-Shannon -> numéricas y categóricas
         - Chi-cuadrado                  -> variables categóricas
       También se mide el drift de la predicción (probabilidad de no pago) y, cuando ya se
       conoce el resultado real, el desempeño del modelo en el periodo.
    4. Análisis temporal: tendencia del drift (regresión lineal sobre el PSI) y cambios
       abruptos entre periodos consecutivos.
    5. Alertas y recomendaciones (revisar variables o reentrenar el modelo).

Los resultados quedan en las tablas `metricas_drift`, `resumen_periodos`, `tendencias` y
`alertas` de la misma base, y en CSV dentro de monitoreo/. La app de Streamlit
(app_monitoreo.py) los lee para mostrarlos.

Uso:
    python mlops_pipeline/model_monitoring.py                    # mensual, datos históricos
    python mlops_pipeline/model_monitoring.py --periodicidad Q   # trimestral
    python mlops_pipeline/model_monitoring.py --simular-drift    # agrega 3 periodos con drift simulado
"""

import argparse
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from ft_eng import RANDOM_STATE, VARIABLE_OBJETIVO, cargar_datos, ft_engineering

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_MONITOREO = RAIZ / "monitoreo"
RUTA_DB = DIR_MONITOREO / "monitoreo.db"
RUTA_MODELO = RAIZ / "modelos" / "mejor_modelo.joblib"
RUTA_METADATA = RAIZ / "modelos" / "metadata_mejor_modelo.json"

# Variables de entrada que usa el modelo (en su forma original, más fácil de interpretar)
VARIABLES_NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
VARIABLES_CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
VARIABLE_PREDICCION = "prob_no_pago"

COLUMNAS_TABLA = (
    ["fecha_registro", "periodo", "origen", "fecha_prestamo"]
    + VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS
    + [VARIABLE_PREDICCION, "prediccion", VARIABLE_OBJETIVO]
)

# Umbrales del semáforo: (amarillo, rojo)
UMBRALES = {
    "psi": (0.10, 0.25),        # regla estándar de la industria
    "ks": (0.10, 0.20),         # estadístico KS (además se exige p-valor < 0,05)
    "js": (0.02, 0.05),         # divergencia JS en base 2 (0 = iguales, 1 = totalmente distintas)
    "cramer_v": (0.05, 0.10),   # tamaño del efecto del chi-cuadrado (además p-valor < 0,05)
}
ALFA = 0.05
N_BINS = 10
TAMANO_MUESTRA = 500           # registros muestreados por periodo
MUESTRA_MINIMA = 100           # por debajo de esto el periodo no es confiable
PORCENTAJE_ROJO_REENTRENAR = 0.30
CAIDA_AUC_REENTRENAR = 0.10      # caída de ROC-AUC frente a la de evaluación
MINIMO_NO_PAGO_AUC = 20          # casos de no pago necesarios para que la AUC sea confiable
SALTO_ABRUPTO_PSI = 0.10

ESTADOS = {0: "Verde", 1: "Amarillo", 2: "Rojo"}
EPS = 1e-4


# ---------------------------------------------------------------------------
# Métricas de drift
# ---------------------------------------------------------------------------
def _bordes_cuantiles(referencia: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bordes de los bins según los cuantiles de la referencia (los extremos quedan abiertos)."""
    bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_bins + 1)))
    bordes[0], bordes[-1] = -np.inf, np.inf
    return bordes


def distribuciones_numericas(referencia, actual, n_bins: int = N_BINS):
    """Proporción de registros por bin en la referencia y en el periodo actual."""
    ref = pd.Series(referencia, dtype=float).dropna().to_numpy()
    act = pd.Series(actual, dtype=float).dropna().to_numpy()
    bordes = _bordes_cuantiles(ref, n_bins)
    if len(bordes) < 3:  # variable casi constante: se compara "valor más frecuente" vs resto
        moda = pd.Series(ref).mode().iloc[0]
        p_ref = np.array([(ref == moda).mean(), (ref != moda).mean()])
        p_act = np.array([(act == moda).mean(), (act != moda).mean()])
        return p_ref, p_act
    p_ref = np.histogram(ref, bordes)[0] / len(ref)
    p_act = np.histogram(act, bordes)[0] / max(len(act), 1)
    return p_ref, p_act


def distribuciones_categoricas(referencia, actual):
    """Proporción por categoría (unión de categorías de ambos conjuntos)."""
    ref = pd.Series(referencia).astype(object).fillna("Sin dato").astype(str)
    act = pd.Series(actual).astype(object).fillna("Sin dato").astype(str)
    categorias = sorted(set(ref) | set(act))
    p_ref = ref.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    p_act = act.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    return p_ref, p_act, categorias


def psi(p_ref, p_act) -> float:
    """Population Stability Index: sum((act - ref) * ln(act / ref))."""
    p_ref = np.clip(np.asarray(p_ref, dtype=float), EPS, None)
    p_act = np.clip(np.asarray(p_act, dtype=float), EPS, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def jensen_shannon(p_ref, p_act) -> float:
    """Divergencia de Jensen-Shannon en base 2 (entre 0 y 1)."""
    distancia = jensenshannon(np.asarray(p_ref, dtype=float), np.asarray(p_act, dtype=float), base=2)
    return float(distancia ** 2)


def kolmogorov_smirnov(referencia, actual):
    """Estadístico y p-valor de la prueba KS de dos muestras."""
    ref = pd.Series(referencia, dtype=float).dropna()
    act = pd.Series(actual, dtype=float).dropna()
    resultado = stats.ks_2samp(ref, act)
    return float(resultado.statistic), float(resultado.pvalue)


def chi_cuadrado(referencia, actual):
    """Prueba chi-cuadrado de homogeneidad. Devuelve estadístico, p-valor y V de Cramér."""
    ref = pd.Series(referencia).astype(object).fillna("Sin dato").astype(str)
    act = pd.Series(actual).astype(object).fillna("Sin dato").astype(str)
    tabla = pd.crosstab(
        pd.concat([ref, act], ignore_index=True),
        np.r_[np.zeros(len(ref)), np.ones(len(act))],
    )
    tabla = tabla[(tabla.sum(axis=1) > 0)]
    if tabla.shape[0] < 2:
        return 0.0, 1.0, 0.0
    chi2, p_valor, _, _ = stats.chi2_contingency(tabla)
    cramer_v = np.sqrt(chi2 / (tabla.to_numpy().sum() * (min(tabla.shape) - 1)))
    return float(chi2), float(p_valor), float(cramer_v)


def _nivel(valor: float, metrica: str) -> int:
    amarillo, rojo = UMBRALES[metrica]
    return 2 if valor >= rojo else 1 if valor >= amarillo else 0


def drift_variable(referencia, actual, tipo: str) -> dict:
    """Calcula todas las métricas de drift de una variable y su estado (semáforo)."""
    if tipo == "numerica":
        p_ref, p_act = distribuciones_numericas(referencia, actual)
        ks_stat, ks_p = kolmogorov_smirnov(referencia, actual)
        valor_psi, valor_js = psi(p_ref, p_act), jensen_shannon(p_ref, p_act)
        nivel_ks = _nivel(ks_stat, "ks") if ks_p < ALFA else 0
        nivel = max(_nivel(valor_psi, "psi"), _nivel(valor_js, "js"), nivel_ks)
        return {"tipo": tipo, "psi": valor_psi, "js": valor_js, "ks_estadistico": ks_stat,
                "ks_p_valor": ks_p, "chi2_p_valor": np.nan, "cramer_v": np.nan,
                "nivel": nivel, "estado": ESTADOS[nivel]}

    p_ref, p_act, _ = distribuciones_categoricas(referencia, actual)
    _, chi_p, cramer_v = chi_cuadrado(referencia, actual)
    valor_psi, valor_js = psi(p_ref, p_act), jensen_shannon(p_ref, p_act)
    nivel_chi = _nivel(cramer_v, "cramer_v") if chi_p < ALFA else 0
    nivel = max(_nivel(valor_psi, "psi"), _nivel(valor_js, "js"), nivel_chi)
    return {"tipo": tipo, "psi": valor_psi, "js": valor_js, "ks_estadistico": np.nan,
            "ks_p_valor": np.nan, "chi2_p_valor": chi_p, "cramer_v": cramer_v,
            "nivel": nivel, "estado": ESTADOS[nivel]}


def calcular_drift(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Tabla con las métricas de drift de todas las variables (y de la predicción)."""
    filas = []
    for variable in VARIABLES_NUMERICAS + [VARIABLE_PREDICCION]:
        if variable in referencia and variable in actual and actual[variable].notna().any():
            filas.append({"variable": variable,
                          **drift_variable(referencia[variable], actual[variable], "numerica")})
    for variable in VARIABLES_CATEGORICAS:
        if variable in referencia and variable in actual:
            filas.append({"variable": variable,
                          **drift_variable(referencia[variable], actual[variable], "categorica")})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Tabla de monitoreo (datos + pronósticos)
# ---------------------------------------------------------------------------
def cargar_modelo(ruta: Path = RUTA_MODELO):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero mlops_pipeline/model_training_evaluation.py"
        )
    return joblib.load(ruta)


def puntuar(df: pd.DataFrame, modelo) -> pd.DataFrame:
    """Agrega al DataFrame la probabilidad de no pago y la predicción del modelo."""
    salida = df.copy()
    salida[VARIABLE_PREDICCION] = modelo.predict_proba(df)[:, 0]
    salida["prediccion"] = modelo.predict(df).astype(int)
    return salida


def _normalizar_tabla(df: pd.DataFrame, origen: str, periodicidad: str = "M") -> pd.DataFrame:
    """Deja el DataFrame con las columnas exactas de la tabla `predicciones`."""
    salida = df.copy()
    if "fecha_registro" not in salida:
        salida["fecha_registro"] = pd.Timestamp.now().isoformat(timespec="seconds")
    fecha = pd.to_datetime(salida.get("fecha_prestamo"), errors="coerce")
    if fecha is None or fecha.isna().all():
        fecha = pd.to_datetime(salida["fecha_registro"], errors="coerce")
    salida["periodo"] = fecha.dt.to_period(periodicidad).astype(str)
    salida["origen"] = origen
    salida["fecha_prestamo"] = fecha.dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in VARIABLES_CATEGORICAS:
        salida[col] = salida[col].astype(object).where(salida[col].notna(), None)
        salida[col] = salida[col].map(lambda v: None if v is None else str(v))
    return salida.reindex(columns=COLUMNAS_TABLA)


def guardar_predicciones(df: pd.DataFrame, origen: str = "api", ruta_db: Path | None = None,
                         reemplazar: bool = False) -> int:
    """Agrega registros (datos + pronóstico) a la tabla `predicciones`."""
    ruta_db = ruta_db or RUTA_DB
    ruta_db.parent.mkdir(parents=True, exist_ok=True)
    tabla = _normalizar_tabla(df, origen)
    with conectar(ruta_db) as conexion:
        if reemplazar and _existe_tabla(conexion, "predicciones"):
            conexion.execute("DELETE FROM predicciones WHERE origen = ?", (origen,))
        tabla.to_sql("predicciones", conexion, if_exists="append", index=False)
    return len(tabla)


@contextmanager
def conectar(ruta_db: Path):
    """Abre la base SQLite, confirma los cambios y siempre cierra la conexión."""
    conexion = sqlite3.connect(ruta_db)
    try:
        yield conexion
        conexion.commit()
    finally:
        conexion.close()


def _existe_tabla(conexion, nombre: str) -> bool:
    consulta = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conexion.execute(consulta, (nombre,)).fetchone() is not None


def leer_tabla(nombre: str, ruta_db: Path | None = None) -> pd.DataFrame:
    ruta_db = ruta_db or RUTA_DB
    if not ruta_db.exists():
        return pd.DataFrame()
    with conectar(ruta_db) as conexion:
        if not _existe_tabla(conexion, nombre):
            return pd.DataFrame()
        return pd.read_sql(f"SELECT * FROM {nombre}", conexion)  # nombre interno, no del usuario


def construir_referencia(modelo) -> pd.DataFrame:
    """Población de referencia: el mismo conjunto de entrenamiento del modelo, con su pronóstico."""
    X_train, _, y_train, _, _ = ft_engineering()
    referencia = puntuar(X_train, modelo)
    referencia[VARIABLE_OBJETIVO] = y_train
    return referencia


def registrar_historico(modelo, ruta_db: Path | None = None) -> int:
    """
    Simula la operación del modelo en el tiempo: puntúa todos los créditos históricos y los
    guarda en la tabla de monitoreo con su resultado real (Pago_atiempo) conocido.
    """
    datos = cargar_datos()
    puntuados = puntuar(datos.drop(columns=[VARIABLE_OBJETIVO]), modelo)
    puntuados[VARIABLE_OBJETIVO] = datos[VARIABLE_OBJETIVO]
    return guardar_predicciones(puntuados, origen="historico", ruta_db=ruta_db, reemplazar=True)


def simular_drift(modelo, n_periodos: int = 3, tamano: int = TAMANO_MUESTRA,
                  ruta_db: Path | None = None) -> int:
    """
    Genera periodos futuros con un cambio progresivo en la población (para demostrar las
    alertas): clientes con menos salario, más consultas al buró, menor puntaje, créditos
    más grandes y más ingresos decrecientes. El resultado real aún no se conoce.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    datos = cargar_datos().drop(columns=[VARIABLE_OBJETIVO])
    ultima_fecha = datos["fecha_prestamo"].max()
    simulados = []
    for k in range(1, n_periodos + 1):
        lote = datos.sample(tamano, random_state=RANDOM_STATE + k, replace=True).copy()
        lote["fecha_prestamo"] = (ultima_fecha + pd.DateOffset(months=k)).normalize()
        lote["salario_cliente"] = lote["salario_cliente"] * (1 - 0.12 * k)
        lote["capital_prestado"] = lote["capital_prestado"] * (1 + 0.15 * k)
        lote["huella_consulta"] = lote["huella_consulta"] + rng.poisson(k, len(lote))
        con_historial = lote["puntaje_datacredito"] > 0
        lote.loc[con_historial, "puntaje_datacredito"] -= 25 * k
        cambia = rng.random(len(lote)) < 0.15 * k
        lote.loc[cambia, "tendencia_ingresos"] = "Decreciente"
        simulados.append(lote)
    puntuados = puntuar(pd.concat(simulados, ignore_index=True), modelo)
    return guardar_predicciones(puntuados, origen="simulado", ruta_db=ruta_db, reemplazar=True)


# ---------------------------------------------------------------------------
# Muestreo periódico y desempeño
# ---------------------------------------------------------------------------
def muestreo_periodico(datos: pd.DataFrame, periodicidad: str = "M",
                       tamano: int = TAMANO_MUESTRA) -> dict:
    """Agrupa los registros por periodo y toma una muestra aleatoria de cada uno."""
    fechas = pd.to_datetime(datos["fecha_prestamo"], errors="coerce")
    periodos = fechas.dt.to_period(periodicidad).astype(str)
    muestras = {}
    for periodo, grupo in datos.groupby(periodos, sort=True):
        muestras[periodo] = grupo.sample(min(tamano, len(grupo)), random_state=RANDOM_STATE)
    return muestras


def desempeno(muestra: pd.DataFrame) -> dict:
    """
    Métricas del modelo en el periodo (solo si ya se conoce el resultado real). La ROC-AUC
    solo se calcula con al menos MINIMO_NO_PAGO_AUC casos de no pago; con menos es muy inestable.
    """
    from sklearn.metrics import recall_score, roc_auc_score

    resultado = {"tasa_no_pago_predicha": float((muestra["prediccion"] == 0).mean()),
                 "prob_no_pago_media": float(muestra[VARIABLE_PREDICCION].mean()),
                 "tasa_no_pago_real": np.nan, "casos_no_pago": 0, "roc_auc": np.nan,
                 "recall_no_pago": np.nan}
    reales = pd.to_numeric(muestra[VARIABLE_OBJETIVO], errors="coerce")
    con_real = muestra[reales.notna()]
    if len(con_real) == 0:
        return resultado
    y = reales.dropna().astype(int)
    resultado["tasa_no_pago_real"] = float((y == 0).mean())
    resultado["casos_no_pago"] = int((y == 0).sum())
    if (y == 0).sum() >= MINIMO_NO_PAGO_AUC and y.nunique() == 2:
        resultado["roc_auc"] = float(roc_auc_score(y, 1 - con_real[VARIABLE_PREDICCION]))
        resultado["recall_no_pago"] = float(recall_score(y, con_real["prediccion"], pos_label=0))
    return resultado


# ---------------------------------------------------------------------------
# Análisis temporal
# ---------------------------------------------------------------------------
def detectar_tendencias(metricas: pd.DataFrame) -> pd.DataFrame:
    """
    Por variable: pendiente del PSI a lo largo de los periodos (tendencia) y cambios
    abruptos (salto de PSI mayor a SALTO_ABRUPTO_PSI entre dos periodos seguidos).
    """
    filas = []
    confiables = metricas[metricas["muestra_suficiente"]]
    for variable, grupo in confiables.groupby("variable"):
        grupo = grupo.sort_values("periodo")
        serie = grupo["psi"].to_numpy()
        pendiente, p_valor = 0.0, 1.0
        if len(serie) >= 3 and np.ptp(serie) > 0:
            regresion = stats.linregress(np.arange(len(serie)), serie)
            pendiente, p_valor = float(regresion.slope), float(regresion.pvalue)
        saltos = np.diff(serie)
        abruptos = grupo["periodo"].to_numpy()[1:][saltos > SALTO_ABRUPTO_PSI]
        if p_valor < ALFA and pendiente > 0:
            tendencia = "Creciente"
        elif p_valor < ALFA and pendiente < 0:
            tendencia = "Decreciente"
        else:
            tendencia = "Estable"
        filas.append({"variable": variable, "pendiente_psi": pendiente, "p_valor_tendencia": p_valor,
                      "tendencia": tendencia, "cambios_abruptos": ", ".join(abruptos),
                      "psi_ultimo_periodo": float(serie[-1]) if len(serie) else np.nan})
    return pd.DataFrame(filas)


def generar_alertas(resumen: pd.DataFrame, metricas: pd.DataFrame, tendencias: pd.DataFrame,
                    auc_referencia: float | None) -> pd.DataFrame:
    """Mensajes automáticos cuando se supera un umbral crítico, con su recomendación."""
    alertas = []
    for _, fila in resumen[resumen["muestra_suficiente"]].iterrows():
        periodo = fila["periodo"]
        del_periodo = metricas[(metricas["periodo"] == periodo) & (metricas["variable"] != VARIABLE_PREDICCION)]
        rojas = del_periodo.loc[del_periodo["nivel"] == 2, "variable"].tolist()
        amarillas = del_periodo.loc[del_periodo["nivel"] == 1, "variable"].tolist()

        if fila["porcentaje_variables_rojo"] >= PORCENTAJE_ROJO_REENTRENAR:
            alertas.append({"periodo": periodo, "nivel": "Crítica",
                            "mensaje": f"{fila['porcentaje_variables_rojo']:.0%} de las variables con drift alto: "
                                       + ", ".join(rojas),
                            "recomendacion": "Reentrenar el modelo con datos recientes y validar de nuevo."})
        elif rojas:
            alertas.append({"periodo": periodo, "nivel": "Alta",
                            "mensaje": "Drift alto en: " + ", ".join(rojas),
                            "recomendacion": "Revisar el origen de estas variables (cambios en captura, "
                                             "en la política comercial o en el perfil de clientes)."})
        elif amarillas:
            alertas.append({"periodo": periodo, "nivel": "Media",
                            "mensaje": "Drift moderado en: " + ", ".join(amarillas),
                            "recomendacion": "Seguir de cerca estas variables en los próximos periodos."})

        if fila["psi_prediccion"] >= UMBRALES["psi"][1]:
            alertas.append({"periodo": periodo, "nivel": "Crítica",
                            "mensaje": f"La distribución de la probabilidad de no pago cambió (PSI = "
                                       f"{fila['psi_prediccion']:.2f}).",
                            "recomendacion": "Reentrenar o recalibrar el modelo; revisar el umbral de decisión."})

        if auc_referencia and pd.notna(fila["roc_auc"]) and fila["roc_auc"] < auc_referencia - CAIDA_AUC_REENTRENAR:
            alertas.append({"periodo": periodo, "nivel": "Alta",
                            "mensaje": f"ROC-AUC del periodo ({fila['roc_auc']:.2f}) por debajo de la de "
                                       f"evaluación ({auc_referencia:.2f}).",
                            "recomendacion": "Revisar el desempeño; si persiste, reentrenar el modelo."})

    for _, fila in tendencias.iterrows():
        if fila["tendencia"] == "Creciente":
            alertas.append({"periodo": "Tendencia", "nivel": "Media",
                            "mensaje": f"El drift de {fila['variable']} viene aumentando periodo a periodo.",
                            "recomendacion": "Revisar la variable antes de que supere el umbral crítico."})
        if fila["cambios_abruptos"]:
            alertas.append({"periodo": fila["cambios_abruptos"], "nivel": "Media",
                            "mensaje": f"Cambio abrupto en {fila['variable']}.",
                            "recomendacion": "Verificar si hubo un cambio en la fuente de datos o en el negocio."})
    return pd.DataFrame(alertas, columns=["periodo", "nivel", "mensaje", "recomendacion"])


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def ejecutar_monitoreo(periodicidad: str = "M", tamano_muestra: int = TAMANO_MUESTRA,
                       incluir_simulacion: bool = False, ruta_db: Path | None = None) -> dict:
    """Ejecuta todo el monitoreo y guarda los resultados en la base y en CSV."""
    ruta_db = ruta_db or RUTA_DB
    DIR_MONITOREO.mkdir(exist_ok=True)
    modelo = cargar_modelo()
    referencia = construir_referencia(modelo)

    registrar_historico(modelo, ruta_db)
    if incluir_simulacion:
        simular_drift(modelo, ruta_db=ruta_db)
    else:
        with conectar(ruta_db) as conexion:
            conexion.execute("DELETE FROM predicciones WHERE origen = 'simulado'")

    tabla = leer_tabla("predicciones", ruta_db)
    muestras = muestreo_periodico(tabla, periodicidad, tamano_muestra)
    periodos = pd.to_datetime(tabla["fecha_prestamo"], errors="coerce").dt.to_period(periodicidad).astype(str)

    filas_metricas, filas_resumen = [], []
    for periodo, muestra in muestras.items():
        suficiente = len(muestra) >= MUESTRA_MINIMA
        drift = calcular_drift(referencia, muestra)
        drift.insert(0, "periodo", periodo)
        drift["muestra_suficiente"] = suficiente
        filas_metricas.append(drift)

        variables = drift[drift["variable"] != VARIABLE_PREDICCION]
        prediccion = drift[drift["variable"] == VARIABLE_PREDICCION]
        nivel_global = int(variables["nivel"].max()) if suficiente else -1
        filas_resumen.append({
            "periodo": periodo,
            "origen": ", ".join(sorted(muestra["origen"].unique())),
            "registros_muestra": len(muestra),
            "muestra_suficiente": suficiente,
            "variables_rojo": int((variables["nivel"] == 2).sum()),
            "variables_amarillo": int((variables["nivel"] == 1).sum()),
            "porcentaje_variables_rojo": float((variables["nivel"] == 2).mean()),
            "psi_promedio": float(variables["psi"].mean()),
            "psi_prediccion": float(prediccion["psi"].iloc[0]) if len(prediccion) else np.nan,
            # el desempeño se mide con todos los registros del periodo (no solo la muestra)
            **desempeno(tabla[periodos == periodo]),
            "estado_global": ESTADOS.get(nivel_global, "Muestra insuficiente"),
        })

    metricas = pd.concat(filas_metricas, ignore_index=True)
    resumen = pd.DataFrame(filas_resumen)
    tendencias = detectar_tendencias(metricas)
    auc_ref = None
    if RUTA_METADATA.exists():
        auc_ref = json.loads(RUTA_METADATA.read_text(encoding="utf-8"))["metricas_test"].get("roc_auc")
    alertas = generar_alertas(resumen, metricas, tendencias, auc_ref)

    referencia_guardar = referencia[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS + [VARIABLE_PREDICCION]]
    configuracion = pd.DataFrame([{
        "fecha_ejecucion": pd.Timestamp.now().isoformat(timespec="seconds"),
        "periodicidad": periodicidad, "tamano_muestra": tamano_muestra,
        "incluye_simulacion": incluir_simulacion, "auc_referencia": auc_ref,
    }])
    with conectar(ruta_db) as conexion:
        for nombre, df in [("metricas_drift", metricas), ("resumen_periodos", resumen),
                           ("tendencias", tendencias), ("alertas", alertas),
                           ("referencia", referencia_guardar), ("configuracion", configuracion)]:
            df.to_sql(nombre, conexion, if_exists="replace", index=False)
    for nombre, df in [("metricas_drift", metricas), ("resumen_periodos", resumen),
                       ("tendencias", tendencias), ("alertas", alertas)]:
        df.to_csv(DIR_MONITOREO / f"{nombre}.csv", index=False, encoding="utf-8-sig")

    return {"metricas": metricas, "resumen": resumen, "tendencias": tendencias, "alertas": alertas}


def main():
    parser = argparse.ArgumentParser(description="Monitoreo de data drift del modelo")
    parser.add_argument("--periodicidad", default="M", choices=["W", "M", "Q"],
                        help="W = semanal, M = mensual (defecto), Q = trimestral")
    parser.add_argument("--tamano-muestra", type=int, default=TAMANO_MUESTRA)
    parser.add_argument("--simular-drift", action="store_true",
                        help="Agrega 3 periodos futuros con drift simulado para probar las alertas")
    args = parser.parse_args()

    resultado = ejecutar_monitoreo(args.periodicidad, args.tamano_muestra, args.simular_drift)
    resumen, alertas = resultado["resumen"], resultado["alertas"]
    columnas = ["periodo", "registros_muestra", "variables_rojo", "variables_amarillo",
                "psi_prediccion", "roc_auc", "estado_global"]
    print("=== Resumen por periodo ===")
    print(resumen[columnas].round(3).to_string(index=False))
    print(f"\n=== Alertas ({len(alertas)}) ===")
    for _, alerta in alertas.iterrows():
        print(f"[{alerta['nivel']}] {alerta['periodo']}: {alerta['mensaje']}\n    -> {alerta['recomendacion']}")
    print(f"\nResultados guardados en {DIR_MONITOREO}")


if __name__ == "__main__":
    main()
