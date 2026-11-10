"""
app_monitoreo.py
================
Aplicación de Streamlit para visualizar el monitoreo del modelo y el data drift.

Ejecutar (desde la raíz del repositorio):
    streamlit run mlops_pipeline/app_monitoreo.py

Secciones:
    1. Resumen        -> semáforo general y estado de cada periodo.
    2. Drift por variable -> tabla de métricas (KS, PSI, JS, chi-cuadrado) y barras de riesgo.
    3. Distribuciones -> comparación histórica (entrenamiento) vs actual (periodo).
    4. Evolución temporal -> drift a lo largo del tiempo, tendencias y cambios abruptos.
    5. Alertas y recomendaciones -> mensajes automáticos, sugerencias de reentrenamiento.
    6. Probar el modelo -> predicción por lotes subiendo un CSV.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_monitoring as mon  # noqa: E402

COLORES = {"Verde": "#2e7d32", "Amarillo": "#f9a825", "Rojo": "#c62828", "Muestra insuficiente": "#9e9e9e"}
ICONOS = {"Verde": "🟢", "Amarillo": "🟡", "Rojo": "🔴", "Muestra insuficiente": "⚪"}
NIVEL_ALERTA = {"Crítica": st.error, "Alta": st.error, "Media": st.warning}

st.set_page_config(page_title="Monitoreo del modelo de crédito", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_resultados(ruta_db: str, marca: float) -> dict:
    """Lee todas las tablas del monitoreo. `marca` (fecha de modificación de la base) invalida
    la caché cuando el monitoreo se vuelve a ejecutar."""
    nombres = ["resumen_periodos", "metricas_drift", "tendencias", "alertas",
               "referencia", "predicciones", "configuracion"]
    tablas = {n: mon.leer_tabla(n, Path(ruta_db)) for n in nombres}
    for df in tablas.values():  # SQLite devuelve None en columnas numéricas vacías
        for col in df.columns:
            if df[col].dtype == object:
                convertida = pd.to_numeric(df[col], errors="coerce")
                if convertida.notna().sum() == df[col].notna().sum() and df[col].notna().any():
                    df[col] = convertida
                elif df[col].isna().all():
                    df[col] = np.nan
    return tablas


def marca_base() -> float:
    return mon.RUTA_DB.stat().st_mtime if mon.RUTA_DB.exists() else 0.0


def periodo_de(fechas: pd.Series, periodicidad: str) -> pd.Series:
    return pd.to_datetime(fechas, errors="coerce").dt.to_period(periodicidad).astype(str)


# ---------------------------------------------------------------------------
# Barra lateral: ejecución del monitoreo
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Monitoreo")
opciones_periodo = {"Mensual": "M", "Trimestral": "Q", "Semanal": "W"}
periodicidad_txt = st.sidebar.selectbox("Periodicidad del muestreo", list(opciones_periodo))
tamano = st.sidebar.slider("Tamaño de muestra por periodo", 100, 1000, mon.TAMANO_MUESTRA, step=50)
simular = st.sidebar.checkbox("Agregar 3 periodos con drift simulado", value=False,
                              help="Crea periodos futuros con cambios en la población para probar las alertas.")
if st.sidebar.button("▶ Ejecutar monitoreo", type="primary", width="stretch"):
    with st.spinner("Calculando métricas de drift..."):
        try:
            mon.ejecutar_monitoreo(opciones_periodo[periodicidad_txt], tamano, simular)
            st.sidebar.success("Monitoreo actualizado")
        except FileNotFoundError as error:
            st.sidebar.error(str(error))

datos = cargar_resultados(str(mon.RUTA_DB), marca_base())
resumen = datos["resumen_periodos"]
if resumen.empty:
    st.title("📊 Monitoreo del modelo de pago a tiempo")
    st.info("Todavía no hay resultados. Pulsa **Ejecutar monitoreo** en la barra lateral.")
    st.stop()

config = datos["configuracion"].iloc[0] if not datos["configuracion"].empty else {}
periodicidad = config.get("periodicidad", "M")
metricas = datos["metricas_drift"]
tendencias = datos["tendencias"]
alertas = datos["alertas"]
referencia = datos["referencia"]
predicciones = datos["predicciones"]
auc_referencia = config.get("auc_referencia")

st.sidebar.caption(f"Última ejecución: {config.get('fecha_ejecucion', '-')}")
periodos_validos = resumen.loc[resumen["muestra_suficiente"].astype(bool), "periodo"].tolist()
periodo_sel = st.sidebar.selectbox("Periodo a analizar", resumen["periodo"].tolist(),
                                   index=resumen["periodo"].tolist().index(periodos_validos[-1])
                                   if periodos_validos else len(resumen) - 1)
fila_periodo = resumen.set_index("periodo").loc[periodo_sel]
metricas_periodo = metricas[metricas["periodo"] == periodo_sel].copy()

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("📊 Monitoreo del modelo de pago a tiempo")
st.caption("Compara la población de cada periodo con la de entrenamiento para detectar data drift "
           "que pueda afectar el desempeño del modelo.")

tabs = st.tabs(["Resumen", "Drift por variable", "Distribuciones", "Evolución temporal",
                "Alertas y recomendaciones", "Probar el modelo"])

# ---------------------------------------------------------------------------
# 1. Resumen
# ---------------------------------------------------------------------------
with tabs[0]:
    estado = fila_periodo["estado_global"]
    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{COLORES[estado]};color:white;'>"
        f"<span style='font-size:1.1rem'>Periodo {periodo_sel}</span><br>"
        f"<span style='font-size:2rem;font-weight:700'>{ICONOS[estado]} Estado: {estado}</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Registros en la muestra", int(fila_periodo["registros_muestra"]))
    c2.metric("Variables en rojo", int(fila_periodo["variables_rojo"]))
    c3.metric("Variables en amarillo", int(fila_periodo["variables_amarillo"]))
    c4.metric("PSI de la predicción", f"{fila_periodo['psi_prediccion']:.3f}")
    auc = fila_periodo["roc_auc"]
    c5.metric("ROC-AUC del periodo", "-" if pd.isna(auc) else f"{auc:.3f}",
              None if pd.isna(auc) or not auc_referencia else f"{auc - auc_referencia:+.3f} vs evaluación")

    st.subheader("Semáforo por periodo")
    fig = go.Figure(go.Bar(
        x=resumen["periodo"], y=[1] * len(resumen),
        marker_color=[COLORES[e] for e in resumen["estado_global"]],
        text=[ICONOS[e] for e in resumen["estado_global"]], textposition="inside",
        hovertext=[f"{p}: {e} | rojo={r}, amarillo={a} | origen={o}" for p, e, r, a, o in zip(
            resumen["periodo"], resumen["estado_global"], resumen["variables_rojo"],
            resumen["variables_amarillo"], resumen["origen"])],
        hoverinfo="text",
    ))
    fig.update_layout(height=170, yaxis_visible=False, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    columnas = ["periodo", "origen", "registros_muestra", "variables_rojo", "variables_amarillo",
                "psi_promedio", "psi_prediccion", "tasa_no_pago_real", "tasa_no_pago_predicha",
                "roc_auc", "estado_global"]
    st.dataframe(
        resumen[columnas].style.format({"psi_promedio": "{:.3f}", "psi_prediccion": "{:.3f}",
                                        "tasa_no_pago_real": "{:.1%}", "tasa_no_pago_predicha": "{:.1%}",
                                        "roc_auc": "{:.3f}"}, na_rep="-")
        .map(lambda e: f"color:{COLORES.get(e, 'inherit')};font-weight:600", subset=["estado_global"]),
        width="stretch", hide_index=True,
    )
    st.caption("Umbrales — PSI: 0,10 amarillo / 0,25 rojo · KS: 0,10 / 0,20 (con p < 0,05) · "
               "Jensen-Shannon: 0,02 / 0,05 · Chi-cuadrado: V de Cramér 0,05 / 0,10 (con p < 0,05). "
               f"Periodos con menos de {mon.MUESTRA_MINIMA} registros se marcan como muestra insuficiente.")

# ---------------------------------------------------------------------------
# 2. Drift por variable
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader(f"Métricas de drift por variable — periodo {periodo_sel}")
    orden = metricas_periodo.sort_values("psi", ascending=False)
    fig = px.bar(orden, x="psi", y="variable", orientation="h", color="estado",
                 color_discrete_map=COLORES, category_orders={"variable": orden["variable"].tolist()})
    fig.add_vline(x=mon.UMBRALES["psi"][0], line_dash="dash", line_color=COLORES["Amarillo"])
    fig.add_vline(x=mon.UMBRALES["psi"][1], line_dash="dash", line_color=COLORES["Rojo"])
    fig.update_layout(height=520, xaxis_title="PSI (barra de riesgo)", yaxis_title="", legend_title="Estado")
    st.plotly_chart(fig, width="stretch")

    columnas = ["variable", "tipo", "ks_estadistico", "ks_p_valor", "psi", "js",
                "chi2_p_valor", "cramer_v", "estado"]
    st.dataframe(
        orden[columnas].style.format({c: "{:.4f}" for c in columnas[2:-1]}, na_rep="-")
        .map(lambda e: f"background-color:{COLORES.get(e, 'transparent')};color:white", subset=["estado"]),
        width="stretch", hide_index=True,
    )

# ---------------------------------------------------------------------------
# 3. Distribuciones: histórico vs actual
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Distribución histórica (entrenamiento) vs actual")
    variables = mon.VARIABLES_NUMERICAS + mon.VARIABLES_CATEGORICAS + [mon.VARIABLE_PREDICCION]
    por_psi = metricas_periodo.sort_values("psi", ascending=False)["variable"].tolist()
    variable = st.selectbox("Variable", por_psi or variables)
    actual = predicciones[periodo_de(predicciones["fecha_prestamo"], periodicidad) == periodo_sel]

    if variable in mon.VARIABLES_CATEGORICAS:
        ref = referencia[variable].fillna("Sin dato").astype(str).value_counts(normalize=True)
        act = actual[variable].fillna("Sin dato").astype(str).value_counts(normalize=True)
        comp = pd.DataFrame({"Histórico": ref, "Actual": act}).fillna(0).reset_index(names="categoria")
        comp = comp.melt(id_vars="categoria", var_name="conjunto", value_name="proporcion")
        fig = px.bar(comp, x="categoria", y="proporcion", color="conjunto", barmode="group",
                     color_discrete_sequence=["#5c6bc0", "#ef6c00"])
        fig.update_yaxes(tickformat=".0%")
    else:
        ref = pd.to_numeric(referencia[variable], errors="coerce").dropna()
        act = pd.to_numeric(actual[variable], errors="coerce").dropna()
        log = st.checkbox("Escala logarítmica (log1p) para montos", value=variable in mon.VARIABLES_NUMERICAS[:6])
        if log:
            ref, act = np.log1p(ref.clip(lower=0)), np.log1p(act.clip(lower=0))
        comp = pd.concat([pd.DataFrame({"valor": ref, "conjunto": "Histórico"}),
                          pd.DataFrame({"valor": act, "conjunto": "Actual"})])
        fig = px.histogram(comp, x="valor", color="conjunto", histnorm="probability density",
                           barmode="overlay", opacity=0.6, nbins=50,
                           color_discrete_sequence=["#5c6bc0", "#ef6c00"])
        fig.update_xaxes(title=f"log(1 + {variable})" if log else variable)
    fig.update_layout(height=430, legend_title="")
    st.plotly_chart(fig, width="stretch")

    fila = metricas_periodo[metricas_periodo["variable"] == variable]
    if not fila.empty:
        f = fila.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PSI", f"{f['psi']:.3f}")
        c2.metric("Jensen-Shannon", f"{f['js']:.4f}")
        if f["tipo"] == "numerica":
            c3.metric("KS (estadístico)", f"{f['ks_estadistico']:.3f}")
            c4.metric("KS p-valor", f"{f['ks_p_valor']:.4f}")
        else:
            c3.metric("Chi-cuadrado p-valor", f"{f['chi2_p_valor']:.4f}")
            c4.metric("V de Cramér", f"{f['cramer_v']:.3f}")
        st.markdown(f"**Estado:** {ICONOS[f['estado']]} {f['estado']}")

# ---------------------------------------------------------------------------
# 4. Evolución temporal
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Evolución del drift a lo largo del tiempo")
    confiables = metricas[metricas["muestra_suficiente"].astype(bool)]
    top = (confiables.groupby("variable")["psi"].mean().sort_values(ascending=False).head(4).index.tolist())
    seleccion = st.multiselect("Variables", sorted(metricas["variable"].unique()), default=top)
    serie = confiables[confiables["variable"].isin(seleccion)]
    fig = px.line(serie, x="periodo", y="psi", color="variable", markers=True)
    fig.add_hline(y=mon.UMBRALES["psi"][0], line_dash="dash", line_color=COLORES["Amarillo"],
                  annotation_text="amarillo")
    fig.add_hline(y=mon.UMBRALES["psi"][1], line_dash="dash", line_color=COLORES["Rojo"], annotation_text="rojo")
    fig.update_layout(height=420, yaxis_title="PSI")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Mapa de calor del PSI (periodo × variable)**")
    mapa = confiables.pivot_table(index="variable", columns="periodo", values="psi")
    fig = px.imshow(mapa, color_continuous_scale=["#e8f5e9", "#fff59d", "#ef5350", "#b71c1c"],
                    zmin=0, zmax=0.5, aspect="auto")
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Drift de la predicción y desempeño**")
        fig = go.Figure()
        validos = resumen[resumen["muestra_suficiente"].astype(bool)]
        fig.add_trace(go.Scatter(x=validos["periodo"], y=validos["psi_prediccion"], name="PSI predicción",
                                 mode="lines+markers"))
        fig.add_trace(go.Scatter(x=validos["periodo"], y=validos["roc_auc"], name="ROC-AUC",
                                 mode="lines+markers", yaxis="y2"))
        fig.update_layout(height=360, yaxis=dict(title="PSI"),
                          yaxis2=dict(title="ROC-AUC", overlaying="y", side="right", range=[0.4, 1]),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Tendencias y cambios abruptos**")
        st.dataframe(
            tendencias.sort_values("pendiente_psi", ascending=False)
            .style.format({"pendiente_psi": "{:+.4f}", "p_valor_tendencia": "{:.3f}",
                           "psi_ultimo_periodo": "{:.3f}"}),
            width="stretch", hide_index=True, height=360,
        )
    st.caption("Tendencia: regresión lineal del PSI contra el tiempo (creciente si la pendiente es "
               f"positiva y p < 0,05). Cambio abrupto: el PSI sube más de {mon.SALTO_ABRUPTO_PSI} "
               "de un periodo al siguiente.")

# ---------------------------------------------------------------------------
# 5. Alertas y recomendaciones
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Recomendación general")
    ultimo = resumen[resumen["muestra_suficiente"].astype(bool)].tail(1)
    criticas = alertas[alertas["nivel"] == "Crítica"]
    if not ultimo.empty and ultimo.iloc[0]["periodo"] in criticas["periodo"].tolist():
        st.error("🔁 **Se recomienda reentrenar el modelo**: el último periodo supera un umbral crítico.")
    elif not ultimo.empty and ultimo.iloc[0]["estado_global"] == "Rojo":
        st.warning("🔎 **Revisar variables**: hay drift alto en el último periodo, aunque no en un nivel "
                   "que obligue a reentrenar todavía.")
    else:
        st.success("✅ El modelo se mantiene estable. Continuar con el monitoreo periódico.")

    variables_revisar = tendencias[(tendencias["tendencia"] == "Creciente")
                                   | (tendencias["psi_ultimo_periodo"] >= mon.UMBRALES["psi"][1])]
    if not variables_revisar.empty:
        st.markdown("**Variables a revisar:** " + ", ".join(variables_revisar["variable"]))

    st.subheader(f"Alertas automáticas ({len(alertas)})")
    filtro = st.multiselect("Nivel", ["Crítica", "Alta", "Media"], default=["Crítica", "Alta", "Media"])
    orden_nivel = {"Crítica": 0, "Alta": 1, "Media": 2}
    visibles = (alertas[alertas["nivel"].isin(filtro)]
                .assign(_orden=lambda d: d["nivel"].map(orden_nivel))
                .sort_values(["_orden", "periodo"], ascending=[True, False]))
    for _, alerta in visibles.iterrows():
        NIVEL_ALERTA[alerta["nivel"]](
            f"**[{alerta['nivel']}] {alerta['periodo']}** — {alerta['mensaje']}\n\n➜ {alerta['recomendacion']}"
        )

# ---------------------------------------------------------------------------
# 6. Probar el modelo
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Predicción por lotes")
    st.write("Sube un CSV con las mismas columnas de `Base_de_datos.csv` (la columna `Pago_atiempo` es opcional).")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        lote = pd.read_csv(archivo)
        modelo = mon.cargar_modelo()
        resultado = mon.puntuar(lote.drop(columns=["Pago_atiempo"], errors="ignore"), modelo)
        resultado["resultado"] = np.where(resultado["prediccion"] == 1, "Paga a tiempo", "Riesgo de no pago")
        c1, c2 = st.columns(2)
        c1.metric("Registros", len(resultado))
        c2.metric("Con riesgo de no pago", f"{(resultado['prediccion'] == 0).mean():.1%}")
        st.dataframe(resultado[["prob_no_pago", "resultado"] + [c for c in lote.columns][:8]],
                     width="stretch")
        st.download_button("Descargar predicciones", resultado.to_csv(index=False).encode("utf-8"),
                           "predicciones.csv", "text/csv")



