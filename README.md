# Predicción de pago a tiempo de créditos — Proyecto final de Machine Learning (MLOps)

**Universidad Católica Luis Amigó** · **Docente:** Juan Sebastián Parra Sánchez · **Entrega:** 11 de noviembre de 2026
**Autor:** Juan Gutiérrez

Modelo de Machine Learning que predice si un cliente pagará su crédito a tiempo, desarrollado bajo un esquema
MLOps: desde la carga y el análisis de los datos hasta el despliegue como API, el monitoreo del data drift,
las pruebas automáticas, el análisis de calidad en SonarCloud y la integración continua.

---

## 1. Caso de negocio

La empresa otorga créditos de consumo y necesita decidir, **al momento de la solicitud**, qué tan probable es que
el cliente no pague a tiempo. Aprobar a un cliente que luego incumple genera pérdidas y costos de cobranza;
rechazar a uno bueno es negocio perdido.

- **Objetivo:** predecir `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).
- **Tipo de problema:** clasificación binaria supervisada.
- **Clase de interés:** el **no pago** (0), que es solo el **4,75 %** de los casos.
- **Uso esperado:** apoyar la aprobación de créditos y priorizar la gestión de cobranza preventiva sobre los clientes
  con mayor riesgo.

Por el desbalance de clases, la exactitud (*accuracy*) no sirve para evaluar: un modelo que diga siempre
"paga" acertaría el 95 % de las veces sin detectar a ningún cliente riesgoso. Por eso se usan ROC-AUC,
PR-AUC, *balanced accuracy* y el *recall* de la clase "no paga".

## 2. Datos

`Base_de_datos.csv`: 10.763 créditos desembolsados entre noviembre de 2024 y abril de 2026, con 23 variables:
características del crédito (tipo, capital, plazo, cuota), del cliente (edad, tipo laboral, salario) y de su
historial en el buró Datacrédito (puntaje, consultas, créditos vigentes, saldos, ingresos reportados).

## 3. Principales hallazgos del EDA (`eda_analisis.ipynb`)

- **Desbalance fuerte:** 95,25 % paga a tiempo y 4,75 % no.
- **Fuga de información:** la variable `puntaje` separa perfectamente a quienes pagan de quienes no
  (no pagadores entre −38 y 62,7; pagadores entre 63,8 y 95,2). Se calculó conociendo el resultado, así que
  **se elimina**. También se elimina `saldo_mora_codeudor` (casi constante) y `saldo_principal`
  (correlación de 0,95 con `saldo_total`).
- **Problemas de calidad:** 150 edades de 121 a 123 años, 24 salarios en 0, 58 valores numéricos en
  `tendencia_ingresos` y puntajes de Datacrédito fuera de rango. Se convierten en datos faltantes.
- **Distribuciones:** ninguna variable numérica es normal; los montos tienen asimetría positiva fuerte
  (se transforman con `log1p` y se limitan en los percentiles 1 y 99).
- **Señales de riesgo:** tener saldo en mora (36,4 % de no pago frente a 4,6 %), puntaje de Datacrédito bajo,
  muchas consultas recientes al buró, ser joven o independiente, ingresos decrecientes, montos y plazos altos,
  el tipo de crédito 6 y los desembolsos en fin de semana.
- **Relaciones débiles y no lineales** con la variable objetivo: ninguna variable sola predice bien el no pago.

## 4. Proceso (pipeline MLOps)

```text
Cargar_datos.ipynb ─► eda_analisis.ipynb ─► ft_eng.py ─► model_training_evaluation.py
                                                                │  modelos/mejor_modelo.joblib
                                                                ▼
                         app_monitoreo.py ◄── model_monitoring.py ◄── model_deploy.py (API)
                         (Streamlit)          (data drift)            cada predicción se guarda
                                                                      en la tabla de monitoreo
```

### 4.1 Feature engineering (`ft_eng.py`)

Un único `Pipeline` de scikit-learn, que se aplica igual en el entrenamiento y en producción:

1. **`LimpiezaYAtributos`:** valores imposibles a `NaN`; agrupa los tipos de crédito poco frecuentes en "Otros";
   crea atributos derivados (`tiene_mora`, `sin_info_datacredito`, `sin_historial_datacredito`, `fin_de_semana`,
   `edad_desconocida`, ratios `cuota_salario`, `endeudamiento`, `capital_salario`, `cuota_ingreso_datacredito`,
   mes, día de la semana y total de créditos por sector); aplica `log1p` a los montos y elimina las variables
   descartadas.
2. **`ColumnTransformer`** con tres ramas:
   - `numeric`: `SimpleImputer(mediana)` → `Winsorizador` (percentiles 1–99) → `StandardScaler`
   - `categoric`: `SimpleImputer(moda)` → `OneHotEncoder`
   - `categoric_ordinales`: `SimpleImputer(moda)` → `OrdinalEncoder` (Decreciente < Estable < Creciente)

La partición es estratificada (80 % entrenamiento, 20 % evaluación), y quedan 35 variables finales.

### 4.2 Entrenamiento y selección del modelo (`model_training_evaluation.py`)

Se comparan cinco algoritmos supervisados: **regresión logística** (equivalente de la regresión lineal para
un objetivo 0/1), **árbol de decisión**, **Random Forest**, **KNN** y **red neuronal (MLP)**.

- `build_model` arma el pipeline completo, busca hiperparámetros con `GridSearchCV` (ROC-AUC) y ajusta el
  umbral de decisión para maximizar la *balanced accuracy* (necesario por el desbalance).
- `summarize_classification` calcula todas las métricas.
- El mejor modelo se elige con un puntaje ponderado de **performance (50 %)**, **consistency (30 %)** y
  **scalability (20 %)**.

| Modelo | ROC-AUC | PR-AUC (no pago) | Balanced acc. | Recall no pago | Precisión no pago |
|---|---|---|---|---|---|
| **Regresión logística** ✅ | 0,696 | 0,183 | 0,644 | 0,667 | 0,080 |
| Random Forest | 0,704 | 0,198 | 0,649 | 0,725 | 0,078 |
| Árbol de decisión | 0,656 | 0,119 | 0,610 | 0,637 | 0,071 |
| KNN | 0,625 | 0,149 | 0,598 | 0,510 | 0,075 |
| Red neuronal (MLP) | 0,611 | 0,139 | 0,564 | 0,225 | 0,102 |

**Modelo seleccionado: regresión logística.** Random Forest predice apenas mejor, pero sobreajusta mucho más
(0,31 de diferencia de AUC entre entrenamiento y validación, frente a 0,03) y es unas 10 veces más lento y
pesado. El modelo detecta 2 de cada 3 clientes que no pagan, aunque con muchas falsas alarmas (precisión del
8 %), lo que es coherente con la señal débil que mostró el EDA. Los gráficos comparativos y la tabla resumen
quedan en `resultados/`.

### 4.3 Despliegue (`model_deploy.py`)

API REST con **FastAPI** + **Uvicorn**:

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y del modelo |
| `GET /modelo` | Metadata del modelo (algoritmo, umbral, métricas) |
| `POST /predict` | Predicción por lotes en JSON: `{"registros": [{...}, {...}]}` |
| `POST /predict/csv` | Predicción por lotes subiendo un CSV (`?formato=csv` devuelve un CSV) |

Cada respuesta incluye `prob_no_pago`, `prediccion`, `resultado` y `nivel_riesgo` (Bajo / Medio / Alto).
Cada predicción se guarda con sus datos en la tabla de monitoreo.

```bash
uvicorn model_deploy:app --app-dir mlops_pipeline --port 8000     # documentación en http://localhost:8000/docs

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"registros":[{"tipo_credito":4,"capital_prestado":1852560,"plazo_meses":12,"cuota_pactada":128650,
       "edad_cliente":32,"tipo_laboral":"Empleado","salario_cliente":3500000,"puntaje_datacredito":795}]}'

curl -F "archivo=@Base_de_datos.csv" "http://localhost:8000/predict/csv?formato=csv" -o predicciones.csv
```

### 4.4 Monitoreo y data drift (`model_monitoring.py`)

1. **Tabla de monitoreo** (`monitoreo/monitoreo.db`, tabla `predicciones`): los datos de cada crédito con el
   pronóstico entregado. La alimentan la API y la simulación histórica (todos los créditos del dataset,
   puntuados por el modelo y ordenados por fecha de desembolso).
2. **Muestreo periódico:** con la periodicidad definida (mensual por defecto; también semanal o trimestral)
   se toma una muestra aleatoria de hasta 500 registros por periodo.
3. **Métricas de drift** de cada periodo frente a la población de entrenamiento:

   | Métrica | Variables | Amarillo | Rojo |
   |---|---|---|---|
   | Population Stability Index (PSI) | todas | ≥ 0,10 | ≥ 0,25 |
   | Kolmogorov-Smirnov (estadístico, con p < 0,05) | numéricas | ≥ 0,10 | ≥ 0,20 |
   | Divergencia de Jensen-Shannon (base 2) | todas | ≥ 0,02 | ≥ 0,05 |
   | Chi-cuadrado (V de Cramér, con p < 0,05) | categóricas | ≥ 0,05 | ≥ 0,10 |

   También se mide el drift de la **predicción** (PSI de la probabilidad de no pago) y, cuando ya se conoce el
   resultado real, la **ROC-AUC del periodo**.
4. **Análisis temporal:** tendencia del PSI (regresión lineal en el tiempo) y cambios abruptos (el PSI sube más
   de 0,10 entre dos periodos seguidos).
5. **Alertas y recomendaciones:** reentrenar si el 30 % o más de las variables están en rojo, si el PSI de la
   predicción supera 0,25 o si la AUC cae más de 0,10; revisar las variables con drift alto o creciente.

```bash
python mlops_pipeline/model_monitoring.py                   # mensual
python mlops_pipeline/model_monitoring.py --periodicidad Q  # trimestral
python mlops_pipeline/model_monitoring.py --simular-drift   # agrega 3 periodos con drift simulado (demo de alertas)
```

**Hallazgos del monitoreo con los datos reales:**

- La predicción del modelo es estable entre 2024-12 y 2025-12 (PSI de la predicción menor a 0,10) y la ROC-AUC
  mensual se mantiene entre 0,65 y 0,76, en línea con la de evaluación (0,70).
- `promedio_ingresos_datacredito` cambia casi todos los meses (en rojo en el 67 % de los periodos): muchos
  ingresos del buró son valores fijos cercanos al salario mínimo (908.526, 916.148, 931.394…) que varían según
  el periodo. Conviene revisar con el área de datos cómo se calcula esta variable.
- `plazo_meses` muestra una **tendencia creciente** de drift: los plazos bajaron a 6 meses a mediados de 2025 y
  volvieron a subir a finales del año. `total_otros_prestamos`, `capital_prestado`, `huella_consulta` y
  `edad_cliente` también tienen drift creciente.
- Desde 2026-01 el volumen de créditos cae mucho y el PSI de la predicción sube a 0,22, cerca del umbral
  crítico. **Recomendación:** reentrenar el modelo incorporando los datos más recientes cuando haya suficientes
  registros. Los periodos con menos de 100 créditos (2026-02 a 2026-04) se marcan como muestra insuficiente.

### 4.5 Aplicación de Streamlit (`app_monitoreo.py`)

```bash
streamlit run mlops_pipeline/app_monitoreo.py
```

- **Resumen:** semáforo del periodo, indicadores clave y semáforo de todos los periodos.
- **Drift por variable:** tabla con KS, PSI, Jensen-Shannon y chi-cuadrado, y barras de riesgo.
- **Distribuciones:** comparación histórica (entrenamiento) vs actual de cada variable.
- **Evolución temporal:** PSI en el tiempo, mapa de calor, tendencias y cambios abruptos.
- **Alertas y recomendaciones:** mensajes automáticos y sugerencia de reentrenamiento o revisión de variables.
- **Probar el modelo:** predicción por lotes subiendo un CSV.

### 4.6 Imagen Docker

`Dockerfile` (Python 3.13 slim, usuario sin privilegios, *healthcheck*) con el código, las dependencias de
`requirements-prod.txt`, el modelo entrenado y el servidor Uvicorn. `.dockerignore` deja fuera el entorno
virtual, los notebooks y las pruebas.

```bash
docker build -t credito-ml .
docker run -p 8000:8000 credito-ml                         # API
docker compose up --build                                  # API (8000) + app de monitoreo (8501)
```

## 5. Calidad del código, pruebas y CI

- **Pruebas** (`tests/`, pytest): estructura del repositorio, feature engineering, funciones de entrenamiento,
  métricas de drift, API (pruebas de integración con `TestClient`) y app de Streamlit (`AppTest`).
  Cobertura cercana al 90 %.

  ```bash
  pip install -r requirements-dev.txt
  pytest --cov=mlops_pipeline --cov-report=term
  flake8 mlops_pipeline tests
  ```

- **GitHub Actions** (`.github/workflows/ci.yml`): en cada push o pull request ejecuta flake8, las pruebas con
  cobertura, el análisis de **SonarCloud** y la construcción de la imagen Docker con una prueba de humo de la API.
- **SonarCloud** (`sonar-project.properties`): calidad (duplicación, complejidad, malas prácticas), seguridad,
  cobertura de pruebas e integridad y estilo (reporte de flake8).
- **Jenkins (bonus)** (`Jenkinsfile`): clona el repositorio, valida la estructura de carpetas, instala
  dependencias, ejecuta estilo y pruebas, construye la imagen Docker, hace una prueba de humo y envía una
  notificación por correo. Se dispara con cada push o merge (webhook de GitHub) o revisando el repositorio cada
  15 minutos.

## 6. Estructura del proyecto

```text
Repositorio_ML-proyecto_Final/
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb             # carga de datos
│   ├── eda_analisis.ipynb             # comprensión de los datos y EDA
│   ├── ft_eng.py                      # feature engineering
│   ├── model_training_evaluation.py   # entrenamiento, evaluación y selección del modelo
│   ├── model_deploy.py                # API FastAPI (/predict)
│   ├── model_monitoring.py            # monitoreo y data drift
│   └── app_monitoreo.py               # app de Streamlit
├── modelos/                           # mejor modelo (.joblib) y su metadata
├── resultados/                        # tabla resumen y gráficos comparativos
├── monitoreo/                         # métricas de drift, alertas y base SQLite
├── tests/                             # pruebas automáticas
├── .github/workflows/ci.yml           # GitHub Actions + SonarCloud
├── Base_de_datos.csv
├── Dockerfile · .dockerignore · docker-compose.yml
├── Jenkinsfile · sonar-project.properties · .flake8 · pytest.ini
├── requirements.txt · requirements-prod.txt · requirements-dev.txt
├── set_up.bat
└── README.md
```

## 7. Cómo ejecutar el proyecto de punta a punta

```bash
set_up.bat                                            # Windows: crea el entorno e instala dependencias
venv\Scripts\activate
python mlops_pipeline\ft_eng.py                       # revisa los conjuntos de entrenamiento y evaluación
python mlops_pipeline\model_training_evaluation.py    # entrena y guarda el mejor modelo (~3 min)
python mlops_pipeline\model_monitoring.py             # calcula el data drift
uvicorn model_deploy:app --app-dir mlops_pipeline     # API en http://localhost:8000/docs
streamlit run mlops_pipeline\app_monitoreo.py         # app de monitoreo en http://localhost:8501
```

## 8. Tecnologías

Python 3.13 · pandas · NumPy · SciPy · scikit-learn · Matplotlib · Plotly · FastAPI · Uvicorn · Streamlit ·
SQLite · Docker · pytest · flake8 · GitHub Actions · SonarCloud · Jenkins
