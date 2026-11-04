"""
ft_engineering.py
=================

Primera componente del flujo de creación de modelos.
 
Toma los datos crudos, aplica la limpieza y los atributos derivados definidos en el
EDA (secciones 4, 6, 11 y 12 de eda_analisis.ipynb), construye el ColumnTransformer
y devuelve los conjuntos de entrenamiento y evaluación.


Uso desde otro script:
 
    from ft_eng import ft_engineering
    X_train, X_test, y_train, y_test, preprocesador = ft_engineering()
 
El preprocesador se devuelve SIN ajustar: en model_training_evaluation.py se une con
cada modelo en un solo Pipeline, de modo que se ajusta solo con los datos de
entrenamiento (sin fuga de información) y queda listo para guardarse y desplegarse.


"""

from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

NOMBRE_ARCHIVO = "Base_de_datos.csv"
VARIABLE_OBJETIVO = "Pago_atiempo"
TEST_SIZE = 0.2
RANDOM_STATE = 42

VARIABLES_ELIMINAR = [
    "puntaje",              # fuga de informacion: separa perfectamente la variable objetivo
    "saldo_mora_codeudor",  # casi constante (solo 3 valores distintos de 0)
    "saldo_principal",      # redundante con saldo_total (correlacion 0,95)
    "fecha_prestamo",       # se reemplaza por mes, dia de la semana y fin de semana
]

# Montos con asimetría positiva fuerte: se transforman con log1p
VARIABLES_MONTO = [
    "capital_prestado", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "saldo_mora", "saldo_total", "promedio_ingresos_datacredito",
]


CATEGORIAS_TENDENCIA = ["Decreciente", "Estable", "Creciente"]
TIPOS_CREDITO_CONOCIDOS = ["4", "6", "9", "10"]  # 7 y 68 (3 registros) pasan a "Otros"



 # Grupos de columnas del ColumnTransformer (despues de todala limpieza)
COLUMNAS_NUMERICAS = [
    # originales
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    # derivadas (seccion 12 del EDA)
    "tiene_mora", "sin_info_datacredito", "sin_historial_datacredito", "fin_de_semana",
    "edad_desconocida", "cuota_salario", "endeudamiento", "capital_salario",
    "cuota_ingreso_datacredito", "mes_prestamo", "dia_semana", "total_creditos_sectores",

]

COLUMNAS_CATEGORICAS = ["tipo_credito", "tipo_laboral"]

COLUMNAS_ORDINALES = ["tendencia_ingresos"]

#-----------------
#Carga de datos 
#-----------------

def obtener_ruta_datos(nombre_archivo: str = NOMBRE_ARCHIVO) -> Path:
    """Busca el CSV en la raíz del repositorio, sin importar desde dónde se ejecute."""
    candidatas = [
        Path(__file__).resolve().parent.parent / nombre_archivo,  # raíz del repo
        Path.cwd() / nombre_archivo,
        Path.cwd().parent / nombre_archivo,
    ]
    for ruta in candidatas:
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró '{nombre_archivo}'. Debe estar en la raíz del repositorio."
    )

def cargar_datos(ruta: Path | None = None) -> pd.DataFrame:
    """Lee el dataset crudo (misma lógica que Cargar_datos.ipynb)."""
    ruta = ruta or obtener_ruta_datos()
    return pd.read_csv(ruta, parse_dates=["fecha_prestamo"])


# Transformadores propios
# -----------------------

"""
se hace una limpieza de datos y se crean atributos derivados. aca no se aprende nada de los datos, solo se transforman.
por eso se aplica igual en entrenamiento y en evaluacion sin que se tenga riesgo de fuga de informacion.

    1. Valores imposibles a NaN: edad > 100, salario <= 0, puntaje_datacredito fuera de rango y tendencia_ingresos fuera de sus 3 categorias.
    2. Agrupa los tipos de credito poco frecuentes en "Otros".
    3. Crea los atributos derivados del EDA (indicadores, ratios y fecha).
    4. Aplica log1p a los montos.
    5. Elimina las variables irrelevantes.


"""
class LimpiezaYAtributos(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()

 #1 valores imposibles a NaN
        df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
            df["tendencia_ingresos"].isin(CATEGORIAS_TENDENCIA), np.nan
        ).astype(object)
 
        edad = pd.to_numeric(df["edad_cliente"], errors="coerce")
        df["edad_desconocida"] = (edad.isna() | (edad > 100)).astype(int)
        df["edad_cliente"] = edad.where(edad.between(18, 100))

        salario = pd.to_numeric(df["salario_cliente"], errors="coerce")
        df["salario_cliente"] = salario.where(salario > 0)
 
        puntaje = pd.to_numeric(df["puntaje_datacredito"], errors="coerce")
        df["puntaje_datacredito"] = puntaje.where((puntaje == 0) | puntaje.between(150, 950))

#2 tipo de credito como categoria agrupando estos como los menos frecuentes en "Otros" 
        
        tipo = df["tipo_credito"].astype(str)
        df["tipo_credito"] = tipo.where(tipo.isin(TIPOS_CREDITO_CONOCIDOS), "Otros").astype(object)
        df["tipo_laboral"] = df["tipo_laboral"].astype(object)

#3 atributos derivados del EDA (indicadores, ratios y fecha)

        fecha = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
        salario = df["salario_cliente"].astype(float)
        ingreso_buro = df["promedio_ingresos_datacredito"].replace(0, np.nan)
 
        df["tiene_mora"] = (df["saldo_mora"].fillna(0) > 0).astype(int)
        df["sin_info_datacredito"] = df["promedio_ingresos_datacredito"].isna().astype(int)
        df["sin_historial_datacredito"] = (df["puntaje_datacredito"] == 0).astype(int)
        df["fin_de_semana"] = (fecha.dt.dayofweek >= 5).astype(int) 

        df["cuota_salario"] = df["cuota_pactada"] / salario
        df["endeudamiento"] = df["total_otros_prestamos"] / salario
        df["capital_salario"] = df["capital_prestado"] / salario
        df["cuota_ingreso_datacredito"] = df["cuota_pactada"] / ingreso_buro
 
        df["mes_prestamo"] = fecha.dt.month
        df["dia_semana"] = fecha.dt.dayofweek
        df["total_creditos_sectores"] = (
            df["creditos_sectorFinanciero"] + df["creditos_sectorCooperativo"]
            + df["creditos_sectorReal"]
        )

#4 log1p a montos (los negativos si llega, se dejan en 0)
         
        for col in VARIABLES_MONTO:
            df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").clip(lower=0))
   
#5 eliminar variables irrelevantes y la viene incluida 

        df = df.drop(columns=VARIABLES_ELIMINAR + [VARIABLE_OBJETIVO], errors="ignore")
        return df[COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + COLUMNAS_ORDINALES]
 
    def get_feature_names_out(self, input_features=None):
        return np.array(COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + COLUMNAS_ORDINALES)

class Winsorizador(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    Limita cada columna a sus percentiles 1 y 99, calculados SOLO con entrenamiento.
    Las columnas binarias (0/1) no se tocan, para no borrar indicadores poco
    frecuentes como tiene_mora.
    """

    def __init__(self, inferior: float = 0.01, superior: float = 0.99):
        self.inferior = inferior
        self.superior = superior
 
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.limite_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.limite_sup_ = np.nanquantile(X, self.superior, axis=0)
        binarias = np.array([np.isin(np.unique(c[~np.isnan(c)]), [0, 1]).all() for c in X.T])
        # Tampoco se limitan las columnas casi constantes (p1 = p99, ej. saldo_mora),
        # porque quedarían en un solo valor y se perdería la información.
        sin_limite = binarias | (self.limite_inf_ == self.limite_sup_)
        self.limite_inf_[sin_limite] = -np.inf
        self.limite_sup_[sin_limite] = np.inf
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.limite_inf_, self.limite_sup_)


#pipeline de preprocesamiento 

def construir_column_transformer() -> ColumnTransformer:
    """ColumnTransformer con las tres ramas: numeric, categoric y categoric ordinales."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizador", Winsorizador(0.01, 0.99)),
        ("scaler", StandardScaler()),  # necesario para KNN, redes neuronales y reg. logística
    ])
    categoric = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    categoric_ordinales = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[CATEGORIAS_TENDENCIA],
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, COLUMNAS_NUMERICAS),
        ("categoric", categoric, COLUMNAS_CATEGORICAS),
        ("categoric_ordinales", categoric_ordinales, COLUMNAS_ORDINALES),
    ], verbose_feature_names_out=False)
 
 
def construir_preprocesador() -> Pipeline:
    """Limpieza + ColumnTransformer. Se combina con cada modelo en el entrenamiento."""
    return Pipeline([
        ("limpieza", LimpiezaYAtributos()),
        ("column_transformer", construir_column_transformer()),
    ])


#funcion principal de la etapa de feature engineering

def ft_engineering(df: pd.DataFrame | None = None, test_size: float = TEST_SIZE,
                   random_state: int = RANDOM_STATE):

    """
    genera los conjuntos de entrenamiento y evaluacion, y devuelve el preprocesador sin ajustar. 
        
    """
    df = cargar_datos() if df is None else df.copy()
    df = df.drop_duplicates()
 
    X = df.drop(columns=[VARIABLE_OBJETIVO])
    y = df[VARIABLE_OBJETIVO].astype(int)
 
    # stratify mantiene la proporción 95 / 5 en ambos conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, construir_preprocesador()

def obtener_datos_transformados(df: pd.DataFrame | None = None):
    """Devuelve X_train y X_test ya transformados (útil para revisar el resultado)."""
    X_train, X_test, y_train, y_test, preprocesador = ft_engineering(df)
    preprocesador.set_output(transform="pandas")
    X_train_t = preprocesador.fit_transform(X_train)  # se ajusta SOLO con entrenamiento
    X_test_t = preprocesador.transform(X_test)
    return X_train_t, X_test_t, y_train, y_test, preprocesador
 
 
if __name__ == "__main__":
    X_train_t, X_test_t, y_train, y_test, _ = obtener_datos_transformados()


    print("=== Conjuntos resultantes ===")
    print(f"X_train: {X_train_t.shape} | X_test: {X_test_t.shape}")
    print(f"y_train -> % no pago: {(1 - y_train.mean()) * 100:.2f}%")
    print(f"y_test  -> % no pago: {(1 - y_test.mean()) * 100:.2f}%")
    print(f"Nulos después de transformar: {int(X_train_t.isna().sum().sum())}")
    print(f"\nVariables finales ({X_train_t.shape[1]}):")
    print(list(X_train_t.columns))
    print("\nPrimeras filas de X_train transformado:")
    print(X_train_t.head().round(3).to_string())




