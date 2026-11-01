"""Funciones reutilizables para el modelado de REGRESIÓN del puntaje BDI-II (bdi_total).

Las usan:
- notebooks/reg_<clave>.ipynb (fase 1): variantes del target por modelo (nativo, sqrt,
  log1p y pérdida Poisson) y, en reg_lineal, reg_poisson y reg_binomial_negativa,
  las pruebas de hipótesis con statsmodels.
- notebooks/reg_comparacion.ipynb (fase 2): comparación de los ganadores, test con
  intervalos bootstrap, permutation importance y el experimento regresión + corte clínico.

Lo común con la clasificación (SEED, predictores, preprocesador, split, GPU) se importa de
Modelado.py. Todo se guarda en data/processed/modelado/regresion/.

Uso para cargar el modelo final:

    from ModeladoRegresion import cargar_modelo_regresion, PREDICTORES
    modelo, metadatos = cargar_modelo_regresion("final")
    bdi_predicho = modelo.predict(df[PREDICTORES])   # ya en puntos BDI, recortado a [0, 63]
"""

import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels
import statsmodels.api as sm
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import sparse, stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, PoissonRegressor, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from xgboost import XGBRegressor

from EstiloGraficos import (
    DEPRIMIDO, LINEA_CORTE, LINEA_MEDIANA, NEUTRO, NO_DEPRIMIDO, TEXTO, TEXTO_SECUNDARIO,
)
from LinearRegressionDiagnostics import diagnosticar_regresion_lineal
from Modelado import (
    COLUMNAS_EXCLUIDAS,
    ESPACIOS_BUSQUEDA as ESPACIOS_CLF,
    GPU_DISPOSITIVOS,
    PREDICTORES,
    PREDICTORES_CATEGORICOS,
    PREDICTORES_NUMERICOS,
    RUTA_FINAL as RUTA_FINAL_CLF,
    RUTA_SALIDA,
    SEED,
    USAR_GPU,
    _describir_valores,
    _versiones,
    cargar_modelo_final,
    cargar_split,
    crear_pipeline_modelo,
    crear_preprocesador,
    detectar_gpu,
    dispositivo_modelo,
    formatear_hiperparametros,
    graficar_matrices,
    intervalos_bootstrap,
)
# SEED, PREDICTORES, USAR_GPU, GPU_DISPOSITIVOS, detectar_gpu, cargar_split, formatear_hiperparametros
# y graficar_matrices se reexportan: los notebooks de regresión importan todo desde este módulo.


TARGET_REG = "bdi_total"
TARGET_CLF = "depressed"
BDI_MIN, BDI_MAX = 0, 63          # rango del BDI-II: todas las predicciones se recortan a él
CORTE_BDI = 14                    # corte clínico: BDI-II >= 14 → deprimido
ALFA = 0.05                       # nivel de significancia de las pruebas de hipótesis

# Modelos de la regresión (clave del notebook reg_<clave>.ipynb → nombre legible)
NOMBRES_REG = {
    "lineal": "Lineal (OLS)",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "poisson": "GLM Poisson",
    "binomial_negativa": "GLM binomial negativa",
    "knn": "KNN",
    "svr": "SVR (RBF)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
CLAVE_BENCHMARK = "benchmark"
NOMBRES_FASE2_REG = {CLAVE_BENCHMARK: "Lineal benchmark (EDA)", **NOMBRES_REG}

# Orden de simplicidad para la regla de empate de la fase 2:
# lineales < Poisson < binomial negativa < KNN < SVR < Random Forest < boosting
SIMPLICIDAD_MODELOS = {
    "lineal": 0, "ridge": 0, "lasso": 0, "elasticnet": 0,
    "poisson": 1, "binomial_negativa": 2, "knn": 3, "svr": 4,
    "random_forest": 5, "xgboost": 6, "lightgbm": 6, "catboost": 6,
}

# Variantes del target, de la más simple a la más compleja (orden de la regla de empate de la fase 1)
VARIANTES = {
    "nativo": "Nativo",
    "sqrt": "sqrt + smearing",
    "log1p": "log1p + smearing",
    "poisson": "Pérdida Poisson",
}
SIMPLICIDAD_VARIANTES = {"nativo": 0, "sqrt": 1, "log1p": 2, "poisson": 3}

MODELOS_CONTEO = ("poisson", "binomial_negativa")
MODELOS_ARBOLES = ("random_forest", "xgboost", "lightgbm", "catboost")

# Métricas (nombre interno → nombre en las tablas). Todas en la escala original (puntos BDI).
METRICAS_REG = {"rmse": "RMSE", "mae": "MAE", "r2": "R²"}
SCORERS_REG = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

RUTA_REGRESION = RUTA_SALIDA / "regresion"
RUTA_FASE1_REG = RUTA_REGRESION / "fase1"
RUTA_FINAL_REG = RUTA_REGRESION / "final"


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def separar_x_y_reg(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Elimina explícitamente las columnas excluidas y devuelve (X, y) con los 6 predictores
    y bdi_total como target."""
    X = df.drop(columns=COLUMNAS_EXCLUIDAS)[PREDICTORES]
    y = df[TARGET_REG]
    return X, y


def crear_cv_reg(seed: int = SEED) -> KFold:
    """Los folds de toda la regresión: KFold(5, shuffle=True, random_state=SEED).
    Como cargar_split() ordena por subject_id, todos los notebooks reciben los mismos folds."""
    return KFold(n_splits=5, shuffle=True, random_state=seed)


def resumen_target(y: pd.Series) -> pd.DataFrame:
    """Descripción del BDI del train: centro, dispersión, asimetría y proporciones clave."""
    y = pd.Series(y)
    return pd.DataFrame({
        "Filas": [len(y)],
        "Media": [round(y.mean(), 2)],
        "Mediana": [y.median()],
        "DE": [round(y.std(), 2)],
        "Asimetría": [round(stats.skew(y), 2)],
        "Ceros (%)": [round((y == 0).mean() * 100, 1)],
        f"BDI ≥ {CORTE_BDI} (%)": [round((y >= CORTE_BDI).mean() * 100, 1)],
    }, index=pd.Index(["Train"], name="Conjunto"))


def _a_matriz(X) -> np.ndarray:
    """Salida del ColumnTransformer (array, matriz dispersa o DataFrame) como array float."""
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float)


# ---------------------------------------------------------------------------
# Estimadores propios (compatibles con scikit-learn)
# ---------------------------------------------------------------------------

def _inversa_sqrt(z):
    # sqrt(y) >= 0: un valor negativo en la escala transformada corresponde a y = 0
    return np.square(np.clip(z, 0, None))


# Transformación → (directa, inversa). Funciones de módulo para que joblib pueda serializarlas.
TRANSFORMACIONES = {"sqrt": (np.sqrt, _inversa_sqrt), "log1p": (np.log1p, np.expm1)}


def _predecir_base(regresor, X) -> np.ndarray:
    """predict() del regresor interno. CatBoost con pérdida Poisson se pide explícitamente en
    escala de la media (prediction_type="Exponent"), no en la escala del log."""
    if isinstance(regresor, CatBoostRegressor) and "Poisson" in str(regresor.get_params().get("loss_function")):
        return np.asarray(regresor.predict(X, prediction_type="Exponent"), dtype=float).ravel()
    return np.asarray(regresor.predict(X), dtype=float).ravel()


def _smearing(prediccion: np.ndarray, residuos: np.ndarray, inversa, bloque: int = 512) -> np.ndarray:
    """Estimador de smearing de Duan: ŷ_j = promedio_i inversa(predicción_j + residuo_i).
    Se calcula por bloques para no crear una matriz n_pred × n_train completa."""
    salida = np.empty(len(prediccion))
    for inicio in range(0, len(prediccion), bloque):
        trozo = prediccion[inicio:inicio + bloque, None] + residuos[None, :]
        salida[inicio:inicio + bloque] = inversa(trozo).mean(axis=1)
    return salida


class RegresorBDI(RegressorMixin, BaseEstimator):
    """Envuelve cualquier regresor para que prediga en puntos BDI recortados a [0, 63].

    - transformacion=None: entrena sobre y sin transformar (variantes nativo y Poisson).
    - transformacion="sqrt" o "log1p": entrena sobre la y transformada y, al predecir,
      invierte la transformación con la corrección de smearing de Duan. Los residuos se
      calculan en la escala transformada con los datos de entrenamiento del propio fit
      (en la validación cruzada, solo con los folds de entrenamiento).

    Con smearing=False se invierte sin corrección (solo para mostrar el sesgo de retransformación).
    """

    def __init__(self, regresor=None, transformacion=None, smearing=True, minimo=BDI_MIN, maximo=BDI_MAX):
        self.regresor = regresor
        self.transformacion = transformacion
        self.smearing = smearing
        self.minimo = minimo
        self.maximo = maximo

    def fit(self, X, y):
        if self.transformacion is not None and self.transformacion not in TRANSFORMACIONES:
            raise ValueError(f"transformacion debe ser None, 'sqrt' o 'log1p'; se recibió {self.transformacion!r}.")
        y = np.asarray(y, dtype=float).ravel()
        regresor = clone(self.regresor) if self.regresor is not None else LinearRegression()
        if self.transformacion is None:
            self.regresor_ = regresor.fit(X, y)
            self.residuos_ = None
        else:
            directa, _ = TRANSFORMACIONES[self.transformacion]
            z = directa(y)
            self.regresor_ = regresor.fit(X, z)
            self.residuos_ = z - _predecir_base(self.regresor_, X)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "regresor_")
        prediccion = _predecir_base(self.regresor_, X)
        if self.transformacion is not None:
            _, inversa = TRANSFORMACIONES[self.transformacion]
            if self.smearing:
                prediccion = _smearing(prediccion, self.residuos_, inversa)
            else:
                prediccion = inversa(prediccion)
        return np.clip(prediccion, self.minimo, self.maximo)


def ajustar_binomial_negativa(y, exog, metodo: str = "bfgs", max_iter: int = 1000):
    """Ajusta una binomial negativa NB2 con statsmodels (alpha por máxima verosimilitud).

    exog debe incluir la constante. Devuelve (resultado, convergio). Los avisos de
    convergencia se capturan para no inundar las salidas del CV; se reflejan en convergio.
    """
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        resultado = NegativeBinomial(y, exog, loglike_method="nb2").fit(method=metodo, maxiter=max_iter, disp=0)
    convergio = bool(resultado.mle_retvals.get("converged", True)) and not any(
        issubclass(aviso.category, ConvergenceWarning) for aviso in avisos
    )
    return resultado, convergio


class RegresorBinomialNegativa(RegressorMixin, BaseEstimator):
    """GLM binomial negativa (NB2, enlace log) de statsmodels con interfaz de scikit-learn.

    Recibe la salida del ColumnTransformer (array) y le agrega la constante. Guarda solo los
    coeficientes (no el objeto de statsmodels con los datos), así el .joblib queda pequeño.
    Sin hiperparámetros que ajustar: alpha se estima por máxima verosimilitud en cada fit.
    """

    def __init__(self, metodo="bfgs", max_iter=1000):
        self.metodo = metodo
        self.max_iter = max_iter

    def fit(self, X, y):
        X = _a_matriz(X)
        exog = sm.add_constant(X, has_constant="add")
        resultado, self.convergio_ = ajustar_binomial_negativa(
            np.asarray(y, dtype=float).ravel(), exog, self.metodo, self.max_iter,
        )
        parametros = np.asarray(resultado.params, dtype=float)   # [constante, coeficientes..., alpha]
        self.intercept_ = parametros[0]
        self.coef_ = parametros[1:-1]
        self.alpha_ = parametros[-1]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        eta = self.intercept_ + _a_matriz(X) @ self.coef_
        return np.exp(np.clip(eta, -30, 30))


# ---------------------------------------------------------------------------
# Modelos, variantes y espacios de búsqueda
# ---------------------------------------------------------------------------

def variantes_modelo(clave: str) -> list[str]:
    """Variantes del target que aplican a cada modelo."""
    if clave in MODELOS_CONTEO:
        return ["nativo"]           # ya modelan conteos: transformar su target no tiene sentido
    if clave in MODELOS_ARBOLES:
        return ["nativo", "sqrt", "log1p", "poisson"]
    return ["nativo", "sqrt", "log1p"]


def crear_regresor(clave: str, variante: str = "nativo", seed: int = SEED):
    """Estimador base de un modelo. Con variante="poisson" (solo árboles) usa la pérdida Poisson.
    Los modelos usan 1 hilo (salvo CatBoost); la paralelización se hace en la búsqueda."""
    poisson = variante == "poisson"
    if clave == "lineal":
        return LinearRegression()
    if clave == "ridge":
        return Ridge()
    if clave == "lasso":
        return Lasso(max_iter=10000, random_state=seed)
    if clave == "elasticnet":
        return ElasticNet(max_iter=10000, random_state=seed)
    if clave == "poisson":
        return PoissonRegressor(max_iter=1000)
    if clave == "binomial_negativa":
        return RegresorBinomialNegativa()
    if clave == "knn":
        return KNeighborsRegressor()
    if clave == "svr":
        return SVR(kernel="rbf", cache_size=1000)
    if clave == "random_forest":
        return RandomForestRegressor(criterion="poisson" if poisson else "squared_error", random_state=seed, n_jobs=1)
    if clave == "xgboost":
        return XGBRegressor(
            objective="count:poisson" if poisson else "reg:squarederror",
            tree_method="hist", random_state=seed, n_jobs=1,
        )
    if clave == "lightgbm":
        # subsample_freq=1: sin él, LightGBM ignora subsample
        return LGBMRegressor(
            objective="poisson" if poisson else "regression",
            subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1,
        )
    if clave == "catboost":
        # CPU siempre: con ~4k filas la GPU es más lenta. allow_writing_files=False evita catboost_info/
        return CatBoostRegressor(
            loss_function="Poisson" if poisson else "RMSE", boosting_type="Plain", task_type="CPU",
            random_seed=seed, thread_count=-1, verbose=0, allow_writing_files=False,
        )
    raise ValueError(f"Clave '{clave}' no válida. Opciones: {', '.join(NOMBRES_REG)}")


def dispositivo_reg(clave: str) -> str:
    """"GPU" si el modelo entrena en GPU en esta máquina; "CPU" en otro caso.
    CatBoost se fuerza a CPU en la regresión (con ~4k filas la GPU es más lenta)."""
    if clave == "catboost":
        return "CPU"
    return dispositivo_modelo(clave)


def n_jobs_reg(clave: str) -> int:
    """Procesos de la búsqueda y del CV: 1 si el modelo usa la GPU o es CatBoost (que ya usa
    todos los hilos con thread_count=-1); todos los núcleos en otro caso."""
    return 1 if clave == "catboost" or dispositivo_reg(clave) == "GPU" else -1


def resumen_dispositivos_reg() -> pd.DataFrame:
    """Tabla con el dispositivo que usará cada modelo de regresión en esta máquina."""
    return pd.DataFrame(
        {"Dispositivo": [dispositivo_reg(c) for c in NOMBRES_REG], "n_jobs búsqueda": [n_jobs_reg(c) for c in NOMBRES_REG]},
        index=pd.Index(list(NOMBRES_REG.values()), name="Modelo"),
    )


def configurar_dispositivo_reg(pipeline, gpu: bool = True):
    """Ajusta el dispositivo de XGBoost/LightGBM dentro del Pipeline (lo modifica y lo devuelve).

    Cambia tanto el regresor "plantilla" (el que se clona en cada fit) como el ya entrenado
    (regresor_). Con gpu=False lo deja listo para cargarse en una máquina sin GPU.
    CatBoost y los modelos de scikit-learn/statsmodels siempre usan CPU.
    """
    modelo = pipeline.named_steps["modelo"]
    for regresor in (modelo.regresor, getattr(modelo, "regresor_", None)):
        if isinstance(regresor, XGBRegressor):
            usar = gpu and GPU_DISPOSITIVOS["xgboost"]
            regresor.set_params(device=GPU_DISPOSITIVOS["xgboost"] if usar else "cpu", tree_method="hist")
        elif isinstance(regresor, LGBMRegressor):
            usar = gpu and GPU_DISPOSITIVOS["lightgbm"]
            regresor.set_params(device=GPU_DISPOSITIVOS["lightgbm"] if usar else "cpu")
    return pipeline


def crear_pipeline_reg(clave: str, variante: str = "nativo", seed: int = SEED):
    """Pipeline preprocesamiento → RegresorBDI(regresor, transformación), listo para la GPU si aplica."""
    if variante not in variantes_modelo(clave):
        raise ValueError(f"La variante '{variante}' no aplica a '{clave}'.")
    transformacion = variante if variante in TRANSFORMACIONES else None
    modelo = RegresorBDI(regresor=crear_regresor(clave, variante, seed), transformacion=transformacion)
    return configurar_dispositivo_reg(crear_pipeline_modelo(modelo), gpu=True)


def crear_benchmark_reg() -> object:
    """Regresión lineal benchmark del EDA (LinearRegression, target nativo, sin tuning), ahora con
    los 6 predictores. Sus predicciones también se recortan a [0, 63]."""
    return crear_pipeline_modelo(RegresorBDI(regresor=LinearRegression()))


# Espacios de búsqueda sin prefijo. Los árboles usan los mismos rangos que en clasificación.
ESPACIOS_REG = {
    "ridge": {"alpha": loguniform(1e-3, 1e3)},
    "lasso": {"alpha": loguniform(1e-4, 1e1)},
    "elasticnet": {"alpha": loguniform(1e-4, 1e1), "l1_ratio": uniform(0.1, 0.8)},   # l1_ratio en [0.1, 0.9]
    "poisson": {"alpha": loguniform(1e-4, 1e1)},
    "knn": {"n_neighbors": randint(15, 301), "weights": ["uniform", "distance"], "p": [1, 2]},
    # Lista de dos espacios: gamma="scale" se prueba en ~la mitad de las combinaciones
    "svr": [
        {"C": loguniform(1e-1, 1e2), "epsilon": loguniform(1e-2, 5), "gamma": ["scale"]},
        {"C": loguniform(1e-1, 1e2), "epsilon": loguniform(1e-2, 5), "gamma": loguniform(1e-3, 1e0)},
    ],
    **{
        clave: {nombre.removeprefix("modelo__"): valores for nombre, valores in ESPACIOS_CLF[clave].items()}
        for clave in MODELOS_ARBOLES
    },
}

# Iteraciones de RandomizedSearchCV (la lineal y la binomial negativa no se ajustan)
N_ITER_REG = {
    "lineal": 0, "ridge": 10, "lasso": 10, "elasticnet": 10, "poisson": 10, "binomial_negativa": 0,
    "knn": 20, "svr": 20, "random_forest": 25, "xgboost": 25, "lightgbm": 25, "catboost": 25,
}

PREFIJO = "modelo__regresor__"   # los hiperparámetros viven en Pipeline → RegresorBDI → regresor


def espacio_pipeline(clave: str):
    """Espacio de búsqueda con el prefijo del Pipeline, o None si el modelo no se ajusta."""
    espacio = ESPACIOS_REG.get(clave)
    if espacio is None:
        return None
    subespacios = espacio if isinstance(espacio, list) else [espacio]
    con_prefijo = [{PREFIJO + nombre: valores for nombre, valores in sub.items()} for sub in subespacios]
    return con_prefijo if isinstance(espacio, list) else con_prefijo[0]


def describir_espacio_reg(clave: str) -> pd.DataFrame:
    """Espacio de búsqueda de un modelo como tabla legible."""
    espacio = ESPACIOS_REG.get(clave) or {}
    subespacios = espacio if isinstance(espacio, list) else [espacio]
    descripciones = {}
    for subespacio in subespacios:
        for nombre, valores in subespacio.items():
            texto = _describir_valores(valores)
            lista = descripciones.setdefault(nombre, [])
            if texto not in lista:
                lista.append(texto)
    return pd.DataFrame(
        {"Valores": [" o ".join(textos) for textos in descripciones.values()]},
        index=pd.Index(list(descripciones), name="Hiperparámetro"),
    )


def _limpiar_parametros_reg(parametros: dict) -> dict:
    """Quita el prefijo del Pipeline y pasa los tipos numpy a tipos nativos (para el JSON)."""
    return {
        nombre.removeprefix(PREFIJO): (valor.item() if isinstance(valor, np.generic) else valor)
        for nombre, valor in parametros.items()
    }


# ---------------------------------------------------------------------------
# Fase 1: variantes del target por modelo
# ---------------------------------------------------------------------------

def _fila_cv_reg(resultados: dict, indice: int | None = None) -> dict:
    """Media y DE de RMSE, MAE y R² desde cross_validate (indice=None) o desde cv_results_ de
    RandomizedSearchCV (indice = combinación elegida). RMSE y MAE vienen con signo negativo."""
    fila = {}
    for metrica, nombre in METRICAS_REG.items():
        if indice is None:
            valores = resultados[f"test_{metrica}"]
            media, de = valores.mean(), valores.std()
        else:
            media, de = resultados[f"mean_test_{metrica}"][indice], resultados[f"std_test_{metrica}"][indice]
        fila[f"{nombre} media"] = float(-media if metrica != "r2" else media)
        fila[f"{nombre} DE"] = float(de)
    return fila


def ejecutar_fase1_reg(clave: str, X_train, y_train, cv, seed: int = SEED) -> tuple[pd.DataFrame, dict, dict]:
    """Tuning y evaluación en CV de un modelo con cada variante del target.

    Por variante, RandomizedSearchCV solo sobre el train (mismos folds, scoring
    neg_root_mean_squared_error). Con scoring múltiple y refit="rmse", cv_results_ trae el
    RMSE, el MAE y el R² de la combinación elegida en los mismos folds, sin repetir el CV.
    Los modelos sin tuning se evalúan con cross_validate. En ambos casos el Pipeline final
    queda reentrenado con todo el train.

    Devuelve (tabla, pipelines, hiperparametros), indexados por variante.
    """
    espacio = espacio_pipeline(clave)
    n_jobs = n_jobs_reg(clave)

    filas, pipelines, hiperparametros = {}, {}, {}
    for variante in variantes_modelo(clave):
        inicio = time.perf_counter()
        pipeline = crear_pipeline_reg(clave, variante, seed=seed)
        if espacio is None:
            resultados = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORERS_REG, n_jobs=n_jobs)
            fila = _fila_cv_reg(resultados)
            pipelines[variante] = pipeline.fit(X_train, y_train)
            hiperparametros[variante] = {}
        else:
            busqueda = RandomizedSearchCV(
                pipeline, espacio, n_iter=N_ITER_REG[clave], scoring=SCORERS_REG, refit="rmse",
                cv=cv, random_state=seed, n_jobs=n_jobs,
            )
            busqueda.fit(X_train, y_train)
            fila = _fila_cv_reg(busqueda.cv_results_, indice=busqueda.best_index_)
            pipelines[variante] = busqueda.best_estimator_
            hiperparametros[variante] = _limpiar_parametros_reg(busqueda.best_params_)
        fila["Tiempo (s)"] = time.perf_counter() - inicio
        filas[variante] = fila
        print(f"✓ {VARIANTES[variante]:<18} RMSE = {fila['RMSE media']:.4f} ± {fila['RMSE DE']:.4f}  "
              f"R² = {fila['R² media']:.4f}  ({fila['Tiempo (s)']:.0f} s)")

    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Variante"
    return tabla, pipelines, hiperparametros


def elegir_mas_simple(tabla: pd.DataFrame, simplicidad: dict, nombres: dict,
                      columna_media: str = "RMSE media", columna_de: str = "RMSE DE") -> tuple[str, str]:
    """Elige la opción con menor RMSE medio de CV con la REGLA DE EMPATE: si una opción más
    simple queda a menos de una DE de CV de la ganadora, se elige la más simple (y, entre las
    igual de simples, la de menor RMSE). Devuelve (opción elegida, explicación en texto)."""
    mejor = tabla[columna_media].idxmin()
    media, de = tabla.loc[mejor, columna_media], tabla.loc[mejor, columna_de]
    candidatas = [
        opcion for opcion in tabla.index
        if simplicidad[opcion] < simplicidad[mejor] and tabla.loc[opcion, columna_media] - media < de
    ]
    if not candidatas:
        return mejor, (
            f"Se elige '{nombres[mejor]}': tiene el menor RMSE medio de CV ({media:.4f} ± {de:.4f}) y "
            f"ninguna opción más simple queda a menos de una desviación estándar ({de:.4f})."
        )
    elegida = min(candidatas, key=lambda opcion: (simplicidad[opcion], tabla.loc[opcion, columna_media]))
    diferencia = tabla.loc[elegida, columna_media] - media
    return elegida, (
        f"El menor RMSE medio de CV es el de '{nombres[mejor]}' ({media:.4f} ± {de:.4f}), pero "
        f"'{nombres[elegida]}' es más simple y su RMSE ({tabla.loc[elegida, columna_media]:.4f}) está a "
        f"solo {diferencia:.4f} puntos, menos que la desviación estándar de CV de la ganadora ({de:.4f}). "
        f"Por la regla de empate se elige '{nombres[elegida]}'."
    )


def elegir_variante(tabla: pd.DataFrame) -> tuple[str, str]:
    """Fase 1: variante ganadora (orden de simplicidad nativo < sqrt < log1p < Poisson)."""
    return elegir_mas_simple(tabla, SIMPLICIDAD_VARIANTES, VARIANTES)


def elegir_modelo_reg(tabla: pd.DataFrame) -> tuple[str, str]:
    """Fase 2: mejor modelo (orden lineales < Poisson < binomial negativa < KNN < SVR < RF < boosting)."""
    return elegir_mas_simple(tabla, SIMPLICIDAD_MODELOS, NOMBRES_REG)


def formatear_tabla_cv_reg(tabla_cv: pd.DataFrame, decimales: int = 3) -> pd.DataFrame:
    """Versión para mostrar: cada métrica como "media ± DE", más las columnas Δ que tenga."""
    vista = pd.DataFrame(index=tabla_cv.index)
    for nombre in METRICAS_REG.values():
        vista[nombre] = [
            f"{media:.{decimales}f} ± {de:.{decimales}f}"
            for media, de in zip(tabla_cv[f"{nombre} media"], tabla_cv[f"{nombre} DE"])
        ]
    for columna in tabla_cv.columns:
        if columna.startswith("Δ"):
            vista[columna] = tabla_cv[columna].round(decimales)
    return vista


def tabla_hiperparametros_reg(hiperparametros: dict) -> pd.DataFrame:
    """Mejores hiperparámetros por variante: una fila por variante y una columna por hiperparámetro."""
    tabla = pd.DataFrame.from_dict(hiperparametros, orient="index")
    tabla = tabla.map(lambda v: float(f"{v:.4g}") if isinstance(v, float) else v)
    tabla.index = pd.Index([VARIANTES[v] for v in tabla.index], name="Variante")
    return tabla


def evaluar_smearing(tabla: pd.DataFrame, pipelines: dict, X_train, y_train, cv) -> tuple[str | None, pd.DataFrame | None]:
    """Muestra el sesgo de retransformación en la mejor variante transformada (sqrt o log1p).

    Con los hiperparámetros ya elegidos, en cada fold se entrena una vez y se predice el fold
    de validación con y sin la corrección de smearing (predicciones out-of-fold). Solo usa el
    train. Devuelve (variante, tabla) o (None, None) si el modelo no tiene variantes transformadas.
    """
    transformadas = [v for v in tabla.index if v in TRANSFORMACIONES]
    if not transformadas:
        return None, None
    variante = tabla.loc[transformadas, "RMSE media"].idxmin()

    y = np.asarray(y_train, dtype=float)
    con, sin = np.empty(len(y)), np.empty(len(y))
    for indices_train, indices_val in cv.split(X_train, y):
        modelo = configurar_dispositivo_reg(clone(pipelines[variante]), gpu=True)
        modelo.fit(X_train.iloc[indices_train], y[indices_train])
        con[indices_val] = modelo.predict(X_train.iloc[indices_val])
        modelo.set_params(modelo__smearing=False)
        sin[indices_val] = modelo.predict(X_train.iloc[indices_val])

    filas = {"BDI real": {"Media": y.mean(), "Sesgo medio": 0.0}}
    for nombre, prediccion in [("Sin smearing", sin), ("Con smearing", con)]:
        filas[nombre] = {
            "Media": prediccion.mean(),
            "Sesgo medio": prediccion.mean() - y.mean(),
            "RMSE": root_mean_squared_error(y, prediccion),
            "MAE": mean_absolute_error(y, prediccion),
            "R²": r2_score(y, prediccion),
        }
    resultado = pd.DataFrame.from_dict(filas, orient="index")
    resultado.index.name = f"Out-of-fold ({VARIANTES[variante]})"
    return variante, resultado


def guardar_fase1_reg(clave: str, tabla: pd.DataFrame, variante: str, pipeline, hiperparametros: dict,
                      explicacion: str, ruta: Path = RUTA_FASE1_REG) -> dict:
    """Guarda en regresion/fase1/<clave>/: resultados_cv.csv, ganador.json y <clave>_<variante>.joblib.
    El modelo se deja configurado para CPU antes de guardarlo."""
    carpeta = Path(ruta) / clave
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "resultados_cv.csv")

    archivo_modelo = f"{clave}_{variante}.joblib"
    joblib.dump(configurar_dispositivo_reg(pipeline, gpu=False), carpeta / archivo_modelo)

    ganador = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "modelo": clave,
        "nombre": NOMBRES_REG[clave],
        "variante": variante,
        "transformacion": variante if variante in TRANSFORMACIONES else None,
        "usa_smearing": variante in TRANSFORMACIONES,
        "hiperparametros": hiperparametros,
        "rmse_media": float(tabla.loc[variante, "RMSE media"]),
        "rmse_de": float(tabla.loc[variante, "RMSE DE"]),
        "mae_media": float(tabla.loc[variante, "MAE media"]),
        "r2_media": float(tabla.loc[variante, "R² media"]),
        "explicacion": explicacion,
        "dispositivo_entrenamiento": dispositivo_reg(clave),
        "archivo": archivo_modelo,
        "seed": SEED,
    }
    with open(carpeta / "ganador.json", "w", encoding="utf-8") as archivo:
        json.dump(ganador, archivo, ensure_ascii=False, indent=2)
    return ganador


def cargar_ganador_reg(clave: str, ruta: Path = RUTA_FASE1_REG) -> tuple[object, dict]:
    """Carga el Pipeline ganador de la fase 1 de un modelo y su ganador.json."""
    carpeta = Path(ruta) / clave
    if not (carpeta / "ganador.json").exists():
        raise FileNotFoundError(
            f"No existe {carpeta / 'ganador.json'}. Ejecuta primero notebooks/reg_{clave}.ipynb."
        )
    with open(carpeta / "ganador.json", encoding="utf-8") as archivo:
        ganador = json.load(archivo)
    return joblib.load(carpeta / ganador["archivo"]), ganador


# ---------------------------------------------------------------------------
# Fase 2: comparación, test y experimento con el corte clínico
# ---------------------------------------------------------------------------

def evaluar_cv_reg(pipelines: dict, X_train, y_train, cv) -> pd.DataFrame:
    """CV (mismos folds) de cada Pipeline con sus hiperparámetros ya elegidos (sin tuning).
    Una fila por clave con media y DE de RMSE, MAE y R², y la diferencia frente al benchmark."""
    filas = {}
    for clave, pipeline in pipelines.items():
        modelo = configurar_dispositivo_reg(clone(pipeline), gpu=True)
        resultados = cross_validate(modelo, X_train, y_train, cv=cv, scoring=SCORERS_REG, n_jobs=n_jobs_reg(clave))
        filas[clave] = _fila_cv_reg(resultados)
        print(f"✓ {NOMBRES_FASE2_REG[clave]:<24} RMSE = {filas[clave]['RMSE media']:.4f} ± {filas[clave]['RMSE DE']:.4f}")
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Modelo"
    for nombre in METRICAS_REG.values():
        tabla[f"Δ {nombre} vs benchmark"] = tabla[f"{nombre} media"] - tabla.loc[CLAVE_BENCHMARK, f"{nombre} media"]
    return tabla.sort_values("RMSE media")


def metricas_reg(y, prediccion) -> dict:
    """RMSE, MAE y R² en puntos BDI."""
    return {
        "RMSE": root_mean_squared_error(y, prediccion),
        "MAE": mean_absolute_error(y, prediccion),
        "R²": r2_score(y, prediccion),
    }


def evaluar_test_reg(y_test, predicciones: dict, n_replicas: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """RMSE, MAE y R² en test de cada modelo, con intervalos bootstrap al 95% (percentiles) del
    RMSE y del R², y de su diferencia pareada frente al benchmark.

    En cada réplica se remuestrea el test con reemplazo y se evalúan TODOS los modelos sobre
    las mismas observaciones: si el intervalo de la diferencia no incluye 0, la diferencia
    frente al benchmark es real con un 95% de confianza.
    """
    y = np.asarray(y_test, dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(y), size=(n_replicas, len(y)))
    y_rep = y[indices]
    sct = ((y_rep - y_rep.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)

    replicas = {}
    for clave, prediccion in predicciones.items():
        errores = (y_rep - np.asarray(prediccion, dtype=float)[indices]) ** 2
        replicas[clave] = {"RMSE": np.sqrt(errores.mean(axis=1)), "R²": 1 - errores.sum(axis=1) / sct}

    filas = {}
    for clave, prediccion in predicciones.items():
        fila = metricas_reg(y, prediccion)
        referencia = metricas_reg(y, predicciones[CLAVE_BENCHMARK])
        for metrica in ("RMSE", "R²"):
            valores = replicas[clave][metrica]
            diferencias = valores - replicas[CLAVE_BENCHMARK][metrica]
            fila[f"{metrica} IC 2.5%"], fila[f"{metrica} IC 97.5%"] = np.percentile(valores, [2.5, 97.5])
            fila[f"Δ {metrica} vs benchmark"] = fila[metrica] - referencia[metrica]
            fila[f"Δ {metrica} IC 2.5%"], fila[f"Δ {metrica} IC 97.5%"] = np.percentile(diferencias, [2.5, 97.5])
            fila[f"Δ {metrica} significativa"] = clave != CLAVE_BENCHMARK and (
                fila[f"Δ {metrica} IC 2.5%"] > 0 or fila[f"Δ {metrica} IC 97.5%"] < 0
            )
        filas[clave] = fila
    orden = ["RMSE", "RMSE IC 2.5%", "RMSE IC 97.5%", "MAE", "R²", "R² IC 2.5%", "R² IC 97.5%",
             "Δ RMSE vs benchmark", "Δ RMSE IC 2.5%", "Δ RMSE IC 97.5%", "Δ RMSE significativa",
             "Δ R² vs benchmark", "Δ R² IC 2.5%", "Δ R² IC 97.5%", "Δ R² significativa"]
    tabla = pd.DataFrame.from_dict(filas, orient="index")[orden]
    tabla.index.name = "Modelo"
    return tabla.sort_values("RMSE")


def importancia_permutacion(pipeline, X, y, scoring: str = "neg_root_mean_squared_error",
                            n_repeats: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Permutation importance: cuánto empeora la métrica al desordenar cada predictor.
    Con scoring neg RMSE, la importancia es el aumento del RMSE en puntos BDI."""
    resultado = permutation_importance(pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    tabla = pd.DataFrame(
        {"Importancia media": resultado.importances_mean, "DE": resultado.importances_std},
        index=pd.Index(X.columns, name="Variable"),
    ).sort_values("Importancia media", ascending=False)
    tabla["Ranking"] = np.arange(1, len(tabla) + 1)
    return tabla


def comparar_jerarquias(importancia_reg: pd.DataFrame, importancia_clf: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ranking de variables del regresor y del clasificador lado a lado, con la correlación
    de Spearman entre ambos rankings."""
    tabla = pd.DataFrame({
        "Ranking regresión": importancia_reg["Ranking"],
        "Ranking clasificación": importancia_clf["Ranking"],
        "Importancia regresión (Δ RMSE)": importancia_reg["Importancia media"],
        "Importancia clasificación (Δ PR-AUC)": importancia_clf["Importancia media"],
    }).sort_values("Ranking regresión")
    tabla["Mismo puesto"] = tabla["Ranking regresión"] == tabla["Ranking clasificación"]
    rho = stats.spearmanr(tabla["Ranking regresión"], tabla["Ranking clasificación"]).statistic
    return tabla, float(rho)


def metricas_riesgo(y_binario, puntuacion, umbral: float) -> dict:
    """Clasificación con un umbral sobre una puntuación de riesgo (BDI predicho o probabilidad):
    PR-AUC y ROC-AUC (no dependen del umbral) y métricas de clase con el umbral dado."""
    puntuacion = np.asarray(puntuacion, dtype=float)
    prediccion = (puntuacion >= umbral).astype(int)
    return {
        "Umbral": umbral,
        "PR-AUC": average_precision_score(y_binario, puntuacion),
        "ROC-AUC": roc_auc_score(y_binario, puntuacion),
        "Precision": precision_score(y_binario, prediccion, zero_division=0),
        "Recall": recall_score(y_binario, prediccion),
        "F1": f1_score(y_binario, prediccion),
        "Accuracy": accuracy_score(y_binario, prediccion),
        "Alertas (%)": prediccion.mean() * 100,
    }


def cargar_clasificador_final():
    """Clasificador final (data/processed/modelado/final/) y sus metadatos, o (None, None) con un
    aviso claro si todavía no existe."""
    try:
        return cargar_modelo_final()
    except FileNotFoundError as error:
        print(f"⚠ No se encontró el clasificador final en {RUTA_FINAL_CLF}.\n  {error}\n"
              "  Las celdas que lo comparan con la regresión se omitirán.")
        return None, None


def comparar_auc_bootstrap(y_binario, proba_clasificador, bdi_predicho, n_replicas: int = 1000,
                           seed: int = SEED) -> pd.DataFrame:
    """Intervalos bootstrap pareados de PR-AUC y ROC-AUC del BDI predicho frente al clasificador
    final (que actúa como referencia). Reutiliza intervalos_bootstrap de Modelado.py."""
    tabla = intervalos_bootstrap(
        y_binario, {"benchmark": proba_clasificador, "regresion": bdi_predicho}, n_replicas=n_replicas, seed=seed,
    )
    tabla = tabla.rename(index={"benchmark": "Clasificador final", "regresion": "Regresor (BDI predicho)"})
    return tabla.rename(columns=lambda c: c.replace("vs benchmark", "vs clasificador"))


# ---------------------------------------------------------------------------
# Pruebas de hipótesis (statsmodels, sobre TODO el train)
# ---------------------------------------------------------------------------

def matriz_statsmodels(X_train: pd.DataFrame) -> pd.DataFrame:
    """Predictores preprocesados con el mismo ColumnTransformer (ajustado en el train) más la
    constante, como DataFrame con nombres legibles (las numéricas quedan en DE)."""
    preprocesador = crear_preprocesador().fit(X_train)
    nombres = [nombre.split("__", 1)[1] for nombre in preprocesador.get_feature_names_out()]
    exog = pd.DataFrame(_a_matriz(preprocesador.transform(X_train)), columns=nombres, index=X_train.index)
    return sm.add_constant(exog, has_constant="add")


def _fila_prueba(prueba: str, h0: str, estadistico: float, p_valor: float, si_rechaza: str, si_no_rechaza: str) -> dict:
    p_valor = float(np.squeeze(p_valor))
    return {
        "Prueba": prueba,
        "Hipótesis nula": h0,
        "Estadístico": float(np.squeeze(estadistico)),
        "p-valor": float(np.squeeze(p_valor)),
        f"Conclusión (α = {ALFA})": f"Se rechaza H0: {si_rechaza}" if p_valor < ALFA else f"No se rechaza H0: {si_no_rechaza}",
    }


def _tabla_pruebas(filas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(filas).set_index("Prueba")


def tabla_coeficientes(resultado, excluir: tuple = ()) -> pd.DataFrame:
    """Coeficientes de un modelo de statsmodels con EE, estadístico (t o z), p-valor e IC 95%."""
    ic = resultado.conf_int(alpha=ALFA)
    tabla = pd.DataFrame({
        "Coeficiente": resultado.params,
        "EE": resultado.bse,
        "Estadístico": resultado.tvalues,
        "p-valor": resultado.pvalues,
        "IC 2.5%": ic.iloc[:, 0],
        "IC 97.5%": ic.iloc[:, 1],
    })
    tabla["Significativo"] = tabla["p-valor"] < ALFA
    tabla.index.name = "Variable"
    return tabla.drop(index=list(excluir), errors="ignore")


def _transformar_y(y: pd.Series, transformacion: str | None) -> pd.Series:
    y = pd.Series(y, dtype=float)
    if transformacion is None:
        return y
    directa, _ = TRANSFORMACIONES[transformacion]
    return directa(y).rename(f"{transformacion}({y.name})")


def pruebas_ols(X_train: pd.DataFrame, y_train: pd.Series, transformacion: str | None = None) -> dict:
    """Supuestos y significancia de la regresión lineal (OLS) sobre todo el train.

    Reutiliza diagnosticar_regresion_lineal (LinearRegressionDiagnostics.py) para Shapiro-Wilk,
    Jarque-Bera, Breusch-Pagan, VIF e influencia: esas pruebas no cambian al estandarizar los
    predictores, porque los residuos y el leverage son los mismos. Complementa con la prueba
    RESET de Ramsey, las pruebas t, la F global, el R² ajustado y, si hay heterocedasticidad,
    los errores estándar robustos HC3.

    Con transformacion="sqrt" o "log1p" todo se calcula en esa escala (donde se asumen los supuestos).
    """
    y = _transformar_y(y_train, transformacion)
    exog = matriz_statsmodels(X_train)
    resultado = sm.OLS(y, exog).fit()

    vif, supuestos, influencia, _ = diagnosticar_regresion_lineal(X_train, y, PREDICTORES_NUMERICOS, PREDICTORES_CATEGORICOS[0])
    supuestos = supuestos.set_index("Supuesto")
    reset = linear_reset(resultado, power=3, test_type="fitted", use_f=True)

    def _de_diagnostico(nombre):
        return supuestos.loc[nombre, "Estadistico"], supuestos.loc[nombre, "p-value"]

    filas = [
        _fila_prueba("Shapiro-Wilk", "Los residuos son normales", *_de_diagnostico("Normalidad (Shapiro-Wilk)"),
                     "los residuos no son normales", "compatible con normalidad"),
        _fila_prueba("Jarque-Bera", "Asimetría 0 y curtosis 3 (normalidad)", *_de_diagnostico("Normalidad (Jarque-Bera)"),
                     "los residuos no son normales", "compatible con normalidad"),
        _fila_prueba("Breusch-Pagan", "Varianza de los residuos constante (homocedasticidad)",
                     *_de_diagnostico("Homocedasticidad (Breusch-Pagan)"),
                     "hay heterocedasticidad", "compatible con homocedasticidad"),
        _fila_prueba("RESET de Ramsey (ŷ², ŷ³)", "La forma lineal está bien especificada",
                     reset.statistic, reset.pvalue,
                     "falta no linealidad o hay variables omitidas", "sin evidencia de mala especificación"),
        _fila_prueba("F global", "Todos los coeficientes (salvo la constante) son 0",
                     resultado.fvalue, resultado.f_pvalue,
                     "al menos un predictor explica el BDI", "ningún predictor explica el BDI"),
    ]

    heterocedasticidad = supuestos.loc["Homocedasticidad (Breusch-Pagan)", "p-value"] < ALFA
    coeficientes = tabla_coeficientes(resultado)
    coeficientes_hc3 = None
    if heterocedasticidad:
        robusto = sm.OLS(y, exog).fit(cov_type="HC3")
        coeficientes_hc3 = pd.DataFrame({
            "Coeficiente": resultado.params,
            "EE OLS": resultado.bse,
            "EE HC3": robusto.bse,
            "p-valor OLS": resultado.pvalues,
            "p-valor HC3": robusto.pvalues,
        })
        coeficientes_hc3["Cambia la conclusión"] = (coeficientes_hc3["p-valor OLS"] < ALFA) != (coeficientes_hc3["p-valor HC3"] < ALFA)
        coeficientes_hc3.index.name = "Variable"

    vif = vif.set_index("Variable")
    vif["Umbral"] = 5
    ajuste = pd.Series({
        "Escala del target": transformacion or "nativa (puntos BDI)",
        "n": int(resultado.nobs),
        "Predictores": int(resultado.df_model),
        "R²": round(resultado.rsquared, 4),
        "R² ajustado": round(resultado.rsquared_adj, 4),
        "F global": round(resultado.fvalue, 2),
        "p-valor F": resultado.f_pvalue,
    }, name="OLS").to_frame()

    return {
        "modelo": resultado,
        "pruebas": _tabla_pruebas(filas),
        "coeficientes": coeficientes,
        "coeficientes_hc3": coeficientes_hc3,
        "vif": vif,
        "influencia": influencia.set_index("Indicador"),
        "ajuste": ajuste,
        "heterocedasticidad": heterocedasticidad,
    }


def graficar_residuos_ols(resultado, titulo: str):
    """Gráfico Q-Q de los residuos estudentizados y residuos vs. ajustados (con suavizado LOWESS)."""
    residuos = resultado.get_influence().resid_studentized_internal
    ajustados = np.asarray(resultado.fittedvalues)
    fig, (ax_qq, ax_res) = plt.subplots(1, 2, figsize=(14, 5.5))

    (teoricos, ordenados), (pendiente, intercepto, _) = stats.probplot(residuos, dist="norm")
    ax_qq.scatter(teoricos, ordenados, s=10, alpha=0.4, color=NEUTRO, edgecolors="none")
    ax_qq.plot(teoricos, pendiente * teoricos + intercepto, color=DEPRIMIDO, linewidth=1.8, label="Normal teórica")
    ax_qq.set_title("Gráfico Q-Q de los residuos", fontweight="bold")
    ax_qq.set_xlabel("Cuantiles teóricos (normal)")
    ax_qq.set_ylabel("Residuos estudentizados")
    ax_qq.legend(loc="upper left")

    suavizado = sm.nonparametric.lowess(residuos, ajustados, frac=0.3)
    ax_res.scatter(ajustados, residuos, s=10, alpha=0.35, color=NEUTRO, edgecolors="none")
    ax_res.axhline(0, color=TEXTO, linewidth=1)
    ax_res.plot(suavizado[:, 0], suavizado[:, 1], color=DEPRIMIDO, linewidth=2, label="LOWESS")
    ax_res.set_title("Residuos vs. ajustados", fontweight="bold")
    ax_res.set_xlabel("Valores ajustados")
    ax_res.set_ylabel("Residuos estudentizados")
    ax_res.legend(loc="upper left")

    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    return fig


def ajustar_poisson_glm(X_train: pd.DataFrame, y_train: pd.Series, escala=None):
    """GLM Poisson (enlace log) de statsmodels sobre todo el train. Con escala="X2" la dispersión
    se estima con el χ² de Pearson (cuasi-Poisson): mismos coeficientes, EE reescalados."""
    exog = matriz_statsmodels(X_train)
    return sm.GLM(pd.Series(y_train, dtype=float), exog, family=sm.families.Poisson()).fit(scale=escala)


def razones_tasas(resultado, excluir: tuple = ("const", "alpha")) -> pd.DataFrame:
    """Coeficientes de un modelo de conteo con enlace log como razones de tasas exp(β), con IC 95%."""
    ic = resultado.conf_int(alpha=ALFA)
    tabla = pd.DataFrame({
        "Razón de tasas": np.exp(resultado.params),
        "IC 2.5%": np.exp(ic.iloc[:, 0]),
        "IC 97.5%": np.exp(ic.iloc[:, 1]),
        "p-valor": resultado.pvalues,
    })
    tabla["Cambio en el BDI esperado (%)"] = (tabla["Razón de tasas"] - 1) * 100
    tabla.index.name = "Variable"
    return tabla.drop(index=list(excluir), errors="ignore")


def interpretar_razones(tabla: pd.DataFrame) -> str:
    """Interpretación en texto (markdown) de una tabla de razones de tasas."""
    lineas = []
    for variable, fila in tabla.iterrows():
        cambio = fila["Cambio en el BDI esperado (%)"]
        ic = f"IC 95%: {(fila['IC 2.5%'] - 1) * 100:+.1f}% a {(fila['IC 97.5%'] - 1) * 100:+.1f}%"
        significancia = "significativo" if fila["p-valor"] < ALFA else "no significativo"
        if variable in PREDICTORES_NUMERICOS:
            lineas.append(f"- Por cada desviación estándar más de **{variable}**, el BDI esperado cambia en "
                          f"**{cambio:+.1f}%** ({ic}; {significancia}), con las demás variables fijas.")
        else:
            lineas.append(f"- **{variable}**: frente a la categoría de referencia, el BDI esperado cambia en "
                          f"**{cambio:+.1f}%** ({ic}; {significancia}), con las demás variables fijas.")
    return "\n".join(lineas)


def pruebas_poisson(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Significancia, bondad de ajuste y equidispersión del GLM Poisson sobre todo el train.

    - Wald (z) de cada coeficiente y razón de verosimilitud frente al modelo nulo.
    - Desviación y χ² de Pearson frente a χ²(n - p) (H0: el modelo ajusta bien).
    - Razón de dispersión (χ² de Pearson / g.l.) y prueba de Cameron-Trivedi: regresión
      auxiliar sin constante ((y - μ)² - y) / μ = α·μ + ε, con H0: α = 0 frente a α > 0 (NB2).
    """
    y = pd.Series(y_train, dtype=float)
    resultado = ajustar_poisson_glm(X_train, y)
    mu = np.asarray(resultado.mu)

    lr = 2 * (resultado.llf - resultado.llnull)
    gl_nulo = int(resultado.df_model)
    gl = resultado.df_resid
    dispersion = resultado.pearson_chi2 / gl

    auxiliar = sm.OLS(((y.to_numpy() - mu) ** 2 - y.to_numpy()) / mu, mu).fit()
    t_ct = float(auxiliar.tvalues[0])
    p_ct = float(stats.t.sf(t_ct, auxiliar.df_resid))   # unilateral: sobredispersión (α > 0)

    filas = [
        _fila_prueba(f"Razón de verosimilitud vs. nulo (χ², {gl_nulo} g.l.)", "Todos los coeficientes (salvo la constante) son 0",
                     lr, stats.chi2.sf(lr, gl_nulo), "al menos un predictor explica el BDI", "ningún predictor explica el BDI"),
        _fila_prueba(f"Desviación (χ², {gl:.0f} g.l.)", "El modelo Poisson ajusta bien los datos",
                     resultado.deviance, stats.chi2.sf(resultado.deviance, gl), "el Poisson no ajusta bien", "ajuste aceptable"),
        _fila_prueba(f"χ² de Pearson ({gl:.0f} g.l.)", "El modelo Poisson ajusta bien los datos",
                     resultado.pearson_chi2, stats.chi2.sf(resultado.pearson_chi2, gl), "el Poisson no ajusta bien", "ajuste aceptable"),
        _fila_prueba("Cameron-Trivedi (t, unilateral)", "Equidispersión: Var(y) = E(y) (α = 0)",
                     t_ct, p_ct, f"hay sobredispersión (α estimado = {auxiliar.params[0]:.3f})", "compatible con equidispersión"),
    ]

    cuasi = ajustar_poisson_glm(X_train, y, escala="X2")
    coeficientes_cuasi = pd.DataFrame({
        "Coeficiente": resultado.params,
        "EE Poisson": resultado.bse,
        "EE cuasi-Poisson": cuasi.bse,
        "p-valor Poisson": resultado.pvalues,
        "p-valor cuasi-Poisson": cuasi.pvalues,
    })
    coeficientes_cuasi["Cambia la conclusión"] = (coeficientes_cuasi["p-valor Poisson"] < ALFA) != (coeficientes_cuasi["p-valor cuasi-Poisson"] < ALFA)
    coeficientes_cuasi.index.name = "Variable"

    equidispersion = pd.Series({
        "Media del BDI (train)": round(y.mean(), 2),
        "Varianza del BDI (train)": round(y.var(), 2),
        "χ² de Pearson / g.l. (razón de dispersión)": round(dispersion, 3),
        "α de Cameron-Trivedi": round(float(auxiliar.params[0]), 4),
        "Factor de los EE (√ dispersión)": round(np.sqrt(dispersion), 3),
    }, name="Equidispersión").to_frame()

    return {
        "modelo": resultado,
        "pruebas": _tabla_pruebas(filas),
        "coeficientes": tabla_coeficientes(resultado),
        "razones_tasas": razones_tasas(resultado),
        "equidispersion": equidispersion,
        "coeficientes_cuasi": coeficientes_cuasi,
        "dispersion": float(dispersion),
    }


def _criterios_informacion(resultado, n: int) -> dict:
    """Log-verosimilitud, AIC y BIC calculados igual para ambos modelos (k = parámetros estimados,
    incluido alpha en la binomial negativa)."""
    k = len(resultado.params)
    return {
        "Log-verosimilitud": float(resultado.llf),
        "Parámetros": k,
        "AIC": float(-2 * resultado.llf + 2 * k),
        "BIC": float(-2 * resultado.llf + k * np.log(n)),
    }


def pruebas_binomial_negativa(X_train: pd.DataFrame, y_train: pd.Series, resultado_poisson=None) -> dict:
    """Pruebas del GLM binomial negativa NB2 (statsmodels) sobre todo el train, comparado con Poisson.

    - alpha con su IC 95% (Wald, error estándar por método delta de statsmodels).
    - LR NB vs. Poisson (H0: alpha = 0). alpha = 0 está en el borde del espacio de parámetros,
      así que el estadístico se distribuye como una mezcla 50:50 de χ²(0) y χ²(1): el p-valor
      correcto es la mitad del p-valor χ²(1).
    - Wald (z) de cada coeficiente y LR frente al modelo nulo (solo constante y alpha).
    - Razones de tasas comparadas con las de Poisson, y log-verosimilitud, AIC y BIC de ambos.
    """
    y = pd.Series(y_train, dtype=float)
    exog = matriz_statsmodels(X_train)
    resultado, convergio = ajustar_binomial_negativa(y, exog)
    if resultado_poisson is None:
        resultado_poisson = ajustar_poisson_glm(X_train, y)

    alpha = float(resultado.params["alpha"])
    ic_alpha = resultado.conf_int(alpha=ALFA).loc["alpha"]

    lr_poisson = max(2 * (resultado.llf - resultado_poisson.llf), 0.0)
    p_chi2 = stats.chi2.sf(lr_poisson, 1)
    lr_nulo = 2 * (resultado.llf - resultado.llnull)
    gl_nulo = len(resultado.params) - 2      # sin constante ni alpha

    filas = [
        _fila_prueba("LR NB vs. Poisson (p corregido = p χ²(1) / 2)", "alpha = 0 (el Poisson basta)",
                     lr_poisson, p_chi2 / 2, "hay sobredispersión: la NB mejora al Poisson", "el Poisson basta"),
        _fila_prueba(f"Razón de verosimilitud vs. nulo (χ², {gl_nulo} g.l.)", "Todos los coeficientes (salvo constante y alpha) son 0",
                     lr_nulo, stats.chi2.sf(lr_nulo, gl_nulo), "al menos un predictor explica el BDI", "ningún predictor explica el BDI"),
    ]
    tabla_lr = pd.DataFrame({
        "LR": [lr_poisson],
        "p-valor χ²(1) sin corregir": [p_chi2],
        "p-valor corregido (borde)": [p_chi2 / 2],
    }, index=pd.Index(["NB vs. Poisson"], name="Prueba"))

    rr_poisson, rr_nb = razones_tasas(resultado_poisson), razones_tasas(resultado)
    comparacion = pd.DataFrame({
        "RR Poisson": rr_poisson["Razón de tasas"],
        "IC Poisson": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(rr_poisson["IC 2.5%"], rr_poisson["IC 97.5%"])],
        "RR NB": rr_nb["Razón de tasas"],
        "IC NB": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(rr_nb["IC 2.5%"], rr_nb["IC 97.5%"])],
    })
    comparacion["Mismo signo"] = np.sign(np.log(comparacion["RR Poisson"])) == np.sign(np.log(comparacion["RR NB"]))
    comparacion["Ancho IC NB / Poisson"] = (rr_nb["IC 97.5%"] - rr_nb["IC 2.5%"]) / (rr_poisson["IC 97.5%"] - rr_poisson["IC 2.5%"])
    comparacion.index.name = "Variable"

    n = int(resultado.nobs)
    ajuste = pd.DataFrame({
        "GLM Poisson": _criterios_informacion(resultado_poisson, n),
        "Binomial negativa (NB2)": _criterios_informacion(resultado, n),
    }).T
    ajuste.index.name = "Modelo"

    mejor_aic = ajuste["AIC"].idxmin()
    mejor_bic = ajuste["BIC"].idxmin()
    if p_chi2 / 2 < ALFA and mejor_aic == mejor_bic == "Binomial negativa (NB2)":
        conclusion = (
            f"La binomial negativa es más apropiada para el BDI: alpha = {alpha:.3f} es claramente mayor que 0 "
            f"(LR = {lr_poisson:.1f}, p corregido = {p_chi2 / 2:.2g}) y tiene menor AIC y BIC. La varianza del BDI "
            f"crece más rápido que su media (Var = μ + α·μ²), así que el Poisson subestima los errores estándar."
        )
    elif p_chi2 / 2 >= ALFA:
        conclusion = (
            f"No hay evidencia de sobredispersión (p corregido = {p_chi2 / 2:.2g}): el Poisson, más simple, es suficiente."
        )
    else:
        conclusion = (
            f"La prueba LR favorece la binomial negativa (p corregido = {p_chi2 / 2:.2g}), pero los criterios de "
            f"información no coinciden (menor AIC: {mejor_aic}; menor BIC: {mejor_bic}). Revisar los gráficos de frecuencias."
        )

    resumen_alpha = pd.Series({
        "alpha (NB2)": round(alpha, 4),
        "IC 95% alpha": f"[{ic_alpha.iloc[0]:.4f}, {ic_alpha.iloc[1]:.4f}]",
        "Convergió": convergio,
        "Varianza implícita con μ = media": round(y.mean() + alpha * y.mean() ** 2, 2),
        "Varianza observada": round(y.var(), 2),
    }, name="Dispersión").to_frame()

    return {
        "modelo": resultado,
        "modelo_poisson": resultado_poisson,
        "alpha": resumen_alpha,
        "pruebas": _tabla_pruebas(filas),
        "lr_poisson": tabla_lr,
        "coeficientes": tabla_coeficientes(resultado, excluir=("alpha",)),
        "razones_tasas": rr_nb,
        "comparacion_poisson": comparacion,
        "ajuste": ajuste,
        "conclusion": conclusion,
    }


def probabilidades_conteo(mu, alpha: float | None = None, maximo: int = BDI_MAX) -> np.ndarray:
    """Probabilidad media predicha de cada valor 0..maximo: Poisson(μ) o NB2(μ, alpha)."""
    k = np.arange(maximo + 1)
    mu = np.asarray(mu, dtype=float)[:, None]
    if alpha is None:
        return stats.poisson.pmf(k[None, :], mu).mean(axis=0)
    n = 1 / alpha
    return stats.nbinom.pmf(k[None, :], n, n / (n + mu)).mean(axis=0)


def graficar_frecuencias_conteo(y, modelos: dict, maximo: int = 40):
    """Frecuencia observada de cada valor del BDI frente a la predicha por los modelos de conteo.
    modelos = {nombre: probabilidades de probabilidades_conteo()}."""
    y = np.asarray(y, dtype=int)
    k = np.arange(maximo + 1)
    observada = np.bincount(y, minlength=BDI_MAX + 1)[: maximo + 1] / len(y)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(k, observada, color=NO_DEPRIMIDO, alpha=0.6, label="Observada (train)")
    for (nombre, probabilidades), color, estilo in zip(modelos.items(), [DEPRIMIDO, TEXTO], ["-", "--"]):
        ax.plot(k, probabilidades[: maximo + 1], color=color, linestyle=estilo, linewidth=2, marker="o", markersize=3, label=nombre)
    ax.axvline(CORTE_BDI - 0.5, color=LINEA_CORTE, linestyle="--", linewidth=1, label=f"Corte clínico (BDI = {CORTE_BDI})")
    ax.set_title("Frecuencia de cada puntaje BDI: observada vs. predicha", fontweight="bold")
    ax.set_xlabel("Puntaje BDI-II")
    ax.set_ylabel("Proporción de adolescentes")
    ax.legend(loc="upper right")
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

def graficar_variantes(tabla: pd.DataFrame, nombre_modelo: str):
    """Fase 1: RMSE, MAE y R² de CV (media ± DE) por variante del target."""
    x = np.arange(len(tabla))
    etiquetas = [VARIANTES[v] for v in tabla.index]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, nombre in zip(axes, METRICAS_REG.values()):
        ax.errorbar(x, tabla[f"{nombre} media"], yerr=tabla[f"{nombre} DE"], fmt="o", color=NEUTRO,
                    capsize=6, markersize=9, linewidth=2)
        ax.set_xticks(x, etiquetas, rotation=15)
        ax.set_title(f"{nombre} de CV (media ± DE)", fontweight="bold")
        ax.set_ylabel(nombre if nombre == "R²" else f"{nombre} (puntos BDI)")
    fig.suptitle(f"{nombre_modelo}: variantes del target", fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_intervalos_bootstrap_reg(tabla: pd.DataFrame):
    """RMSE y R² en test con su intervalo bootstrap al 95%; benchmark resaltado."""
    orden = tabla.sort_values("RMSE", ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(orden) + 2), sharey=True)
    for ax, metrica in zip(axes, ["RMSE", "R²"]):
        for posicion, clave in enumerate(orden):
            fila = tabla.loc[clave]
            error = [[fila[metrica] - fila[f"{metrica} IC 2.5%"]], [fila[f"{metrica} IC 97.5%"] - fila[metrica]]]
            ax.errorbar(fila[metrica], posicion, xerr=error, fmt="o", capsize=4, markersize=7,
                        color=TEXTO if clave == CLAVE_BENCHMARK else NEUTRO)
        ax.axvline(tabla.loc[CLAVE_BENCHMARK, metrica], color=TEXTO, linestyle="--", linewidth=1)
        ax.set_yticks(range(len(orden)), [NOMBRES_FASE2_REG[c] for c in orden])
        ax.set_title(f"{metrica} en test (IC bootstrap 95%)", fontweight="bold")
        ax.set_xlabel(metrica if metrica == "R²" else f"{metrica} (puntos BDI)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_predicho_vs_real(y, predicciones: dict):
    """Predicho vs. real (test) de cada modelo en paneles lado a lado, con la diagonal y el corte clínico.
    predicciones = {título del panel: BDI predicho}."""
    y = np.asarray(y, dtype=float)
    fig, axes = plt.subplots(1, len(predicciones), figsize=(7 * len(predicciones), 6), squeeze=False, sharey=True)
    limite = max(y.max(), max(np.max(p) for p in predicciones.values())) + 2
    for ax, (titulo, prediccion) in zip(axes.flat, predicciones.items()):
        metricas = metricas_reg(y, prediccion)
        ax.scatter(y, prediccion, s=12, alpha=0.35, color=NEUTRO, edgecolors="none")
        ax.plot([0, limite], [0, limite], color=TEXTO, linewidth=1.2, label="Diagonal (predicción perfecta)")
        ax.axvline(CORTE_BDI, color=LINEA_CORTE, linestyle="--", linewidth=1)
        ax.axhline(CORTE_BDI, color=LINEA_CORTE, linestyle="--", linewidth=1, label=f"Corte clínico (BDI = {CORTE_BDI})")
        ax.set_xlim(-1, limite)
        ax.set_ylim(-1, limite)
        ax.set_title(f"{titulo}\nRMSE = {metricas['RMSE']:.2f} · MAE = {metricas['MAE']:.2f} · R² = {metricas['R²']:.3f}",
                     fontweight="bold")
        ax.set_xlabel("BDI real")
        ax.set_ylabel("BDI predicho")
        ax.legend(loc="upper left", fontsize=9)
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_residuos_vs_predicho(y, prediccion, nombre: str):
    """Residuos (real - predicho) vs. predicho en test, con suavizado LOWESS."""
    y = np.asarray(y, dtype=float)
    prediccion = np.asarray(prediccion, dtype=float)
    residuos = y - prediccion
    suavizado = sm.nonparametric.lowess(residuos, prediccion, frac=0.3)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(prediccion, residuos, s=12, alpha=0.35, color=NEUTRO, edgecolors="none")
    ax.axhline(0, color=TEXTO, linewidth=1)
    ax.plot(suavizado[:, 0], suavizado[:, 1], color=DEPRIMIDO, linewidth=2, label="LOWESS")
    ax.set_title(f"{nombre}: residuos vs. predicho (test)", fontweight="bold")
    ax.set_xlabel("BDI predicho")
    ax.set_ylabel("Residuo (real − predicho)")
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_importancias(importancias: dict, unidades: dict):
    """Permutation importance (media ± DE) en paneles lado a lado.
    importancias = {título del panel: tabla de importancia_permutacion()}; unidades = {título: etiqueta del eje}."""
    fig, axes = plt.subplots(1, len(importancias), figsize=(7 * len(importancias), 4.8), squeeze=False)
    for ax, (titulo, tabla) in zip(axes.flat, importancias.items()):
        orden = tabla.sort_values("Importancia media")
        ax.barh(orden.index, orden["Importancia media"], xerr=orden["DE"], color=NEUTRO, capsize=3)
        ax.axvline(0, color=TEXTO, linewidth=1)
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel(unidades[titulo])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_roc_pr_riesgo(y_binario, puntuaciones: dict):
    """Curvas ROC y Precision-Recall de varias puntuaciones de riesgo frente a `depressed` (test)."""
    y = np.asarray(y_binario)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    for (nombre, puntuacion), color, estilo in zip(puntuaciones.items(), [NEUTRO, DEPRIMIDO, LINEA_MEDIANA], ["-", "--", ":"]):
        fpr, tpr, _ = roc_curve(y, puntuacion)
        ax_roc.plot(fpr, tpr, color=color, linestyle=estilo, linewidth=2,
                    label=f"{nombre} (AUC = {roc_auc_score(y, puntuacion):.3f})")
        precision, recall, _ = precision_recall_curve(y, puntuacion)
        ax_pr.plot(recall, precision, color=color, linestyle=estilo, linewidth=2, drawstyle="steps-post",
                   label=f"{nombre} (AP = {average_precision_score(y, puntuacion):.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle=":", color=TEXTO_SECUNDARIO, linewidth=1, label="Azar (AUC = 0.5)")
    ax_pr.axhline(y.mean(), linestyle=":", color=TEXTO_SECUNDARIO, linewidth=1, label=f"Azar (prevalencia = {y.mean():.3f})")
    ax_roc.set_title("Curvas ROC (test)", fontweight="bold")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax_pr.set_title("Curvas Precision-Recall (test)", fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_ylim(0, 1.02)
    ax_roc.legend(fontsize=9, loc="lower right")
    ax_pr.legend(fontsize=9, loc="upper right")
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def _versiones_reg() -> dict:
    return {**_versiones(), "statsmodels": statsmodels.__version__, "python": platform.python_version()}


def guardar_final_reg(pipeline, clave: str, ganador_fase1: dict, metricas_test: pd.DataFrame,
                      explicacion: str, ruta: Path = RUTA_FINAL_REG) -> dict:
    """Guarda en regresion/final/: modelo_final.joblib (configurado para CPU), metricas_test.csv
    y metadatos.json."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    joblib.dump(configurar_dispositivo_reg(pipeline, gpu=False), ruta / "modelo_final.joblib")
    metricas_test.to_csv(ruta / "metricas_test.csv")

    metadatos = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": clave,
        "nombre": NOMBRES_REG[clave],
        "variante_target": ganador_fase1["variante"],
        "transformacion": ganador_fase1["transformacion"],
        "usa_smearing": ganador_fase1["usa_smearing"],
        "recorte_predicciones": [BDI_MIN, BDI_MAX],
        "hiperparametros": ganador_fase1["hiperparametros"],
        "rmse_cv_media": ganador_fase1["rmse_media"],
        "rmse_cv_de": ganador_fase1["rmse_de"],
        "explicacion_eleccion": explicacion,
        "modelo_referencia": "regresión lineal benchmark (LinearRegression, target nativo, sin tuning, 6 predictores)",
        "seed": SEED,
        "predictores": PREDICTORES,
        "target": TARGET_REG,
        "dispositivo_entrenamiento": dispositivo_reg(clave),
        "dispositivo_por_modelo": {c: dispositivo_reg(c) for c in NOMBRES_REG},
        "gpu_detectada": GPU_DISPOSITIVOS,
        "versiones": _versiones_reg(),
    }
    with open(ruta / "metadatos.json", "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    return metadatos


def cargar_modelo_regresion(clave: str = "final") -> tuple[object, dict]:
    """Carga un modelo de regresión guardado y sus metadatos.

    - clave="final": el modelo final de reg_comparacion.ipynb (regresion/final/).
    - clave = un modelo de NOMBRES_REG: su ganador de la fase 1 (regresion/fase1/<clave>/).

    El Pipeline recibe un DataFrame con las columnas originales (PREDICTORES) y predice el BDI
    en puntos, recortado a [0, 63]. Lanza FileNotFoundError si el archivo no existe.
    """
    if clave == "final":
        archivo, archivo_meta = RUTA_FINAL_REG / "modelo_final.joblib", RUTA_FINAL_REG / "metadatos.json"
        notebook = "los notebooks reg_*.ipynb y luego reg_comparacion.ipynb"
    elif clave in NOMBRES_REG:
        archivo_meta = RUTA_FASE1_REG / clave / "ganador.json"
        notebook = f"notebooks/reg_{clave}.ipynb"
        if not archivo_meta.exists():
            raise FileNotFoundError(f"No existe {archivo_meta}. Ejecuta primero {notebook}.")
        with open(archivo_meta, encoding="utf-8") as f:
            archivo = RUTA_FASE1_REG / clave / json.load(f)["archivo"]
    else:
        raise ValueError(f"Clave '{clave}' no válida. Opciones: final, {', '.join(NOMBRES_REG)}")

    if not archivo.exists():
        raise FileNotFoundError(f"No existe el modelo de regresión {archivo}. Ejecuta primero {notebook}.")
    with open(archivo_meta, encoding="utf-8") as f:
        metadatos = json.load(f)
    return joblib.load(archivo), metadatos
