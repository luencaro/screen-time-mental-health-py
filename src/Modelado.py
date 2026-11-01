"""Funciones reutilizables para el modelado de clasificación del estado depresivo.

Las usan:
- notebooks/Modelado.ipynb (versión previa): benchmark conjunto con pesos de clase.
- notebooks/clf_<clave>.ipynb (fase 1): manejo del desbalance por modelo.
- notebooks/clf_comparacion.ipynb (fases 2 y 3): comparación de los ganadores,
  test con intervalos bootstrap y optimización del umbral.

Todo se guarda en data/processed/modelado/.

Uso para cargar el modelo final:

    from Modelado import cargar_modelo_final, PREDICTORES
    modelo, metadatos = cargar_modelo_final()
    proba = modelo.predict_proba(df[PREDICTORES])[:, 1]
    prediccion = (proba >= metadatos["umbral"]).astype(int)
"""

import json
import os
import platform
import shutil
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import catboost
import imblearn
import joblib
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as PipelineRemuestreo
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from EstiloGraficos import (
    CHICAS, CHICOS, CMAP_SECUENCIAL, DEPRIMIDO, LINEA_CORTE, LINEA_MEDIANA,
    NEUTRO, NO_DEPRIMIDO, SUPERFICIE, TEXTO, TEXTO_SECUNDARIO,
)

# SVC(probability=True) está obsoleto desde scikit-learn 1.9 (se eliminará en 1.11), pero se usa
# a propósito en clf_svc.ipynb. Se silencia el aviso para que no inunde las salidas del tuning.
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated", category=FutureWarning)

# XGBoost entrena en la GPU, pero el CV le pasa a predict_proba DataFrames que están en CPU.
# XGBoost lo resuelve solo (predice vía DMatrix) con resultados idénticos; con este volumen
# de datos el costo es despreciable, así que se silencia el aviso.
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix", category=UserWarning)


SEED = 42
UMBRAL = 0.5
RECALL_MINIMO = 0.70  # criterio de tamizaje para elegir el umbral (fase 3)

PREDICTORES_NUMERICOS = [
    "sleep_quality_index",
    "avg_sleep_hours",
    "midsleep_weekend_hours",
    "social_jetlag_hours",
    "screen_time_index",
]
PREDICTORES_CATEGORICOS = ["sex"]
PREDICTORES = PREDICTORES_NUMERICOS + PREDICTORES_CATEGORICOS
TARGET = "depressed"

# Columnas que nunca entran a X: identificador, el target y su origen (bdi_total),
# y est_leisure_screen_hours, redundante con screen_time_index (Spearman r = 0.95)
COLUMNAS_EXCLUIDAS = ["subject_id", "bdi_total", "depressed", "est_leisure_screen_hours"]

MODELO_REFERENCIA = "logistica_benchmark"

# Clave del archivo .joblib → nombre legible para tablas y gráficas (Modelado.ipynb)
NOMBRES_MODELOS = {
    "logistica_benchmark": "Logística benchmark",
    "logistica_ridge": "Logística Ridge (L2)",
    "logistica_lasso": "Logística Lasso (L1)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

# Modelos de las fases 1-3 (clave del notebook clf_<clave>.ipynb → nombre legible).
# Ordenados de más simple a más complejo: logísticas < KNN < SVC < Random Forest < boosting
NOMBRES_CLF = {
    "logistica": "Logística",
    "ridge": "Logística Ridge (L2)",
    "lasso": "Logística Lasso (L1)",
    "knn": "KNN",
    "svc": "SVC (RBF)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
CLAVE_BENCHMARK = "benchmark"
NOMBRES_FASE2 = {CLAVE_BENCHMARK: "Logística benchmark", **NOMBRES_CLF}

# Estrategias de manejo del desbalance, de la más simple a la más compleja
ESTRATEGIAS = {
    "nativo": "Nativo (sin balanceo)",
    "pesos": "Pesos de clase",
    "smotenc": "SMOTENC",
}

# Métricas del benchmarking (nombre en sklearn → nombre en las tablas)
METRICAS = {
    "average_precision": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "accuracy": "Accuracy",
}

RAIZ = Path(__file__).resolve().parent.parent
RUTA_SALIDA = RAIZ / "data" / "processed" / "modelado"
RUTA_MODELOS = RUTA_SALIDA / "modelos"
RUTA_FASE1 = RUTA_SALIDA / "fase1"
RUTA_FINAL = RUTA_SALIDA / "final"


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _hay_gpu_nvidia() -> bool:
    """True si nvidia-smi existe y lista al menos una GPU."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        salida = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return salida.returncode == 0 and "GPU" in salida.stdout


def detectar_gpu() -> dict:
    """Detecta qué librerías pueden entrenar en una GPU CUDA en esta máquina.

    Devuelve {"xgboost": ..., "catboost": ..., "lightgbm": ...} con el valor del
    dispositivo que acepta cada librería ("cuda", "GPU", "gpu") o None si debe usar CPU.
    Cada librería se prueba con un ajuste mínimo dentro de try/except, así que en una
    máquina sin GPU (o con builds solo-CPU) todo queda en None sin errores.

    LightGBM: conda-forge prefiere la build CPU aunque haya CUDA, por eso el Dockerfile
    instala explícitamente la variante cuda* en la imagen GPU. Se prueba primero
    device="cuda" (esa build) y, si falla, device="gpu" (builds OpenCL).
    """
    dispositivos = {"xgboost": None, "catboost": None, "lightgbm": None}
    if not _hay_gpu_nvidia():
        return dispositivos

    X = np.random.default_rng(SEED).normal(size=(64, 3))
    y = np.tile([0, 1], 32)

    try:
        if xgboost.build_info().get("USE_CUDA"):
            XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(X, y)
            dispositivos["xgboost"] = "cuda"
    except Exception:
        pass

    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            CatBoostClassifier(task_type="GPU", devices="0", iterations=2, verbose=0, allow_writing_files=False).fit(X, y)
            dispositivos["catboost"] = "GPU"
    except Exception:
        pass

    for dispositivo in ("cuda", "gpu"):
        try:
            LGBMClassifier(device=dispositivo, n_estimators=2, verbose=-1).fit(X, y)
            dispositivos["lightgbm"] = dispositivo
            break
        except Exception:
            continue

    return dispositivos


def _dispositivos_gpu() -> dict:
    """detectar_gpu() una sola vez por sesión. El resultado se guarda en una variable de
    entorno que heredan los procesos paralelos de joblib (que vuelven a importar este
    módulo), para que no repitan la detección ni compitan por la GPU al hacerlo."""
    guardado = os.environ.get("MODELADO_GPU_DISPOSITIVOS")
    if guardado:
        return json.loads(guardado)
    dispositivos = detectar_gpu()
    os.environ["MODELADO_GPU_DISPOSITIVOS"] = json.dumps(dispositivos)
    return dispositivos


GPU_DISPOSITIVOS = _dispositivos_gpu()
USAR_GPU = any(valor is not None for valor in GPU_DISPOSITIVOS.values())

# Modelo → librería con soporte GPU (los de scikit-learn solo usan CPU)
_LIBRERIA_GPU = {"xgboost": "xgboost", "catboost": "catboost", "lightgbm": "lightgbm"}


def dispositivo_modelo(clave: str) -> str:
    """"GPU" si el modelo entrena en GPU en esta máquina; "CPU" en otro caso."""
    libreria = _LIBRERIA_GPU.get(clave)
    return "GPU" if libreria and GPU_DISPOSITIVOS[libreria] else "CPU"


def n_jobs_modelo(clave: str) -> int:
    """Procesos para el tuning y el CV: 1 si el modelo usa la GPU (varios procesos
    compitiendo por la misma GPU dan errores); todos los núcleos si usa CPU."""
    return 1 if dispositivo_modelo(clave) == "GPU" else -1


def resumen_dispositivos() -> pd.DataFrame:
    """Tabla con el dispositivo que usará cada modelo en esta máquina."""
    return pd.DataFrame(
        {"Dispositivo": [dispositivo_modelo(clave) for clave in NOMBRES_CLF]},
        index=pd.Index(list(NOMBRES_CLF.values()), name="Modelo"),
    )


def configurar_dispositivo(pipeline, gpu: bool = True):
    """Ajusta el dispositivo del modelo dentro del Pipeline (lo modifica y lo devuelve).

    Con gpu=True usa la GPU si la librería la soporta en esta máquina; con gpu=False
    deja el modelo en CPU. Se llama con gpu=False antes de guardar cada .joblib para
    que el modelo se pueda cargar y usar en máquinas sin GPU (como la del dashboard).
    """
    modelo = pipeline.named_steps["modelo"]
    if isinstance(modelo, XGBClassifier):
        usar = gpu and GPU_DISPOSITIVOS["xgboost"]
        modelo.set_params(device=GPU_DISPOSITIVOS["xgboost"] if usar else "cpu", tree_method="hist")
    elif isinstance(modelo, CatBoostClassifier):
        # CatBoost no permite cambiar parámetros de un modelo ya entrenado. Tampoco hace falta:
        # task_type solo afecta al entrenamiento y un modelo entrenado en GPU predice igual en CPU.
        # Al reentrenar (p. ej. clone() en la fase 2) el clon no está entrenado y sí se configura.
        if modelo.is_fitted():
            return pipeline
        if gpu and GPU_DISPOSITIVOS["catboost"]:
            modelo.set_params(task_type="GPU", devices="0")
        else:
            modelo.set_params(task_type="CPU")
    elif isinstance(modelo, LGBMClassifier):
        usar = gpu and GPU_DISPOSITIVOS["lightgbm"]
        modelo.set_params(device=GPU_DISPOSITIVOS["lightgbm"] if usar else "cpu")
    return pipeline


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Elimina explícitamente las columnas excluidas y devuelve (X, y) con los 6 predictores."""
    X = df.drop(columns=COLUMNAS_EXCLUIDAS)[PREDICTORES]
    y = df[TARGET]
    return X, y


def dividir_train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    """Divide el DataFrame completo en train y test (80/20), estratificando por la
    combinación sexo × depresión para conservar las 4 proporciones en ambos conjuntos."""
    estratos = df["sex"].astype(str) + "_" + df[TARGET].astype(str)
    return train_test_split(df, test_size=test_size, stratify=estratos, random_state=seed)


def cargar_split(df: pd.DataFrame, ruta: Path = RUTA_SALIDA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (df_train, df_test) según data/processed/modelado/split_indices.csv.

    Si el archivo no existe, genera el split con dividir_train_test y lo guarda.
    Las filas se ordenan por subject_id, así que todos los notebooks reciben el mismo
    train en el mismo orden y, con crear_cv(), exactamente los mismos folds.
    """
    archivo = Path(ruta) / "split_indices.csv"
    if not archivo.exists():
        df_train, df_test = dividir_train_test(df)
        guardar_split(df_train, df_test, ruta=Path(ruta))
        print(f"No existía {archivo}: se generó el split y se guardó.")

    split = pd.read_csv(archivo)
    ids_train = set(split.loc[split["conjunto"] == "train", "subject_id"])
    ids_test = set(split.loc[split["conjunto"] == "test", "subject_id"])
    sin_asignar = set(df["subject_id"]) - ids_train - ids_test
    if sin_asignar:
        raise ValueError(f"{len(sin_asignar)} subject_id del dataset no aparecen en {archivo}.")

    ordenado = df.sort_values("subject_id")
    return ordenado[ordenado["subject_id"].isin(ids_train)], ordenado[ordenado["subject_id"].isin(ids_test)]


def crear_cv(seed: int = SEED) -> StratifiedKFold:
    """Los folds de todo el modelado: StratifiedKFold(5, shuffle=True, random_state=SEED)."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# Modelos y espacios de búsqueda
# ---------------------------------------------------------------------------

# Espacios de búsqueda (los parámetros llevan el prefijo "modelo__" porque viven dentro del Pipeline)
ESPACIOS_BUSQUEDA = {
    "ridge": {"modelo__C": loguniform(1e-3, 1e2)},
    "lasso": {"modelo__C": loguniform(1e-3, 1e2)},
    "random_forest": {
        "modelo__n_estimators": randint(200, 801),
        "modelo__max_depth": [3, 5, 8, 12, None],
        "modelo__min_samples_leaf": [5, 10, 20, 50],
        "modelo__max_features": ["sqrt", 0.5, 1.0],
    },
    "xgboost": {
        "modelo__n_estimators": randint(100, 601),
        "modelo__max_depth": randint(2, 7),
        "modelo__learning_rate": loguniform(0.01, 0.2),
        "modelo__subsample": uniform(0.6, 0.4),          # uniforme en [0.6, 1.0]
        "modelo__colsample_bytree": uniform(0.6, 0.4),
        "modelo__min_child_weight": randint(1, 11),
    },
    "lightgbm": {
        "modelo__n_estimators": randint(100, 601),
        "modelo__num_leaves": randint(4, 32),
        "modelo__max_depth": [-1, 3, 5, 7],
        "modelo__learning_rate": loguniform(0.01, 0.2),
        "modelo__min_child_samples": randint(10, 101),
        "modelo__subsample": uniform(0.6, 0.4),
        "modelo__colsample_bytree": uniform(0.6, 0.4),
    },
    "catboost": {
        "modelo__iterations": randint(200, 801),
        "modelo__depth": randint(3, 8),
        "modelo__learning_rate": loguniform(0.01, 0.2),
        "modelo__l2_leaf_reg": uniform(1, 9),             # uniforme en [1, 10]
    },
    "knn": {
        "modelo__n_neighbors": randint(15, 301),
        "modelo__weights": ["uniform", "distance"],
        "modelo__p": [1, 2],
    },
    # Lista de dos espacios: RandomizedSearchCV elige uno al azar en cada iteración,
    # así que gamma="scale" se prueba en ~la mitad de las combinaciones
    "svc": [
        {"modelo__C": loguniform(1e-2, 1e2), "modelo__gamma": ["scale"]},
        {"modelo__C": loguniform(1e-2, 1e2), "modelo__gamma": loguniform(1e-3, 1e0)},
    ],
}

def _describir_valores(valores) -> str:
    """Texto legible para una lista de valores o una distribución de scipy."""
    if isinstance(valores, list):
        return ", ".join(str(v) for v in valores)
    tipos = {"randint": "entero", "uniform": "uniforme", "loguniform": "log-uniforme", "reciprocal": "log-uniforme"}
    minimo, maximo = valores.support()
    return f"{tipos.get(valores.dist.name, valores.dist.name)} entre {minimo:g} y {maximo:g}"


def describir_espacio(clave: str) -> pd.DataFrame:
    """Espacio de búsqueda de un modelo como tabla legible (en lugar de los objetos de scipy)."""
    espacio = ESPACIOS_BUSQUEDA.get(clave) or {}
    subespacios = espacio if isinstance(espacio, list) else [espacio]
    descripciones = {}
    for subespacio in subespacios:
        for nombre, valores in subespacio.items():
            texto = _describir_valores(valores)
            lista = descripciones.setdefault(nombre.removeprefix("modelo__"), [])
            if texto not in lista:
                lista.append(texto)
    return pd.DataFrame(
        {"Valores": [" o ".join(textos) for textos in descripciones.values()]},
        index=pd.Index(list(descripciones), name="Hiperparámetro"),
    )


# Iteraciones de RandomizedSearchCV en la fase 1 (la logística no tiene hiperparámetros: sin tuning)
N_ITER_FASE1 = {
    "logistica": 0, "ridge": 10, "lasso": 10, "knn": 20, "svc": 20,
    "random_forest": 25, "xgboost": 25, "lightgbm": 25, "catboost": 25,
}


def crear_preprocesador() -> ColumnTransformer:
    """StandardScaler en las 5 numéricas y OneHotEncoder(drop="first") en sex."""
    return ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), PREDICTORES_NUMERICOS),
            ("categoricas", OneHotEncoder(drop="first", handle_unknown="ignore"), PREDICTORES_CATEGORICOS),
        ]
    )


def crear_pipeline_modelo(modelo) -> Pipeline:
    """Pipeline completo: preprocesamiento + modelo. Al estar todo junto, en la
    validación cruzada el escalado se ajusta solo con los folds de entrenamiento."""
    return Pipeline(steps=[("preprocesamiento", crear_preprocesador()), ("modelo", modelo)])


def definir_modelos(scale_pos_weight: float, seed: int = SEED) -> dict:
    """Devuelve {clave: (pipeline, espacio_de_búsqueda, n_iter)} para los 7 modelos de Modelado.ipynb.

    El benchmark no tiene espacio (None): se entrena tal cual, sin tuning.
    Los modelos usan 1 hilo; la paralelización se hace a nivel de folds/iteraciones.
    """
    return {
        "logistica_benchmark": (
            crear_pipeline_modelo(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
            None,
            0,
        ),
        "logistica_ridge": (
            # l1_ratio=0 equivale a penalty="l2" (penalty está obsoleto desde scikit-learn 1.8)
            crear_pipeline_modelo(LogisticRegression(
                l1_ratio=0, class_weight="balanced", max_iter=1000, random_state=seed,
            )),
            ESPACIOS_BUSQUEDA["ridge"],
            10,
        ),
        "logistica_lasso": (
            # l1_ratio=1 equivale a penalty="l1"
            crear_pipeline_modelo(LogisticRegression(
                l1_ratio=1, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=seed,
            )),
            ESPACIOS_BUSQUEDA["lasso"],
            10,
        ),
        "random_forest": (
            crear_pipeline_modelo(RandomForestClassifier(
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            )),
            ESPACIOS_BUSQUEDA["random_forest"],
            40,
        ),
        "xgboost": (
            crear_pipeline_modelo(XGBClassifier(
                scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=seed, n_jobs=1,
            )),
            ESPACIOS_BUSQUEDA["xgboost"],
            40,
        ),
        "lightgbm": (
            # subsample_freq=1: sin él, LightGBM ignora subsample
            crear_pipeline_modelo(LGBMClassifier(
                class_weight="balanced", subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1,
            )),
            ESPACIOS_BUSQUEDA["lightgbm"],
            40,
        ),
        "catboost": (
            # allow_writing_files=False evita que CatBoost cree la carpeta catboost_info/
            crear_pipeline_modelo(CatBoostClassifier(
                auto_class_weights="Balanced", random_seed=seed, thread_count=1,
                verbose=0, allow_writing_files=False,
            )),
            ESPACIOS_BUSQUEDA["catboost"],
            40,
        ),
    }


def estrategias_modelo(clave: str) -> list[str]:
    """Estrategias de desbalance aplicables a un modelo (KNN no admite pesos de clase)."""
    return [e for e in ESTRATEGIAS if not (clave == "knn" and e == "pesos")]


def crear_modelo(clave: str, pesos: bool, scale_pos_weight: float = 1.0, seed: int = SEED):
    """Estimador de un modelo de las fases 1-3, con o sin pesos de clase.

    Mismos parámetros fijos que en definir_modelos(). Sin pesos, el parámetro de
    pesos se omite (CatBoost no admite auto_class_weights=None al clonar).
    """
    balanceado = "balanced" if pesos else None
    if clave == "logistica":
        return LogisticRegression(class_weight=balanceado, max_iter=1000, random_state=seed)
    if clave == "ridge":
        return LogisticRegression(l1_ratio=0, class_weight=balanceado, max_iter=1000, random_state=seed)
    if clave == "lasso":
        return LogisticRegression(l1_ratio=1, solver="liblinear", class_weight=balanceado, max_iter=1000, random_state=seed)
    if clave == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample" if pesos else None, random_state=seed, n_jobs=1)
    if clave == "xgboost":
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight if pesos else 1, eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=1,
        )
    if clave == "lightgbm":
        return LGBMClassifier(class_weight=balanceado, subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1)
    if clave == "catboost":
        parametros = {"random_seed": seed, "thread_count": 1, "verbose": 0, "allow_writing_files": False}
        if pesos:
            parametros["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(**parametros)
    if clave == "knn":
        if pesos:
            raise ValueError("KNeighborsClassifier no admite pesos de clase.")
        return KNeighborsClassifier()
    if clave == "svc":
        # probability=True: calibración de Platt interna (CV de 5 folds) para tener predict_proba
        return SVC(kernel="rbf", probability=True, class_weight=balanceado, random_state=seed, cache_size=1000)
    raise ValueError(f"Clave '{clave}' no válida. Opciones: {', '.join(NOMBRES_CLF)}")


def crear_smotenc(seed: int = SEED) -> SMOTENC:
    """SMOTENC con `sex` como variable categórica: interpola solo las numéricas y a cada
    caso sintético le asigna el sexo más frecuente entre sus vecinos ("Boy" o "Girl")."""
    return SMOTENC(categorical_features=PREDICTORES_CATEGORICOS, random_state=seed)


def crear_pipeline_clf(clave: str, estrategia: str, scale_pos_weight: float = 1.0, seed: int = SEED):
    """Pipeline de un modelo con una estrategia de desbalance, listo para la GPU si está disponible.

    - nativo / pesos: Pipeline de scikit-learn (preprocesamiento → modelo).
    - smotenc: Pipeline de imbalanced-learn (SMOTENC → preprocesamiento → modelo).
      SMOTENC va antes del preprocesamiento para recibir `sex` sin codificar. Solo se
      aplica en fit (los folds de entrenamiento); en predict y predict_proba se omite,
      así que validación y test nunca tienen casos sintéticos.
    """
    if estrategia not in estrategias_modelo(clave):
        raise ValueError(f"La estrategia '{estrategia}' no aplica a '{clave}'.")
    modelo = crear_modelo(clave, pesos=(estrategia == "pesos"), scale_pos_weight=scale_pos_weight, seed=seed)
    if estrategia == "smotenc":
        pipeline = PipelineRemuestreo(steps=[
            ("remuestreo", crear_smotenc(seed)),
            ("preprocesamiento", crear_preprocesador()),
            ("modelo", modelo),
        ])
    else:
        pipeline = crear_pipeline_modelo(modelo)
    return configurar_dispositivo(pipeline, gpu=True)


def verificar_smotenc(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica SMOTENC directamente a X_train, y_train y comprueba el resultado.

    Se hace fuera del Pipeline porque el Pipeline no expone los datos remuestreados.
    Comprueba que `sex` solo toma "Boy" y "Girl" y que las clases quedan balanceadas;
    si algo falla, lanza ValueError. Devuelve (conteo por clase antes/después,
    sexo × clase después del remuestreo).
    """
    X_rem, y_rem = crear_smotenc(seed).fit_resample(X_train, y_train)
    X_rem = pd.DataFrame(X_rem, columns=X_train.columns)
    y_rem = pd.Series(np.asarray(y_rem), name=TARGET)

    valores_sex = set(X_rem["sex"].unique())
    if not valores_sex <= {"Boy", "Girl"}:
        raise ValueError(f"SMOTENC generó valores de sex no válidos: {sorted(map(str, valores_sex))}")
    conteo = y_rem.value_counts().sort_index()
    if conteo.nunique() != 1:
        raise ValueError(f"Las clases no quedaron balanceadas tras SMOTENC: {conteo.to_dict()}")

    print(f"✓ sex solo toma los valores {sorted(valores_sex)}")
    print(f"✓ Clases balanceadas: {conteo.to_dict()}")

    tabla_clases = pd.DataFrame({"Antes": pd.Series(y_train).value_counts().sort_index(), "Después": conteo})
    tabla_clases["Sintéticos"] = tabla_clases["Después"] - tabla_clases["Antes"]
    tabla_clases.index.name = "Deprimido"
    tabla_sexo = pd.crosstab(X_rem["sex"].to_numpy(), y_rem.to_numpy(), rownames=["Sexo"], colnames=["Deprimido"])
    return tabla_clases, tabla_sexo


def crear_benchmark(seed: int = SEED) -> Pipeline:
    """Logística benchmark original: class_weight="balanced", sin tuning (modelo de referencia)."""
    return crear_pipeline_modelo(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))


def _limpiar_parametros(parametros: dict) -> dict:
    """Quita el prefijo "modelo__" y pasa los tipos numpy a tipos nativos (para el JSON)."""
    return {
        nombre.removeprefix("modelo__"): (valor.item() if isinstance(valor, np.generic) else valor)
        for nombre, valor in parametros.items()
    }


def ajustar_modelos(modelos: dict, X_train, y_train, cv, seed: int = SEED) -> tuple[dict, dict]:
    """Ajusta cada modelo sobre el train (Modelado.ipynb).

    - Benchmark: se entrena directamente, sin tuning.
    - Resto: RandomizedSearchCV con scoring PR-AUC; con refit=True el mejor
      Pipeline queda reentrenado con todo el train.

    Devuelve (pipelines_entrenados, mejores_hiperparametros) indexados por clave.
    """
    entrenados, hiperparametros = {}, {}
    for clave, (pipeline, espacio, n_iter) in modelos.items():
        if espacio is None:
            entrenados[clave] = pipeline.fit(X_train, y_train)
            hiperparametros[clave] = {}
        else:
            busqueda = RandomizedSearchCV(
                pipeline, espacio, n_iter=n_iter, scoring="average_precision",
                cv=cv, random_state=seed, n_jobs=-1, refit=True,
            )
            busqueda.fit(X_train, y_train)
            entrenados[clave] = busqueda.best_estimator_
            hiperparametros[clave] = _limpiar_parametros(busqueda.best_params_)
        print(f"✓ {NOMBRES_MODELOS[clave]}")
    return entrenados, hiperparametros


def coeficientes_logistica(pipeline: Pipeline) -> pd.DataFrame:
    """Coeficientes (escala estandarizada) de una logística dentro del Pipeline,
    con una columna que indica cuáles quedaron exactamente en cero."""
    nombres = pipeline.named_steps["preprocesamiento"].get_feature_names_out()
    coeficientes = pipeline.named_steps["modelo"].coef_.ravel()
    tabla = pd.DataFrame({"Variable": nombres, "Coeficiente": coeficientes})
    tabla["Variable"] = tabla["Variable"].str.replace(r"^(numericas|categoricas)__", "", regex=True)
    tabla["En cero"] = tabla["Coeficiente"] == 0
    return tabla


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

# Métricas de clase con umbral 0.5 sobre predict_proba. Se calculan así (y no con predict)
# para que el umbral sea el mismo en todos los modelos: en el SVC, predict() usa la
# frontera de decisión y no predict_proba >= 0.5.
def _precision_umbral(y, proba):
    return precision_score(y, (proba >= UMBRAL).astype(int), zero_division=0)


def _recall_umbral(y, proba):
    return recall_score(y, (proba >= UMBRAL).astype(int))


def _f1_umbral(y, proba):
    return f1_score(y, (proba >= UMBRAL).astype(int))


def _accuracy_umbral(y, proba):
    return accuracy_score(y, (proba >= UMBRAL).astype(int))


SCORERS = {
    "average_precision": make_scorer(average_precision_score, response_method="predict_proba"),
    "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "precision": make_scorer(_precision_umbral, response_method="predict_proba"),
    "recall": make_scorer(_recall_umbral, response_method="predict_proba"),
    "f1": make_scorer(_f1_umbral, response_method="predict_proba"),
    "accuracy": make_scorer(_accuracy_umbral, response_method="predict_proba"),
}


def metricas_con_umbral(y, proba, umbral: float = UMBRAL) -> dict:
    """PR-AUC y ROC-AUC (no dependen del umbral) y métricas de clase con el umbral dado."""
    pred = (np.asarray(proba) >= umbral).astype(int)
    return {
        "PR-AUC": average_precision_score(y, proba),
        "ROC-AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
    }


def _fila_cv(resultados: dict, indice: int | None = None) -> dict:
    """Media y desviación de cada métrica, desde la salida de cross_validate (indice=None)
    o desde cv_results_ de RandomizedSearchCV (indice = combinación elegida)."""
    fila = {}
    for metrica, nombre in METRICAS.items():
        if indice is None:
            media, de = resultados[f"test_{metrica}"].mean(), resultados[f"test_{metrica}"].std()
        else:
            media, de = resultados[f"mean_test_{metrica}"][indice], resultados[f"std_test_{metrica}"][indice]
        fila[f"{nombre} media"] = float(media)
        fila[f"{nombre} DE"] = float(de)
    return fila


# ---------------------------------------------------------------------------
# Benchmarking (Modelado.ipynb)
# ---------------------------------------------------------------------------

def _agregar_diferencia(tabla: pd.DataFrame, columna_pr_auc: str) -> pd.DataFrame:
    """Agrega la diferencia de PR-AUC frente a la logística benchmark y ordena por PR-AUC."""
    referencia = tabla.loc[NOMBRES_MODELOS[MODELO_REFERENCIA], columna_pr_auc]
    tabla["Δ PR-AUC vs benchmark"] = tabla[columna_pr_auc] - referencia
    return tabla.sort_values(columna_pr_auc, ascending=False)


def evaluar_cv(pipelines: dict, X_train, y_train, cv) -> pd.DataFrame:
    """Validación cruzada de cada Pipeline (con sus mejores hiperparámetros) en el train.
    Devuelve una fila por modelo con la media y la desviación de cada métrica."""
    filas = {}
    for clave, pipeline in pipelines.items():
        resultados = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=list(METRICAS), n_jobs=-1)
        filas[NOMBRES_MODELOS[clave]] = _fila_cv(resultados)
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Modelo"
    return _agregar_diferencia(tabla, "PR-AUC media")


def formatear_tabla_cv(tabla_cv: pd.DataFrame, decimales: int = 3) -> pd.DataFrame:
    """Versión para mostrar de una tabla de CV: cada métrica como "media ± DE",
    más las columnas de diferencias (Δ ...) que tenga."""
    vista = pd.DataFrame(index=tabla_cv.index)
    for nombre in METRICAS.values():
        vista[nombre] = [
            f"{media:.{decimales}f} ± {de:.{decimales}f}"
            for media, de in zip(tabla_cv[f"{nombre} media"], tabla_cv[f"{nombre} DE"])
        ]
    for columna in tabla_cv.columns:
        if columna.startswith("Δ"):
            vista[columna] = tabla_cv[columna].round(decimales)
    return vista


def evaluar_test(pipelines: dict, X_test, y_test, umbral: float = UMBRAL) -> pd.DataFrame:
    """Métricas en el test de cada Pipeline ya entrenado; las de clase usan el umbral dado."""
    filas = {
        NOMBRES_MODELOS[clave]: metricas_con_umbral(y_test, pipeline.predict_proba(X_test)[:, 1], umbral)
        for clave, pipeline in pipelines.items()
    }
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Modelo"
    return _agregar_diferencia(tabla, "PR-AUC")


# ---------------------------------------------------------------------------
# Fase 1: manejo del desbalance por modelo
# ---------------------------------------------------------------------------

def ejecutar_fase1(clave: str, X_train, y_train, cv, seed: int = SEED) -> tuple[pd.DataFrame, dict, dict]:
    """Tuning y evaluación en CV de un modelo con cada estrategia de desbalance.

    Por estrategia, RandomizedSearchCV solo sobre el train (scoring PR-AUC, mismos folds).
    Con scoring múltiple y refit="average_precision", cv_results_ trae todas las métricas
    de la combinación elegida en los mismos folds, sin repetir el CV. La logística no tiene
    hiperparámetros: se evalúa con cross_validate. En ambos casos el Pipeline final queda
    reentrenado con todo el train.

    Devuelve (tabla, pipelines, hiperparametros), indexados por estrategia.
    """
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    espacio = ESPACIOS_BUSQUEDA.get(clave)
    n_jobs = n_jobs_modelo(clave)

    filas, pipelines, hiperparametros = {}, {}, {}
    for estrategia in estrategias_modelo(clave):
        inicio = time.perf_counter()
        pipeline = crear_pipeline_clf(clave, estrategia, scale_pos_weight=n_neg / n_pos, seed=seed)
        if espacio is None:
            resultados = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORERS, n_jobs=n_jobs)
            fila = _fila_cv(resultados)
            pipelines[estrategia] = pipeline.fit(X_train, y_train)
            hiperparametros[estrategia] = {}
        else:
            busqueda = RandomizedSearchCV(
                pipeline, espacio, n_iter=N_ITER_FASE1[clave], scoring=SCORERS, refit="average_precision",
                cv=cv, random_state=seed, n_jobs=n_jobs,
            )
            busqueda.fit(X_train, y_train)
            fila = _fila_cv(busqueda.cv_results_, indice=busqueda.best_index_)
            pipelines[estrategia] = busqueda.best_estimator_
            hiperparametros[estrategia] = _limpiar_parametros(busqueda.best_params_)
        fila["Tiempo (s)"] = time.perf_counter() - inicio
        filas[estrategia] = fila
        print(f"✓ {ESTRATEGIAS[estrategia]:<22} PR-AUC = {fila['PR-AUC media']:.4f} ± {fila['PR-AUC DE']:.4f}  ({fila['Tiempo (s)']:.0f} s)")

    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Estrategia"
    return tabla, pipelines, hiperparametros


def elegir_mejor(tabla: pd.DataFrame, columna_media: str = "PR-AUC media",
                 columna_de: str = "PR-AUC DE") -> tuple[str, str]:
    """Elige la opción con mayor PR-AUC media de CV. Devuelve (opción elegida, explicación en texto)."""
    ganador = tabla[columna_media].idxmax()
    media, de = tabla.loc[ganador, columna_media], tabla.loc[ganador, columna_de]
    return ganador, f"Se elige '{ganador}': tiene la mayor PR-AUC media de CV ({media:.4f} ± {de:.4f})."


def formatear_hiperparametros(params: dict) -> str:
    """Hiperparámetros como texto "k=v, ..." (4 cifras significativas en los decimales)."""
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()) or "sin tuning"


def tabla_hiperparametros(hiperparametros: dict) -> pd.DataFrame:
    """Mejores hiperparámetros por estrategia: una fila por estrategia y una columna por hiperparámetro."""
    tabla = pd.DataFrame.from_dict(hiperparametros, orient="index")
    tabla = tabla.map(lambda v: float(f"{v:.4g}") if isinstance(v, float) else v)
    tabla.index = pd.Index([ESTRATEGIAS[e] for e in tabla.index], name="Estrategia")
    return tabla


def guardar_fase1(clave: str, tabla: pd.DataFrame, estrategia: str, pipeline, hiperparametros: dict,
                  explicacion: str, ruta: Path = RUTA_FASE1) -> dict:
    """Guarda en fase1/<clave>/: resultados_cv.csv, ganador.json y <clave>_<estrategia>.joblib.

    El modelo se deja configurado para CPU antes de guardarlo, para que se pueda
    cargar en máquinas sin GPU.
    """
    carpeta = Path(ruta) / clave
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "resultados_cv.csv")

    archivo_modelo = f"{clave}_{estrategia}.joblib"
    joblib.dump(configurar_dispositivo(pipeline, gpu=False), carpeta / archivo_modelo)

    ganador = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "modelo": clave,
        "nombre": NOMBRES_CLF[clave],
        "estrategia": estrategia,
        "hiperparametros": hiperparametros,
        "pr_auc_media": float(tabla.loc[estrategia, "PR-AUC media"]),
        "pr_auc_de": float(tabla.loc[estrategia, "PR-AUC DE"]),
        "explicacion": explicacion,
        "dispositivo_entrenamiento": dispositivo_modelo(clave),
        "archivo": archivo_modelo,
        "seed": SEED,
    }
    with open(carpeta / "ganador.json", "w", encoding="utf-8") as archivo:
        json.dump(ganador, archivo, ensure_ascii=False, indent=2)
    return ganador


def cargar_ganador_fase1(clave: str, ruta: Path = RUTA_FASE1) -> tuple[object, dict]:
    """Carga el Pipeline ganador de la fase 1 de un modelo y su ganador.json."""
    carpeta = Path(ruta) / clave
    if not (carpeta / "ganador.json").exists():
        raise FileNotFoundError(
            f"No existe {carpeta / 'ganador.json'}. Ejecuta primero notebooks/clf_{clave}.ipynb."
        )
    with open(carpeta / "ganador.json", encoding="utf-8") as archivo:
        ganador = json.load(archivo)
    return joblib.load(carpeta / ganador["archivo"]), ganador


# ---------------------------------------------------------------------------
# Fase 2: comparación de los ganadores
# ---------------------------------------------------------------------------

def evaluar_cv_modelos(pipelines: dict, X_train, y_train, cv) -> pd.DataFrame:
    """CV (mismos folds) de cada Pipeline, indexado por clave, con la diferencia de
    PR-AUC frente al benchmark (CLAVE_BENCHMARK). Cada Pipeline se clona y se
    reentrena en cada fold, en GPU si el modelo la soporta."""
    filas = {}
    for clave, pipeline in pipelines.items():
        modelo = configurar_dispositivo(clone(pipeline), gpu=True)
        resultados = cross_validate(modelo, X_train, y_train, cv=cv, scoring=SCORERS, n_jobs=n_jobs_modelo(clave))
        filas[clave] = _fila_cv(resultados)
        print(f"✓ {NOMBRES_FASE2[clave]}")
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Modelo"
    tabla["Δ PR-AUC vs benchmark"] = tabla["PR-AUC media"] - tabla.loc[CLAVE_BENCHMARK, "PR-AUC media"]
    return tabla.sort_values("PR-AUC media", ascending=False)


def evaluar_test_modelos(probas: dict, y_test, umbral: float = UMBRAL) -> pd.DataFrame:
    """Métricas en test a partir de las probabilidades de cada modelo (indexado por clave)."""
    tabla = pd.DataFrame.from_dict(
        {clave: metricas_con_umbral(y_test, proba, umbral) for clave, proba in probas.items()}, orient="index",
    )
    tabla.index.name = "Modelo"
    tabla["Δ PR-AUC vs benchmark"] = tabla["PR-AUC"] - tabla.loc[CLAVE_BENCHMARK, "PR-AUC"]
    return tabla.sort_values("PR-AUC", ascending=False)


def intervalos_bootstrap(y_test, probas: dict, n_replicas: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """Intervalos bootstrap al 95% (percentiles) del PR-AUC y del ROC-AUC en test.

    En cada réplica se remuestrea el test con reemplazo y se evalúan TODOS los modelos
    sobre las mismas observaciones. Así también se obtiene el intervalo de la diferencia
    pareada frente al benchmark: si no incluye 0, la diferencia es real con un 95% de confianza.
    """
    y = np.asarray(y_test)
    probas = {clave: np.asarray(proba) for clave, proba in probas.items()}
    rng = np.random.default_rng(seed)
    replicas = {clave: {"PR-AUC": [], "ROC-AUC": []} for clave in probas}
    for _ in range(n_replicas):
        indices = rng.integers(0, len(y), len(y))
        if y[indices].min() == y[indices].max():  # réplica con una sola clase: no se puede evaluar
            continue
        for clave, proba in probas.items():
            replicas[clave]["PR-AUC"].append(average_precision_score(y[indices], proba[indices]))
            replicas[clave]["ROC-AUC"].append(roc_auc_score(y[indices], proba[indices]))

    filas = {}
    for clave, proba in probas.items():
        fila = {}
        for metrica, funcion in [("PR-AUC", average_precision_score), ("ROC-AUC", roc_auc_score)]:
            valores = np.array(replicas[clave][metrica])
            diferencias = valores - np.array(replicas[CLAVE_BENCHMARK][metrica])
            fila[metrica] = funcion(y, proba)
            fila[f"{metrica} IC 2.5%"], fila[f"{metrica} IC 97.5%"] = np.percentile(valores, [2.5, 97.5])
            fila[f"Δ {metrica} vs benchmark"] = fila[metrica] - funcion(y, probas[CLAVE_BENCHMARK])
            fila[f"Δ {metrica} IC 2.5%"], fila[f"Δ {metrica} IC 97.5%"] = np.percentile(diferencias, [2.5, 97.5])
            fila[f"Δ {metrica} significativa"] = clave != CLAVE_BENCHMARK and (
                fila[f"Δ {metrica} IC 2.5%"] > 0 or fila[f"Δ {metrica} IC 97.5%"] < 0
            )
        filas[clave] = fila
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Modelo"
    return tabla.sort_values("PR-AUC", ascending=False)


# ---------------------------------------------------------------------------
# Fase 3: optimización del umbral
# ---------------------------------------------------------------------------

def probabilidades_oof(pipeline, clave: str, X_train, y_train, cv) -> np.ndarray:
    """Probabilidades out-of-fold en el train (mismos folds): cada adolescente recibe la
    probabilidad de un modelo que no lo vio al entrenar."""
    modelo = configurar_dispositivo(clone(pipeline), gpu=True)
    return cross_val_predict(modelo, X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs_modelo(clave))[:, 1]


def elegir_umbrales(y, proba, recall_minimo: float = RECALL_MINIMO) -> dict:
    """Umbrales elegidos con probabilidades out-of-fold del train.

    - Principal (tamizaje): la mayor precision entre los umbrales con recall >= recall_minimo.
    - Referencia: el umbral que maximiza F1.
    """
    precision, recall, umbrales = precision_recall_curve(y, proba)
    # precision_recall_curve devuelve un punto más que umbrales (recall = 0): se descarta
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)

    validos = np.flatnonzero(recall >= recall_minimo)   # nunca vacío: el umbral más bajo da recall = 1
    i = validos[np.argmax(precision[validos])]
    j = int(np.argmax(f1))
    return {
        "umbral": float(umbrales[i]),
        "criterio": f"máxima precision con recall >= {recall_minimo:.2f} (probabilidades out-of-fold del train)",
        "precision_oof": float(precision[i]),
        "recall_oof": float(recall[i]),
        "umbral_f1": float(umbrales[j]),
        "f1_oof": float(f1[j]),
    }


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------

# Color y estilo de línea por modelo (colores de EstiloGraficos): logísticas
# discontinuas, árboles continuos y el benchmark en tinta y más grueso.
ESTILO_CURVAS = {
    "logistica_benchmark": {"color": TEXTO, "linestyle": "-", "linewidth": 3.0, "zorder": 10},
    "logistica_ridge": {"color": NO_DEPRIMIDO, "linestyle": "--", "linewidth": 1.6},
    "logistica_lasso": {"color": CHICOS, "linestyle": "--", "linewidth": 1.6},
    "random_forest": {"color": LINEA_CORTE, "linestyle": "-", "linewidth": 1.6},
    "xgboost": {"color": DEPRIMIDO, "linestyle": "-", "linewidth": 1.6},
    "lightgbm": {"color": CHICAS, "linestyle": "-", "linewidth": 1.6},
    "catboost": {"color": LINEA_MEDIANA, "linestyle": ":", "linewidth": 1.8},
}

# Lo mismo para las claves de las fases 1-3 (logísticas discontinuas, KNN/SVC de trazo y punto)
ESTILO_CURVAS_CLF = {
    CLAVE_BENCHMARK: {"color": TEXTO, "linestyle": "-", "linewidth": 3.0, "zorder": 10},
    "logistica": {"color": NO_DEPRIMIDO, "linestyle": "--", "linewidth": 1.6},
    "ridge": {"color": CHICOS, "linestyle": "--", "linewidth": 1.6},
    "lasso": {"color": CHICAS, "linestyle": "--", "linewidth": 1.6},
    "knn": {"color": LINEA_MEDIANA, "linestyle": "-.", "linewidth": 1.6},
    "svc": {"color": TEXTO_SECUNDARIO, "linestyle": "-.", "linewidth": 1.6},
    "random_forest": {"color": LINEA_CORTE, "linestyle": "-", "linewidth": 1.6},
    "xgboost": {"color": DEPRIMIDO, "linestyle": "-", "linewidth": 1.6},
    "lightgbm": {"color": CHICAS, "linestyle": "-", "linewidth": 1.6},
    "catboost": {"color": LINEA_MEDIANA, "linestyle": ":", "linewidth": 1.8},
}


def graficar_curvas_roc_pr(pipelines: dict, X_test, y_test):
    """Curvas ROC y Precision-Recall de todos los modelos superpuestas, lado a lado (Modelado.ipynb)."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    for clave, pipeline in pipelines.items():
        estilo = ESTILO_CURVAS[clave]
        nombre = NOMBRES_MODELOS[clave]
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax_roc, name=nombre, curve_kwargs=estilo)
        PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test, ax=ax_pr, name=nombre, curve_kwargs=estilo)

    # Referencias de un clasificador aleatorio
    ax_roc.plot([0, 1], [0, 1], linestyle=":", color=TEXTO_SECUNDARIO, linewidth=1, label="Azar (AUC = 0.5)")
    prevalencia = float(np.mean(y_test))
    ax_pr.axhline(prevalencia, linestyle=":", color=TEXTO_SECUNDARIO, linewidth=1,
                  label=f"Azar (prevalencia = {prevalencia:.3f})")

    ax_roc.set_title("Curvas ROC (test)", fontweight="bold")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax_pr.set_title("Curvas Precision-Recall (test)", fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=9, loc="lower right" if ax is ax_roc else "upper right")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_curvas_test(probas: dict, y_test):
    """Curvas ROC y Precision-Recall (test) a partir de las probabilidades de cada
    modelo (indexado por clave), con el benchmark resaltado."""
    y = np.asarray(y_test)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    for clave, proba in probas.items():
        estilo = ESTILO_CURVAS_CLF[clave]
        nombre = NOMBRES_FASE2[clave]
        fpr, tpr, _ = roc_curve(y, proba)
        ax_roc.plot(fpr, tpr, label=f"{nombre} (AUC = {roc_auc_score(y, proba):.3f})", **estilo)
        precision, recall, _ = precision_recall_curve(y, proba)
        ax_pr.plot(recall, precision, drawstyle="steps-post",
                   label=f"{nombre} (AP = {average_precision_score(y, proba):.3f})", **estilo)

    ax_roc.plot([0, 1], [0, 1], linestyle=":", color=TEXTO_SECUNDARIO, linewidth=1, label="Azar (AUC = 0.5)")
    ax_pr.axhline(y.mean(), linestyle=":", color=TEXTO_SECUNDARIO, linewidth=1,
                  label=f"Azar (prevalencia = {y.mean():.3f})")

    ax_roc.set_title("Curvas ROC (test)", fontweight="bold")
    ax_roc.set_xlabel("Tasa de falsos positivos")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax_pr.set_title("Curvas Precision-Recall (test)", fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_ylim(0, 1.02)
    ax_roc.legend(fontsize=8.5, loc="lower right")
    ax_pr.legend(fontsize=8.5, loc="upper right")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_estrategias(tabla: pd.DataFrame, nombre_modelo: str):
    """Fase 1: PR-AUC de CV (ranking) y precision/recall con umbral 0.5 por estrategia."""
    x = np.arange(len(tabla))
    etiquetas = [ESTRATEGIAS[e] for e in tabla.index]
    fig, (ax_pr, ax_umbral) = plt.subplots(1, 2, figsize=(14, 5))

    ax_pr.errorbar(x, tabla["PR-AUC media"], yerr=tabla["PR-AUC DE"], fmt="o", color=NEUTRO,
                   capsize=6, markersize=9, linewidth=2)
    ax_pr.set_xticks(x, etiquetas)
    ax_pr.set_title("PR-AUC de CV (media ± DE): ranking", fontweight="bold")
    ax_pr.set_ylabel("PR-AUC")

    ancho = 0.38
    ax_umbral.bar(x - ancho / 2, tabla["Precision media"], ancho, yerr=tabla["Precision DE"],
                  color=NO_DEPRIMIDO, capsize=3, label="Precision")
    ax_umbral.bar(x + ancho / 2, tabla["Recall media"], ancho, yerr=tabla["Recall DE"],
                  color=DEPRIMIDO, capsize=3, label="Recall")
    ax_umbral.set_xticks(x, etiquetas)
    ax_umbral.set_ylim(0, 1)
    ax_umbral.set_title(f"Precision y recall de CV con umbral {UMBRAL}", fontweight="bold")
    ax_umbral.legend(loc="upper right")

    fig.suptitle(f"{nombre_modelo}: estrategias de manejo del desbalance", fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_intervalos_bootstrap(tabla: pd.DataFrame):
    """PR-AUC y ROC-AUC en test con su intervalo bootstrap al 95%; benchmark resaltado."""
    orden = tabla.sort_values("PR-AUC").index
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(orden) + 2), sharey=True)
    for ax, metrica in zip(axes, ["PR-AUC", "ROC-AUC"]):
        for posicion, clave in enumerate(orden):
            fila = tabla.loc[clave]
            error = [[fila[metrica] - fila[f"{metrica} IC 2.5%"]], [fila[f"{metrica} IC 97.5%"] - fila[metrica]]]
            ax.errorbar(fila[metrica], posicion, xerr=error, fmt="o", capsize=4, markersize=7,
                        color=TEXTO if clave == CLAVE_BENCHMARK else NEUTRO)
        ax.axvline(tabla.loc[CLAVE_BENCHMARK, metrica], color=TEXTO, linestyle="--", linewidth=1)
        ax.set_yticks(range(len(orden)), [NOMBRES_FASE2[c] for c in orden])
        ax.set_title(f"{metrica} en test (IC bootstrap 95%)", fontweight="bold")
        ax.set_xlabel(metrica)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_umbral(y, proba, umbrales: dict):
    """Precision, recall y F1 según el umbral (out-of-fold), con los dos umbrales elegidos."""
    precision, recall, valores = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(valores, precision, color=NEUTRO, linewidth=2, label="Precision")
    ax.plot(valores, recall, color=DEPRIMIDO, linewidth=2, label="Recall")
    ax.plot(valores, f1, color=LINEA_MEDIANA, linewidth=2, linestyle="--", label="F1")
    ax.axhline(RECALL_MINIMO, color=DEPRIMIDO, linestyle=":", linewidth=1)
    ax.axvline(umbrales["umbral"], color=TEXTO, linestyle="-", linewidth=1.5,
               label=f"Umbral principal = {umbrales['umbral']:.3f} (recall ≥ {RECALL_MINIMO:.2f})")
    ax.axvline(umbrales["umbral_f1"], color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1.5,
               label=f"Umbral máximo F1 = {umbrales['umbral_f1']:.3f}")
    ax.axvline(UMBRAL, color=LINEA_CORTE, linestyle=":", linewidth=1.5, label=f"Umbral por defecto = {UMBRAL}")
    ax.set_title("Precision, recall y F1 según el umbral (out-of-fold, train)", fontweight="bold")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="center right")
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_matrices(y, predicciones: dict, titulo: str):
    """Matrices de confusión lado a lado; predicciones = {título del panel: clases predichas}."""
    fig, axes = plt.subplots(1, len(predicciones), figsize=(6 * len(predicciones), 5), squeeze=False)
    for ax, (subtitulo, pred) in zip(axes.flat, predicciones.items()):
        sns.heatmap(
            confusion_matrix(y, pred, labels=[0, 1]), annot=True, fmt="d", cmap=CMAP_SECUENCIAL,
            linewidths=2, linecolor=SUPERFICIE, cbar=False, ax=ax, annot_kws={"size": 13},
            xticklabels=["No Deprimido", "Deprimido"], yticklabels=["No Deprimido", "Deprimido"],
        )
        ax.set_title(subtitulo, fontweight="bold")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.grid(False)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def graficar_matrices_confusion(pipelines: dict, clave_mejor: str, X_test, y_test, umbral: float = UMBRAL):
    """Matrices de confusión (test) del benchmark y del mejor modelo, lado a lado (Modelado.ipynb)."""
    predicciones = {
        NOMBRES_MODELOS[clave]: (pipelines[clave].predict_proba(X_test)[:, 1] >= umbral).astype(int)
        for clave in [MODELO_REFERENCIA, clave_mejor]
    }
    return graficar_matrices(y_test, predicciones, f"Matrices de confusión en test (umbral = {umbral})")


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def _versiones() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
        "joblib": joblib.__version__,
    }


def guardar_modelos(pipelines: dict, ruta: Path = RUTA_MODELOS) -> None:
    """Guarda cada Pipeline completo (preprocesamiento + modelo) como <clave>.joblib."""
    ruta.mkdir(parents=True, exist_ok=True)
    for clave, pipeline in pipelines.items():
        joblib.dump(pipeline, ruta / f"{clave}.joblib")


def guardar_metricas(tabla_cv: pd.DataFrame, tabla_test: pd.DataFrame, ruta: Path = RUTA_SALIDA) -> None:
    """Guarda las tablas del benchmarking (valores numéricos, sin formatear)."""
    ruta.mkdir(parents=True, exist_ok=True)
    tabla_cv.to_csv(ruta / "metricas_cv.csv")
    tabla_test.to_csv(ruta / "metricas_test.csv")


def guardar_split(df_train: pd.DataFrame, df_test: pd.DataFrame, ruta: Path = RUTA_SALIDA) -> None:
    """Guarda subject_id y el conjunto (train | test) de cada adolescente."""
    ruta.mkdir(parents=True, exist_ok=True)
    split = pd.concat([
        pd.DataFrame({"subject_id": df_train["subject_id"], "conjunto": "train"}),
        pd.DataFrame({"subject_id": df_test["subject_id"], "conjunto": "test"}),
    ]).sort_values("subject_id")
    split.to_csv(ruta / "split_indices.csv", index=False)


def guardar_metadatos(mejor_modelo: str, hiperparametros: dict, ruta: Path = RUTA_SALIDA) -> dict:
    """Guarda metadatos.json con todo lo necesario para reutilizar los modelos (Modelado.ipynb)."""
    ruta.mkdir(parents=True, exist_ok=True)
    versiones = _versiones()
    versiones.pop("imbalanced-learn")
    metadatos = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "predictores": PREDICTORES,
        "target": TARGET,
        "umbral": UMBRAL,
        "modelo_referencia": MODELO_REFERENCIA,
        "mejor_modelo_pr_auc_cv": mejor_modelo,
        "mejores_hiperparametros": hiperparametros,
        "versiones": versiones,
    }
    with open(ruta / "metadatos.json", "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    return metadatos


def guardar_final(pipeline, clave: str, ganador_fase1: dict, umbrales: dict, metricas_test: pd.DataFrame,
                  explicacion: str, ruta: Path = RUTA_FINAL) -> dict:
    """Guarda en final/: modelo_final.joblib (configurado para CPU), metricas_test.csv y metadatos.json."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    joblib.dump(configurar_dispositivo(pipeline, gpu=False), ruta / "modelo_final.joblib")
    metricas_test.to_csv(ruta / "metricas_test.csv")

    metadatos = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": clave,
        "nombre": NOMBRES_CLF[clave],
        "estrategia_balanceo": ganador_fase1["estrategia"],
        "hiperparametros": ganador_fase1["hiperparametros"],
        "explicacion_eleccion": explicacion,
        "umbral": umbrales["umbral"],
        "criterio_umbral": umbrales["criterio"],
        "umbral_f1": umbrales["umbral_f1"],
        "modelo_referencia": "logística benchmark (class_weight='balanced', sin tuning)",
        "seed": SEED,
        "predictores": PREDICTORES,
        "target": TARGET,
        "dispositivo_por_modelo": {c: dispositivo_modelo(c) for c in NOMBRES_CLF},
        "gpu_detectada": GPU_DISPOSITIVOS,
        "versiones": _versiones(),
    }
    with open(ruta / "metadatos.json", "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    return metadatos


def cargar_modelo(clave: str, ruta: Path = RUTA_MODELOS) -> Pipeline:
    """Carga el Pipeline guardado de un modelo de Modelado.ipynb. Recibe un DataFrame con
    las columnas originales (PREDICTORES) y devuelve predict_proba directamente."""
    if clave not in NOMBRES_MODELOS:
        raise ValueError(f"Clave '{clave}' no válida. Opciones: {', '.join(NOMBRES_MODELOS)}")
    archivo = Path(ruta) / f"{clave}.joblib"
    if not archivo.exists():
        raise FileNotFoundError(
            f"No existe {archivo}. Ejecuta primero notebooks/Modelado.ipynb para entrenar y guardar los modelos."
        )
    return joblib.load(archivo)


def cargar_modelo_final(ruta: Path = RUTA_FINAL) -> tuple[object, dict]:
    """Carga el modelo final (fase 3) y sus metadatos (incluye el umbral elegido)."""
    ruta = Path(ruta)
    if not (ruta / "modelo_final.joblib").exists():
        raise FileNotFoundError(
            f"No existe {ruta / 'modelo_final.joblib'}. Ejecuta primero los notebooks clf_*.ipynb y clf_comparacion.ipynb."
        )
    with open(ruta / "metadatos.json", encoding="utf-8") as archivo:
        metadatos = json.load(archivo)
    return joblib.load(ruta / "modelo_final.joblib"), metadatos
