"""Funciones reutilizables para el modelado de clasificación del estado depresivo.

Las usa notebooks/Modelado.ipynb: división train/test, definición de modelos y
espacios de búsqueda, tuning, benchmarking (CV y test), gráficas y persistencia
de los modelos entrenados en data/processed/modelado/.

Uso para cargar un modelo ya entrenado:

    from Modelado import cargar_modelo, PREDICTORES
    modelo = cargar_modelo("catboost")
    probabilidades = modelo.predict_proba(df[PREDICTORES])[:, 1]
"""

import json
import platform
from datetime import datetime
from pathlib import Path

import catboost
import joblib
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from EstiloGraficos import (
    CHICAS, CHICOS, CMAP_SECUENCIAL, DEPRIMIDO, LINEA_CORTE, LINEA_MEDIANA,
    NO_DEPRIMIDO, SUPERFICIE, TEXTO, TEXTO_SECUNDARIO,
)


SEED = 42
UMBRAL = 0.5

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

# Clave del archivo .joblib → nombre legible para tablas y gráficas
NOMBRES_MODELOS = {
    "logistica_benchmark": "Logística benchmark",
    "logistica_ridge": "Logística Ridge (L2)",
    "logistica_lasso": "Logística Lasso (L1)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
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


# ---------------------------------------------------------------------------
# Modelos y espacios de búsqueda
# ---------------------------------------------------------------------------

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
    """Devuelve {clave: (pipeline, espacio_de_búsqueda, n_iter)} para los 7 modelos.

    El benchmark no tiene espacio (None): se entrena tal cual, sin tuning.
    Los parámetros llevan el prefijo "modelo__" porque viven dentro del Pipeline.
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
            {"modelo__C": loguniform(1e-3, 1e2)},
            10,
        ),
        "logistica_lasso": (
            # l1_ratio=1 equivale a penalty="l1"
            crear_pipeline_modelo(LogisticRegression(
                l1_ratio=1, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=seed,
            )),
            {"modelo__C": loguniform(1e-3, 1e2)},
            10,
        ),
        "random_forest": (
            crear_pipeline_modelo(RandomForestClassifier(
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            )),
            {
                "modelo__n_estimators": randint(200, 801),
                "modelo__max_depth": [3, 5, 8, 12, None],
                "modelo__min_samples_leaf": [5, 10, 20, 50],
                "modelo__max_features": ["sqrt", 0.5, 1.0],
            },
            40,
        ),
        "xgboost": (
            crear_pipeline_modelo(XGBClassifier(
                scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=seed, n_jobs=1,
            )),
            {
                "modelo__n_estimators": randint(100, 601),
                "modelo__max_depth": randint(2, 7),
                "modelo__learning_rate": loguniform(0.01, 0.2),
                "modelo__subsample": uniform(0.6, 0.4),          # uniforme en [0.6, 1.0]
                "modelo__colsample_bytree": uniform(0.6, 0.4),
                "modelo__min_child_weight": randint(1, 11),
            },
            40,
        ),
        "lightgbm": (
            # subsample_freq=1: sin él, LightGBM ignora subsample
            crear_pipeline_modelo(LGBMClassifier(
                class_weight="balanced", subsample_freq=1, random_state=seed, n_jobs=1, verbose=-1,
            )),
            {
                "modelo__n_estimators": randint(100, 601),
                "modelo__num_leaves": randint(4, 32),
                "modelo__max_depth": [-1, 3, 5, 7],
                "modelo__learning_rate": loguniform(0.01, 0.2),
                "modelo__min_child_samples": randint(10, 101),
                "modelo__subsample": uniform(0.6, 0.4),
                "modelo__colsample_bytree": uniform(0.6, 0.4),
            },
            40,
        ),
        "catboost": (
            # allow_writing_files=False evita que CatBoost cree la carpeta catboost_info/
            crear_pipeline_modelo(CatBoostClassifier(
                auto_class_weights="Balanced", random_seed=seed, thread_count=1,
                verbose=0, allow_writing_files=False,
            )),
            {
                "modelo__iterations": randint(200, 801),
                "modelo__depth": randint(3, 8),
                "modelo__learning_rate": loguniform(0.01, 0.2),
                "modelo__l2_leaf_reg": uniform(1, 9),             # uniforme en [1, 10]
            },
            40,
        ),
    }


def ajustar_modelos(modelos: dict, X_train, y_train, cv, seed: int = SEED) -> tuple[dict, dict]:
    """Ajusta cada modelo sobre el train.

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
            # Se quita el prefijo "modelo__" y se pasan los tipos numpy a tipos nativos (para el JSON)
            hiperparametros[clave] = {
                nombre.removeprefix("modelo__"): (valor.item() if isinstance(valor, np.generic) else valor)
                for nombre, valor in busqueda.best_params_.items()
            }
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
# Benchmarking
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
        fila = {}
        for metrica, nombre in METRICAS.items():
            fila[f"{nombre} media"] = resultados[f"test_{metrica}"].mean()
            fila[f"{nombre} DE"] = resultados[f"test_{metrica}"].std()
        filas[NOMBRES_MODELOS[clave]] = fila
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Modelo"
    return _agregar_diferencia(tabla, "PR-AUC media")


def formatear_tabla_cv(tabla_cv: pd.DataFrame, decimales: int = 3) -> pd.DataFrame:
    """Versión para mostrar de la tabla de CV: cada métrica como "media ± DE"."""
    vista = pd.DataFrame(index=tabla_cv.index)
    for nombre in METRICAS.values():
        vista[nombre] = [
            f"{media:.{decimales}f} ± {de:.{decimales}f}"
            for media, de in zip(tabla_cv[f"{nombre} media"], tabla_cv[f"{nombre} DE"])
        ]
    vista["Δ PR-AUC vs benchmark"] = tabla_cv["Δ PR-AUC vs benchmark"].round(decimales)
    return vista


def evaluar_test(pipelines: dict, X_test, y_test, umbral: float = UMBRAL) -> pd.DataFrame:
    """Métricas en el test de cada Pipeline ya entrenado; las de clase usan el umbral dado."""
    filas = {}
    for clave, pipeline in pipelines.items():
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= umbral).astype(int)
        filas[NOMBRES_MODELOS[clave]] = {
            "PR-AUC": average_precision_score(y_test, proba),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "Accuracy": accuracy_score(y_test, pred),
        }
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Modelo"
    return _agregar_diferencia(tabla, "PR-AUC")


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


def graficar_curvas_roc_pr(pipelines: dict, X_test, y_test):
    """Curvas ROC y Precision-Recall de todos los modelos superpuestas, lado a lado."""
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


def graficar_matrices_confusion(pipelines: dict, clave_mejor: str, X_test, y_test, umbral: float = UMBRAL):
    """Matrices de confusión (test) del benchmark y del mejor modelo, lado a lado."""
    claves = [MODELO_REFERENCIA, clave_mejor]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, clave in zip(axes, claves):
        pred = (pipelines[clave].predict_proba(X_test)[:, 1] >= umbral).astype(int)
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=CMAP_SECUENCIAL, linewidths=2, linecolor=SUPERFICIE,
            cbar=False, ax=ax, annot_kws={"size": 13},
            xticklabels=["No Deprimido", "Deprimido"], yticklabels=["No Deprimido", "Deprimido"],
        )
        ax.set_title(NOMBRES_MODELOS[clave], fontweight="bold")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.grid(False)
    fig.suptitle(f"Matrices de confusión en test (umbral = {umbral})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

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
    """Guarda metadatos.json con todo lo necesario para reutilizar los modelos."""
    ruta.mkdir(parents=True, exist_ok=True)
    metadatos = {
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "predictores": PREDICTORES,
        "target": TARGET,
        "umbral": UMBRAL,
        "modelo_referencia": MODELO_REFERENCIA,
        "mejor_modelo_pr_auc_cv": mejor_modelo,
        "mejores_hiperparametros": hiperparametros,
        "versiones": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    with open(ruta / "metadatos.json", "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    return metadatos


def cargar_modelo(clave: str, ruta: Path = RUTA_MODELOS) -> Pipeline:
    """Carga el Pipeline guardado de un modelo. Recibe un DataFrame con las columnas
    originales (PREDICTORES) y devuelve predict_proba directamente."""
    if clave not in NOMBRES_MODELOS:
        raise ValueError(f"Clave '{clave}' no válida. Opciones: {', '.join(NOMBRES_MODELOS)}")
    archivo = Path(ruta) / f"{clave}.joblib"
    if not archivo.exists():
        raise FileNotFoundError(
            f"No existe {archivo}. Ejecuta primero notebooks/Modelado.ipynb para entrenar y guardar los modelos."
        )
    return joblib.load(archivo)
