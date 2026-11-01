"""Modelos por separado para chicos y chicas y comparación de los factores que pesan en cada grupo.

Lo usa notebooks/genero.ipynb. Se entrenan solo los dos modelos ganadores del modelo conjunto:

- Clasificación (`depressed`): CatBoost con estrategia nativa (sin balanceo).
- Regresión (`bdi_total`): ElasticNet con target nativo, predicciones recortadas a [0, 63].

Se usa el MISMO split del modelado (split_indices.csv), filtrado por sexo. Dentro de cada grupo
`sex` es constante, así que los modelos por género usan solo los 5 predictores numéricos.
El tuning y el umbral se eligen solo con el train del grupo; el test se evalúa una única vez.
Todo se guarda en data/processed/modelado/genero/.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from EstiloGraficos import CHICAS, CHICOS, NEUTRO, TEXTO
from Interpretabilidad import (
    importancia_catboost,
    importancia_shap,
    nombres_legibles,
    shap_catboost,
    tabla_coeficientes,
    transformar,
    valores_originales,
)
from Modelado import (
    ESPACIOS_BUSQUEDA,
    N_ITER_FASE1,
    PREDICTORES,
    PREDICTORES_NUMERICOS,
    RECALL_MINIMO,
    RUTA_SALIDA,
    SEED,
    _limpiar_parametros,
    cargar_modelo_final,
    cargar_split,
    crear_cv,
    elegir_umbrales,
    formatear_hiperparametros,
)
from ModeladoRegresion import (
    N_ITER_REG,
    RegresorBDI,
    _limpiar_parametros_reg,
    _versiones_reg,
    cargar_modelo_regresion,
    crear_cv_reg,
    espacio_pipeline,
    metricas_reg,
)

RUTA_GENERO = RUTA_SALIDA / "genero"
RUTA_MODELOS_GENERO = RUTA_GENERO / "modelos"
RUTA_IMPORTANCIAS_CONJUNTO = RUTA_SALIDA / "interpretabilidad" / "importancias.csv"

# Grupo → valor de `sex` en el dataset, nombre legible y color (PALETA_SEXO)
SEXOS = {"chicos": "Boy", "chicas": "Girl"}
NOMBRES_GRUPO = {"chicos": "Chicos", "chicas": "Chicas"}
COLORES_GRUPO = {"chicos": CHICOS, "chicas": CHICAS}

# `sex` es constante dentro de cada grupo: los modelos por género usan solo las 5 numéricas
PREDICTORES_GENERO = PREDICTORES_NUMERICOS

N_ITER_CATBOOST = N_ITER_FASE1["catboost"]      # 25, como el modelo conjunto
N_ITER_ELASTICNET = N_ITER_REG["elasticnet"]    # 10, como el modelo conjunto
N_REPLICAS = 1000

ENFOQUES = {"especifico": "Modelo específico", "conjunto": "Modelo conjunto"}

# Variable de sexo en importancias.csv del modelo conjunto (se excluye del ranking)
NOMBRE_SEXO_CONJUNTO = "Sexo (chica)"

# Método de importancia → (modelo, nombre en las tablas) en importancias.csv
METODOS = {
    "coeficiente": ("elasticnet", "Coeficientes (ElasticNet, |coef.|)"),
    "prediction_values_change": ("catboost", "Importancia nativa (CatBoost)"),
    "shap_global": ("catboost", "Media de |SHAP| (CatBoost)"),
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def preparar_grupos(df: pd.DataFrame) -> dict:
    """Train y test de cada grupo, filtrando por sexo el split de split_indices.csv (no se crea uno nuevo).

    Por grupo devuelve X_train y X_test con los 5 predictores numéricos, los dos targets y
    X_test_conjunto con los 6 predictores (incluido `sex`) para evaluar el modelo conjunto.
    """
    df_train, df_test = cargar_split(df)
    grupos = {}
    for grupo, sexo in SEXOS.items():
        train = df_train[df_train["sex"] == sexo]
        test = df_test[df_test["sex"] == sexo]
        grupos[grupo] = {
            "X_train": train[PREDICTORES_GENERO], "X_test": test[PREDICTORES_GENERO],
            "X_test_conjunto": test[PREDICTORES],
            "y_dep_train": train["depressed"], "y_dep_test": test["depressed"],
            "y_bdi_train": train["bdi_total"], "y_bdi_test": test["bdi_total"],
        }
    return grupos


def tabla_tamanos(grupos: dict) -> pd.DataFrame:
    """Filas, deprimidos y prevalencia de depresión de train y test en cada grupo."""
    filas = []
    for grupo, datos in grupos.items():
        for conjunto in ("train", "test"):
            y = datos[f"y_dep_{conjunto}"]
            filas.append({"Grupo": NOMBRES_GRUPO[grupo], "Conjunto": conjunto.capitalize(), "Filas": len(y),
                          "Deprimidos": int(y.sum()), "Prevalencia": round(float(y.mean()), 3),
                          "BDI medio": round(float(datos[f"y_bdi_{conjunto}"].mean()), 2)})
    return pd.DataFrame(filas).set_index(["Grupo", "Conjunto"])


# ---------------------------------------------------------------------------
# Pipelines y entrenamiento
# ---------------------------------------------------------------------------

def crear_preprocesador_genero() -> ColumnTransformer:
    """StandardScaler en las 5 numéricas. Es un ColumnTransformer con el mismo nombre de paso
    ("numericas") que el modelo conjunto, para reutilizar las funciones de Interpretabilidad.py."""
    return ColumnTransformer(transformers=[("numericas", StandardScaler(), PREDICTORES_GENERO)])


def crear_pipeline_catboost(seed: int = SEED) -> Pipeline:
    """Pipeline StandardScaler → CatBoostClassifier con estrategia nativa (sin pesos de clase)."""
    modelo = CatBoostClassifier(
        task_type="CPU", boosting_type="Plain", allow_writing_files=False,
        thread_count=-1, verbose=0, random_seed=seed,
    )
    return Pipeline(steps=[("preprocesamiento", crear_preprocesador_genero()), ("modelo", modelo)])


def crear_pipeline_elasticnet(seed: int = SEED) -> Pipeline:
    """Pipeline StandardScaler → ElasticNet con target nativo. RegresorBDI recorta las predicciones a [0, 63]."""
    modelo = RegresorBDI(regresor=ElasticNet(max_iter=10000, random_state=seed))
    return Pipeline(steps=[("preprocesamiento", crear_preprocesador_genero()), ("modelo", modelo)])


def ajustar_catboost(X_train, y_train, seed: int = SEED) -> dict:
    """RandomizedSearchCV del CatBoost solo sobre el train del grupo: mismo espacio y n_iter que el
    modelo conjunto, StratifiedKFold(5) y PR-AUC. n_jobs=1 porque CatBoost ya usa todos los hilos.
    El mejor Pipeline queda reentrenado con todo el train del grupo."""
    busqueda = RandomizedSearchCV(
        crear_pipeline_catboost(seed), ESPACIOS_BUSQUEDA["catboost"], n_iter=N_ITER_CATBOOST,
        scoring="average_precision", cv=crear_cv(seed), n_jobs=1, random_state=seed, refit=True,
    )
    busqueda.fit(X_train, y_train)
    indice = busqueda.best_index_
    return {
        "pipeline": busqueda.best_estimator_,
        "hiperparametros": _limpiar_parametros(busqueda.best_params_),
        "pr_auc_cv_media": float(busqueda.cv_results_["mean_test_score"][indice]),
        "pr_auc_cv_de": float(busqueda.cv_results_["std_test_score"][indice]),
    }


def ajustar_elasticnet(X_train, y_train, seed: int = SEED) -> dict:
    """RandomizedSearchCV del ElasticNet solo sobre el train del grupo: KFold(5), RMSE, n_iter=10,
    alpha log-uniforme en [1e-4, 1e1] y l1_ratio en [0.1, 0.9] (el espacio del modelo conjunto)."""
    busqueda = RandomizedSearchCV(
        crear_pipeline_elasticnet(seed), espacio_pipeline("elasticnet"), n_iter=N_ITER_ELASTICNET,
        scoring="neg_root_mean_squared_error", cv=crear_cv_reg(seed), n_jobs=-1, random_state=seed, refit=True,
    )
    busqueda.fit(X_train, y_train)
    indice = busqueda.best_index_
    return {
        "pipeline": busqueda.best_estimator_,
        "hiperparametros": _limpiar_parametros_reg(busqueda.best_params_),
        "rmse_cv_media": float(-busqueda.cv_results_["mean_test_score"][indice]),
        "rmse_cv_de": float(busqueda.cv_results_["std_test_score"][indice]),
    }


def elegir_umbral_grupo(pipeline, X_train, y_train, seed: int = SEED) -> dict:
    """Umbral de tamizaje del grupo con probabilidades out-of-fold de su train (mismos folds del
    tuning): la mayor precision con recall >= 0.70, el mismo criterio del modelo conjunto."""
    proba_oof = cross_val_predict(
        clone(pipeline), X_train, y_train, cv=crear_cv(seed), method="predict_proba", n_jobs=1,
    )[:, 1]
    return elegir_umbrales(y_train, proba_oof, recall_minimo=RECALL_MINIMO)


def tabla_entrenamiento(resultados: dict) -> pd.DataFrame:
    """Resumen por grupo: hiperparámetros elegidos, desempeño de CV y umbral de tamizaje."""
    filas = {}
    for grupo, res in resultados.items():
        clf, reg, umbral = res["catboost"], res["elasticnet"], res["umbral"]
        filas[NOMBRES_GRUPO[grupo]] = {
            "CatBoost: hiperparámetros": formatear_hiperparametros(clf["hiperparametros"]),
            "CatBoost: PR-AUC de CV": f"{clf['pr_auc_cv_media']:.3f} ± {clf['pr_auc_cv_de']:.3f}",
            "Umbral de tamizaje": round(umbral["umbral"], 3),
            "Precision OOF": round(umbral["precision_oof"], 3),
            "Recall OOF": round(umbral["recall_oof"], 3),
            "ElasticNet: hiperparámetros": formatear_hiperparametros(reg["hiperparametros"]),
            "ElasticNet: RMSE de CV": f"{reg['rmse_cv_media']:.3f} ± {reg['rmse_cv_de']:.3f}",
        }
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Grupo"
    return tabla


# ---------------------------------------------------------------------------
# Modelos conjuntos (ya guardados)
# ---------------------------------------------------------------------------

def cargar_modelos_conjuntos() -> dict:
    """Clasificador (CatBoost) y regresor (ElasticNet) del modelo conjunto, sin reentrenar, con el umbral guardado."""
    clasificador, metadatos_clf = cargar_modelo_final()
    regresor, metadatos_reg = cargar_modelo_regresion("final")
    return {"clasificador": clasificador, "regresor": regresor, "umbral": metadatos_clf["umbral"],
            "metadatos_clf": metadatos_clf, "metadatos_reg": metadatos_reg}


# ---------------------------------------------------------------------------
# Evaluación en test con intervalos bootstrap
# ---------------------------------------------------------------------------

def metricas_clasificacion_grupo(y, proba, umbral: float) -> dict:
    """PR-AUC, línea base (prevalencia del test del grupo), PR-AUC relativa a ella, ROC-AUC y
    recall / precision / F1 con el umbral dado."""
    y, proba = np.asarray(y), np.asarray(proba, dtype=float)
    prediccion = (proba >= umbral).astype(int)
    pr_auc, linea_base = average_precision_score(y, proba), y.mean()
    return {
        "PR-AUC": pr_auc,
        "Línea base": linea_base,
        "PR-AUC relativa": pr_auc / linea_base,
        "ROC-AUC": roc_auc_score(y, proba),
        "Umbral": umbral,
        "Recall": recall_score(y, prediccion),
        "Precision": precision_score(y, prediccion, zero_division=0),
        "F1": f1_score(y, prediccion),
    }


def _indices_bootstrap(n: int, n_replicas: int, seed: int) -> np.ndarray:
    """Matriz (réplicas × n) de índices remuestreados con reemplazo. Los dos enfoques se evalúan
    con las mismas réplicas."""
    return np.random.default_rng(seed).integers(0, n, size=(n_replicas, n))


def _intervalo(valores) -> tuple[float, float]:
    """Percentiles 2.5 y 97.5 (intervalo bootstrap al 95%)."""
    inferior, superior = np.percentile(np.asarray(valores, dtype=float), [2.5, 97.5])
    return float(inferior), float(superior)


def evaluar_clasificacion_grupo(grupo: str, y, probas: dict, umbrales: dict,
                                n_replicas: int = N_REPLICAS, seed: int = SEED) -> pd.DataFrame:
    """Métricas en el test del grupo de cada enfoque ({enfoque: probabilidades}), en formato largo,
    con el intervalo bootstrap al 95% de la PR-AUC y la ROC-AUC. Se omiten las réplicas con una sola clase."""
    y = np.asarray(y)
    indices = _indices_bootstrap(len(y), n_replicas, seed)
    indices = indices[[y[fila].min() != y[fila].max() for fila in indices]]
    filas = []
    for enfoque, proba in probas.items():
        proba = np.asarray(proba, dtype=float)
        intervalos = {
            "PR-AUC": _intervalo([average_precision_score(y[fila], proba[fila]) for fila in indices]),
            "ROC-AUC": _intervalo([roc_auc_score(y[fila], proba[fila]) for fila in indices]),
        }
        for metrica, valor in metricas_clasificacion_grupo(y, proba, umbrales[enfoque]).items():
            inferior, superior = intervalos.get(metrica, (np.nan, np.nan))
            filas.append({"grupo": grupo, "tarea": "clasificacion", "enfoque": enfoque, "metrica": metrica,
                          "valor": float(valor), "ic_inf": inferior, "ic_sup": superior})
    return pd.DataFrame(filas)


def evaluar_regresion_grupo(grupo: str, y, predicciones: dict,
                            n_replicas: int = N_REPLICAS, seed: int = SEED) -> pd.DataFrame:
    """RMSE, MAE y R² en el test del grupo de cada enfoque ({enfoque: BDI predicho}), en formato largo,
    con el intervalo bootstrap al 95% del RMSE y del R²."""
    y = np.asarray(y, dtype=float)
    indices = _indices_bootstrap(len(y), n_replicas, seed)
    filas = []
    for enfoque, prediccion in predicciones.items():
        prediccion = np.asarray(prediccion, dtype=float)
        errores = (y[indices] - prediccion[indices]) ** 2
        intervalos = {
            "RMSE": _intervalo(np.sqrt(errores.mean(axis=1))),
            "R²": _intervalo([r2_score(y[fila], prediccion[fila]) for fila in indices]),
        }
        for metrica, valor in metricas_reg(y, prediccion).items():
            inferior, superior = intervalos.get(metrica, (np.nan, np.nan))
            filas.append({"grupo": grupo, "tarea": "regresion", "enfoque": enfoque, "metrica": metrica,
                          "valor": float(valor), "ic_inf": inferior, "ic_sup": superior})
    return pd.DataFrame(filas)


def vista_metricas(metricas: pd.DataFrame, tarea: str, decimales: int = 3) -> pd.DataFrame:
    """Tabla para mostrar: una fila por grupo y enfoque, cada métrica como "valor [ic_inf, ic_sup]"
    cuando tiene intervalo bootstrap."""
    datos = metricas[metricas["tarea"] == tarea].copy()
    datos["texto"] = [
        f"{v:.{decimales}f}" if np.isnan(i) else f"{v:.{decimales}f} [{i:.{decimales}f}, {s:.{decimales}f}]"
        for v, i, s in zip(datos["valor"], datos["ic_inf"], datos["ic_sup"])
    ]
    datos["Grupo"] = datos["grupo"].map(NOMBRES_GRUPO)
    datos["Enfoque"] = datos["enfoque"].map(ENFOQUES)
    orden = list(dict.fromkeys(datos["metrica"]))
    tabla = datos.pivot(index=["Grupo", "Enfoque"], columns="metrica", values="texto")[orden]
    tabla.columns.name = None
    return tabla.reindex([(NOMBRES_GRUPO[g], ENFOQUES[e]) for g in SEXOS for e in ENFOQUES])


def graficar_intervalos_grupos(metricas: pd.DataFrame, tarea: str, metricas_grafico: list[str]):
    """Valor e intervalo bootstrap al 95% de cada métrica por grupo y enfoque. El color indica el
    grupo y el marcador el enfoque: círculo relleno = específico, cuadrado gris = conjunto."""
    datos = metricas[metricas["tarea"] == tarea]
    fig, axes = plt.subplots(1, len(metricas_grafico), figsize=(6.5 * len(metricas_grafico), 3.8), squeeze=False)
    etiquetas = [f"{NOMBRES_GRUPO[g]}: {ENFOQUES[e].lower()}" for g in SEXOS for e in ENFOQUES]
    for ax, metrica in zip(axes.flat, metricas_grafico):
        for posicion, (grupo, enfoque) in enumerate((g, e) for g in SEXOS for e in ENFOQUES):
            fila = datos[(datos["grupo"] == grupo) & (datos["enfoque"] == enfoque) & (datos["metrica"] == metrica)].iloc[0]
            error = [[fila["valor"] - fila["ic_inf"]], [fila["ic_sup"] - fila["valor"]]]
            color = COLORES_GRUPO[grupo] if enfoque == "especifico" else NEUTRO
            ax.errorbar(fila["valor"], posicion, xerr=error, fmt="o" if enfoque == "especifico" else "s",
                        color=color, ecolor=COLORES_GRUPO[grupo], capsize=4, markersize=8, linewidth=2)
        ax.set_yticks(range(len(etiquetas)), etiquetas)
        ax.invert_yaxis()
        ax.set_title(f"{metrica} en test (IC bootstrap 95%)", fontweight="bold")
        ax.set_xlabel(metrica)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interpretabilidad por género
# ---------------------------------------------------------------------------

def explicar_grupo(pipeline_clf, pipeline_reg, X_test: pd.DataFrame, y_dep_test) -> dict:
    """Coeficientes del ElasticNet, importancia nativa y SHAP (log-odds) del CatBoost del grupo,
    calculados sobre el test del grupo. Reutiliza las funciones de Interpretabilidad.py."""
    prep_reg, elasticnet = pipeline_reg.named_steps["preprocesamiento"], pipeline_reg.named_steps["modelo"].regresor_
    prep_clf, catboost = pipeline_clf.named_steps["preprocesamiento"], pipeline_clf.named_steps["modelo"]
    nombres_transformados = list(prep_clf.get_feature_names_out())
    nombres = nombres_legibles(nombres_transformados)

    X_test_clf = transformar(prep_clf, X_test)
    coeficientes = tabla_coeficientes(prep_reg, elasticnet, list(prep_reg.get_feature_names_out()))
    nativa = importancia_catboost(catboost, X_test_clf, np.asarray(y_dep_test), nombres)["PredictionValuesChange"]
    explicacion = shap_catboost(catboost, X_test_clf, valores_originales(X_test, nombres_transformados))
    return {
        "coeficientes": coeficientes,
        "intercepto": float(elasticnet.intercept_),
        "prediction_values_change": nativa.sort_values(ascending=False),
        "shap": explicacion,
        "shap_global": importancia_shap(explicacion),
    }


def graficar_barras_grupo(valores: pd.Series, grupo: str, titulo: str, etiqueta_x: str):
    """Barras horizontales de una importancia de un grupo, en el color del grupo."""
    orden = valores.sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(orden.index, orden.to_numpy(), color=COLORES_GRUPO[grupo])
    for posicion, valor in enumerate(orden.to_numpy()):
        ax.text(valor, posicion, f"  {valor:.3f}", va="center", fontsize=9, color=TEXTO)
    ax.margins(x=0.15)
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(etiqueta_x)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_comparacion_grupos(valores: dict, titulo: str, etiqueta_x: str):
    """Barras horizontales agrupadas chicos vs. chicas ({grupo: Serie por variable}), ordenadas por la
    media de |valor| de los dos grupos. Admite valores con signo (coeficientes)."""
    tabla = pd.DataFrame(valores)
    tabla = tabla.reindex(tabla.abs().mean(axis=1).sort_values().index)
    posiciones, alto = np.arange(len(tabla)), 0.38
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(tabla) + 1.5))
    for desplazamiento, grupo in zip((alto / 2, -alto / 2), SEXOS):
        ax.barh(posiciones + desplazamiento, tabla[grupo], alto, color=COLORES_GRUPO[grupo], label=NOMBRES_GRUPO[grupo])
    if (tabla.to_numpy() < 0).any():
        ax.axvline(0, color=TEXTO, linewidth=1)
    ax.set_yticks(posiciones, tabla.index)
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(etiqueta_x)
    ax.legend(loc="lower right")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


def importancias_largas(explicaciones: dict) -> pd.DataFrame:
    """Formato largo para importancias.csv: grupo, modelo, metodo, variable, valor (coeficientes con signo)."""
    filas = []
    for grupo, exp in explicaciones.items():
        series = {
            "coeficiente": exp["coeficientes"]["Coeficiente (por DE)"],
            "prediction_values_change": exp["prediction_values_change"],
            "shap_global": exp["shap_global"],
        }
        for metodo, serie in series.items():
            for variable, valor in serie.items():
                filas.append({"grupo": grupo, "modelo": METODOS[metodo][0], "metodo": metodo,
                              "variable": variable, "valor": float(valor)})
    return pd.DataFrame(filas)


def cargar_importancias_conjunto(ruta: Path = RUTA_IMPORTANCIAS_CONJUNTO) -> pd.DataFrame | None:
    """importancias.csv del modelo conjunto (interpretabilidad.ipynb), o None con un aviso si no existe."""
    ruta = Path(ruta)
    if not ruta.exists():
        print(f"⚠ No existe {ruta}. Ejecuta primero notebooks/interpretabilidad.ipynb; "
              "la columna del modelo conjunto quedará vacía.")
        return None
    return pd.read_csv(ruta)


def tabla_ranking(importancias: pd.DataFrame, importancias_conjunto: pd.DataFrame | None) -> pd.DataFrame:
    """Puesto de cada variable por método (1 = la más importante) en el modelo conjunto, chicos y chicas.

    Los coeficientes se ordenan por valor absoluto. Del modelo conjunto se excluye `sex` para que
    los puestos sean comparables (5 variables en los tres casos).
    """
    indice = pd.MultiIndex.from_tuples(
        [(nombre, variable) for _, nombre in METODOS.values() for variable in nombres_legibles(PREDICTORES_GENERO)],
        names=["Método", "Variable"],
    )
    columnas = {}
    fuentes = {"conjunto": importancias_conjunto, **{g: importancias[importancias["grupo"] == g] for g in SEXOS}}
    for fuente, datos in fuentes.items():
        puestos = {}
        for metodo, (modelo, nombre_metodo) in METODOS.items():
            if datos is None:
                continue
            serie = datos[(datos["modelo"] == modelo) & (datos["metodo"] == metodo)].set_index("variable")["valor"]
            serie = serie.drop(index=NOMBRE_SEXO_CONJUNTO, errors="ignore").abs()
            for variable, puesto in serie.rank(ascending=False, method="min").items():
                puestos[(nombre_metodo, variable)] = int(puesto)
        # Sin datos (importancias.csv del conjunto ausente) la columna queda vacía (<NA>)
        columna = pd.Series([puestos.get(clave) for clave in indice], index=indice, dtype="Int64")
        columnas["Conjunto" if fuente == "conjunto" else NOMBRES_GRUPO[fuente]] = columna

    tabla = pd.DataFrame(columnas)
    # Dentro de cada método, las variables en el orden del modelo conjunto (si falta, se conserva el orden de PREDICTORES)
    return pd.concat([tabla.loc[[nombre]].sort_values("Conjunto", kind="stable") for _, nombre in METODOS.values()])


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def guardar_resultados(resultados: dict, metricas: pd.DataFrame, importancias: pd.DataFrame,
                       tamanos: pd.DataFrame, ruta: Path = RUTA_GENERO) -> dict:
    """Guarda en genero/: modelos/catboost_<grupo>.joblib y modelos/elasticnet_<grupo>.joblib,
    metricas_test.csv, importancias.csv y metadatos.json."""
    ruta = Path(ruta)
    (ruta / "modelos").mkdir(parents=True, exist_ok=True)
    for grupo, res in resultados.items():
        joblib.dump(res["catboost"]["pipeline"], ruta / "modelos" / f"catboost_{grupo}.joblib")
        joblib.dump(res["elasticnet"]["pipeline"], ruta / "modelos" / f"elasticnet_{grupo}.joblib")
    metricas[["grupo", "tarea", "enfoque", "metrica", "valor", "ic_inf", "ic_sup"]].to_csv(ruta / "metricas_test.csv", index=False)
    importancias[["grupo", "modelo", "metodo", "variable", "valor"]].to_csv(ruta / "importancias.csv", index=False)

    metadatos = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "predictores": PREDICTORES_GENERO,
        "split": "data/processed/modelado/split_indices.csv filtrado por sexo",
        "grupos": {
            grupo: {
                "sexo": SEXOS[grupo],
                "tamanos": {
                    conjunto.lower(): {"filas": int(fila["Filas"]), "deprimidos": int(fila["Deprimidos"]),
                                       "prevalencia": float(fila["Prevalencia"])}
                    for conjunto, fila in tamanos.loc[NOMBRES_GRUPO[grupo]].iterrows()
                },
                "catboost": {
                    "archivo": f"modelos/catboost_{grupo}.joblib",
                    "estrategia_balanceo": "nativo",
                    "hiperparametros": res["catboost"]["hiperparametros"],
                    "pr_auc_cv_media": res["catboost"]["pr_auc_cv_media"],
                    "pr_auc_cv_de": res["catboost"]["pr_auc_cv_de"],
                    "umbral": res["umbral"]["umbral"],
                    "criterio_umbral": res["umbral"]["criterio"],
                    "precision_oof": res["umbral"]["precision_oof"],
                    "recall_oof": res["umbral"]["recall_oof"],
                },
                "elasticnet": {
                    "archivo": f"modelos/elasticnet_{grupo}.joblib",
                    "variante_target": "nativo",
                    "recorte_predicciones": [0, 63],
                    "hiperparametros": res["elasticnet"]["hiperparametros"],
                    "rmse_cv_media": res["elasticnet"]["rmse_cv_media"],
                    "rmse_cv_de": res["elasticnet"]["rmse_cv_de"],
                },
            }
            for grupo, res in resultados.items()
        },
        "versiones": {**_versiones_reg(), "shap": shap.__version__},
    }
    with open(ruta / "metadatos.json", "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    print(f"✓ Guardado en {ruta}: modelos/, metricas_test.csv, importancias.csv y metadatos.json")
    return metadatos
