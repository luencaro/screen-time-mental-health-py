"""Exportación de los resultados del modelado y del clustering para el dashboard.

Lo usa notebooks/exportar_dashboard.ipynb. Se ejecuta UNA vez en el contenedor de Jupyter, después
de todos los notebooks de modelado (clf_*, clf_comparacion, reg_*, reg_comparacion,
interpretabilidad y genero) y de clustering.ipynb.

No reentrena los modelos finales ni repite el tuning: carga los pipelines guardados y replica lo
que hacen clf_comparacion y reg_comparacion (mismo split de split_indices.csv, mismos folds,
mismas métricas y mismo umbral), reutilizando las funciones de Modelado.py y ModeladoRegresion.py.

El contenedor del dashboard solo instala pandas, numpy, scipy, plotly, dash y catboost, así que
todo se escribe en data/processed/dashboard/ como CSV y JSON. Los modelos de la pestaña de
predicción se guardan en formatos portables (CatBoost nativo .cbm y JSON con el preprocesamiento
y los coeficientes) y se verifica que reproducen los pipelines originales con el mismo código que
usa el dashboard (data/prediccion.py).
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import catboost
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))   # para importar data/load_data.py y data/prediccion.py

from data.load_data import ETIQUETAS, ETIQUETAS_SEXO, RUTA_CSV
from Clustering import RUTA_CLUSTERING, VARIABLE_ORDEN, VARIABLES_CLUSTER, datos_estandarizados
from Interpretabilidad import NOMBRES, RUTA_INTERPRETABILIDAD
from Modelado import (
    CLAVE_BENCHMARK,
    ESTRATEGIAS,
    METRICAS,
    NOMBRES_CLF,
    NOMBRES_FASE2,
    PREDICTORES,
    PREDICTORES_NUMERICOS,
    RECALL_MINIMO,
    SEED,
    UMBRAL,
    cargar_cv_fase1,
    cargar_ganador_fase1,
    cargar_modelo_final,
    cargar_split,
    crear_benchmark,
    crear_cv,
    elegir_umbrales,
    evaluar_test_modelos,
    intervalos_bootstrap,
    metricas_con_umbral,
    probabilidades_oof,
    separar_x_y,
)
from ModeladoGenero import NOMBRES_GRUPO, RUTA_GENERO, SEXOS
from ModeladoRegresion import (
    CORTE_BDI,
    METRICAS_REG,
    NOMBRES_FASE2_REG,
    NOMBRES_REG,
    TARGET_CLF,
    VARIANTES,
    cargar_ganador_reg,
    cargar_modelo_regresion,
    comparar_auc_bootstrap,
    crear_benchmark_reg,
    crear_cv_reg,
    evaluar_cv_reg,
    evaluar_test_reg,
    metricas_reg,
    metricas_riesgo,
    separar_x_y_reg,
)

RUTA_DASHBOARD = RAIZ / "data" / "processed" / "dashboard"
RUTA_PREDICCION = RUTA_DASHBOARD / "prediccion"

N_REPLICAS = 1000        # réplicas bootstrap, como en clf_comparacion y reg_comparacion
N_VERIFICACION = 200     # adolescentes del test con los que se verifican los modelos portables

# Logística benchmark: CV del EDA (EDA.ipynb, todo el dataset), copiada de clf_comparacion.ipynb,
# que la usa como fila de referencia de la tabla de CV. No es comparable con la CV del train.
CV_BENCHMARK_EDA = {
    "PR-AUC media": 0.399, "PR-AUC DE": 0.011,
    "ROC-AUC media": 0.766, "ROC-AUC DE": 0.007,
    "Precision media": 0.320, "Precision DE": 0.013,
    "Recall media": 0.666, "Recall DE": 0.035,
    "F1 media": 0.432, "F1 DE": 0.019,
    "Accuracy media": 0.714, "Accuracy DE": 0.009,
}
ESTRATEGIA_BENCHMARK = "pesos"   # LogisticRegression(class_weight="balanced")

# Unidad y descripción de cada predictor numérico para los controles de la pestaña de predicción
UNIDADES_CONTROL = {
    "sleep_quality_index": ("puntos del índice", "Un valor más alto indica peor calidad del sueño."),
    "avg_sleep_hours": ("horas", "Horas de sueño promedio por noche."),
    "midsleep_weekend_hours": ("horas", "Hora del punto medio del sueño en fin de semana."),
    "social_jetlag_hours": ("horas", "Diferencia de horario de sueño entre semana y fin de semana."),
    "screen_time_index": ("puntos del índice", "Un valor más alto indica más tiempo de pantalla."),
}
MAX_VALORES_DISCRETOS = 40   # variables con pocos valores distintos: el control solo toma esos valores
PASO_CONTINUAS = 0.05


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def _a_json(valor):
    """Convierte tipos numpy/pandas a tipos nativos y NaN a None, para un JSON válido."""
    if isinstance(valor, dict):
        return {str(clave): _a_json(v) for clave, v in valor.items()}
    if isinstance(valor, (list, tuple, np.ndarray, pd.Series)):
        return [_a_json(v) for v in list(valor)]
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor


def _escribir_csv(tabla: pd.DataFrame, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return ruta


def _escribir_json(datos: dict, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(_a_json(datos), archivo, ensure_ascii=False, indent=2)
    return ruta


def _leer_json(ruta: Path) -> dict:
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def _exigir(ruta: Path, notebook: str) -> Path:
    """Devuelve la ruta si existe; si no, un error que dice qué notebook ejecutar."""
    if not Path(ruta).exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero {notebook}.")
    return Path(ruta)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def preparar_datos(ruta_csv: Path = RUTA_CSV) -> dict:
    """Dataset crudo (como lo leen los notebooks de modelado) y su split de split_indices.csv."""
    df = pd.read_csv(ruta_csv)
    df_train, df_test = cargar_split(df)
    return {"df": df, "df_train": df_train, "df_test": df_test}


# ---------------------------------------------------------------------------
# 1-2. Clasificación: benchmarking, curvas y modelo final (clf_comparacion.ipynb)
# ---------------------------------------------------------------------------

def _largo_cv(tabla_cv: pd.DataFrame, metricas, nombres: dict, columna_estrategia: str,
              estrategias: dict, clave_final: str) -> pd.DataFrame:
    """Tabla de CV ancha ("<métrica> media" / "<métrica> DE", índice = clave) en formato largo."""
    filas = []
    for clave, fila in tabla_cv.iterrows():
        for metrica in metricas:
            filas.append({
                "modelo": clave, "nombre": nombres[clave], columna_estrategia: estrategias[clave],
                "metrica": metrica, "media": float(fila[f"{metrica} media"]), "desviacion": float(fila[f"{metrica} DE"]),
                "es_referencia": clave == CLAVE_BENCHMARK, "es_final": clave == clave_final,
            })
    return pd.DataFrame(filas)


def _curvas(y, probas: dict) -> pd.DataFrame:
    """Curvas ROC (x = FPR, y = TPR) y Precision-Recall (x = recall, y = precision) en test."""
    partes = []
    for clave, proba in probas.items():
        fpr, tpr, _ = roc_curve(y, proba)
        precision, recall, _ = precision_recall_curve(y, proba)
        partes.append(pd.DataFrame({"modelo": clave, "tipo": "roc", "x": fpr, "y": tpr}))
        partes.append(pd.DataFrame({"modelo": clave, "tipo": "pr", "x": recall, "y": precision}))
    return pd.concat(partes, ignore_index=True)


def exportar_clasificacion(datos: dict, ruta: Path = RUTA_DASHBOARD) -> tuple[list[Path], dict]:
    """Benchmarking de los ganadores de la fase 1 + benchmark, curvas en test, modelo final con su
    umbral, matriz de confusión y curva de métricas vs. umbral (out-of-fold del train)."""
    X_train, y_train = separar_x_y(datos["df_train"])
    X_test, y_test = separar_x_y(datos["df_test"])
    cv = crear_cv(SEED)

    pipelines, ganadores = {}, {}
    for clave in NOMBRES_CLF:
        pipelines[clave], ganadores[clave] = cargar_ganador_fase1(clave)
    # Modelo de referencia fijo: sin hiperparámetros, se entrena con el train como en clf_comparacion
    pipelines = {CLAVE_BENCHMARK: crear_benchmark(SEED).fit(X_train, y_train), **pipelines}
    modelo_final, meta_final = cargar_modelo_final()
    clave_final = meta_final["modelo"]
    estrategias = {CLAVE_BENCHMARK: ESTRATEGIA_BENCHMARK, **{c: g["estrategia"] for c, g in ganadores.items()}}

    # CV: fila de la estrategia ganadora de cada modelo (sin reentrenar) + benchmark del EDA
    tabla_cv = cargar_cv_fase1(ganadores, CV_BENCHMARK_EDA)
    cv_largo = _largo_cv(tabla_cv, METRICAS.values(), NOMBRES_FASE2, "estrategia", estrategias, clave_final)
    cv_largo["fuente_cv"] = np.where(cv_largo["es_referencia"], "eda", "fase1")

    # Test: métricas con umbral 0.5 e intervalos bootstrap pareados de PR-AUC y ROC-AUC
    probas_test = {clave: pipeline.predict_proba(X_test)[:, 1] for clave, pipeline in pipelines.items()}
    tabla_test = evaluar_test_modelos(probas_test, y_test, umbral=UMBRAL)
    tabla_boot = intervalos_bootstrap(y_test, probas_test, n_replicas=N_REPLICAS, seed=SEED)
    filas = []
    for clave in tabla_test.index:
        for metrica in METRICAS.values():
            con_ic = metrica in ("PR-AUC", "ROC-AUC")
            filas.append({
                "modelo": clave, "metrica": metrica, "valor": float(tabla_test.loc[clave, metrica]),
                "ic_inf": float(tabla_boot.loc[clave, f"{metrica} IC 2.5%"]) if con_ic else np.nan,
                "ic_sup": float(tabla_boot.loc[clave, f"{metrica} IC 97.5%"]) if con_ic else np.nan,
                "umbral": UMBRAL,
                "delta_referencia": float(tabla_boot.loc[clave, f"Δ {metrica} vs benchmark"]) if con_ic else np.nan,
                "delta_ic_inf": float(tabla_boot.loc[clave, f"Δ {metrica} IC 2.5%"]) if con_ic else np.nan,
                "delta_ic_sup": float(tabla_boot.loc[clave, f"Δ {metrica} IC 97.5%"]) if con_ic else np.nan,
                "delta_significativa": bool(tabla_boot.loc[clave, f"Δ {metrica} significativa"]) if con_ic else None,
            })
    test_largo = pd.DataFrame(filas)

    # Modelo final: probabilidades del pipeline guardado en final/ y umbral de sus metadatos
    proba_final = modelo_final.predict_proba(X_test)[:, 1]
    umbral = float(meta_final["umbral"])
    prediccion = (proba_final >= umbral).astype(int)
    matriz = confusion_matrix(y_test, prediccion, labels=[0, 1])
    tabla_matriz = pd.DataFrame([
        {"real": real, "predicho": predicho, "n": int(matriz[real, predicho])} for real in (0, 1) for predicho in (0, 1)
    ])

    # Métricas vs. umbral con probabilidades out-of-fold del train (mismos folds, pipeline final)
    proba_oof = probabilidades_oof(modelo_final, clave_final, X_train, y_train, cv)
    precision, recall, valores = precision_recall_curve(y_train, proba_oof)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    tabla_umbral = pd.DataFrame({"umbral": valores, "precision": precision, "recall": recall, "f1": f1})
    umbrales_oof = elegir_umbrales(y_train, proba_oof, recall_minimo=RECALL_MINIMO)
    if not np.isclose(umbrales_oof["umbral"], umbral, atol=1e-3):
        print(f"⚠ El umbral recalculado con las probabilidades out-of-fold ({umbrales_oof['umbral']:.4f}) no "
              f"coincide con el guardado ({umbral:.4f}); puede deberse a diferencias CPU/GPU. Se usa el guardado.")

    alertas = lambda u: float((proba_final >= u).mean() * 100)
    final = {
        "modelo": clave_final,
        "nombre": meta_final["nombre"],
        "estrategia": meta_final["estrategia_balanceo"],
        "estrategia_nombre": ESTRATEGIAS[meta_final["estrategia_balanceo"]],
        "hiperparametros": meta_final["hiperparametros"],
        "explicacion_eleccion": meta_final["explicacion_eleccion"],
        "umbral": umbral,
        "criterio_umbral": meta_final["criterio_umbral"],
        "recall_minimo": RECALL_MINIMO,
        "umbral_f1": float(meta_final["umbral_f1"]),
        "umbral_defecto": UMBRAL,
        "metricas_test_umbral": {**metricas_con_umbral(y_test, proba_final, umbral), "Alertas (%)": alertas(umbral)},
        "metricas_test_05": {**metricas_con_umbral(y_test, proba_final, UMBRAL), "Alertas (%)": alertas(UMBRAL)},
        "prevalencia_test": float(y_test.mean()),
        "prevalencia_train": float(y_train.mean()),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positivos_test": int(y_test.sum()),
        "n_folds": int(cv.get_n_splits()),
        "fecha_modelo": meta_final.get("fecha"),
    }

    archivos = [
        _escribir_csv(cv_largo, ruta / "clasificacion_benchmark_cv.csv"),
        _escribir_csv(test_largo, ruta / "clasificacion_benchmark_test.csv"),
        _escribir_csv(_curvas(y_test, probas_test), ruta / "clasificacion_curvas.csv"),
        _escribir_json(final, ruta / "clasificacion_final.json"),
        _escribir_csv(tabla_matriz, ruta / "clasificacion_matriz_confusion.csv"),
        _escribir_csv(tabla_umbral, ruta / "clasificacion_umbral.csv"),
    ]
    contexto = {"modelo": modelo_final, "metadatos": meta_final, "proba_test": proba_final, "umbral": umbral,
                "X_test": X_test, "X_train": X_train}
    return archivos, contexto


# ---------------------------------------------------------------------------
# 1-2. Regresión: benchmarking, modelo final y experimento con el corte clínico (reg_comparacion.ipynb)
# ---------------------------------------------------------------------------

def exportar_regresion(datos: dict, clasificacion: dict, ruta: Path = RUTA_DASHBOARD) -> tuple[list[Path], dict]:
    """Benchmarking de los ganadores + benchmark (CV y test con IC bootstrap), modelo final,
    predicciones en test y experimento "BDI predicho >= 14" frente al clasificador final."""
    X_train, y_train = separar_x_y_reg(datos["df_train"])
    X_test, y_test = separar_x_y_reg(datos["df_test"])
    y_dep_test = datos["df_test"][TARGET_CLF]
    cv = crear_cv_reg(SEED)

    pipelines, ganadores = {}, {}
    for clave in NOMBRES_REG:
        pipelines[clave], ganadores[clave] = cargar_ganador_reg(clave)
    pipelines = {CLAVE_BENCHMARK: crear_benchmark_reg().fit(X_train, y_train), **pipelines}
    modelo_final, meta_final = cargar_modelo_regresion("final")
    clave_final = meta_final["modelo"]
    variantes = {CLAVE_BENCHMARK: "nativo", **{c: g["variante"] for c, g in ganadores.items()}}

    # CV con los hiperparámetros de la fase 1 (sin tuning), como evaluar_cv_reg en reg_comparacion
    tabla_cv = evaluar_cv_reg(pipelines, X_train, y_train, cv)
    cv_largo = _largo_cv(tabla_cv, METRICAS_REG.values(), NOMBRES_FASE2_REG, "variante", variantes, clave_final)
    cv_largo["variante_nombre"] = cv_largo["variante"].map(VARIANTES)

    predicciones = {clave: pipeline.predict(X_test) for clave, pipeline in pipelines.items()}
    tabla_test = evaluar_test_reg(y_test, predicciones, n_replicas=N_REPLICAS, seed=SEED)
    filas = []
    for clave in tabla_test.index:
        for metrica in METRICAS_REG.values():
            con_ic = metrica in ("RMSE", "R²")
            filas.append({
                "modelo": clave, "metrica": metrica, "valor": float(tabla_test.loc[clave, metrica]),
                "ic_inf": float(tabla_test.loc[clave, f"{metrica} IC 2.5%"]) if con_ic else np.nan,
                "ic_sup": float(tabla_test.loc[clave, f"{metrica} IC 97.5%"]) if con_ic else np.nan,
                "delta_referencia": float(tabla_test.loc[clave, f"Δ {metrica} vs benchmark"]) if con_ic else np.nan,
                "delta_ic_inf": float(tabla_test.loc[clave, f"Δ {metrica} IC 2.5%"]) if con_ic else np.nan,
                "delta_ic_sup": float(tabla_test.loc[clave, f"Δ {metrica} IC 97.5%"]) if con_ic else np.nan,
                "delta_significativa": bool(tabla_test.loc[clave, f"Δ {metrica} significativa"]) if con_ic else None,
            })
    test_largo = pd.DataFrame(filas)

    bdi_final = modelo_final.predict(X_test)
    predicciones_test = pd.DataFrame({
        "subject_id": datos["df_test"]["subject_id"].to_numpy(),
        "sex": datos["df_test"]["sex"].astype(str).to_numpy(),
        "bdi_real": y_test.to_numpy(),
        "bdi_predicho_final": bdi_final,
        "bdi_predicho_benchmark": predicciones[CLAVE_BENCHMARK],
    })

    # Experimento: BDI predicho >= 14 frente al clasificador final en el mismo test
    proba_clf, umbral_clf = clasificacion["proba_test"], clasificacion["umbral"]
    enfoques = {
        "regresion": (f"Regresión: {NOMBRES_REG[clave_final]} (BDI predicho ≥ {CORTE_BDI})",
                      metricas_riesgo(y_dep_test, bdi_final, CORTE_BDI)),
        "clasificacion": (f"Clasificación: {clasificacion['metadatos']['nombre']} (umbral elegido)",
                          metricas_riesgo(y_dep_test, proba_clf, umbral_clf)),
    }
    comparacion = comparar_auc_bootstrap(y_dep_test, proba_clf, bdi_final, n_replicas=N_REPLICAS, seed=SEED)
    filas_bootstrap = {"regresion": "Regresor (BDI predicho)", "clasificacion": "Clasificador final"}
    filas = []
    for enfoque, (nombre, metricas) in enfoques.items():
        boot = comparacion.loc[filas_bootstrap[enfoque]]
        filas.append({
            "enfoque": enfoque, "nombre": nombre, "umbral": metricas["Umbral"],
            "pr_auc": metricas["PR-AUC"], "roc_auc": metricas["ROC-AUC"], "recall": metricas["Recall"],
            "precision": metricas["Precision"], "f1": metricas["F1"], "accuracy": metricas["Accuracy"],
            "alertas_pct": metricas["Alertas (%)"],
            "pr_auc_ic_inf": boot["PR-AUC IC 2.5%"], "pr_auc_ic_sup": boot["PR-AUC IC 97.5%"],
            "roc_auc_ic_inf": boot["ROC-AUC IC 2.5%"], "roc_auc_ic_sup": boot["ROC-AUC IC 97.5%"],
            # Diferencia frente al clasificador final (0 en su propia fila)
            "delta_pr_auc": boot["Δ PR-AUC vs clasificador"],
            "delta_pr_auc_ic_inf": boot["Δ PR-AUC IC 2.5%"], "delta_pr_auc_ic_sup": boot["Δ PR-AUC IC 97.5%"],
            "delta_roc_auc": boot["Δ ROC-AUC vs clasificador"],
            "delta_roc_auc_ic_inf": boot["Δ ROC-AUC IC 2.5%"], "delta_roc_auc_ic_sup": boot["Δ ROC-AUC IC 97.5%"],
        })
    experimento = pd.DataFrame(filas)

    preprocesador = modelo_final.named_steps["preprocesamiento"]
    elasticnet = modelo_final.named_steps["modelo"].regresor_
    nombres_salida = list(preprocesador.get_feature_names_out())
    metricas_test = metricas_reg(y_test, bdi_final)
    final = {
        "modelo": clave_final,
        "nombre": meta_final["nombre"],
        "variante": meta_final["variante_target"],
        "variante_nombre": VARIANTES[meta_final["variante_target"]],
        "hiperparametros": meta_final["hiperparametros"],
        "explicacion_eleccion": meta_final["explicacion_eleccion"],
        "recorte": meta_final["recorte_predicciones"],
        "metricas_test": metricas_test,
        "intervalos_test": {
            metrica: [float(tabla_test.loc[clave_final, f"{metrica} IC 2.5%"]),
                      float(tabla_test.loc[clave_final, f"{metrica} IC 97.5%"])]
            for metrica in ("RMSE", "R²")
        },
        "rmse_cv_media": float(tabla_cv.loc[clave_final, "RMSE media"]),
        "rmse_cv_de": float(tabla_cv.loc[clave_final, "RMSE DE"]),
        # Coeficientes en puntos BDI por DE del predictor (sex: chicas frente a chicos)
        "intercepto": float(elasticnet.intercept_),
        "coeficientes": {nombre.split("__", 1)[1]: float(c) for nombre, c in zip(nombres_salida, elasticnet.coef_)},
        "corte_bdi": CORTE_BDI,
        "n_folds": int(cv.get_n_splits()),
        "fecha_modelo": meta_final.get("fecha"),
    }

    archivos = [
        _escribir_csv(cv_largo, ruta / "regresion_benchmark_cv.csv"),
        _escribir_csv(test_largo, ruta / "regresion_benchmark_test.csv"),
        _escribir_json(final, ruta / "regresion_final.json"),
        _escribir_csv(predicciones_test, ruta / "regresion_predicciones_test.csv"),
        _escribir_csv(experimento, ruta / "regresion_vs_clasificacion.csv"),
    ]
    return archivos, {"modelo": modelo_final, "metadatos": meta_final}


# ---------------------------------------------------------------------------
# 3. Interpretabilidad
# ---------------------------------------------------------------------------

# Nombre legible (como lo guardan interpretabilidad.ipynb y genero.ipynb) → columna del dataset
VARIABLE_DE_ETIQUETA = {etiqueta: variable for variable, etiqueta in NOMBRES.items()}


def _con_variable(tabla: pd.DataFrame) -> pd.DataFrame:
    """Renombra la columna `variable` (nombre legible) a `etiqueta` y añade `variable` con el nombre
    de la columna del dataset."""
    desconocidas = sorted(set(tabla["variable"]) - set(VARIABLE_DE_ETIQUETA))
    if desconocidas:
        raise ValueError(f"Variables sin columna conocida en importancias: {desconocidas}")
    tabla = tabla.rename(columns={"variable": "etiqueta"})
    posicion = tabla.columns.get_loc("etiqueta")
    tabla.insert(posicion, "variable", tabla["etiqueta"].map(VARIABLE_DE_ETIQUETA))
    return tabla


def exportar_interpretabilidad(ruta: Path = RUTA_DASHBOARD) -> list[Path]:
    """importancias.csv (modelo, metodo, variable, etiqueta, valor, desviacion) desde
    interpretabilidad/importancias.csv."""
    origen = _exigir(RUTA_INTERPRETABILIDAD / "importancias.csv", "notebooks/interpretabilidad.ipynb")
    tabla = _con_variable(pd.read_csv(origen))
    return [_escribir_csv(tabla[["modelo", "metodo", "variable", "etiqueta", "valor", "desviacion"]],
                          ruta / "importancias.csv")]


# ---------------------------------------------------------------------------
# 4. Género
# ---------------------------------------------------------------------------

def _metrica_genero(metricas: pd.DataFrame, grupo: str, enfoque: str, metrica: str, tarea: str = "clasificacion") -> dict:
    fila = metricas[(metricas["grupo"] == grupo) & (metricas["tarea"] == tarea)
                    & (metricas["enfoque"] == enfoque) & (metricas["metrica"] == metrica)]
    if len(fila) != 1:
        raise ValueError(f"No se encontró {metrica} ({tarea}, {grupo}, {enfoque}) en genero/metricas_test.csv.")
    fila = fila.iloc[0]
    return {"valor": float(fila["valor"]), "ic_inf": fila["ic_inf"], "ic_sup": fila["ic_sup"]}


def exportar_genero(clasificacion: dict, ruta: Path = RUTA_DASHBOARD) -> list[Path]:
    """Copia genero/metricas_test.csv e importancias.csv y resume umbrales, tamaños y métricas por grupo.

    En genero/metricas_test.csv el enfoque "conjunto" es el modelo conjunto con el umbral conjunto
    y el enfoque "especifico", el modelo de cada grupo con su umbral propio (genero.ipynb).
    """
    metricas = pd.read_csv(_exigir(RUTA_GENERO / "metricas_test.csv", "notebooks/genero.ipynb"))
    importancias = _con_variable(pd.read_csv(_exigir(RUTA_GENERO / "importancias.csv", "notebooks/genero.ipynb")))
    meta = _leer_json(_exigir(RUTA_GENERO / "metadatos.json", "notebooks/genero.ipynb"))

    grupos = {}
    for grupo in SEXOS:
        info = meta["grupos"][grupo]
        clase = lambda enfoque, metrica: _metrica_genero(metricas, grupo, enfoque, metrica)["valor"]
        grupos[grupo] = {
            "nombre": NOMBRES_GRUPO[grupo],
            "sexo": SEXOS[grupo],
            "tamanos": info["tamanos"],
            "umbral_propio": info["catboost"]["umbral"],
            "umbral_conjunto": {m.lower(): clase("conjunto", m) for m in ("Recall", "Precision", "F1")},
            "umbral_propio_metricas": {m.lower(): clase("especifico", m) for m in ("Recall", "Precision", "F1")},
            "roc_auc": {e: _metrica_genero(metricas, grupo, e, "ROC-AUC") for e in ("especifico", "conjunto")},
            "rmse": {e: _metrica_genero(metricas, grupo, e, "RMSE", "regresion") for e in ("especifico", "conjunto")},
        }
    resumen = {
        "umbral_conjunto": clasificacion["umbral"],
        "grupos": grupos,
        "nota": "umbral_conjunto: modelo conjunto con el umbral conjunto; umbral_propio_metricas: modelo "
                "específico del grupo con su propio umbral (mismo criterio: máxima precision con recall >= 0.70).",
    }
    return [
        _escribir_csv(metricas, ruta / "genero_metricas_test.csv"),
        _escribir_csv(importancias, ruta / "genero_importancias.csv"),
        _escribir_json(resumen, ruta / "genero_resumen.json"),
    ]


# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------

def exportar_clustering(datos: dict, ruta: Path = RUTA_DASHBOARD) -> list[Path]:
    """Selección de k, perfiles, proyección PCA de cada adolescente, centroides y metadatos.

    Exige que el clustering se haya ejecutado con el reetiquetado (clusters numerados por la media
    de VARIABLE_ORDEN): si no, pide volver a ejecutar clustering.ipynb.
    """
    notebook = "notebooks/clustering.ipynb"
    meta = _leer_json(_exigir(RUTA_CLUSTERING / "metadatos.json", notebook))
    perfiles = pd.read_csv(_exigir(RUTA_CLUSTERING / "perfiles.csv", notebook)).sort_values("cluster")
    seleccion = pd.read_csv(_exigir(RUTA_CLUSTERING / "seleccion_k.csv", notebook))
    asignaciones = pd.read_csv(_exigir(RUTA_CLUSTERING / "asignaciones.csv", notebook))
    pipeline = joblib.load(_exigir(RUTA_CLUSTERING / "kmeans.joblib", notebook))

    if "mapeo_reetiquetado" not in meta or not perfiles[f"media_{VARIABLE_ORDEN}"].is_monotonic_increasing:
        raise ValueError(f"Los clusters guardados no están numerados por la media de {VARIABLE_ORDEN}. "
                         f"Vuelve a ejecutar {notebook} con la versión actual de src/Clustering.py.")

    X = datos["df"].set_index("subject_id").loc[asignaciones["subject_id"], VARIABLES_CLUSTER]
    if not np.array_equal(pipeline.predict(X), asignaciones["cluster"].to_numpy()):
        raise ValueError(f"kmeans.joblib no reproduce asignaciones.csv. Vuelve a ejecutar {notebook}.")

    # PCA de 2 componentes sobre las variables estandarizadas con el escalador del pipeline (como graficar_pca)
    Z = datos_estandarizados(pipeline, X)
    pca = PCA(n_components=2).fit(Z)
    proyeccion = pca.transform(Z)
    centros = pipeline.named_steps["kmeans"].cluster_centers_
    proyeccion_centros = pca.transform(centros)

    puntos = pd.DataFrame({"subject_id": asignaciones["subject_id"].to_numpy(),
                           "cluster": asignaciones["cluster"].to_numpy(),
                           "pc1": proyeccion[:, 0], "pc2": proyeccion[:, 1]})
    medias_z = pd.DataFrame(Z, columns=VARIABLES_CLUSTER).groupby(asignaciones["cluster"].to_numpy()).mean()
    centroides = pd.DataFrame({"cluster": medias_z.index, "pc1": proyeccion_centros[medias_z.index, 0],
                               "pc2": proyeccion_centros[medias_z.index, 1]})
    for variable in VARIABLES_CLUSTER:
        centroides[variable] = medias_z[variable].to_numpy()

    resumen = {
        "k": int(meta["k_elegido"]),
        "varianza_explicada": [float(v) for v in pca.explained_variance_ratio_],
        "cargas": {variable: [float(c) for c in pca.components_[:, i]] for i, variable in enumerate(VARIABLES_CLUSTER)},
        "nombres": {str(c): n for c, n in zip(perfiles["cluster"], perfiles["nombre"])},
        "variables": VARIABLES_CLUSTER,
        "etiquetas": {variable: ETIQUETAS[variable] for variable in VARIABLES_CLUSTER},
        "orden_clusters": meta["orden_clusters"],
        "n_observaciones": int(meta["n_observaciones"]),
        "n_bootstrap_estabilidad": meta.get("n_bootstrap_estabilidad"),
        "fecha_clustering": meta.get("fecha"),
    }
    return [
        _escribir_csv(seleccion, ruta / "clustering_seleccion_k.csv"),
        _escribir_csv(perfiles, ruta / "clustering_perfiles.csv"),
        _escribir_csv(puntos, ruta / "clustering_puntos.csv"),
        _escribir_csv(centroides, ruta / "clustering_centroides.csv"),
        _escribir_json(resumen, ruta / "clustering_meta.json"),
    ]


# ---------------------------------------------------------------------------
# 6. Modelos portables para la pestaña de predicción
# ---------------------------------------------------------------------------

def describir_preprocesador(preprocesador) -> dict:
    """Todo lo necesario para reproducir con numpy el ColumnTransformer ajustado del pipeline:
    columnas de entrada, orden de las columnas de salida y, por columna de salida, la media y la
    escala del StandardScaler o la categoría de la indicadora del OneHotEncoder.

    La salida se compara con get_feature_names_out(): si el orden no coincide, lanza un error.
    """
    transformaciones, codificaciones = [], {}
    for nombre, transformador, columnas in preprocesador.transformers_:
        if isinstance(transformador, str):
            if transformador == "drop" or len(columnas) == 0:
                continue
            raise TypeError(f"Transformador '{nombre}' = {transformador!r} no soportado.")
        if isinstance(transformador, StandardScaler):
            for i, variable in enumerate(columnas):
                transformaciones.append({
                    "columna_salida": f"{nombre}__{variable}", "tipo": "estandarizar", "variable": variable,
                    "media": float(transformador.mean_[i]) if transformador.with_mean else 0.0,
                    "escala": float(transformador.scale_[i]) if transformador.with_std else 1.0,
                })
        elif isinstance(transformador, OneHotEncoder):
            for i, variable in enumerate(columnas):
                categorias = [str(c) for c in transformador.categories_[i]]
                eliminada = None if transformador.drop_idx_ is None else transformador.drop_idx_[i]
                codificaciones[variable] = {
                    "categorias": categorias,
                    "eliminada": None if eliminada is None else categorias[int(eliminada)],
                    "desconocida": "todas las indicadoras en 0" if transformador.handle_unknown == "ignore" else "error",
                }
                for j, categoria in enumerate(categorias):
                    if eliminada is not None and j == int(eliminada):
                        continue
                    transformaciones.append({
                        "columna_salida": f"{nombre}__{variable}_{categoria}", "tipo": "indicadora",
                        "variable": variable, "categoria": categoria,
                    })
        else:
            raise TypeError(f"Transformador '{nombre}' de tipo {type(transformador).__name__} no soportado.")

    columnas_salida = [t["columna_salida"] for t in transformaciones]
    if columnas_salida != list(preprocesador.get_feature_names_out()):
        raise ValueError(f"El orden reconstruido {columnas_salida} no coincide con get_feature_names_out().")
    return {
        "entradas": list(preprocesador.feature_names_in_),
        "columnas_salida": columnas_salida,
        "transformaciones": transformaciones,
        "codificaciones": codificaciones,
    }


def _rangos(X_train: pd.DataFrame) -> dict:
    """Mínimo, máximo, percentiles 1 y 99, mediana, paso y unidad de cada predictor numérico (train)."""
    variables = {}
    for variable in PREDICTORES_NUMERICOS:
        serie = X_train[variable].astype(float)
        p1, p99 = np.percentile(serie, [1, 99])
        unicos = np.sort(serie.unique())
        discreta = len(unicos) <= MAX_VALORES_DISCRETOS
        unidad, descripcion = UNIDADES_CONTROL[variable]
        variables[variable] = {
            "etiqueta": ETIQUETAS[variable], "unidad": unidad, "descripcion": descripcion,
            "minimo": float(serie.min()), "maximo": float(serie.max()),
            "p1": float(p1), "p99": float(p99), "mediana": float(serie.median()),
            "paso": float(round(np.min(np.diff(unicos)), 3)) if discreta else PASO_CONTINUAS,
            # En las discretas el control solo toma los valores observados entre los percentiles 1 y 99
            "valores": [float(v) for v in unicos if p1 <= v <= p99] if discreta else None,
        }
    sexo = X_train["sex"].astype(str)
    return {
        "orden": PREDICTORES_NUMERICOS,
        "variables": variables,
        "sexo": {
            "opciones": [{"valor": s, "etiqueta": e} for s, e in (("Boy", "Chico"), ("Girl", "Chica"))],
            "inicial": sexo.mode().iloc[0],
            "etiqueta": ETIQUETAS["sex"],
            "grupos": ETIQUETAS_SEXO,
        },
        "fuente": "train de split_indices.csv",
    }


def verificar_prediccion(clasificador, regresor, X_test: pd.DataFrame, carpeta: Path = RUTA_PREDICCION,
                         n: int = N_VERIFICACION, seed: int = SEED) -> pd.DataFrame:
    """Comprueba con el MISMO código del dashboard (data/prediccion.py) que catboost.cbm + su JSON y
    elasticnet.json reproducen los pipelines originales en n adolescentes del test. Si no coinciden,
    lanza un error."""
    from data.prediccion import Predictor   # sin scikit-learn: solo numpy, json y catboost

    muestra = X_test.sample(n=min(n, len(X_test)), random_state=seed)
    predictor = Predictor(carpeta)
    comparaciones = {
        "Probabilidad (CatBoost)": (predictor.probabilidad(muestra), clasificador.predict_proba(muestra)[:, 1]),
        "BDI-II (ElasticNet)": (predictor.bdi(muestra), regresor.predict(muestra)),
    }
    filas = {}
    for nombre, (portable, original) in comparaciones.items():
        diferencia = float(np.max(np.abs(portable - original)))
        if not np.allclose(portable, original):
            raise ValueError(f"{nombre}: el modelo portable no reproduce el pipeline original "
                             f"(máxima diferencia = {diferencia:.3e} en {len(muestra)} adolescentes del test).")
        filas[nombre] = {"Adolescentes": len(muestra), "Máxima diferencia": diferencia, "Coinciden": True}
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Salida"
    return tabla


def exportar_prediccion(clasificacion: dict, regresion: dict, carpeta: Path = RUTA_PREDICCION) -> tuple[list[Path], pd.DataFrame]:
    """catboost.cbm, catboost_preprocesamiento.json, elasticnet.json y rangos.json, y su verificación."""
    clasificador, regresor = clasificacion["modelo"], regresion["modelo"]
    catboost_final = clasificador.steps[-1][1]
    if not isinstance(catboost_final, CatBoostClassifier):
        raise TypeError(f"El clasificador final no es CatBoost ({type(catboost_final).__name__}).")
    modelo_reg = regresor.steps[-1][1]
    elasticnet = getattr(modelo_reg, "regresor_", None)
    if not isinstance(elasticnet, ElasticNet) or modelo_reg.transformacion is not None:
        raise TypeError("El regresor final debe ser un ElasticNet con target nativo (sin transformación).")

    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_cbm = carpeta / "catboost.cbm"
    catboost_final.save_model(str(ruta_cbm), format="cbm")

    meta_clf = clasificacion["metadatos"]
    prep_clf = describir_preprocesador(clasificador.named_steps["preprocesamiento"])
    prep_reg = describir_preprocesador(regresor.named_steps["preprocesamiento"])
    if len(prep_reg["columnas_salida"]) != len(elasticnet.coef_):
        raise ValueError("El número de coeficientes del ElasticNet no coincide con las columnas transformadas.")

    catboost_json = {
        **prep_clf,
        "modelo": ruta_cbm.name,
        "clase_positiva": int(catboost_final.classes_[-1]),
        "umbral": clasificacion["umbral"],
        "criterio_umbral": meta_clf["criterio_umbral"],
        "recall_minimo": RECALL_MINIMO,
    }
    elasticnet_json = {
        **prep_reg,
        "coeficientes": {col: float(c) for col, c in zip(prep_reg["columnas_salida"], elasticnet.coef_)},
        "intercepto": float(elasticnet.intercept_),
        "recorte": [float(modelo_reg.minimo), float(modelo_reg.maximo)],
        "hiperparametros": regresion["metadatos"]["hiperparametros"],
    }

    archivos = [
        ruta_cbm,
        _escribir_json(catboost_json, carpeta / "catboost_preprocesamiento.json"),
        _escribir_json(elasticnet_json, carpeta / "elasticnet.json"),
        _escribir_json(_rangos(clasificacion["X_train"]), carpeta / "rangos.json"),
    ]
    verificacion = verificar_prediccion(clasificador, regresor, clasificacion["X_test"], carpeta)
    return archivos, verificacion


# ---------------------------------------------------------------------------
# 7. Manifest y orquestación
# ---------------------------------------------------------------------------

def _versiones() -> dict:
    return {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "catboost": catboost.__version__,
        "joblib": joblib.__version__,
    }


def exportar_manifest(archivos: list[Path], ruta: Path = RUTA_DASHBOARD) -> Path:
    """manifest.json: fecha, archivos generados (ruta relativa y tamaño) y versiones de librerías."""
    manifest = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "archivos": [{"archivo": a.relative_to(ruta).as_posix(), "bytes": a.stat().st_size} for a in archivos],
        "versiones": _versiones(),
        "nota": "La versión de catboost de requirements.txt (dashboard) debe coincidir con la de esta exportación.",
    }
    return _escribir_json(manifest, ruta / "manifest.json")


def exportar_todo(ruta: Path = RUTA_DASHBOARD) -> dict:
    """Ejecuta toda la exportación en orden y devuelve {"archivos": tabla resumen, "verificacion": tabla}."""
    datos = preparar_datos()
    archivos = []

    print("1/6 Clasificación…")
    nuevos, clasificacion = exportar_clasificacion(datos, ruta)
    archivos += nuevos
    print("2/6 Regresión…")
    nuevos, regresion = exportar_regresion(datos, clasificacion, ruta)
    archivos += nuevos
    print("3/6 Interpretabilidad…")
    archivos += exportar_interpretabilidad(ruta)
    print("4/6 Género…")
    archivos += exportar_genero(clasificacion, ruta)
    print("5/6 Clustering…")
    archivos += exportar_clustering(datos, ruta)
    print("6/6 Modelos portables y verificación…")
    nuevos, verificacion = exportar_prediccion(clasificacion, regresion, ruta / "prediccion")
    archivos += nuevos
    archivos.append(exportar_manifest(archivos, ruta))
    print(f"✓ {len(archivos)} archivos en {ruta}")

    resumen = pd.DataFrame({
        "Archivo": [a.relative_to(ruta).as_posix() for a in archivos],
        "Filas": [len(pd.read_csv(a)) if a.suffix == ".csv" else None for a in archivos],
        "Tamaño (KB)": [round(a.stat().st_size / 1024, 1) for a in archivos],
    }).set_index("Archivo")
    return {"archivos": resumen, "verificacion": verificacion}
