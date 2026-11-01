"""Interpretabilidad de los dos modelos finales, sin reentrenar nada.

- Clasificación (`depressed`): CatBoost, estrategia nativa (data/processed/modelado/final/).
- Regresión (`bdi_total`): ElasticNet con target nativo (data/processed/modelado/regresion/final/).

Lo usa notebooks/interpretabilidad.ipynb. Todas las explicaciones se calculan sobre el TEST
(reconstruido desde split_indices.csv) y las gráficas usan los nombres legibles de
data/load_data.py y los valores originales de las variables (sin estandarizar).
Los resultados se guardan en data/processed/modelado/interpretabilidad/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, Pool
from lime.lime_tabular import LimeTabularExplainer
from scipy.special import logit
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))   # para importar data/load_data.py (etiquetas legibles)

from data.load_data import ETIQUETAS
from EstiloGraficos import DEPRIMIDO, NEUTRO, NO_DEPRIMIDO, TEXTO
from Modelado import (
    PREDICTORES, PREDICTORES_CATEGORICOS, PREDICTORES_NUMERICOS, RUTA_SALIDA, SEED, cargar_modelo_final, cargar_split,
)
from ModeladoRegresion import cargar_modelo_regresion

RUTA_INTERPRETABILIDAD = RUTA_SALIDA / "interpretabilidad"

# Nombre legible de cada predictor. `sex` entra a los modelos como indicadora de "Girl" (sex_Girl),
# así que su etiqueta indica qué significa el valor 1.
NOMBRE_SEXO = f"{ETIQUETAS['sex']} (chica)"
NOMBRES = {**{columna: ETIQUETAS[columna] for columna in PREDICTORES_NUMERICOS}, "sex": NOMBRE_SEXO}

# Unidad original de cada predictor numérico (para leer el efecto por unidad del coeficiente)
UNIDADES = {
    "sleep_quality_index": "punto del índice",
    "avg_sleep_hours": "hora",
    "midsleep_weekend_hours": "hora",
    "social_jetlag_hours": "hora",
    "screen_time_index": "punto del índice",
}

CLASES = ["No deprimido", "Deprimido"]


# ---------------------------------------------------------------------------
# Modelos, datos y componentes del pipeline
# ---------------------------------------------------------------------------

def cargar_modelos() -> dict:
    """Carga los dos modelos finales con las funciones existentes (sin reentrenar)."""
    clasificador, metadatos_clf = cargar_modelo_final()
    regresor, metadatos_reg = cargar_modelo_regresion("final")
    return {
        "clasificador": clasificador, "metadatos_clf": metadatos_clf,
        "regresor": regresor, "metadatos_reg": metadatos_reg,
    }


def extraer_componentes(pipeline) -> tuple:
    """Devuelve (preprocesador ajustado, estimador final, nombres de las variables transformadas).

    En la regresión, el último paso es RegresorBDI (recorta las predicciones a [0, 63]); el
    estimador final es el ElasticNet ajustado que guarda en `regresor_`.
    """
    preprocesador = pipeline.named_steps["preprocesamiento"]
    estimador = pipeline.steps[-1][1]
    estimador = getattr(estimador, "regresor_", estimador)
    return preprocesador, estimador, list(preprocesador.get_feature_names_out())


def verificar_modelos(modelos: dict) -> pd.DataFrame:
    """Comprueba, con los metadatos y el tipo del estimador final, que los modelos son CatBoost y ElasticNet."""
    _, est_clf, _ = extraer_componentes(modelos["clasificador"])
    _, est_reg, _ = extraer_componentes(modelos["regresor"])
    meta_clf, meta_reg = modelos["metadatos_clf"], modelos["metadatos_reg"]
    tabla = pd.DataFrame({
        "Modelo (metadatos)": [meta_clf["nombre"], meta_reg["nombre"]],
        "Estimador final": [type(est_clf).__name__, type(est_reg).__name__],
        "Variante": [f"estrategia {meta_clf['estrategia_balanceo']}", f"target {meta_reg['variante_target']}"],
        "Umbral": [meta_clf["umbral"], None],
        "Hiperparámetros": [meta_clf["hiperparametros"], meta_reg["hiperparametros"]],
    }, index=pd.Index(["Clasificación (depressed)", "Regresión (bdi_total)"], name="Problema"))
    tabla["Correcto"] = [
        meta_clf["modelo"] == "catboost" and isinstance(est_clf, CatBoostClassifier),
        meta_reg["modelo"] == "elasticnet" and isinstance(est_reg, ElasticNet),
    ]
    if not tabla["Correcto"].all():
        raise ValueError("Los modelos guardados no son CatBoost (clasificación) y ElasticNet (regresión).")
    return tabla


def preparar_datos(df: pd.DataFrame) -> dict:
    """Train y test según split_indices.csv (los mismos de todo el modelado), con los 6 predictores
    sin preprocesar y los dos targets. El train solo se usa como referencia (SHAP lineal y LIME)."""
    df_train, df_test = cargar_split(df)
    return {
        "X_train": df_train[PREDICTORES], "X_test": df_test[PREDICTORES],
        "y_dep_test": df_test["depressed"].to_numpy(), "y_bdi_test": df_test["bdi_total"].to_numpy(),
        "subject_id_test": df_test["subject_id"].to_numpy(),
    }


def nombres_legibles(nombres_transformados: list[str]) -> list[str]:
    """`numericas__avg_sleep_hours` → "Horas de sueño promedio"; `categoricas__sex_Girl` → "Sexo (chica)"."""
    legibles = []
    for nombre in nombres_transformados:
        variable = nombre.split("__", 1)[-1]
        legibles.append(NOMBRE_SEXO if variable.startswith("sex_") else ETIQUETAS[variable])
    return legibles


def transformar(preprocesador, X: pd.DataFrame) -> np.ndarray:
    """Aplica el preprocesador ya ajustado (sin volver a ajustarlo)."""
    matriz = preprocesador.transform(X)
    return np.asarray(matriz.toarray() if hasattr(matriz, "toarray") else matriz, dtype=float)


def valores_originales(X: pd.DataFrame, nombres_transformados: list[str]) -> pd.DataFrame:
    """Valores ORIGINALES (sin estandarizar) en el mismo orden que la salida del preprocesador,
    con nombres legibles. `sex` se representa como 0 = chico, 1 = chica."""
    columnas = {}
    for nombre in nombres_transformados:
        variable = nombre.split("__", 1)[-1]
        if variable.startswith("sex_"):
            categoria = variable.split("_", 1)[1]
            columnas[NOMBRE_SEXO] = (X["sex"] == categoria).astype(float).to_numpy()
        else:
            columnas[ETIQUETAS[variable]] = X[variable].astype(float).to_numpy()
    return pd.DataFrame(columnas, index=X.index)


# ---------------------------------------------------------------------------
# Sección 1: coeficientes del ElasticNet
# ---------------------------------------------------------------------------

def tabla_coeficientes(preprocesador, estimador, nombres_transformados: list[str]) -> pd.DataFrame:
    """Coeficientes del ElasticNet (puntos BDI por DE del predictor; para sex, chicas frente a chicos),
    con el efecto por unidad original (coeficiente / DE del train) y los coeficientes anulados por L1."""
    escalador = preprocesador.named_transformers_["numericas"]
    de_train = dict(zip(PREDICTORES_NUMERICOS, escalador.scale_))
    filas = []
    for nombre, coeficiente in zip(nombres_transformados, estimador.coef_):
        variable = nombre.split("__", 1)[-1]
        if variable.startswith("sex_"):
            filas.append({"Variable": NOMBRE_SEXO, "Coeficiente (por DE)": coeficiente, "DE del train": np.nan,
                          "Efecto por unidad original": coeficiente, "Unidad": "chica frente a chico"})
        else:
            filas.append({"Variable": ETIQUETAS[variable], "Coeficiente (por DE)": coeficiente,
                          "DE del train": de_train[variable], "Efecto por unidad original": coeficiente / de_train[variable],
                          "Unidad": UNIDADES[variable]})
    tabla = pd.DataFrame(filas).set_index("Variable")
    tabla["En cero (L1)"] = tabla["Coeficiente (por DE)"] == 0
    return tabla.reindex(tabla["Coeficiente (por DE)"].abs().sort_values(ascending=False).index)


def graficar_coeficientes(tabla: pd.DataFrame, intercepto: float):
    """Coeficientes del ElasticNet (por DE), en rojo los que aumentan el BDI y en azul los que lo reducen."""
    orden = tabla.iloc[::-1]
    colores = [DEPRIMIDO if c > 0 else NO_DEPRIMIDO for c in orden["Coeficiente (por DE)"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(orden.index, orden["Coeficiente (por DE)"], color=colores)
    for posicion, (variable, fila) in enumerate(orden.iterrows()):
        texto = "0 (anulado por L1)" if fila["En cero (L1)"] else f"{fila['Coeficiente (por DE)']:+.2f}"
        ax.text(fila["Coeficiente (por DE)"], posicion, f"  {texto}  ", va="center",
                ha="left" if fila["Coeficiente (por DE)"] >= 0 else "right", fontsize=9, color=TEXTO)
    ax.axvline(0, color=TEXTO, linewidth=1)
    ax.set_title(f"ElasticNet: coeficientes (intercepto = {intercepto:.2f} puntos BDI)", fontweight="bold")
    ax.set_xlabel("Cambio en el BDI esperado por cada DE del predictor (sexo: chicas frente a chicos)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sección 2: importancia nativa de CatBoost
# ---------------------------------------------------------------------------

def importancia_catboost(estimador, X_test_prep: np.ndarray, y_test, nombres: list[str]) -> pd.DataFrame:
    """PredictionValuesChange (no necesita datos) y LossFunctionChange (con un Pool del test preprocesado)."""
    cambio_prediccion = estimador.get_feature_importance(type="PredictionValuesChange")
    cambio_perdida = estimador.get_feature_importance(data=Pool(X_test_prep, label=y_test), type="LossFunctionChange")
    tabla = pd.DataFrame({"PredictionValuesChange": cambio_prediccion, "LossFunctionChange": cambio_perdida},
                         index=pd.Index(nombres, name="Variable"))
    return tabla.sort_values("PredictionValuesChange", ascending=False)


def graficar_importancia_catboost(tabla: pd.DataFrame):
    """Barras de las dos importancias nativas de CatBoost, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    orden = tabla.sort_values("PredictionValuesChange")
    descripciones = {
        "PredictionValuesChange": "Cambio medio de la predicción (%)",
        "LossFunctionChange": "Aumento de la pérdida (Logloss) en test",
    }
    for ax, (columna, etiqueta) in zip(axes, descripciones.items()):
        ax.barh(orden.index, orden[columna], color=NEUTRO)
        ax.set_title(columna, fontweight="bold")
        ax.set_xlabel(etiqueta)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.suptitle("CatBoost: importancia nativa de las variables", fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sección 3: SHAP
# ---------------------------------------------------------------------------

def _construir_explicacion(valores, base, datos_originales: pd.DataFrame) -> shap.Explanation:
    """shap.Explanation con los valores SHAP, la base, los valores originales y nombres legibles."""
    valores = np.asarray(valores, dtype=float)
    return shap.Explanation(
        values=valores,
        base_values=np.full(len(valores), float(base)),
        data=datos_originales.to_numpy(),
        feature_names=list(datos_originales.columns),
    )


def shap_catboost(estimador, X_test_prep: np.ndarray, datos_originales: pd.DataFrame) -> shap.Explanation:
    """SHAP de CatBoost con TreeExplainer sobre el test preprocesado. Los valores están en log-odds."""
    explicador = shap.TreeExplainer(estimador)
    valores = explicador.shap_values(X_test_prep)
    base = explicador.expected_value
    # Según la versión de shap, un clasificador binario devuelve una matriz, una lista por clase
    # o un arreglo (n, variables, clases): se toma siempre la clase positiva.
    if isinstance(valores, list):
        valores = valores[-1]
    valores = np.asarray(valores)
    if valores.ndim == 3:
        valores = valores[:, :, -1]
    base = np.ravel(base)[-1]
    return _construir_explicacion(valores, base, datos_originales)


def shap_lineal(estimador, X_train_prep: np.ndarray, X_test_prep: np.ndarray, datos_originales: pd.DataFrame) -> shap.Explanation:
    """SHAP del ElasticNet con LinearExplainer, con el train preprocesado como fondo. Valores en puntos BDI."""
    explicador = shap.LinearExplainer(estimador, X_train_prep)
    valores = explicador.shap_values(X_test_prep)
    return _construir_explicacion(valores, np.ravel(explicador.expected_value)[0], datos_originales)


def verificar_aditividad(explicacion: shap.Explanation, salida_modelo) -> float:
    """Máxima diferencia entre base + suma de los SHAP y la salida del modelo (debería ser ≈ 0)."""
    reconstruida = explicacion.base_values + explicacion.values.sum(axis=1)
    return float(np.max(np.abs(reconstruida - np.asarray(salida_modelo, dtype=float))))


def salida_logodds_catboost(estimador, X_test_prep: np.ndarray) -> np.ndarray:
    """Salida de CatBoost en log-odds (la escala de sus valores SHAP)."""
    proba = np.clip(estimador.predict_proba(X_test_prep)[:, 1], 1e-12, 1 - 1e-12)
    return logit(proba)


def importancia_shap(explicacion: shap.Explanation) -> pd.Series:
    """Importancia global: media de |SHAP| por variable, ordenada."""
    return pd.Series(np.abs(explicacion.values).mean(axis=0), index=explicacion.feature_names,
                     name="Media de |SHAP|").sort_values(ascending=False)


def graficar_shap_barras(explicacion: shap.Explanation, titulo: str, unidad: str):
    """Media de |SHAP| por variable (gráfico de barras de shap)."""
    shap.plots.bar(explicacion, max_display=len(explicacion.feature_names), show=False)
    fig = plt.gcf()
    fig.axes[0].set_title(titulo, fontweight="bold")
    fig.axes[0].set_xlabel(f"Media de |SHAP| ({unidad})")
    fig.tight_layout()
    return fig


def graficar_shap_beeswarm(explicacion: shap.Explanation, titulo: str, unidad: str):
    """Beeswarm: un punto por adolescente, coloreado por el valor original de la variable."""
    shap.plots.beeswarm(explicacion, max_display=len(explicacion.feature_names), show=False)
    fig = plt.gcf()
    fig.axes[0].set_title(titulo, fontweight="bold")
    fig.axes[0].set_xlabel(f"Valor SHAP ({unidad})")
    fig.tight_layout()
    return fig


def graficar_dependencias(explicacion: shap.Explanation, variables: list[str], titulo: str, unidad: str):
    """Dependence plots (valor original vs. SHAP) de las variables dadas, coloreados por sexo."""
    fig, axes = plt.subplots(1, len(variables), figsize=(6 * len(variables), 4.8))
    for ax, variable in zip(np.atleast_1d(axes), variables):
        color = explicacion[:, NOMBRE_SEXO] if variable != NOMBRE_SEXO else None
        shap.plots.scatter(explicacion[:, variable], color=color, ax=ax, show=False)
        ax.set_ylabel(f"SHAP ({unidad})")
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def graficar_waterfall(explicacion: shap.Explanation, posicion: int, titulo: str):
    """Waterfall de un adolescente: cómo cada variable lleva la predicción desde la base hasta su valor."""
    plt.figure()
    shap.plots.waterfall(explicacion[posicion], max_display=len(explicacion.feature_names), show=False)
    fig = plt.gcf()
    fig.suptitle(titulo, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Elección reproducible de los adolescentes que se explican
# ---------------------------------------------------------------------------

def elegir_casos_clasificacion(y_test, proba, umbral: float, subject_id, seed: int = SEED) -> pd.DataFrame:
    """Un verdadero positivo, un falso negativo y un falso positivo del test según el umbral, elegidos al azar con SEED."""
    y_test, proba = np.asarray(y_test), np.asarray(proba)
    prediccion = (proba >= umbral).astype(int)
    rng = np.random.default_rng(seed)
    criterios = [
        ("Verdadero positivo", (y_test == 1) & (prediccion == 1), f"deprimido y probabilidad ≥ {umbral:.3f}"),
        ("Falso negativo", (y_test == 1) & (prediccion == 0), f"deprimido pero probabilidad < {umbral:.3f}"),
        ("Falso positivo", (y_test == 0) & (prediccion == 1), f"no deprimido pero probabilidad ≥ {umbral:.3f}"),
    ]
    filas = []
    for caso, mascara, motivo in criterios:
        posicion = int(rng.choice(np.flatnonzero(mascara)))
        filas.append({"Caso": caso, "posicion": posicion, "subject_id": int(subject_id[posicion]),
                      "Real": int(y_test[posicion]), "Predicción": round(float(proba[posicion]), 3),
                      "Motivo": f"{caso.lower()}: {motivo} (elegido al azar con SEED = {seed})"})
    return pd.DataFrame(filas).set_index("Caso")


def elegir_casos_regresion(y_test, prediccion, subject_id, seed: int = SEED) -> pd.DataFrame:
    """Tres adolescentes del test con BDI real bajo (≤ 2), cercano al corte (13 a 15) y alto (≥ percentil 95),
    elegidos al azar con SEED dentro de cada grupo."""
    y_test, prediccion = np.asarray(y_test), np.asarray(prediccion)
    alto = np.percentile(y_test, 95)
    rng = np.random.default_rng(seed)
    criterios = [
        ("BDI bajo", y_test <= 2, "BDI real ≤ 2"),
        ("BDI cercano a 14", (y_test >= 13) & (y_test <= 15), "BDI real entre 13 y 15 (cerca del corte clínico)"),
        ("BDI alto", y_test >= alto, f"BDI real ≥ {alto:.0f} (percentil 95 del test)"),
    ]
    filas = []
    for caso, mascara, motivo in criterios:
        posicion = int(rng.choice(np.flatnonzero(mascara)))
        filas.append({"Caso": caso, "posicion": posicion, "subject_id": int(subject_id[posicion]),
                      "Real": int(y_test[posicion]), "Predicción": round(float(prediccion[posicion]), 2),
                      "Motivo": f"{motivo} (elegido al azar con SEED = {seed})"})
    return pd.DataFrame(filas).set_index("Caso")


# ---------------------------------------------------------------------------
# Sección 4: LIME
# ---------------------------------------------------------------------------

INDICE_SEXO = PREDICTORES.index(PREDICTORES_CATEGORICOS[0])


def codificar_para_lime(X: pd.DataFrame) -> np.ndarray:
    """Variables ORIGINALES como matriz numérica para LIME: las numéricas tal cual y sex como 0 (Boy) / 1 (Girl)."""
    matriz = X[PREDICTORES].copy()
    matriz["sex"] = (matriz["sex"] == "Girl").astype(float)
    return matriz.to_numpy(dtype=float)


def decodificar_de_lime(matriz: np.ndarray) -> pd.DataFrame:
    """Matriz de LIME → DataFrame con las columnas y tipos originales (sex de vuelta a "Boy"/"Girl")."""
    df = pd.DataFrame(np.atleast_2d(matriz), columns=PREDICTORES)
    for columna in PREDICTORES_NUMERICOS:
        df[columna] = df[columna].astype(float)
    df["sex"] = np.where(np.rint(df["sex"].astype(float)) == 1, "Girl", "Boy")
    return df


def funcion_prediccion_lime(pipeline, modo: str):
    """Función que LIME llama con un array: lo convierte a DataFrame original y usa el pipeline completo guardado."""
    def predecir(matriz: np.ndarray) -> np.ndarray:
        df = decodificar_de_lime(matriz)
        return pipeline.predict_proba(df) if modo == "classification" else pipeline.predict(df)
    return predecir


def crear_explicador_lime(X_train: pd.DataFrame, modo: str, seed: int = SEED) -> LimeTabularExplainer:
    """LimeTabularExplainer sobre las variables originales (train sin preprocesar), con sex categórica."""
    return LimeTabularExplainer(
        training_data=codificar_para_lime(X_train),
        mode=modo,
        feature_names=[NOMBRES[columna] for columna in PREDICTORES],
        categorical_features=[INDICE_SEXO],
        categorical_names={INDICE_SEXO: ["Chico", "Chica"]},
        class_names=CLASES if modo == "classification" else None,
        discretize_continuous=True,
        random_state=seed,
    )


def explicar_lime(explicador: LimeTabularExplainer, pipeline, modo: str, X_test: pd.DataFrame,
                  casos: pd.DataFrame, num_samples: int = 5000) -> dict:
    """Explicaciones LIME de los casos elegidos ({caso: explicación}); en clasificación, de la clase "Deprimido"."""
    matriz = codificar_para_lime(X_test)
    predecir = funcion_prediccion_lime(pipeline, modo)
    explicaciones = {}
    for caso, fila in casos.iterrows():
        extra = {"labels": (1,)} if modo == "classification" else {}
        explicaciones[caso] = explicador.explain_instance(
            matriz[fila["posicion"]], predecir, num_features=len(PREDICTORES), num_samples=num_samples, **extra,
        )
    return explicaciones


def pesos_lime(explicacion) -> pd.Series:
    """Peso de cada variable en la explicación LIME (para clasificación, de la clase "Deprimido")."""
    mapa = explicacion.as_map()
    clave = 1 if 1 in mapa else next(iter(mapa))
    nombres = explicacion.domain_mapper.feature_names
    return pd.Series({nombres[indice]: peso for indice, peso in mapa[clave]}, name="Peso LIME")


def graficar_lime(explicacion, titulo: str):
    """Gráfico de barras de LIME (reglas locales y su peso) para un adolescente."""
    mapa = explicacion.as_map()
    fig = explicacion.as_pyplot_figure(label=1 if 1 in mapa else next(iter(mapa)))
    fig.set_size_inches(10, 4.5)
    fig.axes[0].set_title(titulo, fontweight="bold")
    fig.tight_layout()
    return fig


def comparar_lime_shap(casos: pd.DataFrame, explicaciones_lime: dict, explicacion_shap: shap.Explanation,
                       top: int = 3) -> pd.DataFrame:
    """Top de variables por |peso| según LIME y según SHAP para cada caso, lado a lado."""
    filas = []
    for caso, fila in casos.iterrows():
        lime_caso = pesos_lime(explicaciones_lime[caso])
        lime_caso = lime_caso.reindex(lime_caso.abs().sort_values(ascending=False).index)
        shap_caso = pd.Series(explicacion_shap.values[fila["posicion"]], index=explicacion_shap.feature_names)
        shap_caso = shap_caso.reindex(shap_caso.abs().sort_values(ascending=False).index)
        for puesto in range(top):
            filas.append({
                "Caso": caso, "Puesto": puesto + 1,
                "LIME: variable": lime_caso.index[puesto], "LIME: peso": lime_caso.iloc[puesto],
                "SHAP: variable": shap_caso.index[puesto], "SHAP: valor": shap_caso.iloc[puesto],
            })
    tabla = pd.DataFrame(filas).set_index(["Caso", "Puesto"])
    tabla["Misma variable"] = tabla["LIME: variable"] == tabla["SHAP: variable"]
    return tabla


# ---------------------------------------------------------------------------
# Sección 5: permutation importance (solo ElasticNet)
# ---------------------------------------------------------------------------

def importancia_permutacion(pipeline, X_test: pd.DataFrame, y_test, n_repeats: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Permutation importance sobre el test con el pipeline completo: aumento del RMSE (puntos BDI) al
    desordenar cada variable, media ± DE de las repeticiones."""
    resultado = permutation_importance(
        pipeline, X_test, y_test, scoring="neg_root_mean_squared_error", n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    tabla = pd.DataFrame(
        {"Aumento del RMSE (media)": resultado.importances_mean, "DE": resultado.importances_std},
        index=pd.Index([NOMBRES[columna] for columna in X_test.columns], name="Variable"),
    )
    return tabla.sort_values("Aumento del RMSE (media)", ascending=False)


def graficar_permutacion(tabla: pd.DataFrame, titulo: str):
    """Barras de la permutation importance con barras de error (DE de las repeticiones)."""
    orden = tabla.sort_values("Aumento del RMSE (media)")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(orden.index, orden["Aumento del RMSE (media)"], xerr=orden["DE"], color=NEUTRO, capsize=3)
    ax.axvline(0, color=TEXTO, linewidth=1)
    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel("Aumento del RMSE al permutar la variable (puntos BDI)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def tabla_importancias(importancias: list[tuple]) -> pd.DataFrame:
    """Formato largo a partir de [(modelo, metodo, Serie de valores por variable, Serie de DE o None), ...]."""
    filas = []
    for modelo, metodo, valores, desviaciones in importancias:
        for variable, valor in valores.items():
            filas.append({
                "modelo": modelo, "metodo": metodo, "variable": variable, "valor": float(valor),
                "desviacion": float(desviaciones[variable]) if desviaciones is not None else np.nan,
            })
    return pd.DataFrame(filas)


def guardar_resultados(importancias: pd.DataFrame, casos: dict, ruta: Path = RUTA_INTERPRETABILIDAD) -> None:
    """Guarda importancias.csv (formato largo) y casos_explicados.csv (subject_id y por qué se eligió cada caso)."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    importancias.to_csv(ruta / "importancias.csv", index=False)
    filas = []
    for modelo, tabla in casos.items():
        for caso, fila in tabla.iterrows():
            filas.append({"modelo": modelo, "caso": caso, "subject_id": fila["subject_id"], "real": fila["Real"],
                          "prediccion": fila["Predicción"], "motivo": fila["Motivo"]})
    pd.DataFrame(filas).to_csv(ruta / "casos_explicados.csv", index=False)
    print(f"✓ Guardado en {ruta}: importancias.csv y casos_explicados.csv")
