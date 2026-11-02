"""Predicción con los modelos portables exportados (pestaña Predicción del dashboard).

Reproduce los pipelines finales sin scikit-learn ni src/:

- Clasificación: el preprocesamiento (StandardScaler + indicadora de sexo) se calcula con numpy a
  partir de catboost_preprocesamiento.json y la probabilidad la da el CatBoostClassifier cargado
  desde catboost.cbm (formato nativo).
- Regresión: ElasticNet con numpy (intercepto + coeficientes · variables transformadas),
  recortado a [0, 63], a partir de elasticnet.json.

notebooks/exportar_dashboard.ipynb usa este mismo módulo para verificar que los resultados
coinciden con los pipelines originales.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from data.load_modelado import CARPETA_PREDICCION, ResultadosNoDisponibles

# Categorías clínicas del BDI-II: (límite superior exclusivo, nombre)
CATEGORIAS_BDI = [(14, "Mínimo"), (20, "Leve"), (29, "Moderado"), (64, "Severo")]
RANGOS_CATEGORIAS_BDI = {"Mínimo": "0–13", "Leve": "14–19", "Moderado": "20–28", "Severo": "29–63"}


def _leer(carpeta: Path, nombre: str) -> dict:
    ruta = Path(carpeta) / nombre
    if not ruta.exists():
        raise ResultadosNoDisponibles(
            f"No se encontró {nombre} en {Path(carpeta).as_posix()}. Hay que ejecutar los notebooks de "
            "modelado y después notebooks/exportar_dashboard.ipynb."
        )
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def transformar(entrada, preprocesamiento: dict) -> np.ndarray:
    """Matriz (n, columnas_salida) idéntica a la del ColumnTransformer del pipeline.

    ``entrada`` es cualquier objeto indexable por nombre de variable: un DataFrame o un dict
    con un valor (o una lista de valores) por variable.
    """
    columnas = []
    for t in preprocesamiento["transformaciones"]:
        valores = np.atleast_1d(np.asarray(entrada[t["variable"]]))
        if t["tipo"] == "estandarizar":
            columnas.append((valores.astype(float) - t["media"]) / t["escala"])
        elif t["tipo"] == "indicadora":
            columnas.append((valores.astype(str) == t["categoria"]).astype(float))
        else:
            raise ValueError(f"Transformación desconocida: {t['tipo']!r}")
    return np.column_stack(columnas)


def categoria_bdi(valor: float) -> str:
    """Categoría clínica del BDI-II (0–13 mínimo, 14–19 leve, 20–28 moderado, 29–63 severo)."""
    for limite, nombre in CATEGORIAS_BDI:
        if valor < limite:
            return nombre
    return CATEGORIAS_BDI[-1][1]


class Predictor:
    """CatBoost (probabilidad de depresión) y ElasticNet (BDI-II) a partir de los archivos portables."""

    def __init__(self, carpeta: Path = CARPETA_PREDICCION):
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise ResultadosNoDisponibles(
                "La librería catboost no está instalada en el contenedor del dashboard. Reconstrúyelo con "
                "docker compose up --build -d (requirements.txt incluye catboost)."
            ) from error
        carpeta = Path(carpeta)
        self.preprocesamiento_clf = _leer(carpeta, "catboost_preprocesamiento.json")
        self.regresion = _leer(carpeta, "elasticnet.json")
        ruta_cbm = carpeta / self.preprocesamiento_clf["modelo"]
        if not ruta_cbm.exists():
            raise ResultadosNoDisponibles(
                f"No se encontró {ruta_cbm.name}. Hay que ejecutar notebooks/exportar_dashboard.ipynb."
            )
        self.clasificador = CatBoostClassifier().load_model(str(ruta_cbm))
        self._coeficientes = np.array([self.regresion["coeficientes"][c] for c in self.regresion["columnas_salida"]])

    @property
    def umbral(self) -> float:
        return float(self.preprocesamiento_clf["umbral"])

    def probabilidad(self, entrada) -> np.ndarray:
        """Probabilidad de depresión (clase positiva, depressed = 1: segunda columna, como en el pipeline)."""
        X = transformar(entrada, self.preprocesamiento_clf)
        return self.clasificador.predict_proba(X)[:, 1]

    def bdi(self, entrada) -> np.ndarray:
        """Puntaje BDI-II estimado, recortado al rango del instrumento."""
        X = transformar(entrada, self.regresion)
        minimo, maximo = self.regresion["recorte"]
        return np.clip(self.regresion["intercepto"] + X @ self._coeficientes, minimo, maximo)


@lru_cache(maxsize=1)
def cargar_predictor() -> Predictor:
    """Predictor del dashboard, cargado una sola vez por proceso."""
    return Predictor(CARPETA_PREDICCION)
