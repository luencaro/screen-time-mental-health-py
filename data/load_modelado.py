"""Carga y caché de los resultados del modelado exportados para el dashboard.

Los archivos los genera notebooks/exportar_dashboard.ipynb en data/processed/dashboard/ (no se
versionan). Si falta alguno, las funciones lanzan ``ResultadosNoDisponibles`` con un mensaje, y
la app muestra un aviso sin romperse. Solo usa pandas y json: nada de scikit-learn ni de src/.
"""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = RAIZ / "data" / "processed" / "dashboard"
CARPETA_PREDICCION = CARPETA_RESULTADOS / "prediccion"

PASOS_EXPORTACION = [
    "Ejecutar en el contenedor de Jupyter los notebooks de modelado (clf_*.ipynb, clf_comparacion, "
    "reg_*.ipynb, reg_comparacion, interpretabilidad y genero) y clustering.ipynb.",
    "Ejecutar notebooks/exportar_dashboard.ipynb, que escribe los resultados en data/processed/dashboard/.",
    "Reconstruir y levantar el dashboard con docker compose up --build -d y recargar la página.",
]


class ResultadosNoDisponibles(Exception):
    """Faltan los resultados exportados del modelado (o una dependencia para usarlos)."""


def _ruta(nombre: str) -> Path:
    ruta = CARPETA_RESULTADOS / nombre
    if not ruta.exists():
        raise ResultadosNoDisponibles(
            f"No se encontró {ruta.relative_to(RAIZ).as_posix()}. Hay que ejecutar los notebooks de "
            "modelado y después notebooks/exportar_dashboard.ipynb."
        )
    return ruta


@lru_cache(maxsize=None)
def leer_csv(nombre: str) -> pd.DataFrame:
    """Lee un CSV de data/processed/dashboard/ una sola vez por proceso (no modificar el resultado)."""
    return pd.read_csv(_ruta(nombre))


@lru_cache(maxsize=None)
def leer_json(nombre: str) -> dict:
    """Lee un JSON de data/processed/dashboard/ una sola vez por proceso (no modificar el resultado)."""
    with open(_ruta(nombre), encoding="utf-8") as archivo:
        return json.load(archivo)


# ── Clasificación ──────────────────────────────────────────────────────
def clasificacion_benchmark_cv() -> pd.DataFrame:
    return leer_csv("clasificacion_benchmark_cv.csv")


def clasificacion_benchmark_test() -> pd.DataFrame:
    return leer_csv("clasificacion_benchmark_test.csv")


def clasificacion_curvas() -> pd.DataFrame:
    return leer_csv("clasificacion_curvas.csv")


def clasificacion_final() -> dict:
    return leer_json("clasificacion_final.json")


def clasificacion_matriz_confusion() -> pd.DataFrame:
    return leer_csv("clasificacion_matriz_confusion.csv")


def clasificacion_umbral() -> pd.DataFrame:
    return leer_csv("clasificacion_umbral.csv")


# ── Regresión ──────────────────────────────────────────────────────────
def regresion_benchmark_cv() -> pd.DataFrame:
    return leer_csv("regresion_benchmark_cv.csv")


def regresion_benchmark_test() -> pd.DataFrame:
    return leer_csv("regresion_benchmark_test.csv")


def regresion_final() -> dict:
    return leer_json("regresion_final.json")


def regresion_predicciones_test() -> pd.DataFrame:
    return leer_csv("regresion_predicciones_test.csv")


def regresion_vs_clasificacion() -> pd.DataFrame:
    return leer_csv("regresion_vs_clasificacion.csv")


# ── Interpretabilidad y género ─────────────────────────────────────────
def importancias() -> pd.DataFrame:
    return leer_csv("importancias.csv")


def genero_metricas_test() -> pd.DataFrame:
    return leer_csv("genero_metricas_test.csv")


def genero_importancias() -> pd.DataFrame:
    return leer_csv("genero_importancias.csv")


def genero_resumen() -> dict:
    return leer_json("genero_resumen.json")


# ── Clustering ─────────────────────────────────────────────────────────
def clustering_seleccion_k() -> pd.DataFrame:
    return leer_csv("clustering_seleccion_k.csv")


def clustering_perfiles() -> pd.DataFrame:
    return leer_csv("clustering_perfiles.csv")


def clustering_puntos() -> pd.DataFrame:
    return leer_csv("clustering_puntos.csv")


def clustering_centroides() -> pd.DataFrame:
    return leer_csv("clustering_centroides.csv")


def clustering_meta() -> dict:
    return leer_json("clustering_meta.json")


# ── Predicción ─────────────────────────────────────────────────────────
def rangos_prediccion() -> dict:
    return leer_json("prediccion/rangos.json")
