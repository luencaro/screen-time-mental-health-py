"""Carga, validación y caché del dataset *Screen Time vs Mental Health*.

El CSV no se versiona. Si no está disponible, ``cargar_datos`` lanza
``DatosNoDisponibles`` con un mensaje y el comando de descarga, para que la
app pueda mostrarlo sin romperse.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "raw" / "screen_time_mental_health.csv"

COMANDO_DESCARGA = (
    "kaggle datasets download -d kylefengkfeng209/screen-time-vs-mental-health-ml-ready "
    "-p data/raw --unzip"
)

# Columnas esperadas y su tipo lógico
COLUMNAS = {
    "subject_id": "entero",
    "sex": "categoria",
    "screen_time_index": "numerico",
    "est_leisure_screen_hours": "numerico",
    "sleep_quality_index": "numerico",
    "avg_sleep_hours": "numerico",
    "midsleep_weekend_hours": "numerico",
    "social_jetlag_hours": "numerico",
    "bdi_total": "entero",
    "depressed": "entero",
}

CATEGORIAS_SEXO = ("Boy", "Girl")
CORTE_BDI = 14  # BDI-II ≥ 14 → depressed = 1

VARIABLES_PANTALLA = ["screen_time_index", "est_leisure_screen_hours"]
VARIABLES_SUENO = [
    "sleep_quality_index",
    "avg_sleep_hours",
    "midsleep_weekend_hours",
    "social_jetlag_hours",
]
VARIABLES_NUMERICAS = VARIABLES_PANTALLA + VARIABLES_SUENO + ["bdi_total"]

ETIQUETAS = {
    "subject_id": "Identificador",
    "sex": "Sexo",
    "screen_time_index": "Índice de tiempo de pantalla",
    "est_leisure_screen_hours": "Horas de pantalla de ocio",
    "sleep_quality_index": "Índice de calidad del sueño",
    "avg_sleep_hours": "Horas de sueño promedio",
    "midsleep_weekend_hours": "Punto medio del sueño (fin de semana)",
    "social_jetlag_hours": "Jet lag social",
    "bdi_total": "Puntaje BDI-II",
    "depressed": "Estado depresivo",
}

ETIQUETAS_SEXO = {"Boy": "Chicos", "Girl": "Chicas"}
ETIQUETAS_ESTADO = {0: "Bajo el corte", 1: "Sobre el corte"}


class DatosNoDisponibles(Exception):
    """El CSV no existe o no tiene la estructura esperada."""


def _validar(df: pd.DataFrame) -> pd.DataFrame:
    """Comprueba columnas, tipos y dominios básicos; devuelve el DataFrame tipado."""
    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        raise DatosNoDisponibles(f"Faltan columnas en el CSV: {', '.join(faltantes)}.")

    df = df[list(COLUMNAS)].copy()
    for col, tipo in COLUMNAS.items():
        if tipo == "numerico":
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        elif tipo == "entero":
            df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
        else:
            df[col] = df[col].astype(str)

    desconocidas = set(df["sex"].unique()) - set(CATEGORIAS_SEXO)
    if desconocidas:
        raise DatosNoDisponibles(f"Valores inesperados en 'sex': {sorted(desconocidas)}.")
    if not set(df["depressed"].unique()) <= {0, 1}:
        raise DatosNoDisponibles("La columna 'depressed' debe ser binaria (0/1).")
    return df


@lru_cache(maxsize=1)
def cargar_datos() -> pd.DataFrame:
    """Lee el CSV una sola vez por proceso y lo devuelve validado.

    Se devuelve siempre el mismo objeto: los consumidores no deben modificarlo.
    """
    if not RUTA_CSV.exists():
        ruta = RUTA_CSV.relative_to(RAIZ) if RUTA_CSV.is_relative_to(RAIZ) else RUTA_CSV
        raise DatosNoDisponibles(f"No se encontró el archivo {ruta.as_posix()}.")
    return _validar(pd.read_csv(RUTA_CSV))

