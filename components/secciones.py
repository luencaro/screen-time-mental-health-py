"""Registro de secciones del dashboard: bloque, número, nombre y descripción.

Lo usan el menú superior (app.py) y el encabezado de cada pestaña (breadcrumb).
"""

BLOQUES = ["Proyecto", "EDA", "Cierre"]

SECCIONES = [
    # Proyecto
    {"slug": "introduccion", "bloque": "Proyecto", "nombre": "Introducción",
     "descripcion": "Contexto del estudio y datos generales de la muestra"},
    {"slug": "antecedentes", "bloque": "Proyecto", "nombre": "Antecedentes",
     "descripcion": "Estudios previos sobre pantallas, sueño y depresión"},
    {"slug": "objetivos", "bloque": "Proyecto", "nombre": "Objetivos",
     "descripcion": "Objetivo general, específicos y alcance de esta fase"},
    {"slug": "metodologia", "bloque": "Proyecto", "nombre": "Metodología",
     "descripcion": "Variables, operacionalización y flujo del EDA"},
    # EDA
    {"slug": "calidad_datos", "bloque": "EDA", "nombre": "Calidad de datos",
     "descripcion": "Dimensiones, nulos, duplicados y descriptivos"},
    {"slug": "variable_objetivo", "bloque": "EDA", "nombre": "Variable objetivo",
     "descripcion": "Distribución de bdi_total y de depressed"},
    {"slug": "analisis_sexo", "bloque": "EDA", "nombre": "Análisis por sexo",
     "descripcion": "Diferencias entre chicas y chicos con pruebas"},
    {"slug": "analisis_numerico", "bloque": "EDA", "nombre": "Variables numéricas",
     "descripcion": "Histogramas interactivos por variable y grupo"},
    {"slug": "correlaciones", "bloque": "EDA", "nombre": "Correlaciones",
     "descripcion": "Spearman entre pantalla, sueño y BDI-II"},
    {"slug": "outliers", "bloque": "EDA", "nombre": "Outliers",
     "descripcion": "Valores atípicos por IQR y su relevancia clínica"},
    # Cierre
    {"slug": "hallazgos", "bloque": "Cierre", "nombre": "Hallazgos",
     "descripcion": "Síntesis de resultados e implicaciones para el modelado"},
    {"slug": "limitaciones", "bloque": "Cierre", "nombre": "Limitaciones",
     "descripcion": "Alcance del análisis y siguiente paso"},
]

for _i, _s in enumerate(SECCIONES, start=1):
    _s["numero"] = f"{_i:02d}"

POR_SLUG = {s["slug"]: s for s in SECCIONES}
SECCION_INICIAL = "introduccion"


def slug_desde_ruta(ruta: str | None) -> str:
    """Convierte una ruta ('/variable_objetivo') en el slug de la sección."""
    slug = (ruta or "/").strip("/").split("/")[0]
    return slug if slug in POR_SLUG else SECCION_INICIAL


def secciones_de(bloque: str) -> list[dict]:
    """Secciones que pertenecen a un bloque del menú."""
    return [s for s in SECCIONES if s["bloque"] == bloque]
