"""Registro de secciones del dashboard: bloque, nombre y descripción.

Lo usan el menú superior (app.py) y el encabezado de cada pestaña (breadcrumb).
"""

BLOQUES = ["Proyecto", "EDA", "Modelado", "Acerca de"]

SECCIONES = [
    # Proyecto
    {"slug": "inicio", "bloque": "Proyecto", "nombre": "Resumen del estudio",
     "descripcion": "Contexto del proyecto y características de la muestra"},
    {"slug": "marco_teorico", "bloque": "Proyecto", "nombre": "Marco teórico",
     "descripcion": "Antecedentes y base estadística del estudio"},
    {"slug": "objetivos", "bloque": "Proyecto", "nombre": "Objetivos",
     "descripcion": "Objetivo general y objetivos específicos"},
    {"slug": "metodologia", "bloque": "Proyecto", "nombre": "Metodología",
     "descripcion": "Dataset, EDA, preprocesamiento, modelado y evaluación"},
    # EDA
    {"slug": "calidad_datos", "bloque": "EDA", "nombre": "Calidad de datos",
     "descripcion": "Dimensiones, nulos, duplicados y descriptivos"},
    {"slug": "variable_objetivo", "bloque": "EDA", "nombre": "Variable objetivo",
     "descripcion": "Distribución del puntaje BDI-II y clasificación según el punto de corte"},
    {"slug": "analisis_sexo", "bloque": "EDA", "nombre": "Análisis por sexo",
     "descripcion": "Diferencias entre chicas y chicos con pruebas"},
    {"slug": "analisis_numerico", "bloque": "EDA", "nombre": "Variables numéricas",
     "descripcion": "Histogramas interactivos por variable y grupo"},
    {"slug": "correlaciones", "bloque": "EDA", "nombre": "Correlaciones",
     "descripcion": "Spearman entre pantalla, sueño y BDI-II"},
    {"slug": "outliers", "bloque": "EDA", "nombre": "Outliers",
     "descripcion": "Valores atípicos por IQR y su relevancia clínica"},
    # Modelado
    {"slug": "benchmarking", "bloque": "Modelado", "nombre": "Benchmarking",
     "descripcion": "Comparación de modelos en validación cruzada y en el conjunto de prueba"},
    {"slug": "modelos_finales", "bloque": "Modelado", "nombre": "Modelos finales",
     "descripcion": "CatBoost y ElasticNet: métricas, umbral e importancia de variables"},
    {"slug": "genero", "bloque": "Modelado", "nombre": "Modelos por género",
     "descripcion": "Chicos vs. chicas: variables que pesan y efecto del umbral"},
    {"slug": "clustering", "bloque": "Modelado", "nombre": "Clustering",
     "descripcion": "Perfiles de sueño y pantalla con K-Means y su validación"},
    {"slug": "prediccion", "bloque": "Modelado", "nombre": "Predicción",
     "descripcion": "Simulación de la clasificación y del puntaje BDI-II para un perfil"},
    # Acerca de
    {"slug": "acerca", "bloque": "Acerca de", "nombre": "Equipo y recursos",
     "descripcion": "Integrantes, repositorio, Jupyter Book y dataset"},
]

POR_SLUG = {s["slug"]: s for s in SECCIONES}
SECCION_INICIAL = "inicio"


def slug_desde_ruta(ruta: str | None) -> str:
    """Convierte una ruta ('/variable_objetivo') en el slug de la sección."""
    slug = (ruta or "/").strip("/").split("/")[0]
    return slug if slug in POR_SLUG else SECCION_INICIAL


def secciones_de(bloque: str) -> list[dict]:
    """Secciones que pertenecen a un bloque del menú."""
    return [s for s in SECCIONES if s["bloque"] == bloque]
