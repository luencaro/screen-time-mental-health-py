"""Fichas estructuradas y breves que complementan el texto del libro.

Resúmenes de estudios (muestra, diseño, hallazgo) extraídos de la sección
Antecedentes de book/intro.md, y la fase en que se atiende cada objetivo.
Los párrafos completos se leen siempre desde book/*.md.
"""

from content.libro import autor_anio, referencias

# Resumen de cada estudio citado en Antecedentes (datos tomados de book/intro.md)
ESTUDIOS = [
    {
        "clave": "hokby2023longitudinal",
        "muestra_n": 4793,
        "muestra": "4.793 adolescentes suecos",
        "pais": "Suecia",
        "diseno": "Longitudinal (0, 3 y 12 meses)",
        "hallazgo": "El tiempo de pantalla no predijo la depresión de forma directa (p = 0,469); sí de "
                    "forma indirecta al interferir con el afrontamiento centrado en el problema, con "
                    "un efecto máximo de 3,4 puntos en el BDI-II.",
    },
    {
        "clave": "lemke2023associations",
        "muestra_n": 8449,
        "muestra": "8.449 adolescentes suecos de 12–16 años",
        "pais": "Suecia",
        "diseno": "Transversal",
        "hallazgo": "La duración del sueño entre semana (OR = 0,773), la calidad del sueño "
                    "(OR = 0,327) y el cronotipo tardío (OR = 1,126) predijeron la depresión "
                    "(BDI-II > 13); la duración en fin de semana no fue significativa.",
    },
    {
        "clave": "saleem2024exploring",
        "muestra_n": None,
        "muestra": "67 estudios (de 4.850 identificados), población de 5 a 18 años",
        "pais": "Varios",
        "diseno": "Revisión narrativa sistemática",
        "hallazgo": "Asociación positiva consistente entre el uso de redes sociales y los síntomas de "
                    "depresión y ansiedad, con alta heterogeneidad metodológica.",
    },
    {
        "clave": "hokby2025adolescents",
        "muestra_n": 4810,
        "muestra": "4.810 adolescentes suecos de 12–16 años",
        "pais": "Suecia",
        "diseno": "Longitudinal · SEM multigrupo",
        "hallazgo": "El tiempo de pantalla deterioró el sueño a los 3 meses. En chicas, el efecto "
                    "sobre la depresión fue mediado por la calidad (57 %), la duración (38 %) y el "
                    "cronotipo (45 %); en chicos fue directo.",
        "estudio_base": True,
    },
    {
        "clave": "chen2026predicting",
        "muestra_n": 1226,
        "muestra": "1.226 adolescentes chinos de 13–25 años",
        "pais": "China",
        "diseno": "Aprendizaje automático (5 algoritmos)",
        "hallazgo": "XGBoost obtuvo el mejor desempeño (AUC = 0,927). Según SHAP, los predictores "
                    "más relevantes fueron el uso de somníferos, la indiferencia parental y el "
                    "nivel educativo.",
    },
]


def estudios() -> list[dict]:
    """Estudios con cita autor-año, año y título desde references.bib."""
    refs = referencias()
    salida = []
    for e in ESTUDIOS:
        ref = refs.get(e["clave"], {})
        salida.append({
            **e,
            "cita": autor_anio(e["clave"]),
            "anio": int(ref["anio"]) if str(ref.get("anio", "")).isdigit() else None,
            "titulo": ref.get("titulo", ""),
            "revista": ref.get("revista", ""),
            "doi": ref.get("doi", ""),
        })
    return salida


# Objetivos específicos que cubre esta fase y sección del dashboard que los atiende
OBJETIVOS_FASE = {
    1: {"fase": "eda", "secciones": ["variable_objetivo", "analisis_numerico", "calidad_datos"]},
    2: {"fase": "eda", "secciones": ["correlaciones"]},
    3: {"fase": "eda", "secciones": ["analisis_sexo"]},
    4: {"fase": "modelado", "secciones": []},
    5: {"fase": "modelado", "secciones": []},
}
