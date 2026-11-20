"""Contexto de todas las secciones para la conversación con Shapiro."""

from components.secciones import POR_SLUG, SECCIONES
from components.shapiro_documentacion import contexto_documentacion
from components.shapiro_eda import contexto_eda
from components.shapiro_resultados import contexto_modelos_finales, contexto_resultados
from components.shapiro_resumenes import contexto_clustering, contexto_genero, contexto_importancias, contexto_prediccion
from data.load_data import ETIQUETAS, VARIABLES_NUMERICAS

SECCIONES_EDA = {"inicio", *(ficha["slug"] for ficha in SECCIONES if ficha["bloque"] == "EDA")}
RESUMENES_MODELO = {
    "benchmarking": (contexto_resultados,),
    "modelos_finales": (contexto_modelos_finales, contexto_importancias),
    "clustering": (contexto_clustering,),
    "genero": (contexto_genero,),
    "prediccion": (contexto_prediccion,),
}
FILTROS_VISIBLES = {
    "analisis_numerico": {
        "variable": ("Variable", {v: ETIQUETAS[v] for v in VARIABLES_NUMERICAS}),
        "agrupacion": ("Agrupación", {"todos": "Sin separar", "sex": "Por sexo", "depressed": "Por estado según BDI-II"}),
    },
    "benchmarking": {
        "tarea": ("Tarea", {"clasificacion": "Clasificación", "regresion": "Regresión"}),
    },
}


def contexto_dashboard(seccion_visible: str | None = None, filtros=None) -> str:
    """Da acceso al catálogo completo y sitúa primero la sección visible."""
    slugs = list(POR_SLUG)
    if seccion_visible in POR_SLUG:
        slugs.remove(seccion_visible)
        slugs.insert(0, seccion_visible)

    bloques = []
    if seccion_visible in POR_SLUG:
        ficha = POR_SLUG[seccion_visible]
        bloques.append(f"Sección visible en el dashboard: {ficha['nombre']} ({seccion_visible}).")
    elif seccion_visible == "portada":
        bloques.append("Sección visible en el dashboard: portada.")

    if filtros:
        for clave, (etiqueta, opciones) in FILTROS_VISIBLES.get(seccion_visible, {}).items():
            valor = opciones.get(filtros.get(clave))
            if valor:
                bloques.append(f"Filtro visible · {etiqueta}: {valor}.")

    for slug in slugs:
        ficha = POR_SLUG[slug]
        partes = [f"{ficha['bloque']} → {ficha['nombre']} ({slug})", ficha["descripcion"]]
        documentacion = contexto_documentacion(slug)
        if documentacion:
            partes.append(documentacion)

        if slug in SECCIONES_EDA:
            resumen = contexto_eda(slug)
        else:
            resumen = "\n\n".join(filter(None, (cargar() for cargar in RESUMENES_MODELO.get(slug, ()))))
        if resumen:
            partes.append(f"Resultados agregados del dashboard:\n{resumen}")
        bloques.append("\n".join(partes))

    return "\n\n".join(bloques)
