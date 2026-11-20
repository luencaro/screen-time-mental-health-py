"""Pestaña 04 · Metodología: contenido de book/methodology.md."""

from dash import html

from components.formato import entero
from components.ui import card, encabezado_seccion, fila_kpis, kpi, markdown
from content import libro
from data.load_data import COLUMNAS, VARIABLES_PANTALLA, VARIABLES_SUENO, DatosNoDisponibles, cargar_datos


def layout():
    """Estructura: KPIs → introducción → una card por apartado de methodology.md.

    Los apartados con tabla van a todo el ancho; el resto, en rejilla tipo mosaico.
    """
    intro, partes = libro.subsecciones("methodology.md", "Metodología")
    con_tabla = [card(markdown(c), titulo=t) for t, c in partes if "|---" in c]
    sin_tabla = [card(markdown(c), titulo=t) for t, c in partes if "|---" not in c]
    predictores = len(VARIABLES_PANTALLA) + len(VARIABLES_SUENO)
    resumen = [
        kpi("Variables", entero(len(COLUMNAS)), "2 objetivos · 1 identificador"),
        kpi("Predictores continuos", entero(predictores),
            f"{len(VARIABLES_PANTALLA)} de pantalla · {len(VARIABLES_SUENO)} de sueño"),
    ]
    try:
        resumen.insert(0, kpi("Registros", entero(len(cargar_datos())), "Un adolescente por fila"))
    except DatosNoDisponibles:
        pass

    return html.Div([
        encabezado_seccion("metodologia"),
        fila_kpis(resumen),
        html.Div(markdown(intro), className="intro-bloque"),
        *[html.Div(c, className="fila") for c in con_tabla],
        html.Div(sin_tabla, className="rejilla-mosaico"),
    ])
