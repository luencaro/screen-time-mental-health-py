"""Pestaña 04 · Metodología: contenido de book/methodology.md."""

from dash import html

from components.formato import entero
from components.ui import card, encabezado_seccion, fila_kpis, kpi, markdown
from content import libro
from data.load_data import VARIABLES_PANTALLA, VARIABLES_SUENO, cargar_datos


def layout():
    """Estructura: KPIs → introducción → una card por apartado de methodology.md.

    Los apartados con tabla van a todo el ancho; el resto, en rejilla tipo mosaico.
    """
    df = cargar_datos()
    intro, partes = libro.subsecciones("methodology.md", "Metodología")
    con_tabla = [card(markdown(c), titulo=t) for t, c in partes if "|---" in c]
    sin_tabla = [card(markdown(c), titulo=t) for t, c in partes if "|---" not in c]
    predictores = len(VARIABLES_PANTALLA) + len(VARIABLES_SUENO)

    return html.Div([
        encabezado_seccion("metodologia", ["Kaggle", f"{df.shape[1]} variables"]),
        fila_kpis([
            kpi("Registros", entero(len(df)), "Un adolescente por fila"),
            kpi("Variables", entero(df.shape[1]), "2 objetivos · 1 identificador"),
            kpi("Predictores continuos", entero(predictores),
                f"{len(VARIABLES_PANTALLA)} de pantalla · {len(VARIABLES_SUENO)} de sueño"),
        ]),
        html.Div(markdown(intro), className="intro-bloque"),
        *[html.Div(c, className="fila") for c in con_tabla],
        html.Div(sin_tabla, className="rejilla-mosaico"),
    ])
