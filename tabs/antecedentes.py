"""Pestaña 02 · Antecedentes: texto del libro y una card por estudio."""

import dash_bootstrap_components as dbc
from dash import html

from components.formato import entero
from components.ui import badge, card, encabezado_seccion, fila_kpis, interpretacion, kpi, markdown
from content import libro
from content.fichas import estudios


def _card_estudio(e: dict) -> html.Div:
    """Card con muestra, diseño y hallazgo clave de un estudio."""
    cabecera = html.Div(
        [html.H2(e["cita"], className="card-titulo mb-0"),
         badge("Estudio base", "primaria") if e.get("estudio_base") else None],
        className="d-flex justify-content-between align-items-start gap-2 mb-1",
    )
    enlace = (html.A("Ver artículo", href=f"https://doi.org/{e['doi']}", target="_blank")
              if e.get("doi") else None)
    return card(
        cabecera,
        html.P(e["titulo"], className="card-subtitulo"),
        html.Dl([
            html.Dt("Muestra"), html.Dd(e["muestra"]),
            html.Dt("Diseño"), html.Dd(e["diseno"]),
            html.Dt("Hallazgo clave"), html.Dd(e["hallazgo"]),
        ], className="dato-lista"),
        html.P(enlace, className="nota mt-3 mb-0") if enlace else None,
        className="estirar",
    )


def layout():
    """Estructura: KPIs → texto de antecedentes + síntesis → cards por estudio."""
    lista = estudios()
    bloques = libro.parrafos(libro.seccion("intro.md", "Antecedentes"))
    cuerpo, sintesis = "\n\n".join(bloques[:-1]), bloques[-1]
    anios = [e["anio"] for e in lista if e["anio"]]
    mayor = max((e for e in lista if e["muestra_n"]), key=lambda e: e["muestra_n"])
    suecos = sum(e["pais"] == "Suecia" for e in lista)

    return html.Div([
        encabezado_seccion("antecedentes", [f"{len(lista)} estudios", f"{min(anios)}–{max(anios)}"]),
        fila_kpis([
            kpi("Estudios revisados", entero(len(lista)), "Citados en el marco del proyecto"),
            kpi("Periodo", f"{min(anios)}–{max(anios)}", "Años de publicación"),
            kpi("Mayor muestra", entero(mayor["muestra_n"]), mayor["cita"]),
            kpi("Estudios con adolescentes suecos", entero(suecos), "Misma población que el dataset"),
        ]),
        dbc.Row([
            dbc.Col(card(markdown(cuerpo), titulo="Evidencia previa"), lg=8),
            dbc.Col(interpretacion(markdown(sintesis), titulo="Síntesis"), lg=4),
        ], className="fila"),
        dbc.Row([dbc.Col(_card_estudio(e), md=6, lg=4) for e in lista], className="fila"),
    ])
