"""Pestaña 02 · Marco teórico: antecedentes y base estadística del estudio."""

import dash_bootstrap_components as dbc
from dash import html

from components.ui import badge, card, encabezado_seccion, markdown
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


def _base_estadistica() -> list:
    """Encabezado, introducción y una card por subsección de book/base_estadistica.md."""
    intro, partes = libro.subsecciones("base_estadistica.md", "Base estadística")
    return [
        html.H2("Base estadística del estudio", className="subtitulo-seccion"),
        html.Div(markdown(intro), className="intro-bloque"),
        html.Div([card(markdown(cuerpo), titulo=titulo) for titulo, cuerpo in partes],
                 className="rejilla-mosaico"),
    ]


def layout():
    """Estructura: antecedentes → base estadística → cards de estudios."""
    lista = estudios()
    antecedentes = libro.seccion("intro.md", "Antecedentes")

    return html.Div([
        encabezado_seccion("marco_teorico", [f"{len(lista)} estudios", "Base estadística"]),
        dbc.Row(dbc.Col(card(markdown(antecedentes, className="dos-columnas"),
                             titulo="Antecedentes")), className="fila"),
        *_base_estadistica(),
        html.H2("Estudios revisados", className="subtitulo-seccion"),
        dbc.Row([dbc.Col(_card_estudio(e), md=6, lg=4) for e in lista], className="fila"),
    ])
