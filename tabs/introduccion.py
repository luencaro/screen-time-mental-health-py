"""Pestaña 01 · Introducción: contexto del estudio y datos generales de la muestra."""

import re

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, pct
from components.secciones import BLOQUES, secciones_de
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi, markdown
from content import libro
from data.load_data import cargar_datos


def _rango_edad(texto: str) -> str:
    """Rango de edad mencionado en la introducción ('entre 12 y 16 años')."""
    m = re.search(r"entre (\d+) y (\d+) años", texto)
    return f"{m.group(1)}–{m.group(2)}" if m else "—"


def _mapa_dashboard() -> html.Div:
    """Lista breve de bloques y secciones con enlaces."""
    return html.Div([
        html.Div([
            html.Div(bloque.upper(), className="overline"),
            html.Ul([html.Li(html.A(s["nombre"], href=f"/{s['slug']}")) for s in secciones_de(bloque)],
                    className="mb-3"),
        ])
        for bloque in BLOQUES
    ])


def layout():
    """Estructura: KPIs → texto de introducción + dona → mapa del dashboard."""
    texto = libro.seccion("intro.md", "Introducción")
    df = cargar_datos()
    r = stats.resumen_muestra(df)

    return html.Div([
        encabezado_seccion("Inicio", [f"n = {entero(r['n'])}", "Estocolmo · Suecia"]),
        fila_kpis([
            kpi("Adolescentes", entero(r["n"]), "Participantes en el dataset"),
            kpi("Rango de edad", _rango_edad(texto), "Años, según el estudio de origen"),
            kpi("Chicas / chicos", f"{pct(r['prop_chicas'], 0)} / {pct(r['prop_chicos'], 0)}",
                f"{entero(r['n_chicas'])} chicas · {entero(r['n_chicos'])} chicos"),
            kpi("Sobre el corte clínico", pct(r["prevalencia"]),
                f"{entero(r['n_deprimidos'])} adolescentes con BDI-II ≥ 14", "deprimido"),
        ]),
        dbc.Row([
            dbc.Col(card(markdown(texto), titulo="Contexto del proyecto"), lg=8),
            dbc.Col(card(
                html.P("Proporción de adolescentes según el corte clínico del BDI-II.",
                       className="card-subtitulo"),
                grafico("in-dona", 280),
                titulo="Estado depresivo en la muestra",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Este dashboard presenta el análisis exploratorio (EDA) del proyecto. No "
                       "incluye modelado: esa fase se desarrollará a partir de los hallazgos aquí "
                       "descritos."),
                titulo="Alcance",
            ), lg=8),
            dbc.Col(card(_mapa_dashboard(), titulo="Contenido del dashboard"), lg=4),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera la dona al cambiar el tema."""

    @app.callback(Output("in-dona", "figure"), Input("tema", "data"))
    def _figura(tema):
        return figures.dona_estado(cargar_datos(), tema)
