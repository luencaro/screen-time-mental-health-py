"""Pestaña 01 · Inicio: contexto del estudio y datos generales de la muestra."""

import re

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, pct
from components.secciones import BLOQUES, secciones_de
from components.ui import (aviso_sin_datos, card, encabezado_seccion, enlace_interno,
                           fila_kpis, grafico, kpi, markdown)
from content import libro
from data.load_data import CORTE_BDI, URL_DATASET, DatosNoDisponibles, cargar_datos


def _rango_edad(texto: str) -> str:
    """Rango de edad mencionado en la introducción ('entre 12 y 16 años')."""
    m = re.search(r"entre (\d+) y (\d+) años", texto)
    return f"{m.group(1)}–{m.group(2)}" if m else "—"


def _mapa_dashboard() -> html.Nav:
    """Índice de secciones agrupadas por bloque."""
    return html.Nav([
        html.Div([
            html.H3(bloque, className="overline"),
            html.Ul([html.Li(enlace_interno(s["nombre"], f"/{s['slug']}"))
                     for s in secciones_de(bloque) if s["slug"] != "inicio"]),
        ])
        for bloque in BLOQUES
    ], className="mapa-navegacion", **{"aria-label": "Índice del dashboard"})


def layout():
    """Presenta el contexto y el índice, con un resumen de la muestra si hay datos."""
    texto = libro.seccion("intro.md", "Introducción")
    contexto = card(markdown(texto), titulo="Contexto del proyecto", className="estirar")
    mapa = card(_mapa_dashboard(), titulo="Explora las secciones")
    try:
        r = stats.resumen_muestra(cargar_datos())
    except DatosNoDisponibles as error:
        return html.Div([
            encabezado_seccion("inicio"),
            dbc.Row([
                dbc.Col(contexto, lg=8),
                dbc.Col(aviso_sin_datos(str(error), URL_DATASET), lg=4),
            ], className="fila"),
            mapa,
        ])

    return html.Div([
        encabezado_seccion("inicio", ["Estocolmo · Suecia"]),
        fila_kpis([
            kpi("Adolescentes", entero(r["n"]), "Participantes en el dataset"),
            kpi("Rango de edad", _rango_edad(texto), "años"),
            kpi("Chicas / chicos", f"{pct(r['prop_chicas'], 0)} / {pct(r['prop_chicos'], 0)}",
                f"{entero(r['n_chicas'])} chicas · {entero(r['n_chicos'])} chicos"),
            kpi(f"BDI-II ≥ {CORTE_BDI}", pct(r["prevalencia"]),
                f"{entero(r['n_deprimidos'])} alcanzan el punto de corte", "deprimido"),
        ]),
        dbc.Row([
            dbc.Col(contexto, lg=7),
            dbc.Col(card(
                html.P("Distribución de la muestra según el punto de corte del estudio.",
                       className="card-subtitulo"),
                grafico("in-dona", 280, animar=False),
                titulo="Clasificación por BDI-II", className="estirar",
            ), lg=5),
        ], className="fila"),
        mapa,
    ])


def register_callbacks(app):
    """Regenera la dona al cambiar el tema."""

    @app.callback(Output("in-dona", "figure"), Input("tema", "data"))
    def _figura(tema):
        return figures.dona_estado(cargar_datos(), tema)
