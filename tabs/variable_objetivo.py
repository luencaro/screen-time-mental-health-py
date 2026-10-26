"""Pestaña 06 · Variable objetivo: distribución de bdi_total y depressed."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, pct
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi, var
from data.load_data import CORTE_BDI, cargar_datos


def layout():
    """Estructura: KPIs → histograma + dona → boxplot."""
    df = cargar_datos()
    r = stats.resumen_muestra(df)
    g = stats.granularidad_corte(df)
    n_bajo = r["n"] - r["n_deprimidos"]

    return html.Div([
        encabezado_seccion("variable_objetivo", [f"n = {entero(r['n'])}", "bdi_total · depressed"]),
        fila_kpis([
            kpi("Deprimidos", pct(r["prevalencia"]),
                f"{entero(r['n_deprimidos'])} adolescentes con BDI-II ≥ {CORTE_BDI}", "deprimido"),
            kpi("No deprimidos", pct(1 - r["prevalencia"]),
                f"{entero(n_bajo)} adolescentes", "no_deprimido"),
            kpi("Mediana de BDI-II", num(r["mediana_bdi"], 0), f"Media {num(r['media_bdi'], 2)}"),
            kpi("Asimetría de BDI-II", num(g["asimetria"], 2), "Positiva: cola hacia la derecha"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P(["Distribución de ", var("bdi_total"), " coloreada por tramo respecto al corte."],
                       className="card-subtitulo"),
                grafico("vo-histograma", 380, flexible=True),
                titulo="Puntaje BDI-II",
                className="estirar grafico-flexible",
            ), lg=8),
            dbc.Col(card(
                html.P(["Proporción de ", var("depressed")], className="card-subtitulo"),
                grafico("vo-dona", 280, animar=False, flexible=True),
                titulo="Estado según el corte",
                className="estirar grafico-flexible",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Mediana punteada y corte clínico discontinuo.", className="card-subtitulo"),
                grafico("vo-boxplot", 220, animar=False),
                titulo="Dispersión de BDI-II",
            ), lg=12),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(
        Output("vo-histograma", "figure"),
        Output("vo-dona", "figure"),
        Output("vo-boxplot", "figure"),
        Input("tema", "data"),
    )
    def _figuras(tema):
        df = cargar_datos()
        return (figures.histograma_bdi(df, tema), figures.dona_estado(df, tema),
                figures.boxplot_bdi(df, tema))
