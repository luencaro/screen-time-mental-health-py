"""Pestaña 06 · Variable objetivo: distribución de bdi_total y depressed."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, pct
from components.ui import card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi, var
from data.load_data import CORTE_BDI, cargar_datos


def layout():
    """Estructura: KPIs → histograma + dona → boxplot + interpretación."""
    df = cargar_datos()
    r = stats.resumen_muestra(df)
    g = stats.granularidad_corte(df)
    n_bajo = r["n"] - r["n_deprimidos"]

    return html.Div([
        encabezado_seccion("variable_objetivo", [f"n = {entero(r['n'])}", "bdi_total · depressed"]),
        fila_kpis([
            kpi("Sobre el corte clínico", pct(r["prevalencia"]),
                f"{entero(r['n_deprimidos'])} adolescentes con BDI-II ≥ {CORTE_BDI}", "deprimido"),
            kpi("Bajo el corte clínico", pct(1 - r["prevalencia"]),
                f"{entero(n_bajo)} adolescentes", "no_deprimido"),
            kpi("Mediana de BDI-II", num(r["mediana_bdi"], 0), f"Media {num(r['media_bdi'], 2)}"),
            kpi("Asimetría de BDI-II", num(g["asimetria"], 2), "Positiva: cola hacia la derecha"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P(["Distribución de ", var("bdi_total"), " coloreada por tramo respecto al corte."],
                       className="card-subtitulo"),
                grafico("vo-histograma", 380),
                titulo="Puntaje BDI-II",
            ), lg=8),
            dbc.Col(card(
                html.P(["Proporción de ", var("depressed")], className="card-subtitulo"),
                grafico("vo-dona", 300),
                titulo="Estado según el corte",
                className="estirar",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Mediana punteada y corte clínico discontinuo.", className="card-subtitulo"),
                grafico("vo-boxplot", 220),
                titulo="Dispersión de BDI-II",
            ), lg=8),
            dbc.Col(interpretacion(
                f"La distribución de `bdi_total` presenta sesgo a la derecha (asimetría "
                f"{num(g['asimetria'], 2)}): la mediana ({num(r['mediana_bdi'], 0)}) queda por debajo "
                f"de la media ({num(r['media_bdi'], 2)}) y el 90 % de la muestra puntúa "
                f"{num(g['p90'], 0)} o menos. El {pct(g['prop_cero'])} registra un puntaje de 0.",
                f"Al binarizar en BDI-II ≥ {CORTE_BDI}, {g['valores_distintos_sobre']} puntajes "
                f"distintos (de {g['min_sobre']} a {g['max_sobre']}) quedan en una sola categoría. "
                f"Además, {entero(g['n_cerca'])} adolescentes ({pct(g['prop_cerca'])}) puntúan entre "
                f"{g['cerca_desde']} y {g['cerca_hasta']}, a {g['margen']} valores o menos del corte, "
                "donde un punto de diferencia cambia la clase. Conservar `bdi_total` como variable "
                "continua evita esa pérdida de granularidad.",
            ), lg=4),
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
