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
            dbc.Col([
                card(
                    html.P(["Proporción de ", var("depressed")], className="card-subtitulo"),
                    grafico("vo-dona", 280, animar=False),
                    titulo="Estado según el corte",
                ),
                interpretacion(
                    f"`depressed` es una simplificación de `bdi_total` mediante el corte clínico "
                    f"≥{CORTE_BDI}. Al binarizar se pierde granularidad (un adolescente con {g['min_sobre']} "
                    f"y otro con {g['max_sobre']} quedan en la misma categoría \"1\"), pero se gana "
                    "interpretabilidad clínica directa y es el formato natural si el objetivo final es "
                    f"clasificación. Tenemos un claro desbalance de clase ({pct(r['prevalencia'])} de "
                    "deprimidos), lo cual nos lleva a plantearnos la idea de usar métodos de balanceo "
                    "para los algoritmos de clasificación."
                ),
            ], lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Mediana punteada y corte clínico discontinuo.", className="card-subtitulo"),
                grafico("vo-boxplot", 220, animar=False),
                titulo="Dispersión de BDI-II",
            ), lg=8),
            dbc.Col(interpretacion(
                f"`bdi_total` presenta una distribución fuertemente sesgada a la derecha (skew de "
                f"{num(g['asimetria'], 2)}, media {num(r['media_bdi'], 2)} y desviación estándar de "
                f"{num(df['bdi_total'].std(), 2)}): la mayoría de los adolescentes reporta síntomas "
                f"mínimos o nulos, un 50 % de los adolescentes presenta un puntaje menor o igual a "
                f"{num(r['mediana_bdi'], 0)}, y una minoría concentra puntajes altos, generando la cola "
                "larga de outliers que se ve en el boxplot.",
                className="estirar",
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
