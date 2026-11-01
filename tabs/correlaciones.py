"""Pestaña 09 · Correlaciones de Spearman entre pantalla, sueño y BDI-II."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, valor_p
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi, tabla
from data.load_data import VARIABLES_NUMERICAS, VARIABLES_PANTALLA, VARIABLES_SUENO, cargar_datos

def _columnas_rho(clave_var: str, titulo_var: str) -> list[dict]:
    return [
        {"clave": clave_var, "titulo": titulo_var, "tipo": "variable"},
        {"clave": "n", "titulo": "N", "tipo": "numero", "formato": entero},
        {"clave": "rho", "titulo": "Spearman r", "tipo": "numero", "formato": lambda v: num(v, 3)},
        {"clave": "p", "titulo": "Valor p", "tipo": "numero", "formato": valor_p},
        {"clave": "sig", "titulo": "Significativo (p < 0,05)"},
    ]


def _ordenar(t):
    """Orden del notebook: por |r| descendente, con columna de significancia."""
    t = t.assign(abs_rho=t["rho"].abs(), sig=["Sí" if p < 0.05 else "No" for p in t["p"]])
    return t.sort_values("abs_rho", ascending=False)


def layout():
    """Estructura: KPIs → heatmap + r con BDI-II → tablas por dominio."""
    df = cargar_datos()
    matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
    pantalla_sueno = _ordenar(stats.spearman_pares(df, ["screen_time_index"], VARIABLES_SUENO))
    sueno_bdi = _ordenar(stats.spearman_pares(df, VARIABLES_SUENO, ["bdi_total"]))
    rho = matriz["bdi_total"]

    return html.Div([
        encabezado_seccion("correlaciones", [f"n = {entero(len(df))}", "Spearman"]),
        fila_kpis([
            kpi("r calidad del sueño · BDI-II", num(rho["sleep_quality_index"], 3),
                f"Efecto {stats.magnitud(rho['sleep_quality_index'])}"),
            kpi("r horas de sueño · BDI-II", num(rho["avg_sleep_hours"], 3),
                f"Efecto {stats.magnitud(rho['avg_sleep_hours'])}"),
            kpi("r cronotipo · BDI-II", num(rho["midsleep_weekend_hours"], 3),
                f"Efecto {stats.magnitud(rho['midsleep_weekend_hours'])}"),
            kpi("r tiempo de pantalla · BDI-II", num(rho["screen_time_index"], 3),
                f"Efecto {stats.magnitud(rho['screen_time_index'])}"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Triángulo inferior sin diagonal. Durazno: positiva; azul: negativa.",
                       className="card-subtitulo"),
                grafico("co-heatmap", 470, ancho_minimo=640, animar=False),
                titulo="Matriz de correlación de Spearman",
            ), lg=8),
            dbc.Col(card(
                html.P("Correlación de cada predictor con el puntaje BDI-II.",
                       className="card-subtitulo"),
                grafico("co-rho-bdi", 380, flexible=True),
                titulo="Asociación con bdi_total",
                className="estirar grafico-flexible",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col([
                card(
                    html.P("Objetivo específico 2: cada dominio por separado.", className="card-subtitulo"),
                    tabla(pantalla_sueno.to_dict("records"), _columnas_rho("y", "Dominio de sueño")),
                    titulo="screen_time_index frente a los dominios del sueño",
                    className="estirar",
                ),
            ], lg=6),
            dbc.Col([
                card(
                    html.P("Asociación de cada dominio con el puntaje BDI-II.", className="card-subtitulo"),
                    tabla(sueno_bdi.to_dict("records"), _columnas_rho("x", "Dominio de sueño")),
                    titulo="Dominios del sueño frente a bdi_total",
                    className="estirar",
                ),
            ], lg=6),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(
        Output("co-heatmap", "figure"),
        Output("co-rho-bdi", "figure"),
        Input("tema", "data"),
    )
    def _figuras(tema):
        df = cargar_datos()
        matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
        con_bdi = stats.spearman_pares(df, VARIABLES_PANTALLA + VARIABLES_SUENO, ["bdi_total"])
        return figures.heatmap_spearman(matriz, tema), figures.barras_rho(con_bdi, tema)
