"""Pestaña 09 · Correlaciones de Spearman entre pantalla, sueño y BDI-II."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, valor_p
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           tabla, var)
from data.load_data import VARIABLES_NUMERICAS, VARIABLES_PANTALLA, VARIABLES_SUENO, cargar_datos

UMBRAL_REDUNDANCIA = 0.7


def _columnas_rho(clave_var: str, titulo_var: str) -> list[dict]:
    return [
        {"clave": clave_var, "titulo": titulo_var, "tipo": "variable"},
        {"clave": "rho", "titulo": "ρ", "tipo": "numero", "formato": lambda v: num(v, 3)},
        {"clave": "p", "titulo": "Valor p", "tipo": "numero", "formato": valor_p},
        {"clave": "magnitud", "titulo": "Magnitud"},
    ]


def _con_magnitud(t):
    return t.assign(magnitud=t["rho"].map(stats.magnitud))


def layout():
    """Estructura: KPIs → heatmap + ρ con BDI-II → tablas por dominio → redundancias."""
    df = cargar_datos()
    matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
    pantalla_sueno = _con_magnitud(stats.spearman_pares(df, ["screen_time_index"], VARIABLES_SUENO))
    sueno_bdi = _con_magnitud(stats.spearman_pares(df, VARIABLES_SUENO, ["bdi_total"]))
    con_bdi = stats.spearman_pares(df, VARIABLES_PANTALLA + VARIABLES_SUENO, ["bdi_total"])
    redundantes = stats.pares_redundantes(matriz, UMBRAL_REDUNDANCIA)
    rho = matriz["bdi_total"]

    mas_fuerte = con_bdi.loc[con_bdi["rho"].abs().idxmax()]
    pantalla_max = con_bdi[con_bdi["x"].isin(VARIABLES_PANTALLA)]["rho"].abs().max()
    sueno_max = pantalla_sueno.loc[pantalla_sueno["rho"].abs().idxmax()]
    # Par más fuerte entre predictores sin contar la segunda variable de cada par redundante
    siguiente = stats.par_mas_fuerte(matriz, excluir=["bdi_total", *redundantes["y"]])

    if redundantes.empty:
        texto_redundantes = html.P(f"Ningún par supera |ρ| ≥ {num(UMBRAL_REDUNDANCIA, 1)}.")
    else:
        filas = [{"par": html.Div([var(f["x"]), html.Br(), var(f["y"])]), "rho": f["rho"]}
                 for _, f in redundantes.iterrows()]
        texto_redundantes = tabla(filas, [
            {"clave": "par", "titulo": "Par de variables"},
            {"clave": "rho", "titulo": "ρ", "tipo": "numero", "formato": lambda v: num(v, 3)},
        ])

    return html.Div([
        encabezado_seccion("correlaciones", [f"n = {entero(len(df))}", "Spearman"]),
        fila_kpis([
            kpi("ρ calidad del sueño · BDI-II", num(rho["sleep_quality_index"], 3),
                f"Efecto {stats.magnitud(rho['sleep_quality_index'])}"),
            kpi("ρ horas de sueño · BDI-II", num(rho["avg_sleep_hours"], 3),
                f"Efecto {stats.magnitud(rho['avg_sleep_hours'])}"),
            kpi("ρ tiempo de pantalla · BDI-II", num(rho["screen_time_index"], 3),
                f"Efecto {stats.magnitud(rho['screen_time_index'])}"),
            kpi("Pares redundantes", entero(len(redundantes)),
                f"|ρ| ≥ {num(UMBRAL_REDUNDANCIA, 1)} entre predictores"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Triángulo inferior sin diagonal. Durazno: positiva; azul: negativa.",
                       className="card-subtitulo"),
                grafico("co-heatmap", 470, ancho_minimo=640),
                titulo="Matriz de Spearman",
            ), lg=8),
            dbc.Col(card(
                html.P("Correlación de cada predictor con el puntaje BDI-II.",
                       className="card-subtitulo"),
                grafico("co-rho-bdi", 470),
                titulo="Asociación con bdi_total",
                className="estirar",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Objetivo específico 2: cada dominio por separado.", className="card-subtitulo"),
                tabla(pantalla_sueno.to_dict("records"), _columnas_rho("y", "Dominio del sueño")),
                titulo="screen_time_index frente a los dominios del sueño",
            ), lg=6),
            dbc.Col(card(
                html.P("Asociación de cada dominio con el puntaje BDI-II.", className="card-subtitulo"),
                tabla(sueno_bdi.to_dict("records"), _columnas_rho("x", "Dominio del sueño")),
                titulo="Dominios del sueño frente a bdi_total",
            ), lg=6),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P(f"Pares de variables con |ρ| ≥ {num(UMBRAL_REDUNDANCIA, 1)}.",
                       className="card-subtitulo"),
                texto_redundantes,
                titulo="Pares redundantes",
                className="estirar",
            ), lg=4),
            dbc.Col(interpretacion(
                f"La asociación más fuerte con `bdi_total` corresponde a `{mas_fuerte['x']}` "
                f"(ρ = {num(mas_fuerte['rho'], 3)}). Las variables de pantalla alcanzan como máximo "
                f"|ρ| = {num(pantalla_max, 3)}, por lo que los dominios del sueño concentran más "
                "asociación monótona con el puntaje depresivo que el tiempo de pantalla.",
                f"`screen_time_index` se asocia sobre todo con `{sueno_max['y']}` "
                f"(ρ = {num(sueno_max['rho'], 3)}); con los demás dominios la asociación es menor, lo que "
                "respalda analizar cada dominio por separado.",
                *[
                    f"`{f['x']}` y `{f['y']}` son prácticamente redundantes (ρ = {num(f['rho'], 3)}): "
                    "conviene conservar solo una de ellas en un modelo lineal."
                    for _, f in redundantes.iterrows()
                ],
                "Sin contar los pares redundantes, la correlación más alta entre predictores es "
                f"`{siguiente['x']}` – "
                f"`{siguiente['y']}` (ρ = {num(siguiente['rho'], 3)}).",
            ), lg=8),
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
