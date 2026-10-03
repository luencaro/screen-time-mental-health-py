"""Pestaña 10 · Outliers: detección por IQR y relevancia clínica."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, pct
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           tabla, var)
from data.load_data import VARIABLES_NUMERICAS, VARIABLES_PANTALLA, VARIABLES_SUENO, cargar_datos

PREDICTORES = VARIABLES_PANTALLA + VARIABLES_SUENO


def _dos(v):
    return num(v, 2)


def layout():
    """Estructura: KPIs → tabla IQR + justificación → outliers por estado + interpretación."""
    df = cargar_datos()
    iqr = stats.outliers_iqr(df, VARIABLES_NUMERICAS)
    por_estado = stats.outliers_por_estado(df, PREDICTORES)
    filas = stats.filas_con_outliers(df, PREDICTORES)
    prevalencia = df["depressed"].mean()
    bdi = iqr[iqr["variable"] == "bdi_total"].iloc[0]
    sin_outliers = iqr.loc[iqr["n"] == 0, "variable"].tolist()

    # Variables donde los outliers se concentran en adolescentes sobre el corte
    ancho = por_estado.pivot(index="variable", columns="depressed", values="proporcion")
    prev_out = por_estado.drop_duplicates("variable").set_index("variable")["prevalencia_en_outliers"]
    con_outliers = [v for v in PREDICTORES if v not in sin_outliers]
    frases_estado = [
        f"`{v}`: {pct(ancho.loc[v, 1])} de outliers entre quienes están sobre el corte frente a "
        f"{pct(ancho.loc[v, 0])} bajo el corte; entre sus outliers, {pct(prev_out[v])} supera el corte."
        for v in sorted(con_outliers, key=lambda v: -prev_out[v])
    ]

    return html.Div([
        encabezado_seccion("outliers", ["Regla IQR · 1,5", f"{len(VARIABLES_NUMERICAS)} variables"]),
        fila_kpis([
            kpi("Filas con algún outlier", entero(filas["n"]),
                f"{pct(filas['proporcion'])} de la muestra (predictores)"),
            kpi("Sobre el corte en esas filas", pct(filas["prevalencia"]),
                f"Frente a {pct(filas['prevalencia_resto'])} en el resto", "deprimido"),
            kpi("Outliers en bdi_total", entero(bdi["n"]),
                f"{pct(bdi['proporcion'])} · puntaje > {_dos(bdi['lim_sup'])}"),
            kpi("Variables sin outliers", entero(len(sin_outliers)),
                ", ".join(sin_outliers) if sin_outliers else "Todas presentan alguno"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Límites de Tukey: Q1 − 1,5·IQR y Q3 + 1,5·IQR.", className="card-subtitulo"),
                tabla(iqr.to_dict("records"), [
                    {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
                    {"clave": "q1", "titulo": "Q1", "tipo": "numero", "formato": _dos},
                    {"clave": "q3", "titulo": "Q3", "tipo": "numero", "formato": _dos},
                    {"clave": "iqr", "titulo": "IQR", "tipo": "numero", "formato": _dos},
                    {"clave": "lim_inf", "titulo": "Lím. inf.", "tipo": "numero", "formato": _dos},
                    {"clave": "lim_sup", "titulo": "Lím. sup.", "tipo": "numero", "formato": _dos},
                    {"clave": "n_bajo", "titulo": "Bajo", "tipo": "numero", "formato": entero},
                    {"clave": "n_alto", "titulo": "Alto", "tipo": "numero", "formato": entero},
                    {"clave": "proporcion", "titulo": "% outliers", "tipo": "numero", "formato": pct},
                ]),
                titulo="Outliers por variable (IQR)",
            ), lg=8),
            dbc.Col(interpretacion(
                f"Los valores atípicos no se eliminan. Son valores plausibles: `bdi_total` llega a "
                f"{entero(df['bdi_total'].max())} (escala 0–63) y `avg_sleep_hours` va de "
                f"{_dos(df['avg_sleep_hours'].min())} a {_dos(df['avg_sleep_hours'].max())} h, sin "
                "indicios de errores de registro.",
                f"Las filas con algún outlier en pantalla o sueño tienen una proporción sobre el corte "
                f"clínico de {pct(filas['prevalencia'])}, frente a {pct(filas['prevalencia_resto'])} en "
                f"el resto (prevalencia global {pct(prevalencia)}). Eliminarlas reduciría justo los "
                "casos de mayor interés clínico y acentuaría el desbalance de clases.",
                "En su lugar se usan métodos robustos (medianas, Spearman, Mann-Whitney) y, en la fase "
                "de modelado, escalado robusto o modelos basados en árboles.",
                titulo="Por qué se conservan",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P(["Predictores de pantalla y sueño. Se excluye ", var("bdi_total"),
                        " porque define el estado."], className="card-subtitulo"),
                grafico("ou-estado", 380, ancho_minimo=520),
                titulo="Proporción de outliers según estado depresivo",
            ), lg=8),
            dbc.Col(interpretacion(*frases_estado), lg=4),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera la figura al cambiar el tema."""

    @app.callback(Output("ou-estado", "figure"), Input("tema", "data"))
    def _figura(tema):
        df = cargar_datos()
        return figures.barras_outliers_estado(stats.outliers_por_estado(df, PREDICTORES), tema)
