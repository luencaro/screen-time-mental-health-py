"""Pestaña 08 · Variables numéricas: histogramas con selector de variable y grupo."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           tabla)
from data.load_data import (ETIQUETAS, ETIQUETAS_ESTADO, ETIQUETAS_SEXO, VARIABLES_NUMERICAS,
                            cargar_datos)

OPCIONES_SEPARAR = [
    {"label": "Sin separar", "value": "todos"},
    {"label": "Por sexo", "value": "sex"},
    {"label": "Por estado depresivo", "value": "depressed"},
]
VARIABLE_INICIAL = "sleep_quality_index"


def _dos(v):
    return num(v, 2)


def _separar(valor):
    return None if valor == "todos" else valor


def _por_grupo(df, variable, separar):
    """Descriptivos por grupo con etiquetas legibles."""
    t = stats.descriptivos_por_grupo(df, variable, separar)
    etiquetas = ETIQUETAS_SEXO if separar == "sex" else ETIQUETAS_ESTADO
    t["grupo"] = t["grupo"].map(etiquetas)
    return t


def _texto(df, variable, separar):
    """Interpretación calculada para la selección actual."""
    d = stats.descriptivos(df, [variable]).iloc[0]
    forma = ("aproximadamente simétrica" if abs(d["asimetria"]) < 0.5
             else f"con cola a la {'derecha' if d['asimetria'] > 0 else 'izquierda'}")
    frases = [
        f"`{variable}` tiene media {num(d['media'], 2)} y mediana {num(d['mediana'], 2)}; "
        f"su distribución es {forma} (asimetría {num(d['asimetria'], 2)}). "
        f"La mitad central de los valores está entre {num(d['p25'], 2)} y {num(d['p75'], 2)}."
    ]
    if separar:
        g = _por_grupo(df, variable, separar)
        mw_grupo = "sex" if separar == "sex" else "depressed"
        niveles = ("Boy", "Girl") if separar == "sex" else (0, 1)
        mw = stats.mann_whitney(df, [variable], grupo=mw_grupo, niveles=niveles).iloc[0]
        a, b = g.iloc[0], g.iloc[1]
        frases.append(
            f"Mediana en {a['grupo'].lower()}: {num(a['mediana'], 2)}; en {b['grupo'].lower()}: "
            f"{num(b['mediana'], 2)}. Mann-Whitney: r biserial = {num(mw['r_biserial'], 3)} "
            f"(efecto {stats.magnitud(mw['r_biserial'])}). Cada histograma se expresa en % de su "
            "propio grupo para compararlos pese a tamaños distintos."
        )
    return frases


def layout():
    """Estructura: controles → KPIs dinámicos → histograma + boxplot → interpretación."""
    df = cargar_datos()
    opciones = [{"label": f"{ETIQUETAS[v]} · {v}", "value": v} for v in VARIABLES_NUMERICAS]
    controles = card(html.Div([
        html.Div([
            html.Label("Variable", htmlFor="num-variable", className="control-etiqueta"),
            dbc.Select(id="num-variable", options=opciones, value=VARIABLE_INICIAL),
        ], style={"flex": "1 1 320px"}),
        html.Div([
            html.Span("Separar", className="control-etiqueta"),
            dbc.RadioItems(id="num-separar", options=OPCIONES_SEPARAR, value="todos", inline=True,
                           class_name="segmentado", label_checked_class_name="activo"),
        ]),
    ], className="controles"))

    return html.Div([
        encabezado_seccion("analisis_numerico",
                           [f"n = {entero(len(df))}", f"{len(VARIABLES_NUMERICAS)} variables"]),
        html.Div(controles, className="fila"),
        fila_kpis([
            kpi("Media", "", "", id_valor="num-kpi-media", id_contexto="num-ctx-media"),
            kpi("Mediana", "", "", id_valor="num-kpi-mediana", id_contexto="num-ctx-mediana"),
            kpi("Desviación estándar", "", "", id_valor="num-kpi-desv", id_contexto="num-ctx-desv"),
            kpi("Asimetría", "", "", id_valor="num-kpi-asim", id_contexto="num-ctx-asim"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P(id="num-subtitulo", className="card-subtitulo"),
                grafico("num-histograma", 400),
                titulo="Histograma",
            ), lg=8),
            dbc.Col([
                card(grafico("num-boxplot", 200), titulo="Dispersión"),
                html.Div(id="num-interpretacion", className="estirar"),
            ], lg=4),
        ], className="fila"),
        dbc.Row(dbc.Col(card(html.Div(id="num-tabla"), titulo="Descriptivos por grupo"), lg=8),
                className="fila", id="num-fila-tabla"),
    ])


def register_callbacks(app):
    """Actualiza gráficos, KPIs y textos según variable, agrupación y tema."""

    @app.callback(
        Output("num-histograma", "figure"),
        Output("num-boxplot", "figure"),
        Output("num-subtitulo", "children"),
        Output("num-kpi-media", "children"), Output("num-ctx-media", "children"),
        Output("num-kpi-mediana", "children"), Output("num-ctx-mediana", "children"),
        Output("num-kpi-desv", "children"), Output("num-ctx-desv", "children"),
        Output("num-kpi-asim", "children"), Output("num-ctx-asim", "children"),
        Output("num-interpretacion", "children"),
        Output("num-tabla", "children"),
        Output("num-fila-tabla", "style"),
        Input("num-variable", "value"),
        Input("num-separar", "value"),
        Input("tema", "data"),
    )
    def _actualizar(variable, separar_valor, tema):
        df = cargar_datos()
        variable = variable if variable in VARIABLES_NUMERICAS else VARIABLE_INICIAL
        separar = _separar(separar_valor)
        d = stats.descriptivos(df, [variable]).iloc[0]

        contextos = [ETIQUETAS[variable]] * 4
        tabla_grupo, estilo_tabla = None, {"display": "none"}
        if separar:
            g = _por_grupo(df, variable, separar)
            def _ctx(col):
                return " · ".join(f"{f['grupo']} {num(f[col], 2)}" for _, f in g.iterrows())
            contextos = [_ctx("media"), _ctx("mediana"), _ctx("desv"), _ctx("asimetria")]
            tabla_grupo = tabla(g.to_dict("records"), [
                {"clave": "grupo", "titulo": "Grupo"},
                {"clave": "n", "titulo": "n", "tipo": "numero", "formato": entero},
                {"clave": "media", "titulo": "Media", "tipo": "numero", "formato": _dos},
                {"clave": "mediana", "titulo": "Mediana", "tipo": "numero", "formato": _dos},
                {"clave": "desv", "titulo": "Desv. estándar", "tipo": "numero", "formato": _dos},
                {"clave": "asimetria", "titulo": "Asimetría", "tipo": "numero", "formato": _dos},
            ])
            estilo_tabla = {}

        subtitulo = ETIQUETAS[variable] + (
            "" if not separar else f" · {'por sexo' if separar == 'sex' else 'por estado depresivo'}")
        return (
            figures.histograma_variable(df, variable, separar, tema),
            figures.boxplot_variable(df, variable, separar, tema),
            subtitulo,
            num(d["media"], 2), contextos[0],
            num(d["mediana"], 2), contextos[1],
            num(d["desv"], 2), contextos[2],
            num(d["asimetria"], 2), contextos[3],
            interpretacion(*_texto(df, variable, separar)),
            tabla_grupo, estilo_tabla,
        )
