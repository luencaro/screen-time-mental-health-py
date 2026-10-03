"""Pestaña 08 · Variables numéricas: histogramas con selector de variable y grupo."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, estadistico, num, texto_p
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           tabla)
from data.load_data import (CORTE_BDI, ETIQUETAS, ETIQUETAS_ESTADO, ETIQUETAS_SEXO,
                            VARIABLES_NUMERICAS, cargar_datos)

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


# Qué significa un valor más alto o más bajo de cada variable
GLOSA = {
    "screen_time_index": ("más tiempo de pantalla", "menos tiempo de pantalla"),
    "est_leisure_screen_hours": ("más horas de pantalla de ocio", "menos horas de pantalla de ocio"),
    "sleep_quality_index": ("peor calidad de sueño", "mejor calidad de sueño"),
    "avg_sleep_hours": ("más horas de sueño", "menos horas de sueño"),
    "midsleep_weekend_hours": ("horarios de sueño más tardíos", "horarios de sueño más tempranos"),
    "social_jetlag_hours": ("mayor jet lag social", "menor jet lag social"),
    "bdi_total": ("más síntomas depresivos", "menos síntomas depresivos"),
}
PREDICTORES = [v for v in VARIABLES_NUMERICAS if v != "bdi_total"]


def _linea_mw(mw, grupo_a: str, grupo_b: str, frase_a: str) -> str:
    """Resultado de Mann-Whitney en el formato del notebook."""
    return (f"Mann-Whitney U: Mdn = {num(mw['mediana_a'], 2)} ({grupo_a}) vs "
            f"{num(mw['mediana_b'], 2)} ({grupo_b}), U = {estadistico(mw['u'])}, "
            f"{texto_p(mw['p'])}, r = {num(mw['r_biserial'], 3)}"
            f"{'' if mw['p'] < 0.05 else ' (sin diferencia significativa)'}. "
            f"r negativo indica valores mayores en {frase_a}.")


def _texto(df, variable, separar):
    """Interpretación para la selección actual, siguiendo el texto del notebook."""
    d = stats.descriptivos(df, [variable]).iloc[0]
    forma = ("es aproximadamente simétrica" if abs(d["asimetria"]) < 0.5
             else f"presenta cola a la {'derecha' if d['asimetria'] > 0 else 'izquierda'}")
    mayor, menor = GLOSA[variable]

    if separar is None:
        frases = [
            f"`{variable}` tiene media {num(d['media'], 2)} y mediana {num(d['mediana'], 2)}; su "
            f"distribución {forma} (skew {num(d['asimetria'], 2)})."
        ]
        if variable != "bdi_total":
            rho = stats.spearman_pares(df, [variable], ["bdi_total"]).iloc[0]
            signo = "positiva" if rho["rho"] > 0 else "negativa"
            efecto = "más" if rho["rho"] > 0 else "menos"
            frases.append(
                f"Frente a `bdi_total`, la relación es {signo} (r = {num(rho['rho'], 3)}, "
                f"{texto_p(rho['p'])}, efecto {stats.magnitud(rho['rho'])}): a {mayor}, {efecto} "
                "síntomas depresivos."
            )
        return frases

    if separar == "sex":
        mw = stats.mann_whitney(df, [variable], grupo="sex", niveles=("Girl", "Boy")).iloc[0]
        if variable == "bdi_total" or abs(mw["r_biserial"]) >= 0.2:
            hacia = "más altos" if mw["r_biserial"] < 0 else "más bajos"
            glosa = mayor if mw["r_biserial"] < 0 else menor
            primera = (f"La distribución de las chicas está desplazada hacia valores {hacia} de "
                       f"`{variable}`, es decir, {glosa}.")
        else:
            primera = ("Las curvas de chicos y chicas se solapan casi por completo. Este hallazgo, en "
                       "conjunto con la brecha de género en depresión, sugiere que hay factores externos "
                       "al dataset (hormonales, sociales u otros no capturados aquí) detrás de esa brecha.")
        return [primera, _linea_mw(mw, "chicas", "chicos", "las chicas")]

    # Por estado depresivo
    if variable == "bdi_total":
        return [f"`bdi_total` define el estado depresivo (corte ≥ {CORTE_BDI}), por lo que ambos grupos "
                "quedan separados por construcción."]
    mw = stats.mann_whitney(df, [variable], grupo="depressed", niveles=(1, 0)).iloc[0]
    r = mw["r_biserial"]
    hacia, glosa = ("más altos", mayor) if r < 0 else ("más bajos", menor)
    if abs(r) >= 0.3:
        primera = (f"Acá sí hay separación visible: el grupo deprimido está claramente desplazado hacia "
                   f"valores {hacia}, es decir, {glosa}.")
    elif abs(r) >= 0.1:
        primera = (f"Hay una separación moderada: el grupo deprimido está desplazado hacia valores "
                   f"{hacia}, es decir, {glosa}.")
    else:
        primera = (f"La separación es pequeña: el grupo deprimido apenas se desplaza hacia valores "
                   f"{hacia} ({glosa}).")
    frases = [primera, _linea_mw(mw, "deprimidos", "no deprimidos", "el grupo con depresión")]
    ranking = (stats.mann_whitney(df, PREDICTORES, grupo="depressed", niveles=(1, 0))
               .assign(abs_r=lambda t: t["r_biserial"].abs()).sort_values("abs_r", ascending=False))
    orden = list(ranking["variable"])
    if variable == orden[0]:
        frases.append(f"Esto sugiere que `{orden[0]}` es el predictor con más señal, seguido de "
                      f"`{orden[1]}`.")
    elif variable == orden[1]:
        frases.append(f"Es el segundo predictor con más señal, después de `{orden[0]}`.")
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
                card(grafico("num-boxplot", 200, animar=False), titulo="Dispersión"),
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
