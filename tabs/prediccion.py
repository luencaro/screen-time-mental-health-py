"""Pestaña · Predicción: probabilidad de depresión (CatBoost) y BDI-II estimado (ElasticNet) para un
perfil. Usa los modelos portables de data/processed/dashboard/prediccion/ (sin scikit-learn)."""

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from analysis import figuras_modelado as figuras
from components.formato import num, pct
from components.ui import badge, card, encabezado_seccion, grafico, var
from data import load_modelado as datos
from data.load_data import VARIABLES_SUENO
from data.prediccion import RANGOS_CATEGORIAS_BDI, cargar_predictor, categoria_bdi

# Mismas 5 variables que los controles (rangos.json puede no existir al registrar los callbacks)
VARIABLES = VARIABLES_SUENO + ["screen_time_index"]


def _decimales(paso):
    return 0 if float(paso).is_integer() else 2


def _control(variable, info):
    """Slider de una variable: límites en los percentiles 1 y 99 del train y valor inicial en la mediana.
    En las variables con pocos valores distintos solo se pueden elegir los valores observados.
    El valor elegido se muestra junto a la etiqueta (sin tooltip, que se superponía con las marcas)."""
    d = _decimales(info["paso"])
    if info["valores"]:
        minimo, maximo = info["valores"][0], info["valores"][-1]
        marcas = {v: "" for v in info["valores"]}
        inicial = min(info["valores"], key=lambda v: abs(v - info["mediana"]))
        props = {"step": None}
    else:
        minimo, maximo, marcas, inicial = info["p1"], info["p99"], {}, info["mediana"]
        props = {"step": info["paso"]}
    marcas.update({minimo: num(minimo, d), maximo: num(maximo, d)})
    return html.Div([
        html.Div([
            html.Label([info["etiqueta"], " ", var(variable)], htmlFor=f"pr-{variable}", className="control-etiqueta"),
            html.Span(id=f"pr-valor-{variable}", className="valor-control"),
        ], className="etiqueta-control"),
        dcc.Slider(id=f"pr-{variable}", min=minimo, max=maximo, value=inicial, marks=marcas, **props),
        html.Div(f"{info['unidad'].capitalize()}. {info['descripcion']}", className="nota"),
    ], className="control-prediccion")


def layout():
    """Estructura: formulario + resultados (clasificación, regresión)."""
    rangos = datos.rangos_prediccion()
    predictor = cargar_predictor()
    clf, reg = datos.clasificacion_final(), datos.regresion_final()
    umbral = predictor.umbral

    formulario = card(
        html.Div([
            html.Span(rangos["sexo"]["etiqueta"], className="control-etiqueta"),
            dbc.RadioItems(id="pr-sexo", options=[{"label": o["etiqueta"], "value": o["valor"]}
                                                  for o in rangos["sexo"]["opciones"]],
                           value=rangos["sexo"]["inicial"], inline=True, class_name="segmentado",
                           label_checked_class_name="activo"),
        ], className="control-prediccion"),
        *[_control(v, rangos["variables"][v]) for v in rangos["orden"]],
        dbc.Button("Calcular", id="pr-calcular", n_clicks=0, class_name="boton-primario"),
        titulo="Perfil del adolescente",
    )
    resultados = [
        card(
            html.P(f"CatBoost. Positivo si la probabilidad es ≥ {pct(umbral)} (umbral de tamizaje).",
                   className="card-subtitulo"),
            html.Div([html.Div(id="pr-probabilidad", className="kpi-valor"), html.Div(id="pr-resultado")],
                     className="resultado-prediccion"),
            grafico("pr-fig-probabilidad", 130, animar=False),
            titulo="1 · Clasificación: probabilidad de depresión",
        ),
        card(
            html.P("ElasticNet. Puntaje recortado al rango del instrumento (0–63).", className="card-subtitulo"),
            html.Div([html.Div(id="pr-bdi", className="kpi-valor"), html.Div(id="pr-categoria")],
                     className="resultado-prediccion"),
            grafico("pr-fig-bdi", 130, animar=False),
            titulo="2 · Regresión: puntaje BDI-II estimado",
        ),
    ]

    return html.Div([
        encabezado_seccion("prediccion", [f"{clf['nombre']} · {reg['nombre']}", "Herramienta académica"]),
        dbc.Row([dbc.Col(formulario, lg=4), dbc.Col(resultados, lg=8)], className="fila"),
    ])


def register_callbacks(app):
    """Muestra el valor de cada slider y calcula las dos predicciones al pulsar Calcular (y con los
    valores iniciales al entrar)."""

    @app.callback(
        [Output(f"pr-valor-{v}", "children") for v in VARIABLES],
        [Input(f"pr-{v}", "value") for v in VARIABLES],
    )
    def _valores(*valores):
        rangos = datos.rangos_prediccion()["variables"]
        return [num(valor, _decimales(rangos[v]["paso"])) if valor is not None else "—"
                for v, valor in zip(VARIABLES, valores)]

    @app.callback(
        Output("pr-probabilidad", "children"), Output("pr-resultado", "children"),
        Output("pr-fig-probabilidad", "figure"),
        Output("pr-bdi", "children"), Output("pr-categoria", "children"), Output("pr-fig-bdi", "figure"),
        Input("pr-calcular", "n_clicks"), Input("tema", "data"),
        State("pr-sexo", "value"), *[State(f"pr-{v}", "value") for v in VARIABLES],
    )
    def _predecir(_clics, tema, sexo, *valores):
        predictor = cargar_predictor()
        entrada = {"sex": sexo, **dict(zip(VARIABLES, valores))}
        probabilidad = float(predictor.probabilidad(entrada)[0])
        bdi = float(predictor.bdi(entrada)[0])
        positivo = probabilidad >= predictor.umbral
        categoria = categoria_bdi(bdi)
        return (
            pct(probabilidad),
            badge("Positivo en tamizaje" if positivo else "Negativo en tamizaje",
                  "primaria" if positivo else "neutra"),
            figuras.indicador_probabilidad(probabilidad, predictor.umbral, tema),
            num(bdi, 1),
            badge(f"{categoria} · {RANGOS_CATEGORIAS_BDI[categoria]}", "neutra"),
            figuras.escala_bdi(bdi, tema),
        )
