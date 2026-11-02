"""Pestaña · Benchmarking: ganadores de cada algoritmo frente al benchmark (CV y test)."""

import dash_bootstrap_components as dbc
from dash import Input, Output, ctx, html

from analysis import figuras_modelado as figuras
from components.formato import entero, num
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi, tabla
from data import load_modelado as datos

OPCIONES_TAREA = [
    {"label": "Clasificación · depressed", "value": "clasificacion"},
    {"label": "Regresión · bdi_total", "value": "regresion"},
]
METRICAS_CLF = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Accuracy"]
METRICAS_REG = ["RMSE", "MAE", "R²"]
NOMBRES_ESTRATEGIA = {"nativo": "Nativo", "pesos": "Pesos de clase", "smotenc": "SMOTENC"}
OCULTO = {"display": "none"}


def _ancha(cv):
    """CV en formato largo → una fila por modelo con '<métrica>' y '<métrica> DE'."""
    medias = cv.pivot(index="modelo", columns="metrica", values="media")
    desv = cv.pivot(index="modelo", columns="metrica", values="desviacion").add_suffix(" DE")
    info = cv.drop_duplicates("modelo").set_index("modelo").drop(columns=["metrica", "media", "desviacion"])
    return info.join(medias).join(desv).reset_index()


def _ancha_clf():
    return _ancha(datos.clasificacion_benchmark_cv()).sort_values("PR-AUC", ascending=False)


def _ancha_reg():
    return _ancha(datos.regresion_benchmark_cv()).sort_values("RMSE")


def _media_de(media, de, decimales=3):
    return f"{num(media, decimales)} ± {num(de, decimales)}"


def _nombre_tabla(fila):
    if fila["es_referencia"]:
        return f"{fila['nombre']} (referencia)"
    return f"{fila['nombre']} · final" if fila["es_final"] else fila["nombre"]


def _final(ancha):
    """Clave del modelo final (selección inicial del filtro)."""
    return list(ancha.loc[ancha["es_final"], "modelo"])


def _selector_modelos(id_, ancha):
    """Checklist que empieza solo con el modelo final; se agregan los demás modelos que se quieran ver."""
    return dbc.Checklist(
        id=id_, options=[{"label": _nombre_tabla(f), "value": f["modelo"]} for _, f in ancha.iterrows()],
        value=_final(ancha), inline=True, class_name="segmentado selector-modelos",
        label_checked_class_name="activo",
    )


def _tabla_cv(ancha, metricas, columna_extra, titulo_extra, nombres_extra):
    filas = []
    for _, f in ancha.iterrows():
        fila = {"modelo": _nombre_tabla(f), "extra": nombres_extra.get(f[columna_extra], f[columna_extra]),
                "_clase": "resaltada" if f["es_final"] else None}
        for m in metricas:
            fila[m] = _media_de(f[m], f[f"{m} DE"])
        filas.append(fila)
    columnas = [{"clave": "modelo", "titulo": "Modelo"}, {"clave": "extra", "titulo": titulo_extra}]
    columnas += [{"clave": m, "titulo": m, "tipo": "numero"} for m in metricas]
    return tabla(filas, columnas)


def _sin_seleccion():
    return card(html.P("Selecciona al menos un modelo para ver sus resultados.", className="nota"))


def _clasificacion(tema, seleccion):
    test, curvas, final = datos.clasificacion_benchmark_test(), datos.clasificacion_curvas(), datos.clasificacion_final()
    ancha = _ancha_clf()
    fila_final = ancha[ancha["es_final"]].iloc[0]
    competidores = ancha[~ancha["es_referencia"]]
    kpis = fila_kpis([
        kpi("Modelo final", fila_final["nombre"], NOMBRES_ESTRATEGIA.get(fila_final["estrategia"], "")),
        kpi("PR-AUC de CV del final", _media_de(fila_final["PR-AUC"], fila_final["PR-AUC DE"]),
            f"Media ± DE en {final['n_folds']} folds"),
        kpi("Rango de PR-AUC entre ganadores", num(competidores["PR-AUC"].max() - competidores["PR-AUC"].min(), 3),
            f"DE entre folds: {num(competidores['PR-AUC DE'].min(), 3)}–{num(competidores['PR-AUC DE'].max(), 3)}"),
        kpi("Prevalencia (azar)", num(final["prevalencia_train"], 3), "PR-AUC de un clasificador aleatorio"),
    ])
    elegidos = ancha[ancha["modelo"].isin(seleccion)]
    if elegidos.empty:
        return html.Div([kpis, _sin_seleccion()])

    auc = {tipo: test[test["metrica"] == m].set_index("modelo")["valor"].to_dict()
           for tipo, m in (("roc", "ROC-AUC"), ("pr", "PR-AUC"))}
    modelos = elegidos[["modelo", "nombre", "es_final", "es_referencia"]]
    puntos = elegidos.rename(columns={"PR-AUC": "media", "PR-AUC DE": "desviacion"})
    alto_puntos = max(220, 44 * len(elegidos) + 100)

    return html.Div([
        kpis,
        dbc.Row(dbc.Col(card(
            html.P("Media ± desviación estándar en los folds del train. Precision, recall, F1 y accuracy con "
                   f"umbral {num(final['umbral_defecto'], 1)}. La fila de referencia es la CV del EDA (todo el "
                   "dataset), no estrictamente comparable.", className="card-subtitulo"),
            _tabla_cv(elegidos, METRICAS_CLF, "estrategia", "Balanceo", NOMBRES_ESTRATEGIA),
            titulo="Validación cruzada de los ganadores de cada algoritmo",
        ), lg=12), className="fila"),
        dbc.Row(dbc.Col(card(
            html.P(f"Rombo: modelo final. Como referencia, la PR-AUC de un clasificador aleatorio es la "
                   f"prevalencia del train ({num(final['prevalencia_train'], 3)}).", className="card-subtitulo"),
            grafico("bm-puntos", alto_puntos, figura=figuras.puntos_cv(puntos, "PR-AUC", tema), animar=False),
            titulo="PR-AUC de CV por modelo",
        ), lg=12), className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Test. Modelo final en pizarra y grueso; referencia en gris discontinuo.",
                       className="card-subtitulo"),
                grafico("bm-roc", 520, figura=figuras.curvas_test(curvas, modelos, auc["roc"], "roc", tema),
                        animar=False),
                titulo="Curvas ROC",
            ), lg=6),
            dbc.Col(card(
                html.P("Test. AP = precisión promedio (PR-AUC).", className="card-subtitulo"),
                grafico("bm-pr", 520, figura=figuras.curvas_test(curvas, modelos, auc["pr"], "pr", tema,
                                                                 final["prevalencia_test"]), animar=False),
                titulo="Curvas Precision-Recall",
            ), lg=6),
        ], className="fila"),
    ])


def _regresion(tema, seleccion):
    final = datos.regresion_final()
    ancha = _ancha_reg()
    fila_final = ancha[ancha["es_final"]].iloc[0]
    mejor = ancha[~ancha["es_referencia"]].iloc[0]
    referencia = ancha[ancha["es_referencia"]].iloc[0]
    variantes = dict(zip(ancha["variante"], ancha["variante_nombre"]))
    kpis = fila_kpis([
        kpi("Modelo final", fila_final["nombre"], f"Target {fila_final['variante_nombre'].lower()}"),
        kpi("RMSE de CV del final", _media_de(fila_final["RMSE"], fila_final["RMSE DE"]),
            f"Puntos BDI · media ± DE en {final['n_folds']} folds"),
        kpi("Menor RMSE de CV", num(mejor["RMSE"], 3),
            f"{mejor['nombre']} · {num(fila_final['RMSE'] - mejor['RMSE'], 3)} puntos menos que el final"),
        kpi("R² de CV del final", _media_de(fila_final["R²"], fila_final["R² DE"]),
            f"Benchmark: {num(referencia['R²'], 3)}"),
    ])
    elegidos = ancha[ancha["modelo"].isin(seleccion)]
    if elegidos.empty:
        return html.Div([kpis, _sin_seleccion()])

    puntos = elegidos.rename(columns={"RMSE": "media", "RMSE DE": "desviacion"})
    return html.Div([
        kpis,
        dbc.Row(dbc.Col(card(
            html.P("Media ± desviación estándar en los folds del train, en puntos BDI (R² sin unidades).",
                   className="card-subtitulo"),
            _tabla_cv(elegidos, METRICAS_REG, "variante", "Variante del target", variantes),
            titulo="Validación cruzada de los ganadores de cada algoritmo",
        ), lg=12), className="fila"),
        dbc.Row(dbc.Col(card(
            html.P(f"Rombo: modelo final. RMSE del benchmark: {num(referencia['RMSE'], 3)} puntos.",
                   className="card-subtitulo"),
            grafico("bm-puntos-reg", max(220, 40 * len(elegidos) + 100),
                    figura=figuras.puntos_cv(puntos, "RMSE", tema, unidad=" puntos"), animar=False),
            titulo="RMSE de CV por modelo",
        ), lg=12), className="fila"),
    ])


def layout():
    """Estructura: selector de tarea y de modelos (fijo bajo el menú) → KPIs → tabla de CV → puntos de CV → curvas (clasificación)."""
    final = datos.clasificacion_final()
    ancha_clf, ancha_reg = _ancha_clf(), _ancha_reg()
    controles = card(html.Div([
        html.Div([
            html.Span("Tarea", className="control-etiqueta"),
            dbc.RadioItems(id="bm-tarea", options=OPCIONES_TAREA, value="clasificacion", inline=True,
                           class_name="segmentado", label_checked_class_name="activo"),
        ]),
        html.Div([
            html.Div([
                html.Span("Modelos", className="control-etiqueta"),
                html.Button("Solo el modelo final", id="bm-final", n_clicks=0, className="enlace-boton"),
                html.Button("Mostrar todos", id="bm-todos", n_clicks=0, className="enlace-boton"),
            ], className="etiqueta-con-accion"),
            html.Div(_selector_modelos("bm-modelos-clf", ancha_clf), id="bm-caja-clf"),
            html.Div(_selector_modelos("bm-modelos-reg", ancha_reg), id="bm-caja-reg", style=OCULTO),
        ], style={"flex": "1 1 100%"}),
    ], className="controles"))
    return html.Div([
        encabezado_seccion("benchmarking", [f"{len(ancha_clf) - 1} algoritmos + benchmark",
                                            f"CV de {final['n_folds']} folds",
                                            f"Test · n = {entero(final['n_test'])}"]),
        # Filtro fijo bajo el menú superior: sigue a mano al bajar por las tablas y gráficos
        html.Div(controles, className="fila filtro-fijo"),
        html.Div(id="bm-contenido"),
    ])


def register_callbacks(app):
    """Filtro de modelos por tarea y contenido de la tarea elegida con las figuras del tema activo."""

    @app.callback(Output("bm-caja-clf", "style"), Output("bm-caja-reg", "style"), Input("bm-tarea", "value"))
    def _cajas(tarea):
        return (OCULTO, {}) if tarea == "regresion" else ({}, OCULTO)

    @app.callback(Output("bm-modelos-clf", "value"), Output("bm-modelos-reg", "value"),
                  Input("bm-todos", "n_clicks"), Input("bm-final", "n_clicks"), prevent_initial_call=True)
    def _accesos(_clics_todos, _clics_final):
        if ctx.triggered_id == "bm-final":
            return _final(_ancha_clf()), _final(_ancha_reg())
        return list(_ancha_clf()["modelo"]), list(_ancha_reg()["modelo"])

    @app.callback(
        Output("bm-contenido", "children"),
        Input("bm-tarea", "value"), Input("bm-modelos-clf", "value"), Input("bm-modelos-reg", "value"),
        Input("tema", "data"),
    )
    def _contenido(tarea, modelos_clf, modelos_reg, tema):
        if tarea == "regresion":
            return _regresion(tema, modelos_reg or [])
        return _clasificacion(tema, modelos_clf or [])
