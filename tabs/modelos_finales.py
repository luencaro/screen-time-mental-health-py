"""Pestaña · Modelos finales: CatBoost (clasificación) y ElasticNet (regresión), umbral, importancia
de variables y experimento "BDI predicho ≥ 14" frente al clasificador."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figuras_modelado as figuras
from components.formato import entero, num, pct
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi, tabla
from data import load_modelado as datos


def _importancia(modelo, metodo):
    t = datos.importancias()
    return t[(t["modelo"] == modelo) & (t["metodo"] == metodo)]


def _ic(test, modelo, metrica):
    fila = test[(test["modelo"] == modelo) & (test["metrica"] == metrica)].iloc[0]
    return f"IC 95 %: {num(fila['ic_inf'], 3)}–{num(fila['ic_sup'], 3)}"


def _tabla_experimento(exp):
    filas = exp.to_dict("records")
    columnas = [{"clave": "nombre", "titulo": "Enfoque (test)"},
                {"clave": "pr_auc", "titulo": "PR-AUC", "tipo": "numero", "formato": lambda v: num(v, 3)},
                {"clave": "roc_auc", "titulo": "ROC-AUC", "tipo": "numero", "formato": lambda v: num(v, 3)},
                {"clave": "recall", "titulo": "Recall", "tipo": "numero", "formato": lambda v: num(v, 3)},
                {"clave": "precision", "titulo": "Precision", "tipo": "numero", "formato": lambda v: num(v, 3)},
                {"clave": "f1", "titulo": "F1", "tipo": "numero", "formato": lambda v: num(v, 3)},
                {"clave": "alertas_pct", "titulo": "Alertas", "tipo": "numero", "formato": lambda v: pct(v / 100)}]
    return tabla(filas, columnas)


def layout():
    """Estructura: clasificación (KPIs, matriz, umbral, importancia) → regresión (KPIs, predicho vs.
    real, coeficientes, permutación) → experimento."""
    clf, reg = datos.clasificacion_final(), datos.regresion_final()
    test_clf = datos.clasificacion_benchmark_test()
    exp = datos.regresion_vs_clasificacion()
    m, m05 = clf["metricas_test_umbral"], clf["metricas_test_05"]
    mr, ic = reg["metricas_test"], reg["intervalos_test"]
    umbral = num(clf["umbral"], 3)
    en_cero = [k for k, v in reg["coeficientes"].items() if v == 0]

    return html.Div([
        encabezado_seccion("modelos_finales", [f"{clf['nombre']} · {reg['nombre']}", f"Umbral {umbral}",
                                               f"Test · n = {entero(clf['n_test'])}"]),
        html.H2(f"Clasificación · {clf['nombre']} ({clf['estrategia_nombre'].lower()})", className="subtitulo-seccion"),
        fila_kpis([
            kpi("PR-AUC (test)", num(m["PR-AUC"], 3), _ic(test_clf, clf["modelo"], "PR-AUC")),
            kpi("ROC-AUC (test)", num(m["ROC-AUC"], 3), _ic(test_clf, clf["modelo"], "ROC-AUC")),
            kpi(f"Recall · umbral {umbral}", num(m["Recall"], 3), f"Con umbral 0,5: {num(m05['Recall'], 3)}", "deprimido"),
            kpi(f"Precision · umbral {umbral}", num(m["Precision"], 3), f"Con umbral 0,5: {num(m05['Precision'], 3)}"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P(f"Test, umbral {umbral}. % dentro de cada fila real.", className="card-subtitulo"),
                grafico("mf-matriz", 340, animar=False, flexible=True),
                titulo="Matriz de confusión", className="estirar grafico-flexible",
            ), lg=4),
            dbc.Col(card(
                html.P(f"Probabilidades out-of-fold del train. Criterio: {clf['criterio_umbral']}.",
                       className="card-subtitulo"),
                grafico("mf-umbral", 340),
                titulo="Precision, recall y F1 según el umbral",
            ), lg=8),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(html.P("Test, en log-odds.", className="card-subtitulo"), grafico("mf-shap", 320),
                         titulo="Media de |SHAP|"), lg=6),
            dbc.Col(card(html.P("PredictionValuesChange: cambio medio de la predicción (%).", className="card-subtitulo"),
                         grafico("mf-nativa", 320), titulo="Importancia nativa de CatBoost"), lg=6),
        ], className="fila"),
        html.H2(f"Regresión · {reg['nombre']} (target {reg['variante_nombre'].lower()})", className="subtitulo-seccion"),
        fila_kpis([
            kpi("RMSE (test)", num(mr["RMSE"], 3), f"Puntos BDI · IC 95 %: {num(ic['RMSE'][0], 2)}–{num(ic['RMSE'][1], 2)}"),
            kpi("MAE (test)", num(mr["MAE"], 3), "Puntos BDI"),
            kpi("R² (test)", num(mr["R²"], 3), f"IC 95 %: {num(ic['R²'][0], 3)}–{num(ic['R²'][1], 3)}"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Test, con la diagonal (predicción perfecta) y el corte clínico.", className="card-subtitulo"),
                grafico("mf-predicho", 400, animar=False),
                titulo="BDI-II predicho vs. real",
            ), lg=6),
            dbc.Col(card(
                html.P(f"Puntos BDI por cada DE del predictor (sexo: chicas frente a chicos). Intercepto: "
                       f"{num(reg['intercepto'], 2)} puntos."
                       + (f" Anulados por L1: {', '.join(en_cero)}." if en_cero else ""), className="card-subtitulo"),
                grafico("mf-coeficientes", 400),
                titulo="Coeficientes del ElasticNet",
            ), lg=6),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Aumento del RMSE (puntos BDI) al permutar cada variable en el test; media ± DE de las "
                       "repeticiones.", className="card-subtitulo"),
                grafico("mf-permutacion", 320),
                titulo="Permutation importance",
            ), lg=12),
        ], className="fila"),
        html.H2(f"Experimento · BDI predicho ≥ {reg['corte_bdi']} vs. clasificador final", className="subtitulo-seccion"),
        dbc.Row([
            dbc.Col(card(_tabla_experimento(exp), titulo="Mismo test, frente a depressed", className="estirar"), lg=6),
            dbc.Col(card(grafico("mf-experimento", 380, animar=False), titulo="Métricas en test"), lg=6),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(
        Output("mf-matriz", "figure"), Output("mf-umbral", "figure"), Output("mf-shap", "figure"),
        Output("mf-nativa", "figure"), Output("mf-predicho", "figure"), Output("mf-coeficientes", "figure"),
        Output("mf-permutacion", "figure"), Output("mf-experimento", "figure"),
        Input("tema", "data"),
    )
    def _figuras(tema):
        clf = datos.clasificacion_final()
        return (
            figuras.matriz_confusion(datos.clasificacion_matriz_confusion(), tema),
            figuras.metricas_umbral(datos.clasificacion_umbral(), clf["umbral"], clf["umbral_f1"],
                                    clf["recall_minimo"], tema),
            figuras.barras_importancia(_importancia("catboost", "shap_global"), "Media de |SHAP| (log-odds)", tema),
            figuras.barras_importancia(_importancia("catboost", "prediction_values_change"),
                                       "PredictionValuesChange (%)", tema, decimales=1),
            figuras.predicho_vs_real(datos.regresion_predicciones_test(), tema),
            figuras.barras_importancia(_importancia("elasticnet", "coeficiente"), "Puntos BDI por DE", tema, signo=True),
            figuras.barras_importancia(_importancia("elasticnet", "permutacion"), "Aumento del RMSE (puntos BDI)", tema),
            figuras.barras_experimento(datos.regresion_vs_clasificacion(), tema),
        )
