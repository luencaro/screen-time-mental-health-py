"""Pestaña · Modelos por género: modelos específicos de chicos y chicas frente al modelo conjunto."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figuras_modelado as figuras
from components.formato import entero, num
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi
from data import load_modelado as datos


def _shap_por_grupo():
    """Media de |SHAP| del CatBoost de cada grupo: una fila por variable con columnas chicos y chicas."""
    t = datos.genero_importancias()
    t = t[(t["modelo"] == "catboost") & (t["metodo"] == "shap_global")]
    return t.pivot(index=["variable", "etiqueta"], columns="grupo", values="valor").reset_index()


def layout():
    """Estructura: KPIs → |SHAP| chicos vs. chicas → efecto del umbral."""
    resumen = datos.genero_resumen()
    g = resumen["grupos"]
    u = num(resumen["umbral_conjunto"], 3)
    recall_minimo = datos.clasificacion_final()["recall_minimo"]
    kpis = []
    for grupo in ("chicos", "chicas"):
        kpis.append(kpi(f"ROC-AUC · {g[grupo]['nombre'].lower()}", num(g[grupo]["roc_auc"]["especifico"]["valor"], 3),
                        f"Modelo específico · conjunto: {num(g[grupo]['roc_auc']['conjunto']['valor'], 3)}"))
    for grupo in ("chicos", "chicas"):
        kpis.append(kpi(f"Recall · {g[grupo]['nombre'].lower()} · umbral {u}",
                        num(g[grupo]["umbral_conjunto"]["recall"], 3),
                        f"Umbral propio ({num(g[grupo]['umbral_propio'], 3)}): "
                        f"{num(g[grupo]['umbral_propio_metricas']['recall'], 3)}", "deprimido"))
    n_test = {k: entero(v["tamanos"]["test"]["filas"]) for k, v in g.items()}

    return html.Div([
        encabezado_seccion("genero", [f"Test · {n_test['chicos']} chicos · {n_test['chicas']} chicas",
                                      "CatBoost por grupo"]),
        fila_kpis(kpis),
        dbc.Row(dbc.Col(card(
            html.P("CatBoost de cada grupo, test del grupo, en log-odds.", className="card-subtitulo"),
            grafico("ge-shap", 400),
            titulo="Media de |SHAP| por variable: chicos vs. chicas",
        ), lg=12), className="fila"),
        dbc.Row(dbc.Col(card(
            html.P(f"Gris: modelo conjunto con el umbral único ({u}). Color del grupo: modelo específico con su "
                   f"umbral propio (mismo criterio: máxima precision con recall ≥ {num(recall_minimo, 2)} en el train del grupo).",
                   className="card-subtitulo"),
            grafico("ge-umbral", 360),
            titulo="Recall y precision con el umbral conjunto y con el umbral propio",
        ), lg=12), className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(Output("ge-shap", "figure"), Output("ge-umbral", "figure"), Input("tema", "data"))
    def _figuras(tema):
        return (figuras.importancia_por_sexo(_shap_por_grupo(), "Media de |SHAP| (log-odds)", tema),
                figuras.umbral_por_sexo(datos.genero_resumen(), tema))
