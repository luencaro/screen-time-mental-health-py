"""Pestaña · Clustering: perfiles de sueño y tiempo de pantalla con K-Means y su validación externa."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figuras_modelado as figuras
from components.formato import entero, num, pct
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi
from data import load_modelado as datos
from data.load_data import cargar_datos


def _con_bdi():
    """Cluster de cada adolescente (clustering_puntos.csv) unido al CSV crudo."""
    puntos = datos.clustering_puntos()[["subject_id", "cluster"]]
    return puntos.merge(cargar_datos()[["subject_id", "bdi_total", "depressed", "sex"]], on="subject_id")


def layout():
    """Estructura: KPIs → perfiles → PCA + prevalencia → BDI + sexo."""
    meta, perfiles, seleccion = datos.clustering_meta(), datos.clustering_perfiles(), datos.clustering_seleccion_k()
    datos.clustering_puntos()
    datos.clustering_centroides()
    df = cargar_datos()
    k = meta["k"]
    fila_k = seleccion.set_index("k").loc[k]
    riesgo = perfiles.loc[perfiles["prevalencia_depresion"].idxmax()]

    return html.Div([
        encabezado_seccion("clustering", [f"K-Means · k = {k}", f"n = {entero(meta['n_observaciones'])}",
                                          f"{len(meta['variables'])} variables de sueño y pantalla"]),
        fila_kpis([
            kpi("Clusters", str(k), "K-Means sobre las variables estandarizadas"),
            kpi("Silueta", num(fila_k["silueta"], 3), f"Con k = {k}"),
            kpi("Estabilidad (ARI)", num(fila_k["ari_media"], 3), f"± {num(fila_k['ari_desviacion'], 3)} · "
                f"{meta['n_bootstrap_estabilidad']} réplicas bootstrap"),
            kpi("Prevalencia en el perfil de mayor riesgo", pct(riesgo["prevalencia_depresion"]),
                f"{riesgo['nombre']} · global {pct(df['depressed'].mean())}", "deprimido"),
        ]),
        dbc.Row(dbc.Col(card(
            html.P("Media de cada variable estandarizada por cluster (0 = media global). Un índice de calidad del "
                   "sueño alto indica peor calidad.", className="card-subtitulo"),
            grafico("cl-perfiles", 400),
            titulo="Perfiles de los clusters",
        ), lg=12), className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Cada punto es un adolescente; X = centroide.", className="card-subtitulo"),
                grafico("cl-pca", 440, animar=False),
                titulo="Proyección en las dos primeras componentes principales",
            ), lg=7),
            dbc.Col(card(
                html.P("Validación externa: la depresión no se usó para formar los clusters.", className="card-subtitulo"),
                grafico("cl-prevalencia", 440, animar=False, flexible=True),
                titulo="Prevalencia de depresión por cluster", className="estirar grafico-flexible",
            ), lg=5),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(grafico("cl-bdi", 380, animar=False), titulo="BDI-II por cluster"), lg=6),
            dbc.Col(card(grafico("cl-sexo", 380, animar=False), titulo="Proporción de chicas por cluster"), lg=6),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(
        Output("cl-perfiles", "figure"), Output("cl-pca", "figure"),
        Output("cl-prevalencia", "figure"), Output("cl-bdi", "figure"), Output("cl-sexo", "figure"),
        Input("tema", "data"),
    )
    def _figuras(tema):
        meta, perfiles = datos.clustering_meta(), datos.clustering_perfiles()
        centroides, nombres = datos.clustering_centroides(), meta["nombres"]
        df = cargar_datos()
        return (
            figuras.perfiles_clusters(centroides, perfiles, nombres, meta["variables"], meta["etiquetas"], tema),
            figuras.pca_clusters(datos.clustering_puntos(), centroides, nombres, meta["varianza_explicada"], tema),
            figuras.prevalencia_clusters(perfiles, nombres, df["depressed"].mean(), tema),
            figuras.bdi_clusters(_con_bdi(), nombres, tema),
            figuras.sexo_clusters(perfiles, nombres, (df["sex"] == "Girl").mean(), tema),
        )
