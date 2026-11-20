"""Figuras Plotly del bloque de Modelado.

Igual que analysis/figures.py: cada función recibe tablas ya cargadas (data/load_modelado.py) y el
``tema`` ("claro" u "oscuro"), y usa la plantilla ``uninorte_<tema>``. Reglas de color: sexo con
MAPA_SEXO, estado depresivo con MAPA_ESTADO, clusters con MAPA_CLUSTERS, modelos del benchmarking con
COLORES[tema]["modelos"] (final en pizarra, referencia en gris cálido y el resto por familia) y el
modelo conjunto de género en el color neutro (linea_corte). El texto va siempre en colores de texto.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.figures import _base, _linea_vertical
from components.formato import entero, num, pct
from components.plotly_theme import COLORES, MAPA_CLUSTERS, MAPA_ESTADO, MAPA_SEXO
from data.load_data import CORTE_BDI, ETIQUETAS_ESTADO, ETIQUETAS_SEXO


def _rgba(hex_color: str, alfa: float) -> str:
    """'#4F5B7A' → 'rgba(79,91,122,alfa)'."""
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alfa})"


# Familia y posición fija de cada modelo (clave de clasificación o de regresión). La posición elige el
# tono y el trazo dentro de la familia y no cambia al filtrar, así cada modelo conserva su estilo.
FAMILIA_MODELO = {
    "logistica": ("lineales", 0), "lineal": ("lineales", 0), "lasso": ("lineales", 1), "ridge": ("lineales", 2),
    "elasticnet": ("lineales", 3),
    "random_forest": ("ensambles", 0), "lightgbm": ("ensambles", 1), "xgboost": ("ensambles", 2),
    "catboost": ("ensambles", 3),
    "svc": ("otros", 0), "svr": ("otros", 0), "knn": ("otros", 1), "poisson": ("otros", 2),
    "binomial_negativa": ("otros", 3),
}
TRAZOS = ["solid", "dash", "dot", "dashdot"]


def estilo_modelo(clave: str, es_final: bool, es_referencia: bool, tema: str) -> dict:
    """Color, grosor y trazo de un modelo: el final en pizarra y grueso, la referencia en gris cálido
    discontinua y el resto con el tono de su familia (lineales, ensambles, otros) y un trazo propio."""
    paleta = COLORES[tema]["modelos"]
    if es_final:
        return {"color": paleta["final"], "width": 3.5, "dash": "solid"}
    if es_referencia:
        return {"color": paleta["referencia"], "width": 2.0, "dash": "dash"}
    familia, posicion = FAMILIA_MODELO.get(clave, ("otros", 0))
    tonos = paleta[familia]
    return {"color": tonos[posicion if posicion < len(tonos) else 1], "width": 1.8, "dash": TRAZOS[posicion]}


def _estilos(tabla: pd.DataFrame, tema: str) -> list[dict]:
    return [estilo_modelo(f["modelo"], f["es_final"], f["es_referencia"], tema) for _, f in tabla.iterrows()]


# ── Benchmarking ───────────────────────────────────────────────────────
def puntos_cv(tabla: pd.DataFrame, metrica: str, tema: str = "claro", unidad: str = "") -> go.Figure:
    """Media ± DE de CV de una métrica por modelo (una fila por modelo, en el orden dado: el primero
    queda arriba). Columnas: modelo, nombre, media, desviacion, es_final, es_referencia.
    El eje se ajusta a los puntos y el modelo final se resalta (rombo grande y nombre en negrita)."""
    c = COLORES[tema]
    t = tabla.iloc[::-1]
    estilos = _estilos(t, tema)
    etiquetas = [f"<b>{f['nombre']} · final</b>" if f["es_final"] else f["nombre"] for _, f in t.iterrows()]
    fig = go.Figure(go.Scatter(
        x=t["media"], y=etiquetas, mode="markers",
        error_x=dict(type="data", array=t["desviacion"], color=c["texto_secundario"], thickness=1.5, width=5),
        marker=dict(color=[e["color"] for e in estilos],
                    size=[16 if f["es_final"] else 10 for _, f in t.iterrows()],
                    symbol=["diamond" if f["es_final"] else "circle" for _, f in t.iterrows()],
                    line=dict(color=c["superficie"], width=1)),
        customdata=[[n, float(d)] for n, d in zip(t["nombre"], t["desviacion"])],
        hovertemplate="%{customdata[0]} · <b>%{x:.3f}</b> ± %{customdata[1]:.3f}" + unidad + "<extra></extra>",
    ))
    extremos = np.r_[t["media"] - t["desviacion"], t["media"] + t["desviacion"]]
    margen = (extremos.max() - extremos.min()) * 0.08 or 0.01
    fig.update_xaxes(title=f"{metrica} de CV (media ± DE)", showgrid=True, gridcolor=c["cuadricula"],
                     range=[extremos.min() - margen, extremos.max() + margen])
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(color=c["texto"], size=12))
    return _base(fig, tema, showlegend=False, margin=dict(l=12, r=24, t=16, b=44))


def curvas_test(curvas: pd.DataFrame, modelos: pd.DataFrame, auc: dict, tipo: str, tema: str = "claro",
                prevalencia: float | None = None) -> go.Figure:
    """Curvas ROC (tipo="roc") o Precision-Recall (tipo="pr") en test de los modelos dados.

    ``modelos``: una fila por modelo con modelo, nombre, es_final, es_referencia. Cada modelo tiene
    el color de su familia y su trazo; el final va en pizarra y grueso, dibujado encima de todos.
    """
    c = COLORES[tema]
    fig = go.Figure()
    metrica = "AUC" if tipo == "roc" else "AP"
    orden = modelos.assign(_prioridad=modelos["es_final"] * 2 + modelos["es_referencia"]).sort_values("_prioridad")
    for _, m in orden.iterrows():
        datos = curvas[(curvas["modelo"] == m["modelo"]) & (curvas["tipo"] == tipo)]
        estilo = estilo_modelo(m["modelo"], m["es_final"], m["es_referencia"], tema)
        nombre = f"{m['nombre']}{' · final' if m['es_final'] else ''} · {metrica} = {num(auc[m['modelo']], 3)}"
        fig.add_scatter(
            x=datos["x"], y=datos["y"], mode="lines", name=f"<b>{nombre}</b>" if m["es_final"] else nombre,
            line=dict(color=estilo["color"], width=estilo["width"], dash=estilo["dash"],
                      shape="hv" if tipo == "pr" else "linear"),
            hovertemplate=f"{nombre}<extra></extra>",
        )
    if tipo == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", hoverinfo="skip", showlegend=False,
                        line=dict(color=c["cuadricula"], width=1.2, dash="dot"))
        fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])
        fig.update_yaxes(title="Recall (tasa de verdaderos positivos)", range=[0, 1.02])
    else:
        if prevalencia is not None:
            fig.add_hline(y=prevalencia, line_dash="dot", line_color=c["linea_mediana"], line_width=1,
                          annotation_text=f"Azar · prevalencia {num(prevalencia, 3)}",
                          annotation_position="bottom left", annotation_font=dict(size=12, color=c["texto"]))
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
    fig.update_xaxes(showgrid=True, gridcolor=c["cuadricula"])
    # Leyenda debajo del gráfico: con varios modelos no tapa las curvas
    return _base(fig, tema, legend=dict(orientation="h", x=0, y=-0.22, yanchor="top", font=dict(size=12)),
                 margin=dict(l=56, r=16, t=16, b=44))


# ── Modelo final de clasificación ──────────────────────────────────────
def matriz_confusion(tabla: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Matriz de confusión (real × predicho) con el conteo y el % de cada fila real."""
    c = COLORES[tema]
    m = tabla.pivot(index="real", columns="predicho", values="n").reindex(index=[0, 1], columns=[0, 1]).to_numpy()
    por_fila = m / m.sum(axis=1, keepdims=True)
    etiquetas = [ETIQUETAS_ESTADO[0], ETIQUETAS_ESTADO[1]]
    nombres = [["Verdaderos negativos", "Falsos positivos"], ["Falsos negativos", "Verdaderos positivos"]]
    texto = [[f"<b>{entero(m[i, j])}</b><br>{pct(por_fila[i, j])}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=m, x=etiquetas, y=etiquetas, text=texto, texttemplate="%{text}", customdata=nombres,
        textfont=dict(size=15, color=c["texto"]), xgap=3, ygap=3, showscale=False,
        colorscale=[[0, c["superficie"]], [1, _rgba(c["linea_corte"], 0.5)]],
        hovertemplate="%{customdata} · <b>%{z:,}</b><extra></extra>",
    ))
    fig.update_xaxes(title="Predicción", side="bottom", showline=False)
    fig.update_yaxes(title="Real", autorange="reversed", showgrid=False)
    return _base(fig, tema, margin=dict(l=96, r=12, t=16, b=56))


def metricas_umbral(tabla: pd.DataFrame, umbral: float, umbral_f1: float, recall_minimo: float,
                    tema: str = "claro") -> go.Figure:
    """Precision, recall y F1 (out-of-fold, train) según el umbral, con los umbrales elegidos."""
    c = COLORES[tema]
    fig = go.Figure()
    series = [("precision", "Precision", c["linea_corte"], "solid"), ("recall", "Recall", c["texto"], "dash"),
              ("f1", "F1", c["texto_secundario"], "dot")]
    for columna, nombre, color, trazo in series:
        fig.add_scatter(x=tabla["umbral"], y=tabla[columna], mode="lines", name=nombre,
                        line=dict(color=color, width=2, dash=trazo),
                        hovertemplate=f"Umbral %{{x:.3f}} · {nombre} <b>%{{y:.3f}}</b><extra></extra>")
    fig.add_hline(y=recall_minimo, line_dash="dot", line_color=c["cuadricula"], line_width=1.5,
                  annotation_text=f"Recall mínimo · {num(recall_minimo, 2)}", annotation_position="bottom right",
                  annotation_font=dict(size=12, color=c["texto_secundario"]))
    _linea_vertical(fig, umbral, f"Umbral elegido · {num(umbral, 3)}", c["texto"], "solid", tema)
    _linea_vertical(fig, umbral_f1, f"Máximo F1 · {num(umbral_f1, 3)}", c["linea_mediana"], "dash", tema, nivel=1)
    fig.update_xaxes(title="Umbral de probabilidad", range=[0, min(1, float(tabla["umbral"].max()) * 1.02)],
                     showgrid=True, gridcolor=c["cuadricula"])
    fig.update_yaxes(title="Valor de la métrica", range=[0, 1.02])
    _base(fig, tema)
    return fig.update_layout(hovermode="x unified")


def barras_importancia(tabla: pd.DataFrame, titulo_x: str, tema: str = "claro", signo: bool = False,
                       decimales: int = 3) -> go.Figure:
    """Barras horizontales de una importancia por variable (columnas etiqueta, valor y, opcional,
    desviacion). Con ``signo`` (coeficientes) se colorean como las correlaciones del EDA: positivo
    (aumenta el BDI) en el color de "deprimido" y negativo en el de "no deprimido"; si no, neutro."""
    c = COLORES[tema]
    t = tabla.assign(_abs=tabla["valor"].abs()).sort_values("_abs")
    if signo:
        colores = [c["deprimido"] if v > 0 else c["no_deprimido"] for v in t["valor"]]
        textos = ["0 (anulado por L1)" if v == 0 else f"{'+' if v > 0 else ''}{num(v, 2)}" for v in t["valor"]]
    else:
        colores = c["linea_corte"]
        textos = [num(v, decimales) for v in t["valor"]]
    error = None
    if "desviacion" in t and t["desviacion"].notna().any():
        error = dict(type="data", array=t["desviacion"].fillna(0), color=c["texto_secundario"], thickness=1.2, width=4)
    fig = go.Figure(go.Bar(
        x=t["valor"], y=t["etiqueta"], orientation="h", marker_color=colores, error_x=error,
        text=textos, textposition="outside", cliponaxis=False, textfont=dict(color=c["texto"], size=12),
        hovertemplate="%{y} · <b>%{x:." + str(decimales) + "f}</b><extra></extra>",
    ))
    if signo:
        limite = max(float(t["_abs"].max()) * 1.45, 0.5)
        fig.update_xaxes(range=[-limite, limite], zeroline=True, zerolinecolor=c["texto_secundario"])
    else:
        extremo = float((t["valor"] + (t["desviacion"].fillna(0) if error else 0)).max())
        fig.update_xaxes(range=[min(0, float(t["valor"].min())), extremo * 1.3])
    fig.update_xaxes(title=titulo_x, showgrid=True, gridcolor=c["cuadricula"])
    fig.update_yaxes(showgrid=False)
    return _base(fig, tema, showlegend=False, bargap=0.35, margin=dict(l=230, r=24, t=16, b=44))


# ── Modelo final de regresión ──────────────────────────────────────────
def predicho_vs_real(tabla: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """BDI predicho vs. real en test, con la diagonal y el corte clínico en ambos ejes."""
    c = COLORES[tema]
    limite = float(max(tabla["bdi_real"].max(), tabla["bdi_predicho_final"].max())) + 2
    fig = go.Figure(go.Scatter(
        x=tabla["bdi_real"], y=tabla["bdi_predicho_final"], mode="markers", name="Adolescentes del test",
        marker=dict(color=c["texto_secundario"], size=6, opacity=0.4, line=dict(width=0)),
        hovertemplate="Real %{x} · predicho <b>%{y:.1f}</b><extra></extra>",
    ))
    fig.add_scatter(x=[0, limite], y=[0, limite], mode="lines", name="Predicción perfecta", hoverinfo="skip",
                    line=dict(color=c["texto"], width=1.5))
    fig.add_vline(x=CORTE_BDI, line_dash="dash", line_color=c["linea_corte"], line_width=1.5)
    fig.add_hline(y=CORTE_BDI, line_dash="dash", line_color=c["linea_corte"], line_width=1.5,
                  annotation_text=f"Corte clínico · {CORTE_BDI}", annotation_position="top right",
                  annotation_font=dict(size=12, color=c["texto"]))
    fig.update_xaxes(title="BDI-II real", range=[-1, limite], showgrid=True, gridcolor=c["cuadricula"])
    fig.update_yaxes(title="BDI-II predicho", range=[-1, limite])
    return _base(fig, tema)


def barras_experimento(tabla: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Métricas en test de la regresión con el corte 14 y del clasificador final con su umbral."""
    c = COLORES[tema]
    metricas = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("recall", "Recall"), ("precision", "Precision"),
                ("f1", "F1")]
    # Sin patrones de relleno: la animación de entrada (assets/animacion_graficos.js) no los redibuja
    colores = {"clasificacion": c["linea_corte"], "regresion": _rgba(c["linea_corte"], 0.4)}
    fig = go.Figure()
    for _, fila in tabla.iterrows():
        valores = [fila[m] for m, _ in metricas]
        fig.add_bar(
            x=[n for _, n in metricas], y=valores, name=fila["nombre"], marker_color=colores[fila["enfoque"]],
            text=[num(v, 3) for v in valores], textposition="outside", cliponaxis=False,
            textfont=dict(color=c["texto"], size=12),
            hovertemplate="%{x} · <b>%{y:.3f}</b><extra>" + fila["nombre"] + "</extra>",
        )
    fig.update_yaxes(title="Valor en test", range=[0, 1.08])
    return _base(fig, tema, barmode="group", bargap=0.3, bargroupgap=0.08,
                 legend=dict(orientation="h", x=0, y=1.18), margin=dict(l=48, r=12, t=64, b=44))


# ── Género ─────────────────────────────────────────────────────────────
def importancia_por_sexo(tabla: pd.DataFrame, titulo_x: str, tema: str = "claro") -> go.Figure:
    """Barras horizontales agrupadas chicos vs. chicas (columnas etiqueta, chicos, chicas), ordenadas
    por la media de los dos grupos."""
    c = COLORES[tema]
    colores = MAPA_SEXO(tema)
    t = tabla.assign(_media=(tabla["chicos"] + tabla["chicas"]) / 2).sort_values("_media")
    fig = go.Figure()
    for grupo, sexo in (("chicas", "Girl"), ("chicos", "Boy")):
        fig.add_bar(
            y=t["etiqueta"], x=t[grupo], orientation="h", name=ETIQUETAS_SEXO[sexo], marker_color=colores[sexo],
            text=[num(v, 3) for v in t[grupo]], textposition="outside", cliponaxis=False,
            textfont=dict(color=c["texto"], size=12),
            hovertemplate="%{y} · " + ETIQUETAS_SEXO[sexo].lower() + " · <b>%{x:.3f}</b><extra></extra>",
        )
    fig.update_xaxes(title=titulo_x, showgrid=True, gridcolor=c["cuadricula"],
                     range=[0, float(t[["chicos", "chicas"]].to_numpy().max()) * 1.25])
    fig.update_yaxes(showgrid=False)
    return _base(fig, tema, barmode="group", bargap=0.28, bargroupgap=0.06,
                 legend=dict(orientation="h", x=0, y=1.1, traceorder="reversed"),
                 margin=dict(l=230, r=24, t=40, b=44))


def umbral_por_sexo(resumen: dict, tema: str = "claro") -> go.Figure:
    """Recall y precision de cada grupo con el umbral conjunto (neutro) y con su umbral propio
    (color del grupo)."""
    c = COLORES[tema]
    colores = MAPA_SEXO(tema)
    grupos = resumen["grupos"]
    claves = list(grupos)
    nombres = [grupos[g]["nombre"] for g in claves]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Recall", "Precision"], horizontal_spacing=0.1)
    for columna, metrica in enumerate(("recall", "precision"), start=1):
        conjunto = [grupos[g]["umbral_conjunto"][metrica] for g in claves]
        propio = [grupos[g]["umbral_propio_metricas"][metrica] for g in claves]
        umbrales = [num(grupos[g]["umbral_propio"], 3) for g in claves]
        fig.add_bar(
            x=nombres, y=conjunto, name=f"Umbral conjunto ({num(resumen['umbral_conjunto'], 3)})",
            marker_color=c["linea_corte"], showlegend=columna == 1, legendgroup="conjunto",
            text=[num(v, 2) for v in conjunto], textposition="outside", cliponaxis=False,
            textfont=dict(color=c["texto"], size=12),
            hovertemplate="%{x} · umbral conjunto · <b>%{y:.3f}</b><extra></extra>", row=1, col=columna,
        )
        fig.add_bar(
            x=nombres, y=propio, name="Umbral propio de cada grupo", customdata=umbrales,
            marker_color=[colores[grupos[g]["sexo"]] for g in claves], showlegend=columna == 1, legendgroup="propio",
            text=[num(v, 2) for v in propio], textposition="outside", cliponaxis=False,
            textfont=dict(color=c["texto"], size=12),
            hovertemplate="%{x} · umbral propio (%{customdata}) · <b>%{y:.3f}</b><extra></extra>", row=1, col=columna,
        )
    fig.update_yaxes(range=[0, 1.12], gridcolor=c["cuadricula"])
    fig.update_annotations(font=dict(size=13, color=c["texto"]))
    return _base(fig, tema, barmode="group", bargap=0.3, bargroupgap=0.08,
                 legend=dict(orientation="h", x=0, y=1.2), margin=dict(l=48, r=12, t=72, b=44))


# ── Clustering ─────────────────────────────────────────────────────────
def _nombre_cluster(cluster: int, nombres: dict) -> str:
    return f"{cluster} · {nombres[str(cluster)]}"


def perfiles_clusters(centroides: pd.DataFrame, perfiles: pd.DataFrame, nombres: dict, variables: list,
                      etiquetas: dict, tema: str = "claro") -> go.Figure:
    """Barras agrupadas: media estandarizada de cada variable (eje x) por cluster (color)."""
    c = COLORES[tema]
    colores = MAPA_CLUSTERS(tema)
    tamanos = perfiles.set_index("cluster")
    fig = go.Figure()
    for _, fila in centroides.sort_values("cluster").iterrows():
        cluster = int(fila["cluster"])
        n, porcentaje = tamanos.loc[cluster, "n"], tamanos.loc[cluster, "porcentaje"]
        fig.add_bar(
            x=[etiquetas[v] for v in variables], y=[fila[v] for v in variables], marker_color=colores[cluster],
            name=f"{_nombre_cluster(cluster, nombres)} (n = {entero(n)}, {num(porcentaje, 1)} %)",
            hovertemplate="%{x} · <b>%{y:.2f} DE</b><extra>" + _nombre_cluster(cluster, nombres) + "</extra>",
        )
    fig.add_hline(y=0, line_color=c["texto_secundario"], line_width=1)
    fig.update_yaxes(title="Media estandarizada (DE respecto a la media global)")
    return _base(fig, tema, barmode="group", bargap=0.22, bargroupgap=0.05,
                 legend=dict(orientation="h", x=0, y=1.16), margin=dict(l=56, r=12, t=64, b=64))


def pca_clusters(puntos: pd.DataFrame, centroides: pd.DataFrame, nombres: dict, varianza: list,
                 tema: str = "claro") -> go.Figure:
    """Proyección PCA coloreada por cluster (Scattergl si hay muchos puntos), con los centroides."""
    c = COLORES[tema]
    colores = MAPA_CLUSTERS(tema)
    Traza = go.Scattergl if len(puntos) > 1500 else go.Scatter
    fig = go.Figure()
    for cluster in sorted(puntos["cluster"].unique()):
        p = puntos[puntos["cluster"] == cluster]
        fig.add_trace(Traza(
            x=p["pc1"], y=p["pc2"], mode="markers", name=_nombre_cluster(int(cluster), nombres),
            marker=dict(color=colores[int(cluster)], size=5, opacity=0.55, line=dict(width=0)),
            hovertemplate="CP1 %{x:.2f} · CP2 %{y:.2f}<extra>" + _nombre_cluster(int(cluster), nombres) + "</extra>",
        ))
    fig.add_scatter(
        x=centroides["pc1"], y=centroides["pc2"], mode="markers", name="Centroides",
        marker=dict(symbol="x", size=18, color=[colores[int(k)] for k in centroides["cluster"]],
                    line=dict(color=c["texto"], width=2)),
        customdata=[_nombre_cluster(int(k), nombres) for k in centroides["cluster"]],
        hovertemplate="Centroide · %{customdata}<extra></extra>",
    )
    fig.update_xaxes(title=f"Componente principal 1 ({pct(varianza[0])} de la varianza)", showgrid=True,
                     gridcolor=c["cuadricula"], zeroline=False)
    fig.update_yaxes(title=f"Componente principal 2 ({pct(varianza[1])} de la varianza)")
    return _base(fig, tema, legend=dict(orientation="h", x=0, y=1.14, itemsizing="constant"),
                 margin=dict(l=56, r=12, t=64, b=44))


def prevalencia_clusters(perfiles: pd.DataFrame, nombres: dict, prevalencia_global: float,
                         tema: str = "claro") -> go.Figure:
    """Prevalencia de depresión por cluster (color de "deprimido") con IC 95 % de Wilson y la global."""
    c = COLORES[tema]
    t = perfiles.sort_values("cluster")
    x = [_nombre_cluster(int(k), nombres) for k in t["cluster"]]
    fig = go.Figure(go.Bar(
        x=x, y=t["prevalencia_depresion"] * 100, marker_color=MAPA_ESTADO(tema)[1],
        error_y=dict(type="data", symmetric=False, color=c["texto_secundario"], thickness=1.5, width=6,
                     array=(t["prevalencia_ic_sup"] - t["prevalencia_depresion"]) * 100,
                     arrayminus=(t["prevalencia_depresion"] - t["prevalencia_ic_inf"]) * 100),
        customdata=np.column_stack([t["prevalencia_ic_inf"] * 100, t["prevalencia_ic_sup"] * 100]),
        hovertemplate="%{x} · <b>%{y:.1f} %</b> (IC 95 %: %{customdata[0]:.1f}–%{customdata[1]:.1f})<extra></extra>",
    ))
    fig.add_hline(y=prevalencia_global * 100, line_dash="dash", line_color=c["linea_corte"], line_width=1.5,
                  annotation_text=f"Global · {pct(prevalencia_global)}", annotation_position="top left",
                  annotation_font=dict(size=12, color=c["texto"]))
    fig.update_yaxes(title="% de deprimidos (IC 95 % de Wilson)", ticksuffix=" %", rangemode="tozero")
    fig.update_xaxes(tickfont=dict(size=11))
    return _base(fig, tema, showlegend=False, bargap=0.4)


def bdi_clusters(tabla: pd.DataFrame, nombres: dict, tema: str = "claro") -> go.Figure:
    """Boxplot del BDI-II por cluster (columnas cluster y bdi_total) con el corte clínico."""
    c = COLORES[tema]
    colores = MAPA_CLUSTERS(tema)
    fig = go.Figure()
    for cluster in sorted(tabla["cluster"].unique()):
        color = colores[int(cluster)]
        fig.add_box(
            y=tabla.loc[tabla["cluster"] == cluster, "bdi_total"], name=_nombre_cluster(int(cluster), nombres),
            boxpoints="outliers", marker=dict(color=color, size=4), line=dict(color=c["texto_secundario"], width=1.2),
            fillcolor=color, hoverinfo="y",
        )
    fig.add_hline(y=CORTE_BDI, line_dash="dash", line_color=c["linea_corte"], line_width=1.5,
                  annotation_text=f"Corte clínico · {CORTE_BDI}", annotation_position="top left",
                  annotation_font=dict(size=12, color=c["texto"]))
    fig.update_yaxes(title="Puntaje BDI-II")
    fig.update_xaxes(tickfont=dict(size=11))
    return _base(fig, tema, showlegend=False)


def sexo_clusters(perfiles: pd.DataFrame, nombres: dict, proporcion_global: float, tema: str = "claro") -> go.Figure:
    """Barras apiladas al 100 %: chicos y chicas en cada cluster, con la proporción global de chicas."""
    c = COLORES[tema]
    colores = MAPA_SEXO(tema)
    t = perfiles.sort_values("cluster")
    x = [_nombre_cluster(int(k), nombres) for k in t["cluster"]]
    fig = go.Figure()
    for sexo, valores in (("Girl", t["proporcion_chicas"]), ("Boy", 1 - t["proporcion_chicas"])):
        fig.add_bar(
            x=x, y=valores * 100, name=ETIQUETAS_SEXO[sexo], marker_color=colores[sexo],
            hovertemplate="%{x} · " + ETIQUETAS_SEXO[sexo].lower() + " · <b>%{y:.1f} %</b><extra></extra>",
        )
    fig.add_hline(y=proporcion_global * 100, line_dash="dash", line_color=c["linea_corte"], line_width=1.5,
                  annotation_text=f"Chicas en la muestra · {pct(proporcion_global)}", annotation_position="top left",
                  annotation_font=dict(size=12, color=c["texto"]))
    fig.update_yaxes(title="% del cluster", ticksuffix=" %", range=[0, 100])
    fig.update_xaxes(tickfont=dict(size=11))
    return _base(fig, tema, barmode="stack", bargap=0.4)


# ── Predicción ─────────────────────────────────────────────────────────
def indicador_probabilidad(probabilidad: float, umbral: float, tema: str = "claro") -> go.Figure:
    """Barra 0–100 % con la probabilidad estimada y la línea del umbral de tamizaje."""
    c = COLORES[tema]
    color = MAPA_ESTADO(tema)[int(probabilidad >= umbral)]
    fig = go.Figure()
    fig.add_bar(x=[100], y=[0], orientation="h", marker_color=c["cuadricula"], hoverinfo="skip", width=0.6)
    fig.add_bar(x=[probabilidad * 100], y=[0], orientation="h", marker_color=color, width=0.6,
                hovertemplate="Probabilidad · <b>%{x:.1f} %</b><extra></extra>")
    fig.add_vline(x=umbral * 100, line_dash="dash", line_color=c["linea_corte"], line_width=2,
                  annotation_text=f"Umbral · {pct(umbral)}", annotation_position="top",
                  annotation_font=dict(size=12, color=c["texto"]))
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(range=[-0.4, 0.4], showticklabels=False, showgrid=False)
    return _base(fig, tema, barmode="overlay", showlegend=False, margin=dict(l=12, r=16, t=30, b=28))


# Categorías del BDI-II: (inicio, fin, nombre) para la barra de la escala
TRAMOS_BDI = [(0, 14, "Mínimo"), (14, 20, "Leve"), (20, 29, "Moderado"), (29, 64, "Severo")]


def escala_bdi(bdi: float, tema: str = "claro") -> go.Figure:
    """Escala del BDI-II (0–63) por categorías clínicas (nombres en el eje), con el corte 14 y un
    marcador sobre la barra en el puntaje estimado."""
    c = COLORES[tema]
    fig = go.Figure()
    for (inicio, fin, nombre), alfa in zip(TRAMOS_BDI, (0.12, 0.24, 0.36, 0.48)):
        fig.add_bar(x=[fin - inicio], base=[inicio], y=[0], orientation="h", name=nombre, width=0.5,
                    marker=dict(color=_rgba(c["linea_corte"], alfa), line=dict(color=c["superficie"], width=2)),
                    hovertemplate=f"{nombre} · {inicio}–{fin - 1}<extra></extra>")
    fig.add_vline(x=CORTE_BDI, line_dash="dash", line_color=c["linea_corte"], line_width=2,
                  annotation_text=f"Corte · {CORTE_BDI}", annotation_position="top",
                  annotation_font=dict(size=12, color=c["texto"]))
    fig.add_scatter(x=[bdi], y=[0.42], mode="markers", marker=dict(symbol="triangle-down", size=14, color=c["texto"]),
                    hovertemplate="BDI-II estimado · <b>%{x:.1f}</b><extra></extra>")
    fig.update_xaxes(range=[0, 63], showgrid=False, tickfont=dict(size=11, color=c["texto_secundario"]),
                     tickvals=[(inicio + min(fin, 63)) / 2 for inicio, fin, _ in TRAMOS_BDI],
                     ticktext=[n for _, _, n in TRAMOS_BDI])
    fig.update_yaxes(range=[-0.35, 0.6], showticklabels=False, showgrid=False)
    return _base(fig, tema, barmode="overlay", showlegend=False, margin=dict(l=12, r=16, t=30, b=28))
