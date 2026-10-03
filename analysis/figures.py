"""Figuras Plotly del EDA.

Cada función recibe los datos (o una tabla de analysis.stats) y el ``tema``
("claro" u "oscuro"), y usa la plantilla ``uninorte_<tema>`` junto con
``MAPA_SEXO`` / ``MAPA_ESTADO``. Ningún color se define aquí.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis import stats
from components.formato import entero, num, pct
from components.plotly_theme import COLORES, MAPA_ESTADO, MAPA_SEXO
from data.load_data import CORTE_BDI, ETIQUETAS, ETIQUETAS_ESTADO, ETIQUETAS_SEXO

FUENTE_MONO = "IBM Plex Mono, monospace"


def _base(fig: go.Figure, tema: str, **layout) -> go.Figure:
    """Aplica la plantilla del tema y opciones comunes."""
    fig.update_layout(template=f"uninorte_{tema}", hovermode="closest", **layout)
    return fig


def _linea_vertical(fig: go.Figure, x: float, texto: str, color: str, estilo: str,
                    tema: str, posicion: str = "top right", nivel: int = 0) -> None:
    """Línea vertical anotada (corte discontinuo o mediana punteada).

    ``nivel`` baja la etiqueta una línea de texto por nivel, para que las
    etiquetas de dos líneas cercanas (mediana y corte) no se solapen.
    """
    fig.add_vline(
        x=x, line_dash=estilo, line_color=color, line_width=1.5,
        annotation_text=texto, annotation_position=posicion, annotation_yshift=-18 * nivel,
        annotation_font=dict(size=12, color=COLORES[tema]["texto"]),
    )


def _mediana_texto(valor: float) -> str:
    """Mediana sin decimales si es entera (5), con dos si no (1,75)."""
    return num(valor, 0) if float(valor).is_integer() else num(valor, 2)


def _bins(serie: pd.Series) -> dict:
    """Bins comunes para una variable; en variables discretas, uno por valor."""
    unicos = np.sort(serie.unique())
    if len(unicos) <= 40:
        paso = float(np.min(np.diff(unicos))) if len(unicos) > 1 else 1.0
        return dict(start=unicos[0] - paso / 2, end=unicos[-1] + paso / 2, size=paso)
    bordes = np.histogram_bin_edges(serie, bins="fd")
    if len(bordes) > 60:
        bordes = np.histogram_bin_edges(serie, bins=40)
    return dict(start=bordes[0], end=bordes[-1], size=bordes[1] - bordes[0])


# ── Variable objetivo ──────────────────────────────────────────────────
def dona_estado(df: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Dona de ``depressed`` con hueco del 70 % y prevalencia al centro."""
    dist = stats.distribucion_estado(df)
    colores = MAPA_ESTADO(tema)
    fig = go.Figure(go.Pie(
        labels=dist["etiqueta"], values=dist["n"], hole=0.7, sort=False, direction="clockwise",
        marker=dict(colors=[colores[e] for e in dist["depressed"]],
                    line=dict(color=COLORES[tema]["superficie"], width=3)),
        textinfo="none",
        hovertemplate="%{label} · <b>%{value:,}</b> (%{percent})<extra></extra>",
    ))
    prevalencia = dist.loc[dist["depressed"] == 1, "proporcion"].iloc[0]
    fig.add_annotation(
        text=f"<b>{pct(prevalencia)}</b><br><span style='font-size:12px'>deprimidos</span>",
        showarrow=False, font=dict(size=24, color=COLORES[tema]["texto"]),
    )
    return _base(fig, tema, margin=dict(l=8, r=8, t=40, b=8), showlegend=True)


def histograma_bdi(df: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Histograma de ``bdi_total`` por tramo, con KDE, mediana y corte clínico."""
    c = COLORES[tema]
    colores = MAPA_ESTADO(tema)
    conteo = df["bdi_total"].value_counts().sort_index()
    fig = go.Figure()
    for estado in (0, 1):
        tramo = conteo[conteo.index >= CORTE_BDI] if estado else conteo[conteo.index < CORTE_BDI]
        fig.add_bar(
            x=tramo.index, y=tramo.values, name=ETIQUETAS_ESTADO[estado],
            marker_color=colores[estado],
            hovertemplate="Puntaje %{x} · <b>%{y:,} adolescentes</b><extra></extra>",
        )
    densidad = stats.kde(df["bdi_total"], limites=(0, df["bdi_total"].max()))
    fig.add_scatter(
        x=densidad["x"], y=densidad["densidad"] * len(df), mode="lines", name="Densidad (KDE)",
        line=dict(color=c["texto_secundario"], width=2),
        hovertemplate="Puntaje %{x:.1f} · <b>%{y:,.0f}</b> (KDE)<extra></extra>",
    )
    mediana = df["bdi_total"].median()
    _linea_vertical(fig, mediana, f"Mediana · {num(mediana, 0)}", c["linea_mediana"], "dot", tema)
    _linea_vertical(fig, CORTE_BDI - 0.5, f"Corte clínico · {CORTE_BDI}", c["linea_corte"], "dash", tema,
                    nivel=1)
    fig.update_xaxes(title="Puntaje BDI-II")
    fig.update_yaxes(title="Adolescentes")
    return _base(fig, tema, bargap=0.12, barmode="overlay")


def boxplot_bdi(df: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Boxplot horizontal de ``bdi_total`` con mediana y corte clínico anotados."""
    c = COLORES[tema]
    fig = go.Figure(go.Box(
        x=df["bdi_total"], name="", orientation="h", boxpoints="outliers",
        fillcolor=c["cuadricula"], line=dict(color=c["texto_secundario"], width=1.5),
        marker=dict(color=c["texto_secundario"], size=5, opacity=0.6),
        hoverinfo="x",
    ))
    mediana = df["bdi_total"].median()
    _linea_vertical(fig, mediana, f"Mediana · {num(mediana, 0)}", c["linea_mediana"], "dot", tema)
    _linea_vertical(fig, CORTE_BDI, f"Corte clínico · {CORTE_BDI}", c["linea_corte"], "dash", tema,
                    nivel=1)
    fig.update_xaxes(title="Puntaje BDI-II", showgrid=True, gridcolor=c["cuadricula"])
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return _base(fig, tema, showlegend=False, margin=dict(l=16, r=16, t=36, b=44))


# ── Sexo ───────────────────────────────────────────────────────────────
def barras_sexo(df: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Número de adolescentes por sexo."""
    dist = stats.distribucion_sexo(df)
    colores = MAPA_SEXO(tema)
    fig = go.Figure()
    for _, fila in dist.iterrows():
        fig.add_bar(
            x=[fila["etiqueta"]], y=[fila["n"]], name=fila["etiqueta"],
            marker_color=colores[fila["sex"]],
            text=[f"{entero(fila['n'])} · {pct(fila['proporcion'])}"], textposition="outside",
            textfont=dict(color=COLORES[tema]["texto"], size=13), cliponaxis=False,
            hovertemplate="%{x} · <b>%{y:,}</b><extra></extra>",
        )
    fig.update_yaxes(title="Adolescentes", rangemode="tozero")
    return _base(fig, tema, bargap=0.45, showlegend=False)


def barras_depresion_sexo(df: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Porcentaje de deprimidos y no deprimidos dentro de cada sexo."""
    tabla = stats.depresion_por_sexo(df)
    colores = MAPA_ESTADO(tema)
    fig = go.Figure()
    for estado in (0, 1):
        t = tabla[tabla["depressed"] == estado]
        fig.add_bar(
            x=t["etiqueta_sexo"], y=t["proporcion"] * 100, name=ETIQUETAS_ESTADO[estado],
            marker_color=colores[estado], customdata=t["n"],
            texttemplate="%{y:.1f} %", textposition="outside", cliponaxis=False,
            textfont=dict(color=COLORES[tema]["texto"], size=13),
            hovertemplate="%{x} · " + ETIQUETAS_ESTADO[estado].lower()
                          + " · <b>%{y:.1f} %</b> (%{customdata:,})<extra></extra>",
        )
    fig.update_yaxes(title="% dentro del grupo", range=[0, 105], ticksuffix=" %")
    return _base(fig, tema, barmode="group", bargap=0.3, bargroupgap=0.08)


def boxplot_por_sexo(df: pd.DataFrame, columna: str = "bdi_total", tema: str = "claro") -> go.Figure:
    """Boxplot de una variable por sexo; con ``bdi_total`` añade el corte clínico."""
    c = COLORES[tema]
    colores = MAPA_SEXO(tema)
    fig = go.Figure()
    for sexo, etiqueta in ETIQUETAS_SEXO.items():
        fig.add_box(
            y=df.loc[df["sex"] == sexo, columna], name=etiqueta, boxpoints="outliers",
            marker=dict(color=colores[sexo], size=4), line=dict(color=colores[sexo], width=1.5),
            fillcolor=colores[sexo], opacity=0.9, hoverinfo="y",
        )
    if columna == "bdi_total":
        fig.add_hline(y=CORTE_BDI, line_dash="dash", line_color=c["linea_corte"], line_width=1.5,
                      annotation_text=f"Corte clínico · {CORTE_BDI}", annotation_position="top left",
                      annotation_font=dict(size=12, color=c["texto"]))
    fig.update_yaxes(title=ETIQUETAS[columna])
    return _base(fig, tema, showlegend=False)


# ── Variables numéricas ────────────────────────────────────────────────
def histograma_variable(df: pd.DataFrame, columna: str, separar: str | None = None,
                        tema: str = "claro") -> go.Figure:
    """Histograma de una variable, opcionalmente separado por sexo o por estado.

    Al separar, cada grupo se normaliza a % del grupo para compararlos pese a
    tamaños distintos. Los conteos se calculan aquí y se dibujan como barras,
    siempre con dos trazas (la segunda vacía si no se separa): así dcc.Graph
    puede animar la transición al cambiar de variable o de agrupación.
    """
    c = COLORES[tema]
    bins = _bins(df[columna])
    bordes = np.arange(bins["start"], bins["end"] + bins["size"] * 0.5, bins["size"])
    centros = (bordes[:-1] + bordes[1:]) / 2
    rangos = [f"{num(a, 2)}–{num(b, 2)}" for a, b in zip(bordes[:-1], bordes[1:])]

    if separar == "sex":
        grupos = [(df["sex"] == s, ETIQUETAS_SEXO[s], MAPA_SEXO(tema)[s]) for s in ETIQUETAS_SEXO]
    elif separar == "depressed":
        grupos = [(df["depressed"] == e, ETIQUETAS_ESTADO[e], MAPA_ESTADO(tema)[e]) for e in (0, 1)]
    else:
        grupos = [(pd.Series(True, index=df.index), "Todos", c["linea_corte"]),
                  (pd.Series(False, index=df.index), "", c["linea_corte"])]

    fig = go.Figure()
    for mascara, nombre, color in grupos:
        valores = df.loc[mascara, columna]
        conteo, _ = np.histogram(valores, bins=bordes)
        y = conteo / len(valores) * 100 if separar else conteo
        if not len(valores):
            y = np.zeros_like(centros)
        fig.add_bar(
            x=centros, y=y, width=bins["size"] * 0.88, name=nombre, customdata=rangos,
            marker_color=color, opacity=0.65 if separar else 0.85, showlegend=bool(separar),
            hovertemplate=(f"{nombre} · %{{customdata}} · <b>%{{y:.1f}} %</b><extra></extra>" if separar
                           else "%{customdata} · <b>%{y:,} adolescentes</b><extra></extra>"),
        )
    mediana = df[columna].median()
    _linea_vertical(fig, mediana, f"Mediana · {_mediana_texto(mediana)}", c["linea_mediana"], "dot", tema)
    if columna == "bdi_total":
        _linea_vertical(fig, CORTE_BDI - 0.5, f"Corte clínico · {CORTE_BDI}", c["linea_corte"],
                        "dash", tema, nivel=1)
    # Rangos explícitos: Plotly.animate no recalcula el autorango al animar
    y_max = max(float(np.max(t.y)) for t in fig.data) * 1.12
    fig.update_xaxes(title=ETIQUETAS[columna], range=[bordes[0], bordes[-1]])
    fig.update_yaxes(title="% del grupo" if separar else "Adolescentes",
                     ticksuffix=" %" if separar else "", range=[0, y_max])
    return _base(fig, tema, barmode="overlay", showlegend=bool(separar))


def boxplot_variable(df: pd.DataFrame, columna: str, separar: str | None = None,
                     tema: str = "claro") -> go.Figure:
    """Boxplot horizontal compacto de una variable, por grupo si se separa."""
    c = COLORES[tema]
    if separar == "sex":
        grupos = [(df["sex"] == s, ETIQUETAS_SEXO[s], MAPA_SEXO(tema)[s]) for s in ETIQUETAS_SEXO]
    elif separar == "depressed":
        grupos = [(df["depressed"] == e, ETIQUETAS_ESTADO[e], MAPA_ESTADO(tema)[e]) for e in (0, 1)]
    else:
        grupos = [(pd.Series(True, index=df.index), "Todos", c["linea_corte"])]
    fig = go.Figure()
    for mascara, nombre, color in grupos:
        fig.add_box(
            x=df.loc[mascara, columna], name=nombre, orientation="h", boxpoints="outliers",
            marker=dict(color=color, size=4), line=dict(color=color, width=1.5),
            fillcolor=color, opacity=0.85, hoverinfo="x",
        )
    margen = (df[columna].max() - df[columna].min()) * 0.04
    fig.update_xaxes(title=ETIQUETAS[columna], showgrid=True, gridcolor=c["cuadricula"],
                     range=[df[columna].min() - margen, df[columna].max() + margen])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _base(fig, tema, showlegend=False, margin=dict(l=96, r=12, t=16, b=44))


# ── Correlaciones ──────────────────────────────────────────────────────
def _escala_divergente(tema: str) -> list:
    c = COLORES[tema]
    return [[0.0, c["no_deprimido"]], [0.5, c["superficie"]], [1.0, c["deprimido"]]]


def heatmap_spearman(matriz: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """Heatmap del triángulo inferior de la matriz de Spearman."""
    c = COLORES[tema]
    inferior = matriz.iloc[1:, :-1]
    valores = inferior.to_numpy().copy()
    valores[np.triu_indices_from(valores, k=1)] = np.nan
    fig = go.Figure(go.Heatmap(
        z=valores, x=list(inferior.columns), y=list(inferior.index), zmin=-1, zmax=1, colorscale=_escala_divergente(tema),
        xgap=2, ygap=2, texttemplate="%{z:.2f}", textfont=dict(size=12, color=c["texto"]),
        hovertemplate="%{y} × %{x} · <b>ρ = %{z:.3f}</b><extra></extra>", hoverongaps=False,
        colorbar=dict(title=dict(text="ρ", side="top"), thickness=10, outlinewidth=0,
                      tickvals=[-1, -0.5, 0, 0.5, 1], tickfont=dict(color=c["texto_secundario"])),
    ))
    fig.update_xaxes(tickfont=dict(family=FUENTE_MONO, size=11), tickangle=-35, showline=False)
    fig.update_yaxes(tickfont=dict(family=FUENTE_MONO, size=11), autorange="reversed",
                     showgrid=False)
    return _base(fig, tema, margin=dict(l=180, r=12, t=16, b=150))


def barras_rho(tabla: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """ρ de Spearman de cada variable con un destino (tabla de spearman_pares)."""
    c = COLORES[tema]
    t = tabla.assign(abs_rho=tabla["rho"].abs()).sort_values("abs_rho")
    colores = [c["deprimido"] if r > 0 else c["no_deprimido"] for r in t["rho"]]
    fig = go.Figure(go.Bar(
        x=t["rho"], y=t["x"], orientation="h", marker_color=colores,
        texttemplate="%{x:.3f}", textposition="outside", cliponaxis=False,
        textfont=dict(color=c["texto"], size=12),
        hovertemplate="%{y} · <b>ρ = %{x:.3f}</b><extra></extra>",
    ))
    limite = max(0.5, t["abs_rho"].max() * 1.35)
    fig.update_xaxes(title="ρ de Spearman", range=[-limite, limite], zeroline=True,
                     zerolinecolor=c["cuadricula"], showgrid=True, gridcolor=c["cuadricula"])
    fig.update_yaxes(tickfont=dict(family=FUENTE_MONO, size=11), showgrid=False)
    return _base(fig, tema, showlegend=False, bargap=0.35, margin=dict(l=170, r=16, t=16, b=44))


# ── Outliers ───────────────────────────────────────────────────────────
def barras_outliers_estado(tabla: pd.DataFrame, tema: str = "claro") -> go.Figure:
    """% de outliers por variable dentro de cada estado depresivo."""
    colores = MAPA_ESTADO(tema)
    fig = go.Figure()
    for estado in (0, 1):
        t = tabla[tabla["depressed"] == estado]
        fig.add_bar(
            y=t["variable"], x=t["proporcion"] * 100, name=ETIQUETAS_ESTADO[estado], orientation="h",
            marker_color=colores[estado], customdata=t["n"],
            texttemplate="%{x:.1f} %", textposition="outside", cliponaxis=False,
            textfont=dict(color=COLORES[tema]["texto"], size=12),
            hovertemplate="%{y} · " + ETIQUETAS_ESTADO[estado].lower()
                          + " · <b>%{x:.1f} %</b> (%{customdata:,})<extra></extra>",
        )
    fig.update_xaxes(title="% de outliers dentro del grupo", ticksuffix=" %", showgrid=True,
                     gridcolor=COLORES[tema]["cuadricula"])
    fig.update_yaxes(tickfont=dict(family=FUENTE_MONO, size=11), autorange="reversed", showgrid=False)
    return _base(fig, tema, barmode="group", bargap=0.3, bargroupgap=0.08,
                 margin=dict(l=170, r=40, t=36, b=44))
