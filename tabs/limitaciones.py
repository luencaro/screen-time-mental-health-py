"""Pestaña 12 · Limitaciones: alcance del análisis y siguiente paso."""

import re

import dash_bootstrap_components as dbc
from dash import html

from analysis import stats
from components.formato import entero, num, pct
from components.ui import card, encabezado_seccion, fila_kpis, interpretacion, kpi, markdown, rico
from content import libro
from data.load_data import (CORTE_BDI, VARIABLES_NUMERICAS, VARIABLES_PANTALLA, VARIABLES_SUENO,
                            cargar_datos)


def _limitaciones(df) -> list:
    """Limitaciones con cifras calculadas desde los datos."""
    r = stats.resumen_muestra(df)
    matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
    par = stats.par_mas_fuerte(matriz, excluir=["bdi_total"])
    redundantes = stats.pares_redundantes(matriz, 0.7)
    rho_max = matriz["bdi_total"].drop("bdi_total").abs().max()
    una_medicion = df["subject_id"].is_unique
    n_bajo = r["n"] - r["n_deprimidos"]
    edad = re.search(r"entre (\d+) y (\d+) años", libro.seccion("intro.md", "Introducción"))
    rango = f"de {edad.group(1)} a {edad.group(2)} años" if edad else "escolares"

    items = [
        ("Corte transversal.",
         ("El dataset contiene una sola medición por adolescente "
          f"({entero(df['subject_id'].nunique())} identificadores únicos en {entero(len(df))} filas). "
          if una_medicion else "")
         + "Aunque el estudio de origen es prospectivo, este análisis no permite establecer el "
           "orden temporal entre pantalla, sueño y síntomas depresivos."),
        ("Variables autorreportadas.",
         f"Las {len(VARIABLES_PANTALLA)} variables de pantalla y las {len(VARIABLES_SUENO)} de sueño "
         "provienen de cuestionarios, sujetos a sesgos de recuerdo y de deseabilidad social. El "
         f"BDI-II es un instrumento de cribado: un puntaje ≥ {CORTE_BDI} indica síntomas sobre el "
         "corte clínico, no un diagnóstico."),
        ("Desbalance de clases.",
         f"{entero(r['n_deprimidos'])} adolescentes ({pct(r['prevalencia'])}) están sobre el corte "
         f"frente a {entero(n_bajo)} bajo el corte (1 : {num(n_bajo / r['n_deprimidos'], 1)}). La "
         "exactitud global sería una métrica engañosa en la fase de modelado."),
        ("Multicolinealidad.",
         f"{len(redundantes)} pares de predictores superan |ρ| = 0,7; el más alto es `{par['x']}` – "
         f"`{par['y']}` (ρ = {num(par['rho'], 3)}). Sus efectos individuales no son separables en "
         "un modelo lineal."),
        ("Muestra sueca.",
         f"Los participantes son adolescentes {rango} de centros escolares de Estocolmo; la "
         "generalización a otros países o contextos educativos es limitada."),
        ("Correlación no implica causalidad.",
         f"La asociación más fuerte con `bdi_total` es |ρ| = {num(rho_max, 3)} "
         f"(efecto {stats.magnitud(rho_max)}). Factores no incluidos en el dataset, como el "
         "contexto familiar o escolar, pueden explicar parte de las asociaciones observadas."),
    ]
    return [html.Li([html.Strong(t), html.Br(), *rico(d)]) for t, d in items]


def layout():
    """Estructura: KPIs → limitaciones + siguiente paso (desde methodology.md)."""
    df = cargar_datos()
    r = stats.resumen_muestra(df)
    matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
    par = stats.par_mas_fuerte(matriz, excluir=["bdi_total"])
    n_bajo = r["n"] - r["n_deprimidos"]
    siguiente = "\n\n".join(
        f"**{titulo}**\n\n{libro.seccion('methodology.md', titulo)}"
        for titulo in ("Preprocesamiento", "Modelado", "Evaluación")
    )

    return html.Div([
        encabezado_seccion("limitaciones", ["Alcance del EDA", "Siguiente paso: modelado"]),
        fila_kpis([
            kpi("Mediciones por adolescente", entero(len(df) / df["subject_id"].nunique()),
                "Diseño de corte transversal"),
            kpi("Desbalance", f"1 : {num(n_bajo / r['n_deprimidos'], 1)}",
                f"{pct(r['prevalencia'])} deprimidos", "deprimido"),
            kpi("Máxima |ρ| entre predictores", num(abs(par["rho"]), 3), f"{par['x']} · {par['y']}"),
            kpi("Población", "Suecia", "Centros escolares de Estocolmo"),
        ]),
        dbc.Row([
            dbc.Col(card(html.Ul(_limitaciones(df), className="lista-hallazgos"),
                         titulo="Limitaciones del análisis"), lg=8),
            dbc.Col(interpretacion(
                "La siguiente fase construye modelos predictivos a partir de estos hallazgos, según "
                "la metodología del proyecto:",
                markdown(siguiente),
                titulo="Siguiente paso · modelado",
            ), lg=4),
        ], className="fila"),
    ])
