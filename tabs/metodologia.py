"""Pestaña 04 · Metodología: dataset, variables, operacionalización y flujo del EDA."""

import dash_bootstrap_components as dbc
from dash import html

from components.formato import entero
from components.secciones import secciones_de
from components.ui import card, encabezado_seccion, fila_kpis, flujo, kpi, markdown, tabla
from content import libro
from content.fichas import OPERACIONALIZACION
from data.load_data import VARIABLES_PANTALLA, VARIABLES_SUENO, cargar_datos


def _pasos_eda() -> list[dict]:
    """Pasos del flujo del EDA a partir del registro de secciones."""
    pasos = [
        {"numero": str(i), "titulo": s["nombre"], "descripcion": s["descripcion"],
         "href": f"/{s['slug']}"}
        for i, s in enumerate(secciones_de("EDA"), start=1)
    ]
    pasos.append({"numero": "→", "titulo": "Hallazgos y modelado",
                  "descripcion": "Síntesis de resultados e insumos para la fase de modelado",
                  "href": "/hallazgos"})
    return pasos


def layout():
    """Estructura: KPIs → descripción y tabla de variables + flujo → operacionalización."""
    df = cargar_datos()
    descripcion = libro.seccion("methodology.md", "Descripción del dataset")
    eda = libro.seccion("methodology.md", "EDA")
    roles = [f["rol"] for f in OPERACIONALIZACION]

    return html.Div([
        encabezado_seccion("metodologia", ["Kaggle", f"{df.shape[1]} variables"]),
        fila_kpis([
            kpi("Registros", entero(len(df)), "Un adolescente por fila"),
            kpi("Variables", entero(df.shape[1]),
                f"{sum(r.startswith('Objetivo') for r in roles)} objetivos · 1 identificador"),
            kpi("Predictores continuos", entero(sum(r == "Predictor" for r in roles)),
                f"{len(VARIABLES_PANTALLA)} de pantalla · {len(VARIABLES_SUENO)} de sueño"),
            kpi("Pasos del EDA", entero(len(secciones_de("EDA"))), "Ver diagrama de flujo"),
        ]),
        dbc.Row([
            dbc.Col([
                card(markdown(descripcion), titulo="Descripción del dataset"),
                card(markdown(eda), titulo="Análisis exploratorio"),
            ], lg=8),
            dbc.Col(card(flujo(_pasos_eda()), titulo="Flujo del EDA", className="estirar"), lg=4),
        ], className="fila"),
        dbc.Row(dbc.Col(card(
            tabla(OPERACIONALIZACION, [
                {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
                {"clave": "dimension", "titulo": "Dimensión"},
                {"clave": "tipo", "titulo": "Tipo"},
                {"clave": "escala", "titulo": "Escala / unidad"},
                {"clave": "indicador", "titulo": "Indicador"},
                {"clave": "rol", "titulo": "Rol"},
            ]),
            titulo="Operacionalización de variables",
        )), className="fila"),
    ])
