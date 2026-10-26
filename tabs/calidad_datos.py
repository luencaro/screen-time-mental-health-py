"""Pestaña 05 · Calidad de datos: dimensiones, nulos, duplicados y descriptivos."""

import dash_bootstrap_components as dbc
from dash import html

from analysis import stats
from components.formato import entero, num
from components.ui import card, encabezado_seccion, fila_kpis, kpi, tabla
from data.load_data import VARIABLES_NUMERICAS, cargar_datos


def _dos(v):
    return num(v, 2)


def layout():
    """Estructura: KPIs → descriptivos + tipos y nulos."""
    df = cargar_datos()
    q = stats.calidad(df)
    d = stats.duplicados(df)
    desc = stats.descriptivos(df, VARIABLES_NUMERICAS)

    tabla_desc = tabla(desc.to_dict("records"),[
        {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
        {"clave": "N", "titulo": "N", "tipo": "numero", "formato": entero},
        {"clave": "media", "titulo": "Media", "tipo": "numero", "formato": _dos},
        {"clave": "mediana", "titulo": "Mediana", "tipo": "numero", "formato": _dos},
        {"clave": "desv", "titulo": "SD", "tipo": "numero", "formato": _dos},
        {"clave": "min", "titulo": "Mín.", "tipo": "numero", "formato": _dos},
        {"clave": "p25", "titulo": "Q1", "tipo": "numero", "formato": _dos},
        {"clave": "p75", "titulo": "Q3", "tipo": "numero", "formato": _dos},
        {"clave": "max", "titulo": "Máx.", "tipo": "numero", "formato": _dos},
    ]
)
    tabla_tipos = tabla(q.to_dict("records"),
    [{       "clave": "variable",
            "titulo": "Variable",
            "tipo": "variable"
        },

        {
            "clave": "tipo",
            "titulo": "Tipo"
        },

        {
            "clave": "unicos",
            "titulo": "Únicos",
            "tipo": "numero",
            "formato": entero
        },

        {
            "clave": "pct_unicos",
            "titulo": "% únicos",
            "tipo": "numero",
            "formato": _dos
        },

        {
            "clave": "nulos",
            "titulo": "Nulos",
            "tipo": "numero",
            "formato": entero
        },

        {
            "clave": "completitud",
            "titulo": "Completitud",
            "tipo": "numero",
            "formato": lambda x: f"{x:.1f}%"
        },
    ])
    

    return html.Div([
        encabezado_seccion("calidad_datos", [f"{entero(df.shape[0])} filas", f"{df.shape[1]} variables"]),
        fila_kpis([
            kpi("Filas", entero(df.shape[0]), "Un registro por adolescente"),
            kpi("Columnas", entero(df.shape[1]), f"{len(VARIABLES_NUMERICAS)} numéricas de análisis"),
            kpi("Valores nulos", entero(d["nulos_totales"]), "En todas las columnas"),
            kpi("Duplicados en subject_id", entero(d["claves_duplicadas"]),
                f"{entero(d['filas_duplicadas'])} filas duplicadas completas"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Variables numéricas de análisis; subject_id y depressed se excluyen.",
                       className="card-subtitulo"),
                tabla_desc,
                titulo="Estadística descriptiva",
                className="estirar",
            ), lg=7),
            dbc.Col(card(tabla_tipos, titulo="Tipos, valores únicos y nulos", className="estirar"),
                    lg=5),
        ], className="fila"),
    ])
