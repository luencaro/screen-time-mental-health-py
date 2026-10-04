"""Pestaña 05 · Calidad de datos: dimensiones, nulos, duplicados y descriptivos."""

import dash_bootstrap_components as dbc
from dash import html

from analysis import stats
from components.formato import entero, num
from components.ui import card, encabezado_seccion, fila_kpis, interpretacion, kpi, tabla
from data.load_data import VARIABLES_NUMERICAS, cargar_datos


def _dos(v):
    return num(v, 2)


def _texto_calidad(df, d) -> list[str]:
    """Interpretación del notebook sobre dimensiones, nulos y duplicados."""
    frases = [f"Tenemos un dataset con {entero(df.shape[0])} registros y {df.shape[1]} variables."]
    if d["nulos_totales"] == 0 and d["claves_duplicadas"] == 0:
        frases.append("El dataset se encuentra sin valores nulos ni duplicados en `subject_id`, lo que "
                      "significa que el conjunto de datos está listo para análisis.")
    else:
        frases.append(f"El dataset tiene {entero(d['nulos_totales'])} valores nulos y "
                      f"{entero(d['claves_duplicadas'])} duplicados en `subject_id`, que deben tratarse "
                      "antes del análisis.")
    return frases


def _texto_descriptivo(df, desc) -> list[str]:
    """Interpretación del notebook sobre la tabla descriptiva."""
    f = desc.set_index("variable")
    sueno, jetlag = f.loc["avg_sleep_hours"], f.loc["social_jetlag_hours"]
    return [
        f"Tenemos que `avg_sleep_hours` tiene una media de {_dos(sueno['media'])} h con asimetría "
        f"negativa (skew {_dos(sueno['asimetria'])}), reflejando un grupo que duerme muy poco y arrastra "
        f"la distribución hacia la izquierda, con un mínimo de {_dos(sueno['min'])} h que es clínicamente "
        "extremo y candidato a revisión como outlier; y que `social_jetlag_hours` presenta un mínimo "
        f"negativo ({_dos(jetlag['min'])}), un valor atípico que conviene decidir si tratar como dato "
        "válido o como posible error de captura antes de avanzar.",
        f"Además, varias variables ya muestran asimetría notable (`sleep_quality_index` con skew "
        f"{_dos(f.loc['sleep_quality_index', 'asimetria'])}, `est_leisure_screen_hours` con "
        f"{_dos(f.loc['est_leisure_screen_hours', 'asimetria'])}), lo que anticipa colas largas en las "
        "distribuciones y sugiere priorizar la mediana sobre la media al resumir estas variables en las "
        "siguientes secciones.",
    ]


def layout():
    """Estructura: KPIs → descriptivos + tipos y nulos → interpretación."""
    df = cargar_datos()
    q = stats.calidad(df)
    d = stats.duplicados(df)
    desc = stats.descriptivos(df, VARIABLES_NUMERICAS)

    tabla_desc = tabla(desc.to_dict("records"), [
        {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
        {"clave": "media", "titulo": "Media", "tipo": "numero", "formato": _dos},
        {"clave": "mediana", "titulo": "Mediana", "tipo": "numero", "formato": _dos},
        {"clave": "desv", "titulo": "Desv. estándar", "tipo": "numero", "formato": _dos},
        {"clave": "min", "titulo": "Mín.", "tipo": "numero", "formato": _dos},
        {"clave": "max", "titulo": "Máx.", "tipo": "numero", "formato": _dos},
        {"clave": "asimetria", "titulo": "Asimetría", "tipo": "numero", "formato": _dos},
    ]),
    tabla_tipos = tabla(
    q.to_dict("records"),
    [
        {
            "clave": "variable",
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
    ]
),
    

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
            ), lg=8),
            dbc.Col(card(tabla_tipos, titulo="Tipos, valores únicos y nulos", className="estirar"),
                    lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(interpretacion(*_texto_calidad(df, d), *_texto_descriptivo(df, desc)), lg=8),
            dbc.Col(card(
                html.P([
                    "Los valores extremos se revisan en detalle en la sección ",
                    html.A("Outliers", href="/outliers"), ". Aquí solo se verifica que estén dentro "
                    "del rango válido de cada escala.",
                ]),
                titulo="Siguiente paso",
            ), lg=4),
        ], className="fila"),
    ])
