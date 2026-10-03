"""Pestaña 05 · Calidad de datos: dimensiones, nulos, duplicados y descriptivos."""

import dash_bootstrap_components as dbc
from dash import html

from analysis import stats
from components.formato import entero, num
from components.ui import card, encabezado_seccion, fila_kpis, interpretacion, kpi, tabla
from data.load_data import VARIABLES_NUMERICAS, cargar_datos


def _dos(v):
    return num(v, 2)


def _comentarios_extremos(desc):
    """Frases sobre asimetría y valores extremos, calculadas desde los descriptivos."""
    frases = []
    asimetricas = desc[desc["asimetria"].abs() >= 1].sort_values("asimetria", ascending=False)
    for _, f in asimetricas.iterrows():
        cola = "derecha" if f["asimetria"] > 0 else "izquierda"
        frases.append(
            f"`{f['variable']}` presenta asimetría {num(f['asimetria'], 2)} (cola a la {cola}): "
            f"el 75 % de los valores está entre {_dos(f['min'])} y {_dos(f['p75'])}, "
            f"mientras que el máximo llega a {_dos(f['max'])}."
        )
    moderadas = desc[desc["asimetria"].abs().between(0.5, 1, inclusive="left")]
    if not moderadas.empty:
        lista = ", ".join(f"`{v}` ({num(a, 2)})" for v, a in zip(moderadas["variable"],
                                                                  moderadas["asimetria"]))
        frases.append(f"Con asimetría moderada: {lista}.")
    return frases


def layout():
    """Estructura: KPIs → descriptivos + tipos y nulos → interpretación."""
    df = cargar_datos()
    q = stats.calidad(df)
    d = stats.duplicados(df)
    desc = stats.descriptivos(df, VARIABLES_NUMERICAS)
    negativos = int((df["social_jetlag_hours"] < 0).sum())

    tabla_desc = tabla(desc.to_dict("records"), [
        {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
        {"clave": "media", "titulo": "Media", "tipo": "numero", "formato": _dos},
        {"clave": "mediana", "titulo": "Mediana", "tipo": "numero", "formato": _dos},
        {"clave": "desv", "titulo": "Desv. estándar", "tipo": "numero", "formato": _dos},
        {"clave": "min", "titulo": "Mín.", "tipo": "numero", "formato": _dos},
        {"clave": "max", "titulo": "Máx.", "tipo": "numero", "formato": _dos},
        {"clave": "asimetria", "titulo": "Asimetría", "tipo": "numero", "formato": _dos},
    ])
    tabla_tipos = tabla(q.to_dict("records"), [
        {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
        {"clave": "tipo", "titulo": "Tipo"},
        {"clave": "unicos", "titulo": "Únicos", "tipo": "numero", "formato": entero},
        {"clave": "nulos", "titulo": "Nulos", "tipo": "numero", "formato": entero},
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
            ), lg=8),
            dbc.Col(card(tabla_tipos, titulo="Tipos, valores únicos y nulos", className="estirar"),
                    lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(interpretacion(
                f"El dataset tiene {entero(df.shape[0])} filas y {df.shape[1]} columnas, sin valores "
                f"nulos ({entero(d['nulos_totales'])}) ni identificadores repetidos en `subject_id` "
                f"({entero(d['claves_duplicadas'])}). No se requiere imputación ni deduplicación.",
                *_comentarios_extremos(desc),
                f"`social_jetlag_hours` admite valores negativos ({entero(negativos)} casos, mínimo "
                f"{_dos(df['social_jetlag_hours'].min())} h): indican un punto medio del sueño más "
                "temprano en fin de semana que entre semana, no un error de registro.",
                f"`screen_time_index` toma valores entre {_dos(df['screen_time_index'].min())} y "
                f"{_dos(df['screen_time_index'].max())}, dentro de la escala 1–6 esperada; "
                f"`bdi_total` queda en {entero(df['bdi_total'].min())}–{entero(df['bdi_total'].max())}, "
                "dentro del rango 0–63 del instrumento.",
            ), lg=8),
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
