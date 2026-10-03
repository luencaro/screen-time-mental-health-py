"""Pestaña 07 · Análisis por sexo: prevalencia, chi-cuadrado y Mann-Whitney."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, pct, texto_p, valor_p
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           tabla, var)
from data.load_data import ETIQUETAS_SEXO, VARIABLES_NUMERICAS, cargar_datos


def _tabla_contingencia(df):
    """Tabla sexo × estado con % sobre el corte dentro de cada fila."""
    ct = stats.tabla_contingencia(df)
    filas = []
    for sexo in ["Girl", "Boy", "Total"]:
        fila = ct.loc[sexo]
        filas.append({
            "grupo": ETIQUETAS_SEXO.get(sexo, "Total"),
            "bajo": fila[0], "sobre": fila[1], "total": fila["Total"],
            "pct": fila[1] / fila["Total"],
            "_clase": "resaltada" if sexo == "Total" else None,
        })
    return tabla(filas, [
        {"clave": "grupo", "titulo": "Grupo"},
        {"clave": "bajo", "titulo": "Bajo el corte", "tipo": "numero", "formato": entero},
        {"clave": "sobre", "titulo": "Sobre el corte", "tipo": "numero", "formato": entero},
        {"clave": "total", "titulo": "Total", "tipo": "numero", "formato": entero},
        {"clave": "pct", "titulo": "% sobre", "tipo": "numero", "formato": pct},
    ])


def _tabla_chi(chi):
    filas = [
        {"medida": "χ² (sin corrección)", "valor": num(chi["chi2"], 2)},
        {"medida": "Grados de libertad", "valor": entero(chi["gl"])},
        {"medida": "Valor p", "valor": valor_p(chi["p"])},
        {"medida": "V de Cramér", "valor": num(chi["v_cramer"], 3)},
        {"medida": "Razón de prevalencias (chicas / chicos)", "valor": num(chi["razon_prevalencias"], 2)},
        {"medida": "Odds ratio (chicas / chicos)", "valor": num(chi["odds_ratio"], 2)},
    ]
    return tabla(filas, [
        {"clave": "medida", "titulo": "Prueba de independencia"},
        {"clave": "valor", "titulo": "Valor", "tipo": "numero"},
    ])


def _tabla_mann_whitney(mw):
    return tabla(mw.to_dict("records"), [
        {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
        {"clave": "mediana_a", "titulo": "Mediana chicos", "tipo": "numero", "formato": lambda v: num(v, 2)},
        {"clave": "mediana_b", "titulo": "Mediana chicas", "tipo": "numero", "formato": lambda v: num(v, 2)},
        {"clave": "u", "titulo": "U", "tipo": "numero", "formato": entero},
        {"clave": "p", "titulo": "Valor p", "tipo": "numero", "formato": valor_p},
        {"clave": "r_biserial", "titulo": "r biserial", "tipo": "numero", "formato": lambda v: num(v, 3)},
    ])


def _texto_mann_whitney(mw):
    """Describe las diferencias por sexo de mayor tamaño de efecto."""
    predictores = mw[mw["variable"] != "bdi_total"].copy()
    predictores["abs_r"] = predictores["r_biserial"].abs()
    top = predictores.sort_values("abs_r", ascending=False).head(2)
    partes = []
    for _, f in top.iterrows():
        sentido = "más altos en chicas" if f["r_biserial"] > 0 else "más altos en chicos"
        partes.append(f"`{f['variable']}` (r = {num(f['r_biserial'], 3)}, valores {sentido})")
    no_sig = predictores.loc[predictores["p"] >= 0.05, "variable"].tolist()
    texto = ("Entre las variables de pantalla y sueño, las diferencias de mayor tamaño se observan en "
             + " y ".join(partes) + ".")
    if no_sig:
        texto += " Sin diferencia significativa (p ≥ 0,05): " + ", ".join(f"`{v}`" for v in no_sig) + "."
    else:
        texto += (" Todas las comparaciones son significativas (p < 0,05); con muestras de este "
                  "tamaño, el tamaño de efecto r es más informativo que el valor p.")
    return texto


def layout():
    """Estructura: KPIs → barras de depresión + distribución → boxplot + contingencia → pruebas."""
    df = cargar_datos()
    r = stats.resumen_muestra(df)
    chi = stats.chi_cuadrado(df).iloc[0]
    mw = stats.mann_whitney(df, VARIABLES_NUMERICAS)
    bdi = mw[mw["variable"] == "bdi_total"].iloc[0]
    n_dep = df.groupby("sex")["depressed"].sum()

    return html.Div([
        encabezado_seccion("analisis_sexo", [f"n = {entero(r['n'])}", "sex × depressed"]),
        fila_kpis([
            kpi("Chicas", entero(r["n_chicas"]), f"{pct(r['prop_chicas'])} de la muestra"),
            kpi("Chicos", entero(r["n_chicos"]), f"{pct(r['prop_chicos'])} de la muestra"),
            kpi("Sobre el corte · chicas", pct(r["prevalencia_chicas"]),
                f"{entero(n_dep['Girl'])} de {entero(r['n_chicas'])} chicas", "deprimido"),
            kpi("Sobre el corte · chicos", pct(r["prevalencia_chicos"]),
                f"{entero(n_dep['Boy'])} de {entero(r['n_chicos'])} chicos", "deprimido"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Porcentaje dentro de cada sexo.", className="card-subtitulo"),
                grafico("sx-depresion", 360),
                titulo="Estado depresivo por sexo",
            ), lg=8),
            dbc.Col(card(
                html.P(["Distribución de ", var("sex")], className="card-subtitulo"),
                grafico("sx-distribucion", 360),
                titulo="Composición de la muestra",
                className="estirar",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P(["Distribución de ", var("bdi_total"), " con el corte clínico."],
                       className="card-subtitulo"),
                grafico("sx-boxplot", 520),
                titulo="Puntaje BDI-II por sexo",
            ), lg=8),
            dbc.Col([
                card(_tabla_contingencia(df), titulo="Tabla de contingencia"),
                card(_tabla_chi(chi), titulo="Chi-cuadrado", className="estirar"),
            ], lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Mann-Whitney U bilateral. r biserial positivo: valores más altos en chicas.",
                       className="card-subtitulo"),
                _tabla_mann_whitney(mw),
                titulo="Comparación por sexo · pantalla, sueño y BDI-II",
            ), lg=8),
            dbc.Col(interpretacion(
                f"La proporción de adolescentes con puntaje sobre el corte clínico es "
                f"{pct(r['prevalencia_chicas'])} en chicas y {pct(r['prevalencia_chicos'])} en chicos, "
                f"una razón de {num(chi['razon_prevalencias'], 2)}. La asociación entre sexo y estado "
                f"es significativa (χ² = {num(chi['chi2'], 1)}, gl = {entero(chi['gl'])}, "
                f"{texto_p(chi['p'])}) con un tamaño de efecto {stats.magnitud(chi['v_cramer'])} "
                f"(V de Cramér = {num(chi['v_cramer'], 3)}).",
                f"La mediana de `bdi_total` es {num(bdi['mediana_b'], 0)} en chicas y "
                f"{num(bdi['mediana_a'], 0)} en chicos (r = {num(bdi['r_biserial'], 3)}).",
                _texto_mann_whitney(mw),
            ), lg=4),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(
        Output("sx-depresion", "figure"),
        Output("sx-distribucion", "figure"),
        Output("sx-boxplot", "figure"),
        Input("tema", "data"),
    )
    def _figuras(tema):
        df = cargar_datos()
        return (figures.barras_depresion_sexo(df, tema), figures.barras_sexo(df, tema),
                figures.boxplot_por_sexo(df, "bdi_total", tema))
