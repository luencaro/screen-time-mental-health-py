"""Pestaña 07 · Análisis por sexo: prevalencia, chi-cuadrado y Mann-Whitney."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, estadistico, num, pct, valor_p
from components.ui import card, encabezado_seccion, fila_kpis, grafico, kpi, tabla, var
from data.load_data import ETIQUETAS_SEXO, VARIABLES_NUMERICAS, cargar_datos

# Orden del notebook: chicas como primer grupo (r < 0 → valores mayores en chicas)
NIVELES = ("Girl", "Boy")


def _tabla_contingencia(df):
    """Tabla sexo × estado con % de deprimidos dentro de cada fila."""
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
        {"clave": "bajo", "titulo": "No deprimido", "tipo": "numero", "formato": entero},
        {"clave": "sobre", "titulo": "Deprimido", "tipo": "numero", "formato": entero},
        {"clave": "total", "titulo": "Total", "tipo": "numero", "formato": entero},
        {"clave": "pct", "titulo": "% deprimido", "tipo": "numero", "formato": pct},
    ])


def _tabla_chi(chi):
    filas = [
        {"medida": "χ²", "valor": num(chi["chi2"], 2)},
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
        {"clave": "mediana_a", "titulo": "Mediana chicas", "tipo": "numero", "formato": lambda v: num(v, 2)},
        {"clave": "mediana_b", "titulo": "Mediana chicos", "tipo": "numero", "formato": lambda v: num(v, 2)},
        {"clave": "u", "titulo": "U", "tipo": "numero", "formato": estadistico},
        {"clave": "p", "titulo": "Valor p", "tipo": "numero", "formato": valor_p},
        {"clave": "r_biserial", "titulo": "r rango-biserial", "tipo": "numero", "formato": lambda v: num(v, 3)},
    ])


def layout():
    """Estructura: KPIs → depresión por sexo + composición → boxplot + chi-cuadrado → Mann-Whitney."""
    df = cargar_datos()
    r = stats.resumen_muestra(df)
    chi = stats.chi_cuadrado(df).iloc[0]
    mw = stats.mann_whitney(df, VARIABLES_NUMERICAS, niveles=NIVELES).sort_values("p")
    n_dep = df.groupby("sex")["depressed"].sum()

    return html.Div([
        encabezado_seccion("analisis_sexo", [f"n = {entero(r['n'])}", "sex × depressed"]),
        fila_kpis([
            kpi("Chicas", entero(r["n_chicas"]), f"{pct(r['prop_chicas'])} de la muestra"),
            kpi("Chicos", entero(r["n_chicos"]), f"{pct(r['prop_chicos'])} de la muestra"),
            kpi("Deprimidas · chicas", pct(r["prevalencia_chicas"]),
                f"{entero(n_dep['Girl'])} de {entero(r['n_chicas'])} chicas", "deprimido"),
            kpi("Deprimidos · chicos", pct(r["prevalencia_chicos"]),
                f"{entero(n_dep['Boy'])} de {entero(r['n_chicos'])} chicos", "deprimido"),
        ]),
        dbc.Row([
            dbc.Col([
                card(
                    html.P("Porcentaje dentro de cada sexo.", className="card-subtitulo"),
                    grafico("sx-depresion", 360),
                    titulo="Estado depresivo por sexo",
                ),
            ], lg=8),
            dbc.Col([
                card(
                    html.P(["Distribución de ", var("sex")], className="card-subtitulo"),
                    grafico("sx-distribucion", 300, flexible=True),
                    titulo="Composición de la muestra",
                    className="estirar grafico-flexible",
                ),
            ], lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col([
                card(
                    html.P(["Distribución de ", var("bdi_total"), " con el corte clínico."],
                           className="card-subtitulo"),
                    grafico("sx-boxplot", 420, animar=False, flexible=True),
                    titulo="Puntaje BDI-II por sexo",
                    className="estirar grafico-flexible",
                ),
            ], lg=8),
            dbc.Col([
                card(_tabla_contingencia(df), titulo="Tabla de contingencia"),
                card(_tabla_chi(chi), titulo="Test chi-cuadrado: estado depresivo vs. sexo"),
            ], lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Mann-Whitney U bilateral. r negativo: valores mayores en las chicas.",
                       className="card-subtitulo"),
                _tabla_mann_whitney(mw),
                titulo="Mann-Whitney U: variables numéricas por sexo",
                className="estirar",
            ), lg=12),
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
