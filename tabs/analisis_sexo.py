"""Pestaña 07 · Análisis por sexo: prevalencia, chi-cuadrado y Mann-Whitney.

Las interpretaciones siguen el texto del notebook EDA, con las cifras
calculadas desde los datos.
"""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, estadistico, num, pct, texto_p, valor_p
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           rico, tabla, var)
from content import libro
from data.load_data import CORTE_BDI, ETIQUETAS_SEXO, VARIABLES_NUMERICAS, cargar_datos

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


def _veces(razon: float) -> str:
    """'casi 3 veces' cuando la razón queda por debajo del entero más cercano."""
    return f"{'casi ' if razon < round(razon) else ''}{round(razon)} veces"


def _texto_composicion(r) -> str:
    balanceada = abs(r["prop_chicos"] - r["prop_chicas"]) < 0.05
    return (f"Tenemos que la muestra está {'prácticamente balanceada' if balanceada else 'desbalanceada'} "
            f"por género: un {pct(r['prop_chicos'])} de los adolescentes son chicos y un "
            f"{pct(r['prop_chicas'])} son chicas.")


def _texto_prevalencia(r, chi) -> str:
    return (f"Las chicas tienen una tasa de depresión de {pct(r['prevalencia_chicas'])}, "
            f"{_veces(chi['razon_prevalencias'])} más que los chicos, que es "
            f"{pct(r['prevalencia_chicos'])}. Esto es un indicio de que `sex` oculta una señal fuerte "
            "para predicción.")


def _texto_chi(chi) -> str:
    if chi["p"] >= 0.05:
        return (f"El test de independencia chi-cuadrado no mostró una asociación significativa entre el "
                f"sexo y el estado depresivo ({texto_p(chi['p'])}, {entero(chi['gl'])} grado de libertad).")
    gl = entero(chi["gl"])
    return (f"El test de independencia chi-cuadrado mostró una asociación significativa entre el sexo y "
            f"el estado depresivo ({texto_p(chi['p'])}, {gl} grado{'s' if gl != '1' else ''} de libertad). "
            "La prevalencia de depresión fue mayor en el grupo de chicas que en el de chicos. Este "
            "resultado es coherente con lo encontrado en la literatura revisada "
            f"({libro.autor_anio('hokby2025adolescents')}).")


def _texto_boxplot(df) -> str:
    q = df.groupby("sex")["bdi_total"].quantile([0.25, 0.75]).unstack()
    iqr = q[0.75] - q[0.25]
    med = df.groupby("sex")["bdi_total"].median()
    mas_ancha = " y es visiblemente más ancha" if iqr["Girl"] > iqr["Boy"] else ""
    return (f"La caja de las chicas está desplazada hacia arriba{mas_ancha}. Ambos grupos muestran la "
            "misma cola larga de outliers hacia arriba, pero la mediana de las chicas "
            f"({num(med['Girl'], 0)}) ya está bastante más cerca del corte clínico ({CORTE_BDI}) que la "
            f"de los chicos ({num(med['Boy'], 0)}).")


def _bloque_mann_whitney(df, mw):
    """Interpretación de la prueba U, como en el notebook: método, lista y conclusión."""
    n = df["sex"].value_counts()
    sig = mw[mw["p"] < 0.05]
    no_sig = mw[mw["p"] >= 0.05]
    items = [
        html.Li(rico(
            f"`{f['variable']}`: Mdn = {num(f['mediana_a'], 2)} (chicas) vs {num(f['mediana_b'], 2)} "
            f"(chicos), U = {estadistico(f['u'])}, {texto_p(f['p'])}, r = {num(f['r_biserial'], 3)}."
        ))
        for _, f in sig.iterrows()
    ]
    mayor = mw.loc[mw["r_biserial"].abs().idxmax()]
    sentido = "mayor en las chicas que en los chicos" if mayor["r_biserial"] < 0 else \
        "mayor en los chicos que en las chicas"
    cierre = (f"La diferencia más relevante se observó en `{mayor['variable']}`, cuya mediana fue "
              f"{sentido}, con un tamaño de efecto {stats.magnitud(mayor['r_biserial'])} "
              f"(r = {num(mayor['r_biserial'], 3)}).")
    if not no_sig.empty:
        nombres = [f"`{v}`" for v in no_sig["variable"]]
        lista = nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " ni en " + nombres[-1]
        cierre += f" En cambio, no se encontraron diferencias significativas entre grupos en {lista}."
    return interpretacion(
        f"Se aplicó la prueba U de Mann-Whitney (bilateral, α = 0,05) para comparar chicas "
        f"(n = {entero(n['Girl'])}) y chicos (n = {entero(n['Boy'])}). Al ser una prueba no paramétrica "
        "basada en rangos, no tiene grados de libertad; se reporta el estadístico U, el valor p y el "
        "tamaño del efecto r rango-biserial (r negativo indica valores mayores en las chicas; r "
        "positivo, mayores en los chicos).",
        html.P("La prueba mostró diferencias significativas por sexo en:") if items else None,
        html.Ul(items) if items else None,
        cierre,
    )


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
                interpretacion(_texto_prevalencia(r, chi)),
            ], lg=8),
            dbc.Col([
                card(
                    html.P(["Distribución de ", var("sex")], className="card-subtitulo"),
                    grafico("sx-distribucion", 300, flexible=True),
                    titulo="Composición de la muestra",
                    className="estirar grafico-flexible",
                ),
                interpretacion(_texto_composicion(r)),
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
                interpretacion(_texto_boxplot(df)),
            ], lg=8),
            dbc.Col([
                card(_tabla_contingencia(df), titulo="Tabla de contingencia"),
                card(_tabla_chi(chi), titulo="Test chi-cuadrado: estado depresivo vs. sexo"),
                interpretacion(_texto_chi(chi)),
            ], lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P("Mann-Whitney U bilateral. r negativo: valores mayores en las chicas.",
                       className="card-subtitulo"),
                _tabla_mann_whitney(mw),
                titulo="Mann-Whitney U: variables numéricas por sexo",
                className="estirar",
            ), lg=8),
            dbc.Col(_bloque_mann_whitney(df, mw), lg=4),
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
