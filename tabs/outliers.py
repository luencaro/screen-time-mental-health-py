"""Pestaña 10 · Outliers: detección por IQR y relevancia clínica."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, pct
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           tabla, var)
from data.load_data import VARIABLES_NUMERICAS, VARIABLES_PANTALLA, VARIABLES_SUENO, cargar_datos

PREDICTORES = VARIABLES_PANTALLA + VARIABLES_SUENO


def _dos(v):
    return num(v, 2)


UNIDAD = {"bdi_total": " puntos", "est_leisure_screen_hours": " h", "avg_sleep_hours": " h",
          "midsleep_weekend_hours": " h", "social_jetlag_hours": " h"}


def _lim(v: float) -> str:
    """Límite sin decimales si es entero (22), con dos si no (3,75)."""
    return num(v, 0) if float(v).is_integer() else _dos(v)


def _lista(nombres: list[str], conector: str = "y") -> str:
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + f" {conector} " + nombres[-1]


def _texto_iqr(df, iqr) -> str:
    """Descripción de la tabla IQR en el estilo del notebook."""
    t = iqr.sort_values("proporcion", ascending=False)
    con = [f for _, f in t.iterrows() if f["n"] > 0]
    sin = [f["variable"] for _, f in t.iterrows() if f["n"] == 0]

    def base(f):
        return f"n = {entero(f['n'])}; {pct(f['proporcion'], 2)}"

    def cola(f):
        u = UNIDAD.get(f["variable"], "")
        if f["n_bajo"] == 0:
            return f"> {_lim(f['lim_sup'])}{u}", ""
        if f["n_alto"] == 0:
            return f"< {_lim(f['lim_inf'])}{u}", ""
        return "", (f", donde {entero(f['n_bajo'])} casos estuvieron por debajo de {_lim(f['lim_inf'])}{u} "
                    f"(mínimo = {_dos(df[f['variable']].min())}{u}) y {entero(f['n_alto'])} por encima de "
                    f"{_lim(f['lim_sup'])}{u}")

    texto = ("Los valores atípicos se identificaron mediante el criterio de Tukey (valores por debajo de "
             "Q1 − 1,5·IQR o por encima de Q3 + 1,5·IQR).")
    if not con:
        return texto + " Ninguna variable presentó valores atípicos."
    top, siguientes, resto = con[0], con[1:3], con[3:]
    limite, detalle = cola(top)
    if top["n_bajo"] == 0:
        ubicacion = f", todos en la cola superior ({limite})"
    elif top["n_alto"] == 0:
        ubicacion = f", todos en la cola inferior ({limite})"
    else:
        ubicacion = detalle
    texto += f" La mayor proporción se observó en `{top['variable']}` ({base(top)}){ubicacion}."
    if top["variable"] == "bdi_total":
        texto += (" Esto es coherente con su distribución asimétrica positiva y corresponde a casos con "
                  "sintomatología depresiva severa, no a errores de medición.")
    if siguientes:
        partes, detalles = [], ""
        for f in siguientes:
            limite, detalle = cola(f)
            partes.append(f"`{f['variable']}` ({base(f)}{'; ' + limite if limite else ''})")
            detalles += detalle
        texto += f" Le siguieron {_lista(partes)}{detalles}."
    if resto or sin:
        frase = ""
        if resto:
            frase = _lista([f"`{f['variable']}` ({base(f)})" for f in resto])
            frase += " mostraron proporciones menores" if len(resto) > 1 else " mostró una proporción menor"
        if sin:
            nombres = _lista([f"`{v}`" for v in sin])
            verbo = "no presentaron" if len(sin) > 1 else "no presentó"
            frase += (", y " if frase else "") + f"{nombres} {verbo} valores atípicos"
        texto += " " + frase[0].upper() + frase[1:] + "."
    return texto


def _texto_concentracion(por_estado) -> str:
    """¿Se concentran los atípicos de sueño en el grupo con depresión?"""
    ancho = por_estado.pivot(index="variable", columns="depressed", values="proporcion")
    sueno = [v for v in VARIABLES_SUENO if v in ancho.index and ancho.loc[v].sum() > 0]
    if sueno and all(ancho.loc[v, 1] > ancho.loc[v, 0] for v in sueno):
        return "Los valores atípicos en las variables de sueño se concentraron en el grupo con depresión."
    return "Los valores atípicos en las variables de sueño no se concentraron en un solo grupo."


DECISION = [
    "Dado que los valores extremos son plausibles y parecen reflejar la señal clínica de interés, se "
    "decidió conservarlos.",
    "Para los modelos de clasificación se priorizaron métodos robustos a colas largas (árboles y "
    "ensambles). Para los modelos de regresión se consideraron transformaciones que reduzcan la "
    "asimetría de las distribuciones.",
]


def layout():
    """Estructura: KPIs → tabla IQR + justificación → outliers por estado + interpretación."""
    df = cargar_datos()
    iqr = stats.outliers_iqr(df, VARIABLES_NUMERICAS)
    por_estado = stats.outliers_por_estado(df, PREDICTORES)
    filas = stats.filas_con_outliers(df, PREDICTORES)
    bdi = iqr[iqr["variable"] == "bdi_total"].iloc[0]
    sin_outliers = iqr.loc[iqr["n"] == 0, "variable"].tolist()


    return html.Div([
        encabezado_seccion("outliers", ["Regla IQR · 1,5", f"{len(VARIABLES_NUMERICAS)} variables"]),
        fila_kpis([
            kpi("Filas con algún outlier", entero(filas["n"]),
                f"{pct(filas['proporcion'])} de la muestra (predictores)"),
            kpi("Deprimidos en esas filas", pct(filas["prevalencia"]),
                f"Frente a {pct(filas['prevalencia_resto'])} en el resto", "deprimido"),
            kpi("Outliers en bdi_total", entero(bdi["n"]),
                f"{pct(bdi['proporcion'])} · puntaje > {_dos(bdi['lim_sup'])}"),
            kpi("Variables sin outliers", entero(len(sin_outliers)),
                ", ".join(sin_outliers) if sin_outliers else "Todas presentan alguno"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Límites de Tukey: Q1 − 1,5·IQR y Q3 + 1,5·IQR.", className="card-subtitulo"),
                tabla(iqr.to_dict("records"), [
                    {"clave": "variable", "titulo": "Variable", "tipo": "variable"},
                    {"clave": "q1", "titulo": "Q1", "tipo": "numero", "formato": _dos},
                    {"clave": "q3", "titulo": "Q3", "tipo": "numero", "formato": _dos},
                    {"clave": "iqr", "titulo": "IQR", "tipo": "numero", "formato": _dos},
                    {"clave": "lim_inf", "titulo": "Lím. inf.", "tipo": "numero", "formato": _dos},
                    {"clave": "lim_sup", "titulo": "Lím. sup.", "tipo": "numero", "formato": _dos},
                    {"clave": "n_bajo", "titulo": "Bajo", "tipo": "numero", "formato": entero},
                    {"clave": "n_alto", "titulo": "Alto", "tipo": "numero", "formato": entero},
                    {"clave": "proporcion", "titulo": "% outliers", "tipo": "numero", "formato": pct},
                ]),
                titulo="Outliers por variable (IQR)",
            ), lg=8),
            dbc.Col(interpretacion(_texto_iqr(df, iqr)), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(
                html.P(["Predictores de pantalla y sueño. Se excluye ", var("bdi_total"),
                        " porque define el estado."], className="card-subtitulo"),
                grafico("ou-estado", 380, ancho_minimo=520),
                titulo="Proporción de outliers según estado depresivo",
            ), lg=8),
            dbc.Col([
                interpretacion(_texto_concentracion(por_estado)),
                interpretacion(*DECISION, titulo="Decisión"),
            ], lg=4),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera la figura al cambiar el tema."""

    @app.callback(Output("ou-estado", "figure"), Input("tema", "data"))
    def _figura(tema):
        df = cargar_datos()
        return figures.barras_outliers_estado(stats.outliers_por_estado(df, PREDICTORES), tema)
