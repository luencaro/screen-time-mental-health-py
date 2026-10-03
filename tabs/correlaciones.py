"""Pestaña 09 · Correlaciones de Spearman entre pantalla, sueño y BDI-II.

Las interpretaciones siguen el texto del notebook EDA, con las cifras
calculadas desde los datos.
"""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, valor_p
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           tabla)
from content import libro
from data.load_data import VARIABLES_NUMERICAS, VARIABLES_PANTALLA, VARIABLES_SUENO, cargar_datos

# Nombre de cada dominio del sueño en el texto
DOMINIO = {
    "sleep_quality_index": "la calidad del sueño",
    "avg_sleep_hours": "la duración de sueño",
    "midsleep_weekend_hours": "el cronotipo",
    "social_jetlag_hours": "el jet lag social",
}
# Qué implica una correlación positiva o negativa con el tiempo de pantalla
IMPLICA = {
    "sleep_quality_index": ("peor calidad de sueño", "mejor calidad de sueño"),
    "avg_sleep_hours": ("más horas de sueño", "menos horas de sueño"),
    "midsleep_weekend_hours": ("horarios de sueño más tardíos", "horarios de sueño más tempranos"),
    "social_jetlag_hours": ("mayor jet lag social", "menor jet lag social"),
}


def _columnas_rho(clave_var: str, titulo_var: str) -> list[dict]:
    return [
        {"clave": clave_var, "titulo": titulo_var, "tipo": "variable"},
        {"clave": "n", "titulo": "N", "tipo": "numero", "formato": entero},
        {"clave": "rho", "titulo": "Spearman r", "tipo": "numero", "formato": lambda v: num(v, 3)},
        {"clave": "p", "titulo": "Valor p", "tipo": "numero", "formato": valor_p},
        {"clave": "sig", "titulo": "Significativo (p < 0,05)"},
    ]


def _ordenar(t):
    """Orden del notebook: por |r| descendente, con columna de significancia."""
    t = t.assign(abs_rho=t["rho"].abs(), sig=["Sí" if p < 0.05 else "No" for p in t["p"]])
    return t.sort_values("abs_rho", ascending=False)


def _r(v: float) -> str:
    return f"r = {num(v, 3)}"


def _texto_matriz(con_bdi) -> str:
    top = con_bdi.iloc[0]
    return (f"`{top['x']}` es el predictor individual más fuerte de `bdi_total` "
            f"({_r(top['rho'])}).")


def _texto_pantalla_sueno(ps) -> str:
    n_sig = int((ps["p"] < 0.05).sum())
    alcance = ("los cuatro dominios del sueño" if n_sig == len(ps)
               else f"{n_sig} de los {len(ps)} dominios del sueño")
    filas = list(ps.itertuples())
    primero, *resto = filas
    texto = (f"La correlación de Spearman mostró asociaciones estadísticamente significativas entre el "
             f"índice de tiempo de pantalla y {alcance}. La relación más fuerte se observó con "
             f"`{primero.y}` ({_r(primero.rho)})")
    medios, ultimo = resto[:-1], resto[-1] if resto else None
    if medios:
        texto += ", seguida de " + " y ".join(f"`{f.y}` ({_r(f.rho)})" for f in medios)
    texto += "."
    if ultimo is not None:
        signo = "positiva" if ultimo.rho > 0 else "negativa"
        debil = f", pero {'muy débil' if abs(ultimo.rho) < 0.1 else 'débil'}" if abs(ultimo.rho) < 0.2 else ""
        texto += f" La relación con `{ultimo.y}` fue {signo}{debil} ({_r(ultimo.rho)})."
    principales = [IMPLICA[f.y][0 if f.rho > 0 else 1] for f in filas[:3]]
    texto += (" Estos resultados sugieren que un mayor tiempo de pantalla se relaciona principalmente "
              f"con {principales[0]}, {principales[1]} y {principales[2]}.")
    return texto


def _texto_sueno_bdi(sb, df) -> str:
    filas = list(sb.itertuples())
    primero, *resto = filas
    partes = [f"{DOMINIO[f.x]} ({_r(f.rho)}{', el más débil' if f is filas[-1] else ''})" for f in resto]
    texto = ("La asociación bivariada entre cada dominio de sueño y la severidad de síntomas depresivos "
             f"(BDI-II total) confirmó a `{primero.x}` como el predictor más fuerte ({_r(primero.rho)}), "
             f"seguido de {', '.join(partes[:-1])} y {partes[-1]}.")
    # ¿Coincide el orden con el obtenido sobre la variable binarizada (Mann-Whitney por estado)?
    mw = stats.mann_whitney(df, VARIABLES_SUENO, grupo="depressed", niveles=(1, 0))
    orden_binario = list(mw.assign(a=mw["r_biserial"].abs()).sort_values("a", ascending=False)["variable"])
    if orden_binario == [f.x for f in filas]:
        texto += (" Este orden de magnitud replica consistentemente los resultados obtenidos con la "
                  "variable de depresión binarizada y coincide con la jerarquía de importancia reportada "
                  f"por {libro.autor_anio('hokby2025adolescents').replace(', ', ' (')}) en su modelo de "
                  "mediación SEM.")
    if primero.x == "sleep_quality_index":
        texto += (" Esta convergencia sugiere que la calidad del sueño constituye la vía más robusta y "
                  "consistente entre sueño y depresión en esta población.")
    return texto


def layout():
    """Estructura: KPIs → heatmap + r con BDI-II → tablas por dominio con su interpretación."""
    df = cargar_datos()
    matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
    pantalla_sueno = _ordenar(stats.spearman_pares(df, ["screen_time_index"], VARIABLES_SUENO))
    sueno_bdi = _ordenar(stats.spearman_pares(df, VARIABLES_SUENO, ["bdi_total"]))
    con_bdi = _ordenar(stats.spearman_pares(df, VARIABLES_PANTALLA + VARIABLES_SUENO, ["bdi_total"]))
    rho = matriz["bdi_total"]

    return html.Div([
        encabezado_seccion("correlaciones", [f"n = {entero(len(df))}", "Spearman"]),
        fila_kpis([
            kpi("r calidad del sueño · BDI-II", num(rho["sleep_quality_index"], 3),
                f"Efecto {stats.magnitud(rho['sleep_quality_index'])}"),
            kpi("r horas de sueño · BDI-II", num(rho["avg_sleep_hours"], 3),
                f"Efecto {stats.magnitud(rho['avg_sleep_hours'])}"),
            kpi("r cronotipo · BDI-II", num(rho["midsleep_weekend_hours"], 3),
                f"Efecto {stats.magnitud(rho['midsleep_weekend_hours'])}"),
            kpi("r tiempo de pantalla · BDI-II", num(rho["screen_time_index"], 3),
                f"Efecto {stats.magnitud(rho['screen_time_index'])}"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("Triángulo inferior sin diagonal. Durazno: positiva; azul: negativa.",
                       className="card-subtitulo"),
                grafico("co-heatmap", 470, ancho_minimo=640, animar=False),
                titulo="Matriz de correlación de Spearman",
            ), lg=8),
            dbc.Col([
                card(
                    html.P("Correlación de cada predictor con el puntaje BDI-II.",
                           className="card-subtitulo"),
                    grafico("co-rho-bdi", 380),
                    titulo="Asociación con bdi_total",
                ),
                interpretacion(_texto_matriz(con_bdi)),
            ], lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col([
                card(
                    html.P("Objetivo específico 2: cada dominio por separado.", className="card-subtitulo"),
                    tabla(pantalla_sueno.to_dict("records"), _columnas_rho("y", "Dominio de sueño")),
                    titulo="screen_time_index frente a los dominios del sueño",
                ),
                interpretacion(_texto_pantalla_sueno(pantalla_sueno)),
            ], lg=6),
            dbc.Col([
                card(
                    html.P("Asociación de cada dominio con el puntaje BDI-II.", className="card-subtitulo"),
                    tabla(sueno_bdi.to_dict("records"), _columnas_rho("x", "Dominio de sueño")),
                    titulo="Dominios del sueño frente a bdi_total",
                ),
                interpretacion(_texto_sueno_bdi(sueno_bdi, df)),
            ], lg=6),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(
        Output("co-heatmap", "figure"),
        Output("co-rho-bdi", "figure"),
        Input("tema", "data"),
    )
    def _figuras(tema):
        df = cargar_datos()
        matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
        con_bdi = stats.spearman_pares(df, VARIABLES_PANTALLA + VARIABLES_SUENO, ["bdi_total"])
        return figures.heatmap_spearman(matriz, tema), figures.barras_rho(con_bdi, tema)
