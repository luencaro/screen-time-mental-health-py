"""Pestaña 11 · Hallazgos: síntesis del EDA e implicaciones para el modelado."""

import dash_bootstrap_components as dbc
from dash import Input, Output, html

from analysis import figures, stats
from components.formato import entero, num, pct, texto_p
from components.ui import (card, encabezado_seccion, fila_kpis, grafico, interpretacion, kpi,
                           rico)
from data.load_data import (VARIABLES_NUMERICAS, VARIABLES_PANTALLA, VARIABLES_SUENO,
                            cargar_datos)

PREDICTORES = VARIABLES_PANTALLA + VARIABLES_SUENO


def _cifras(df) -> dict:
    """Reúne las cifras que alimentan la síntesis."""
    r = stats.resumen_muestra(df)
    con_bdi = stats.spearman_pares(df, PREDICTORES, ["bdi_total"]).set_index("x")
    pantalla_sueno = stats.spearman_pares(df, ["screen_time_index"], VARIABLES_SUENO).set_index("y")
    matriz = stats.matriz_spearman(df, VARIABLES_NUMERICAS)
    mw = stats.mann_whitney(df, ["bdi_total"]).iloc[0]
    return {
        "r": r,
        "con_bdi": con_bdi,
        "sueno_top": con_bdi.loc[VARIABLES_SUENO, "rho"].abs().idxmax(),
        "sueno_max": con_bdi.loc[VARIABLES_SUENO, "rho"].abs().max(),
        "pantalla_max": con_bdi.loc[VARIABLES_PANTALLA, "rho"].abs().max(),
        "pantalla_sueno": pantalla_sueno,
        "chi": stats.chi_cuadrado(df).iloc[0],
        "mw_bdi": mw,
        "outliers": stats.filas_con_outliers(df, PREDICTORES),
        "redundantes": stats.pares_redundantes(matriz, 0.7),
        "asimetria_bdi": df["bdi_total"].skew(),
    }


def _hallazgos(c) -> list:
    """Lista de hallazgos, cada uno con título en negrita y texto calculado."""
    r, cb, ps = c["r"], c["con_bdi"], c["pantalla_sueno"]
    top_ps = ps["rho"].abs().idxmax()
    razon = c["chi"]["razon_prevalencias"]
    veces = f"{'casi ' if razon < round(razon) else ''}{round(razon)} veces"
    items = [
        ("El sueño concentra más señal que el tiempo de pantalla."
         if c["sueno_max"] > c["pantalla_max"]
         else "El tiempo de pantalla muestra una asociación similar o mayor que el sueño.",
         f"`{c['sueno_top']}` es la variable con mayor asociación con `bdi_total` "
         f"(ρ = {num(cb.loc[c['sueno_top'], 'rho'], 3)}), seguida de `avg_sleep_hours` "
         f"(ρ = {num(cb.loc['avg_sleep_hours', 'rho'], 3)}). Las variables de pantalla no superan "
         f"|ρ| = {num(c['pantalla_max'], 3)}."),
        ("El tiempo de pantalla se asocia más con el sueño que con el puntaje depresivo."
         if abs(ps.loc[top_ps, "rho"]) > abs(cb.loc["screen_time_index", "rho"])
         else "El tiempo de pantalla se asocia con el sueño y con el puntaje depresivo.",
         f"`screen_time_index` alcanza ρ = {num(ps.loc[top_ps, 'rho'], 3)} con `{top_ps}` y "
         f"ρ = {num(ps.loc['avg_sleep_hours', 'rho'], 3)} con `avg_sleep_hours`, frente a "
         f"ρ = {num(cb.loc['screen_time_index', 'rho'], 3)} con `bdi_total`. El patrón es compatible "
         "con la hipótesis de mediación por sueño, aunque un análisis transversal no permite "
         "comprobarla."),
        (f"Las chicas presentan {veces} más casos sobre el corte clínico.",
         f"{pct(r['prevalencia_chicas'])} frente a {pct(r['prevalencia_chicos'])} "
         f"(razón {num(c['chi']['razon_prevalencias'], 2)}; χ² = {num(c['chi']['chi2'], 1)}, "
         f"{texto_p(c['chi']['p'])}). La mediana de `bdi_total` es {num(c['mw_bdi']['mediana_b'], 0)} "
         f"en chicas y {num(c['mw_bdi']['mediana_a'], 0)} en chicos."),
        ("Los outliers son clínicamente relevantes.",
         f"Entre los {entero(c['outliers']['n'])} adolescentes con algún valor atípico en pantalla o "
         f"sueño, {pct(c['outliers']['prevalencia'])} está sobre el corte clínico, frente a "
         f"{pct(c['outliers']['prevalencia_resto'])} en el resto. Se conservan."),
        ("La variable objetivo está desbalanceada y sesgada.",
         f"Solo {pct(r['prevalencia'])} supera el corte y `bdi_total` tiene asimetría "
         f"{num(c['asimetria_bdi'], 2)}, con mediana {num(r['mediana_bdi'], 0)} y media "
         f"{num(r['media_bdi'], 2)}."),
    ]
    return [html.Li([html.Strong(t), html.Br(), *rico(d)]) for t, d in items]


def _implicaciones(c) -> list[str]:
    r = c["r"]
    pares = " y ".join(f"`{f['x']}` – `{f['y']}`" for _, f in c["redundantes"].iterrows())
    frases = [
        f"Tratar el desbalance ({pct(r['prevalencia'])} positivos) con `class_weight` o SMOTE solo en "
        "entrenamiento, y evaluar con AUC-PR, recall y F1 de la clase positiva.",
        "Dar prioridad a los dominios del sueño, en particular a la calidad, como predictores; "
        "contrastar su importancia con la del tiempo de pantalla.",
        "Incluir `sex` como predictor y entrenar además modelos separados por sexo.",
        "Conservar los outliers y preferir escalado robusto o modelos basados en árboles.",
    ]
    if pares:
        frases.insert(2, f"Evitar pares redundantes en modelos lineales ({pares}): conservar una "
                         "variable de cada par o usar regularización.")
    return frases


def layout():
    """Estructura: KPIs → ρ con BDI-II + depresión por sexo → hallazgos + implicaciones."""
    df = cargar_datos()
    c = _cifras(df)
    r = c["r"]

    return html.Div([
        encabezado_seccion("hallazgos", [f"n = {entero(r['n'])}", "Síntesis del EDA"]),
        fila_kpis([
            kpi("ρ calidad del sueño · BDI-II", num(c["con_bdi"].loc["sleep_quality_index", "rho"], 3),
                "Mayor asociación observada"),
            kpi("ρ máxima de pantalla · BDI-II", num(c["pantalla_max"], 3),
                f"Efecto {stats.magnitud(c['pantalla_max'])}"),
            kpi("Razón chicas / chicos", f"{num(c['chi']['razon_prevalencias'], 2)}×",
                f"{pct(r['prevalencia_chicas'])} frente a {pct(r['prevalencia_chicos'])}"),
            kpi("Sobre el corte entre outliers", pct(c["outliers"]["prevalencia"]),
                f"Frente a {pct(c['outliers']['prevalencia_resto'])} en el resto", "deprimido"),
        ]),
        dbc.Row([
            dbc.Col(card(
                html.P("ρ de Spearman de cada predictor con el puntaje BDI-II.",
                       className="card-subtitulo"),
                grafico("ha-rho", 380),
                titulo="Sueño frente a pantalla",
            ), lg=8),
            dbc.Col(card(
                html.P("Porcentaje dentro de cada sexo.", className="card-subtitulo"),
                grafico("ha-sexo", 380),
                titulo="Estado depresivo por sexo",
                className="estirar",
            ), lg=4),
        ], className="fila"),
        dbc.Row([
            dbc.Col(card(html.Ul(_hallazgos(c), className="lista-hallazgos"),
                         titulo="Hallazgos principales"), lg=8),
            dbc.Col(interpretacion(
                html.Ul([html.Li(rico(f)) for f in _implicaciones(c)]),
                titulo="Implicaciones para el modelado",
            ), lg=4),
        ], className="fila"),
    ])


def register_callbacks(app):
    """Regenera las figuras al cambiar el tema."""

    @app.callback(Output("ha-rho", "figure"), Output("ha-sexo", "figure"), Input("tema", "data"))
    def _figuras(tema):
        df = cargar_datos()
        con_bdi = stats.spearman_pares(df, PREDICTORES, ["bdi_total"])
        return figures.barras_rho(con_bdi, tema), figures.barras_depresion_sexo(df, tema)
