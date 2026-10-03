"""Pestaña 03 · Objetivos: objetivo general y objetivos específicos."""

import dash_bootstrap_components as dbc
from dash import html

from components.secciones import POR_SLUG
from components.ui import badge, card, encabezado_seccion, fila_kpis, kpi, markdown
from content import libro
from content.fichas import OBJETIVOS_FASE
from data.load_data import VARIABLES_SUENO


def _objetivo(numero: int, texto: str) -> html.Div:
    """Objetivo específico con su fase y enlaces a las secciones que lo atienden."""
    info = OBJETIVOS_FASE.get(numero, {"fase": "modelado", "secciones": []})
    es_eda = info["fase"] == "eda"
    enlaces = []
    for slug in info["secciones"]:
        enlaces += [", " if enlaces else "", html.A(POR_SLUG[slug]["nombre"], href=f"/{slug}")]
    return html.Div([
        html.Span(f"{numero:02d}", className="objetivo-num"),
        html.Div([
            markdown(texto),
            html.Div([
                badge("EDA" if es_eda else "Fase de modelado", "primaria"),
                html.Span([" Ver: ", *enlaces], className="nota ms-2") if enlaces else None,
            ], className="mt-2"),
        ], className="objetivo-cuerpo"),
    ], className="objetivo")


def layout():
    """Estructura: KPIs → objetivo general → objetivos específicos."""
    general = libro.seccion("intro.md", "Objetivo General")
    especificos = libro.items_numerados(libro.seccion("intro.md", "Objetivos específicos"))

    return html.Div([
        encabezado_seccion("objetivos", [f"{len(especificos)} objetivos específicos", "Fase EDA"]),
        fila_kpis([
            kpi("Dominios del sueño", str(len(VARIABLES_SUENO)), "Analizados por separado"),
            kpi("Objetivos específicos", str(len(especificos)), "Definidos en el proyecto"),
        ]),
        dbc.Row(dbc.Col([
            card(markdown(general), titulo="Objetivo general"),
            card(*[_objetivo(i, t) for i, t in enumerate(especificos, start=1)],
                 titulo="Objetivos específicos"),
        ]), className="fila"),
    ])
