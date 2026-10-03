"""Pestaña 03 · Objetivos: general, específicos y alcance de esta fase."""

import dash_bootstrap_components as dbc
from dash import html

from components.secciones import POR_SLUG
from components.ui import badge, card, encabezado_seccion, fila_kpis, interpretacion, kpi, markdown
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
                badge("Esta fase · EDA" if es_eda else "Fase de modelado",
                      "primaria" if es_eda else "neutra"),
                html.Span([" Ver: ", *enlaces], className="nota ms-2") if enlaces else None,
            ], className="mt-2"),
        ], className="objetivo-cuerpo"),
    ], className="objetivo")


def layout():
    """Estructura: KPIs → objetivo general y específicos + alcance."""
    general = libro.seccion("intro.md", "Objetivo General")
    especificos = libro.items_numerados(libro.seccion("intro.md", "Objetivos específicos"))
    cubiertos = [n for n, i in OBJETIVOS_FASE.items() if i["fase"] == "eda" and n <= len(especificos)]
    pendientes = [n for n in range(1, len(especificos) + 1) if n not in cubiertos]

    def _lista(nums):
        return ", ".join(map(str, nums[:-1])) + f" y {nums[-1]}" if len(nums) > 1 else str(nums[0])

    return html.Div([
        encabezado_seccion("objetivos", [f"{len(especificos)} objetivos específicos", "Fase EDA"]),
        fila_kpis([
            kpi("Dominios del sueño", str(len(VARIABLES_SUENO)), "Analizados por separado"),
            kpi("Objetivos específicos", str(len(especificos)), "Definidos en el proyecto"),
            kpi("Cubiertos en esta fase", str(len(cubiertos)), f"Objetivos {_lista(cubiertos)}"),
            kpi("Fase de modelado", str(len(pendientes)), f"Objetivos {_lista(pendientes)}"),
        ]),
        dbc.Row([
            dbc.Col([
                card(markdown(general), titulo="Objetivo general"),
                card(*[_objetivo(i, t) for i, t in enumerate(especificos, start=1)],
                     titulo="Objetivos específicos"),
            ], lg=8),
            dbc.Col(interpretacion(
                f"Esta fase corresponde al análisis exploratorio y cubre los objetivos "
                f"{_lista(cubiertos)}: caracterizar las distribuciones de pantalla, sueño y BDI-II; "
                "evaluar la asociación entre pantalla y cada dominio del sueño por separado; y "
                "comparar a chicas y chicos.",
                f"Los objetivos {_lista(pendientes)} (modelos de aprendizaje automático e importancia de "
                "variables) quedan para la fase de modelado, que partirá de los hallazgos del EDA.",
                titulo="Alcance de esta fase",
            ), lg=4),
        ], className="fila"),
    ])
