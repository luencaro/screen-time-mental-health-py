"""Dashboard del EDA: tiempo de pantalla, sueño y síntomas depresivos.

Ejecutar desde la raíz del proyecto:  python app.py  →  http://127.0.0.1:8050
"""

import importlib

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, clientside_callback, dcc, html

from components.secciones import BLOQUES, POR_SLUG, SECCIONES, secciones_de, slug_desde_ruta
from components.ui import aviso_sin_datos, encabezado_seccion
from data.load_data import URL_DATASET, DatosNoDisponibles

# Una pestaña por sección; cada módulo expone layout() y, si aplica, register_callbacks(app)
PESTANAS = {s["slug"]: importlib.import_module(f"tabs.{s['slug']}") for s in SECCIONES}

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="EDA · Pantallas, sueño y depresión",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


# ── Header ─────────────────────────────────────────────────────────────
def _item_menu(seccion: dict) -> dbc.DropdownMenuItem:
    """Ítem del desplegable: número (mono), nombre y descripción de una línea."""
    return dbc.DropdownMenuItem(
        html.Div(
            [
                html.Span(seccion["numero"], className="menu-num"),
                html.Div([
                    html.Div(seccion["nombre"], className="menu-nombre"),
                    html.Div(seccion["descripcion"], className="menu-desc"),
                ]),
            ],
            className="menu-item",
        ),
        href=f"/{seccion['slug']}",
        id=f"nav-item-{seccion['slug']}",
    )


def _menu_bloque(bloque: str) -> dbc.DropdownMenu:
    return dbc.DropdownMenu(
        [_item_menu(s) for s in secciones_de(bloque)],
        label=bloque,
        id=f"nav-{bloque}",
        nav=True,
        class_name="nav-bloque",
    )


cabecera = html.Header(
    html.Div(
        [
            html.Div(
                html.A(
                    html.Img(src=app.get_asset_url("logo_uninorte.png"), alt="Universidad del Norte"),
                    href="/",
                ),
                className="logo-placa",
            ),
            html.Ul([_menu_bloque(b) for b in BLOQUES], className="menu-superior nav"),
            dbc.RadioItems(
                id="selector-tema",
                options=[{"label": "Claro", "value": "claro"}, {"label": "Oscuro", "value": "oscuro"}],
                value="claro",
                inline=True,
                class_name="segmentado selector-tema",
                label_checked_class_name="activo",
                persistence=True,
                persistence_type="local",
            ),
        ],
        className="cabecera-interior",
    ),
    className="cabecera",
)

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="tema", data="claro"),
    html.Div(id="tema-aplicado", hidden=True),
    cabecera,
    html.Main(html.Div(id="contenido"), className="pagina"),
])


# ── Routing ────────────────────────────────────────────────────────────
@app.callback(
    Output("contenido", "children"),
    [Output(f"nav-{b}", "toggle_class_name") for b in BLOQUES],
    [Output(f"nav-item-{s['slug']}", "active") for s in SECCIONES],
    Input("url", "pathname"),
)
def renderizar(ruta):
    """Renderiza la sección activa y marca su bloque e ítem en el menú."""
    slug = slug_desde_ruta(ruta)
    try:
        contenido = PESTANAS[slug].layout()
    except DatosNoDisponibles as error:
        contenido = html.Div([encabezado_seccion(slug), aviso_sin_datos(str(error), URL_DATASET)])
    bloque = POR_SLUG[slug]["bloque"]
    clases = ["activo" if b == bloque else "" for b in BLOQUES]
    activos = [s["slug"] == slug for s in SECCIONES]
    return [contenido, *clases, *activos]


# ── Tema claro / oscuro ────────────────────────────────────────────────
@app.callback(Output("tema", "data"), Input("selector-tema", "value"))
def guardar_tema(valor):
    """Guarda el tema activo; los gráficos lo reciben como Input."""
    return valor if valor in ("claro", "oscuro") else "claro"


clientside_callback(
    """
    function (tema) {
        document.documentElement.setAttribute("data-bs-theme", tema === "oscuro" ? "dark" : "light");
        return tema;
    }
    """,
    Output("tema-aplicado", "children"),
    Input("tema", "data"),
)

for _modulo in PESTANAS.values():
    if hasattr(_modulo, "register_callbacks"):
        _modulo.register_callbacks(app)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
