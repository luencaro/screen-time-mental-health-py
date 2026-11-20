"""Dashboard del EDA y del modelado: tiempo de pantalla, sueño y síntomas depresivos.

Ejecutar desde la raíz del proyecto:  python app.py  →  http://127.0.0.1:8050
"""

import importlib
import os

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, clientside_callback, dcc, html

from components.secciones import BLOQUES, POR_SLUG, SECCIONES, secciones_de, slug_desde_ruta
from components.shapiro import boton_shapiro, dialogo_shapiro
from components.shapiro_gemini import registrar_shapiro
from components.ui import aviso_sin_datos, aviso_sin_resultados, encabezado_seccion, enlace_interno, icono
from data.load_data import URL_DATASET, DatosNoDisponibles
from data.load_modelado import ResultadosNoDisponibles
from tabs import portada

# Una pestaña por sección; cada módulo expone layout() y, si aplica, register_callbacks(app)
PESTANAS = {s["slug"]: importlib.import_module(f"tabs.{s['slug']}") for s in SECCIONES}

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="EDA y modelado · Pantallas, sueño y depresión",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.index_string = app.index_string.replace("<html>", '<html lang="es">', 1)
server = app.server
shapiro_gemini = registrar_shapiro(app)


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
        href=app.get_relative_path(f"/{seccion['slug']}"),
        external_link=False,
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
                enlace_interno(
                    [
                        html.Img(src=app.get_asset_url("logo_uninorte.png"), alt="Universidad del Norte",
                                 className="logo-claro"),
                        html.Img(src=app.get_asset_url("logo_uninorte_oscuro.png"),
                                 alt="Universidad del Norte", className="logo-oscuro"),
                    ],
                    "/",
                    title="Volver a la portada",
                ),
                className="logo-placa",
            ),
            html.Ul([_menu_bloque(b) for b in BLOQUES], className="menu-superior nav"),
            dbc.RadioItems(
                id="selector-tema",
                options=[
                    {"label": html.Span([
                        icono("sol", 14), html.Span("Claro", className="etiqueta-tema"),
                    ], className="con-icono", title="Tema claro"), "value": "claro"},
                    {"label": html.Span([
                        icono("luna", 14), html.Span("Oscuro", className="etiqueta-tema"),
                    ], className="con-icono", title="Tema oscuro"),
                     "value": "oscuro"},
                ],
                value="claro",
                inline=True,
                class_name="segmentado selector-tema",
                label_checked_class_name="activo",
                persistence=True,
                persistence_type="local",
            ),
            boton_shapiro(),
        ],
        className="cabecera-interior",
    ),
    className="cabecera",
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="tema", data="claro"),
    html.Div(id="tema-aplicado", hidden=True),
    cabecera,
    html.Main(html.Div(id="contenido"), id="pagina", className="pagina"),
    dialogo_shapiro(shapiro_gemini),
])


# ── Routing ────────────────────────────────────────────────────────────
@app.callback(
    Output("contenido", "children"),
    Output("pagina", "className"),
    [Output(f"nav-{b}", "toggle_class_name") for b in BLOQUES],
    [Output(f"nav-item-{s['slug']}", "active") for s in SECCIONES],
    Input("url", "pathname"),
)
def renderizar(ruta):
    """Renderiza la sección activa y marca su bloque e ítem en el menú."""
    ruta = app.strip_relative_path(ruta)
    if not ruta:
        return [portada.layout(), "pagina pagina--portada", *["" for _ in BLOQUES],
                *[False for _ in SECCIONES]]

    slug = slug_desde_ruta(ruta)
    try:
        contenido = PESTANAS[slug].layout()
    except DatosNoDisponibles as error:
        contenido = html.Div([encabezado_seccion(slug), aviso_sin_datos(str(error), URL_DATASET)])
    except ResultadosNoDisponibles as error:
        contenido = html.Div([encabezado_seccion(slug), aviso_sin_resultados(str(error))])
    bloque = POR_SLUG[slug]["bloque"]
    clases = ["activo" if b == bloque else "" for b in BLOQUES]
    activos = [s["slug"] == slug for s in SECCIONES]
    return [contenido, "pagina", *clases, *activos]


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
    # DASH_DEBUG=1 recarga el dashboard al guardar cambios (solo para desarrollo)
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", host="0.0.0.0", port=8050)
