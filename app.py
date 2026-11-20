"""Dashboard del EDA y del modelado: tiempo de pantalla, sueño y síntomas depresivos.

Ejecutar desde la raíz del proyecto:  python app.py  →  http://127.0.0.1:8050
"""

import importlib
import os

import dash_bootstrap_components as dbc
from dash import ClientsideFunction, Dash, Input, Output, clientside_callback, dcc, html

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
    title=portada.TITULO,
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, interactive-widget=resizes-content"}],
)
app.index_string = app.index_string.replace("<html>", '<html lang="es">', 1).replace(
    "{%app_entry%}",
    '{%app_entry%}<div class="carga-arranque" role="status"><span aria-hidden="true"></span>'
    'Abriendo el dashboard…</div>',
)
server = app.server
shapiro_gemini = registrar_shapiro(app)


# ── Header ─────────────────────────────────────────────────────────────
def _item_menu(seccion: dict) -> dbc.DropdownMenuItem:
    """Ítem del desplegable con el nombre y el contenido de la sección."""
    return dbc.DropdownMenuItem(
        [html.Span(seccion["nombre"], className="menu-nombre"),
         html.Span(seccion["descripcion"], className="menu-desc")],
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
        align_end=bloque == "Acerca de",
    )


cabecera = html.Header(
    html.Div(
        [
            enlace_interno(
                [html.Span(html.Span(className="marca-grafica"), className="marca-icono", **{"aria-hidden": "true"}),
                 html.Span([html.Strong("Pantallas y bienestar"), html.Small("Universidad del Norte")],
                           className="marca-texto")],
                "/", className="marca-proyecto", title="Volver a la portada",
            ),
            html.Nav(
                html.Ul([_menu_bloque(b) for b in BLOQUES], className="menu-superior nav"),
                className="navegacion-principal", **{"aria-label": "Navegación principal"},
            ),
            html.Div([
                html.Div(
                    dbc.RadioItems(
                        id="selector-tema",
                        options=[
                            {"label": html.Span([icono("sol", 17), html.Span("Claro", className="visualmente-oculto")],
                                                className="opcion-tema", title="Tema claro"), "value": "claro"},
                            {"label": html.Span([icono("luna", 17), html.Span("Oscuro", className="visualmente-oculto")],
                                                className="opcion-tema", title="Tema oscuro"), "value": "oscuro"},
                        ],
                        value="claro", inline=True, class_name="segmentado selector-tema",
                        label_checked_class_name="activo", persistence=True, persistence_type="local",
                    ),
                    role="group", **{"aria-label": "Tema de la interfaz"},
                ),
                boton_shapiro(),
            ], className="cabecera-acciones"),
        ],
        className="cabecera-interior",
    ),
    className="cabecera",
)

def _contenido_cargando():
    return html.Div([
        html.P("Preparando la sección…", role="status"),
        html.Div([
            html.Div(className="esqueleto-titulo"),
            html.Div(className="esqueleto-texto"),
            html.Div([html.Div() for _ in range(3)], className="esqueleto-paneles"),
        ], **{"aria-hidden": "true"}),
    ], className="esqueleto-vista")


def _carga_vista():
    return dcc.Loading(
        html.Div(_contenido_cargando(), id="contenido"),
        id="carga-vista-capa",
        delay_show=160,
        show_initially=False,
        target_components={"contenido": "children"},
        overlay_style={"visibility": "visible"},
        custom_spinner=html.Div([
            html.Span(className="carga-vista-linea", **{"aria-hidden": "true"}),
            html.Span("Cargando sección…", className="carga-vista-texto"),
        ], id="carga-vista", role="status"),
    )


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="tema", data="claro"),
    dcc.Store(id="vista-activa"),
    html.Div(id="vista-aplicada", hidden=True),
    html.A("Saltar al contenido", href="#contenido-principal", className="salto-contenido"),
    cabecera,
    html.Main(_carga_vista(), id="contenido-principal", className="pagina", tabIndex=-1,
              **{"aria-labelledby": "titulo-vista"}),
    html.Footer([
        html.Div([
            html.Img(src=app.get_asset_url("logo_uninorte.png"), alt="Universidad del Norte", className="logo-claro"),
            html.Img(src=app.get_asset_url("logo_uninorte_oscuro.png"), alt="Universidad del Norte", className="logo-oscuro"),
        ], className="pie-logo"),
        enlace_interno("Equipo y recursos", "/acerca"),
    ], className="pie-pagina"),
    dialogo_shapiro(shapiro_gemini),
])


# ── Routing ────────────────────────────────────────────────────────────
@app.callback(
    Output("contenido", "children"),
    Output("contenido-principal", "className"),
    Output("vista-activa", "data"),
    [Output(f"nav-{b}", "toggle_class_name") for b in BLOQUES],
    [Output(f"nav-item-{s['slug']}", "active") for s in SECCIONES],
    Input("url", "pathname"),
)
def renderizar(ruta):
    """Renderiza la portada o la sección activa y actualiza el menú."""
    pathname = ruta
    ruta = app.strip_relative_path(ruta)
    if not ruta:
        return [portada.layout(), "pagina pagina--portada", {"titulo": portada.TITULO, "ruta": pathname},
                *["" for _ in BLOQUES], *[False for _ in SECCIONES]]

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
    titulo = f"{POR_SLUG[slug]['nombre']} · {portada.TITULO}"
    return [contenido, "pagina", {"titulo": titulo, "ruta": pathname}, *clases, *activos]


clientside_callback(
    ClientsideFunction(namespace="navegacion", function_name="presentar"),
    Output("vista-aplicada", "children"),
    Input("vista-activa", "data"),
)


# ── Tema claro / oscuro ────────────────────────────────────────────────
clientside_callback(
    """
    function (valor) {
        const tema = valor === "oscuro" ? "oscuro" : "claro";
        document.documentElement.setAttribute("data-bs-theme", tema === "oscuro" ? "dark" : "light");
        return tema;
    }
    """,
    Output("tema", "data"),
    Input("selector-tema", "value"),
)

for _modulo in PESTANAS.values():
    if hasattr(_modulo, "register_callbacks"):
        _modulo.register_callbacks(app)


if __name__ == "__main__":
    # DASH_DEBUG=1 recarga el dashboard al guardar cambios (solo para desarrollo)
    app.run(debug=os.environ.get("DASH_DEBUG") == "1", host="0.0.0.0", port=8050)
