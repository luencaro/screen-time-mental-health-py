"""Portada y accesos a las secciones principales."""

from dash import html

from components.ui import enlace_interno, icono
from data.load_data import COLUMNAS, CORTE_BDI

TITULO = "Pantallas, sueño y salud mental"


def _ilustracion():
    """Visualización conceptual que acompaña la presentación del estudio."""
    return html.Div([
        html.Div([
            html.Div(className="telefono-camara"),
            html.Div([html.Span("Bienestar digital"), html.Span(className="telefono-estado")],
                     className="telefono-titulo"),
            html.Div([
                html.Span("Tiempo de pantalla", className="telefono-etiqueta"),
                html.Div([html.Span(style={"--altura-barra": f"{altura}%"})
                          for altura in (32, 47, 61, 46, 81, 53, 93)], className="telefono-barras"),
            ], className="telefono-tarjeta telefono-tarjeta--pantalla"),
            html.Div([
                html.Span([html.Span("+", className="telefono-cruz"), "Síntomas depresivos"],
                          className="telefono-etiqueta"),
                html.Span(f"Corte BDI-II ≥ {CORTE_BDI}", className="telefono-detalle"),
            ], className="telefono-tarjeta telefono-tarjeta--salud"),
            html.Div(className="telefono-inicio"),
        ], className="telefono"),
        html.Span("Ilustración conceptual", className="portada-ilustracion-nota"),
    ], className="portada-ilustracion", role="img",
        **{"aria-label": "Ilustración conceptual de un teléfono con un gráfico de tiempo de pantalla "
                         "y el punto de corte BDI-II del estudio."})


def _acceso(nombre: str, descripcion: str, ruta: str, signo):
    return enlace_interno([
        html.Span(signo, className="acceso-signo", **{"aria-hidden": "true"}),
        html.Span([html.Strong(nombre), html.Span(descripcion)], className="acceso-texto"),
        html.Span("→", className="acceso-flecha", **{"aria-hidden": "true"}),
    ], ruta, className="acceso-portada")


def layout():
    return html.Div([
        html.Section([
            html.Div([
                html.P("Investigación · Ciencia de datos", className="portada-etiqueta"),
                html.H1([html.Span("Pantallas, sueño y"), " salud mental"], id="titulo-vista"),
                html.P("Explora la relación entre el uso de pantallas, los hábitos de sueño "
                       "y los síntomas depresivos en adolescentes.", className="portada-descripcion"),
                html.Ul([html.Li("Población adolescente"), html.Li(f"{len(COLUMNAS)} variables")],
                        className="portada-ficha", **{"aria-label": "Características del estudio"}),
                html.Div([
                    enlace_interno(["Explorar dashboard", html.Span("→", **{"aria-hidden": "true"})],
                                   "/inicio", className="portada-cta"),
                    enlace_interno("Ver metodología", "/metodologia", className="portada-secundario"),
                ], className="portada-acciones"),
            ], className="portada-texto"),
            _ilustracion(),
        ], className="portada-hero", **{"aria-labelledby": "titulo-vista"}),
        html.Nav([
            _acceso("El proyecto", "Contexto y objetivos del estudio", "/inicio", icono("libro", 20)),
            _acceso("Análisis exploratorio", "Distribuciones y asociaciones", "/calidad_datos", icono("datos", 20)),
            _acceso("Modelado", "Comparación de modelos predictivos", "/benchmarking",
                    html.Span(className="marca-grafica")),
        ], className="portada-accesos", **{"aria-label": "Accesos principales"}),
    ])
