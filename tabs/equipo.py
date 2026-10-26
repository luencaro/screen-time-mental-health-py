"""Pestaña 11 · Equipo: integrantes del proyecto con foto, carrera, semestre y enlaces.

Para cambiar la información basta con editar INTEGRANTES. Las fotos van en
assets/equipo/ (cuadradas, idealmente ≥ 400 px); si falta una foto se muestran
las iniciales.
"""

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import get_asset_url, html

from components.ui import badge, card, encabezado_seccion

CARPETA_FOTOS = Path(__file__).resolve().parent.parent / "assets" / "equipo"

INTEGRANTES = [
    {
        "nombre": "Luis Cabarcas",
        "foto": "luis.jpg",
        "carreras": ["Matemáticas", "Ciencia de Datos"],
        "semestre": "8vo semestre",
        "github": "https://github.com/luencaro",
        "linkedin": "https://www.linkedin.com/in/luencaro/",
    },
    {
        "nombre": "Santiago",
        "foto": "integrante_2.jpg",
        "carreras": ["Ciencia de Datos"],
        "semestre": "Semestre",
        "github": "https://github.com/usuario",
        "linkedin": "https://www.linkedin.com/in/usuario",
    },
]


def _iniciales(nombre: str) -> str:
    return "".join(p[0] for p in nombre.split()[:2]).upper()


def _foto(persona: dict) -> html.Div:
    """Foto circular o, si el archivo no existe, las iniciales."""
    archivo = persona.get("foto")
    if archivo and (CARPETA_FOTOS / archivo).is_file():
        contenido = html.Img(src=get_asset_url(f"equipo/{archivo}"), alt=persona["nombre"])
    else:
        contenido = html.Span(_iniciales(persona["nombre"]), **{"aria-hidden": "true"})
    return html.Div(contenido, className="equipo-foto")


def _enlace(texto: str, url: str | None):
    if not url:
        return None
    return html.A(texto, href=url, target="_blank", rel="noopener noreferrer", className="equipo-enlace")


def _tarjeta(persona: dict) -> html.Div:
    carreras = persona["carreras"]
    return card(
        html.Div([
            _foto(persona),
            html.H2(persona["nombre"], className="equipo-nombre"),
            html.Div(badge(persona["semestre"]), className="equipo-semestre"),
            html.Dl([
                html.Dt("Carreras" if len(carreras) > 1 else "Carrera"),
                *[html.Dd(c) for c in carreras],
            ], className="dato-lista equipo-datos"),
            html.Div([_enlace("GitHub", persona.get("github")),
                      _enlace("LinkedIn", persona.get("linkedin"))], className="equipo-enlaces"),
        ], className="equipo-perfil"),
        className="estirar",
    )


def layout():
    """Estructura: una card por integrante."""
    return html.Div([
        encabezado_seccion("equipo", [f"{len(INTEGRANTES)} integrantes"]),
        dbc.Row([dbc.Col(_tarjeta(p), md=6, lg=5) for p in INTEGRANTES],
                className="fila justify-content-center"),
    ])
