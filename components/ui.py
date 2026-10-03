"""Componentes de interfaz según la guía 4d Pizarra cálida.

Card, KPI, badge, bloque de interpretación, tabla, encabezado de sección y
envoltorio de gráficos. Todo el estilo vive en assets/*.css con variables.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from components.secciones import POR_SLUG

CONFIG_GRAFICO = {"displayModeBar": False, "responsive": True}


def var(nombre: str) -> html.Code:
    """Nombre de variable del dataset en IBM Plex Mono."""
    return html.Code(nombre, className="variable")


def rico(texto: str) -> list:
    """Convierte `nombre` entre comillas invertidas en nombres de variable (mono)."""
    partes = texto.split("`")
    return [var(t) if i % 2 else t for i, t in enumerate(partes) if t]


def card(*hijos, titulo: str | None = None, className: str = "") -> html.Div:
    """Card de superficie con título opcional arriba a la izquierda."""
    contenido = [html.H2(titulo, className="card-titulo")] if titulo else []
    return html.Div(contenido + list(hijos), className=f"tarjeta {className}".strip())


def kpi(etiqueta: str, valor: str, contexto: str = "", categoria: str | None = None,
        id_valor: str | None = None, id_contexto: str | None = None) -> html.Div:
    """KPI: etiqueta → valor → contexto.

    ``categoria`` ('deprimido', 'no_deprimido') aplica el color "texto" de esa
    categoría al valor; sin categoría se usa el color de texto general.
    """
    clase_valor = "kpi-valor" + (f" kpi-valor--{categoria}" if categoria else "")
    props_valor = {"id": id_valor} if id_valor else {}
    props_contexto = {"id": id_contexto} if id_contexto else {}
    return html.Div(
        [
            html.Div(etiqueta, className="kpi-etiqueta"),
            html.Div(valor, className=clase_valor, **props_valor),
            html.Div(contexto, className="kpi-contexto", **props_contexto),
        ],
        className="tarjeta kpi",
    )


def fila_kpis(kpis: list) -> dbc.Row:
    """Fila de KPIs de 3 columnas cada uno (se apilan en móvil)."""
    return dbc.Row(
        [dbc.Col(k, xs=12, sm=6, lg=3) for k in kpis],
        className="fila",
    )


def badge(texto: str, variante: str = "primaria") -> html.Span:
    """Píldora de 12 px. Variantes: 'primaria' o 'neutra'."""
    return html.Span(texto, className=f"insignia insignia--{variante}")


def interpretacion(*hijos, titulo: str = "Interpretación") -> html.Div:
    """Bloque de interpretación: overline en primario y texto de cuerpo."""
    parrafos = [html.P(rico(h)) if isinstance(h, str) else h for h in hijos]
    return html.Div(
        [html.Div(titulo.upper(), className="overline")] + parrafos,
        className="interpretacion",
    )


def encabezado_seccion(slug: str, badges: list[str] | None = None) -> html.Div:
    """Breadcrumb (bloque / sección) + H1 de sección + badges a la derecha."""
    seccion = POR_SLUG[slug]
    return html.Div(
        [
            html.Nav(
                [
                    html.Span(seccion["bloque"]),
                    html.Span("/", className="breadcrumb-sep"),
                    html.Span(seccion["nombre"], className="breadcrumb-actual"),
                ],
                className="migas",
                **{"aria-label": "Ruta de navegación"},
            ),
            html.Div(
                [
                    html.H1(seccion["nombre"], className="titulo-seccion"),
                    html.Div([badge(b) for b in (badges or [])], className="badges"),
                ],
                className="encabezado-fila",
            ),
        ],
        className="encabezado-seccion",
    )


def grafico(id_: str | dict, altura: int = 360, figura=None, ancho_minimo: int | None = None):
    """dcc.Graph con la barra de herramientas oculta y altura fija.

    Con ``ancho_minimo`` el gráfico se desplaza horizontalmente en pantallas
    estrechas en lugar de comprimirse.
    """
    props = {"figure": figura} if figura is not None else {}
    estilo = {"height": f"{altura}px"}
    if ancho_minimo:
        estilo["minWidth"] = f"{ancho_minimo}px"
    g = dcc.Graph(id=id_, config=CONFIG_GRAFICO, style=estilo, **props)
    return html.Div(g, className="grafico-desplazable") if ancho_minimo else g


def tabla(filas: list[dict], columnas: list[dict], className: str = "") -> html.Div:
    """Tabla HTML según la guía.

    ``columnas``: lista de dicts con 'clave', 'titulo' y opcionalmente
    'tipo' ('texto', 'numero' o 'variable') y 'formato' (función valor → str).
    """
    cabecera = html.Thead(html.Tr([
        html.Th(c["titulo"], className="num" if c.get("tipo") == "numero" else None)
        for c in columnas
    ]))
    cuerpo = []
    for fila in filas:
        celdas = []
        for c in columnas:
            valor = fila[c["clave"]]
            texto = c["formato"](valor) if "formato" in c else valor
            tipo = c.get("tipo", "texto")
            if not isinstance(texto, (str, int, float)):
                celdas.append(html.Td(texto, className="num" if tipo == "numero" else None))
            elif tipo == "variable":
                celdas.append(html.Td(var(texto)))
            else:
                celdas.append(html.Td(texto, className="num" if tipo == "numero" else None))
        cuerpo.append(html.Tr(celdas, className=fila.get("_clase")))
    return html.Div(
        html.Table([cabecera, html.Tbody(cuerpo)], className="tabla"),
        className=f"tabla-contenedor {className}".strip(),
    )


def markdown(texto: str, className: str = "") -> dcc.Markdown:
    """Texto del libro renderizado con estilos de cuerpo."""
    return dcc.Markdown(texto, className=f"texto-libro {className}".strip(), link_target="_blank")


def aviso_sin_datos(motivo: str, comando: str) -> html.Div:
    """Mensaje claro cuando el CSV no está disponible."""
    return card(
        html.P(motivo),
        html.P("Descarga el dataset desde la raíz del proyecto con Kaggle CLI:"),
        html.Pre(html.Code(comando), className="bloque-codigo"),
        html.P("O descárgalo manualmente desde Kaggle y descomprímelo en data/raw/. "
               "Después recarga la página."),
        titulo="Dataset no disponible",
        className="aviso",
    )


def flujo(pasos: list[dict]) -> html.Ol:
    """Diagrama de flujo vertical: cada paso con número, título, descripción y enlace."""
    return html.Ol([
        html.Li([
            html.Span(p["numero"], className="flujo-marca"),
            html.Div([
                html.Div(html.A(p["titulo"], href=p["href"]) if p.get("href") else p["titulo"],
                         className="flujo-titulo"),
                html.Div(p["descripcion"], className="flujo-desc"),
            ]),
        ], className="flujo-paso")
        for p in pasos
    ], className="flujo")
