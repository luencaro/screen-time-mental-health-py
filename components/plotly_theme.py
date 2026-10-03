# Paleta de gráficos · 4d Pizarra cálida
import plotly.graph_objects as go
import plotly.io as pio

COLORES = {
    "claro": {
        "chicos": "#9FC9C0",
        "chicas": "#D9A7C7",
        "no_deprimido": "#A7C4E5",
        "deprimido": "#F0B08F",
        "linea_corte": "#4F5B7A",
        "linea_mediana": "#5F5A54",
        "texto": "#2A2724",
        "texto_secundario": "#5F5A54",
        "cuadricula": "#E8E3DC",
        "superficie": "#FFFFFF",
    },
    "oscuro": {
        "chicos": "#8FBAB1",
        "chicas": "#CB98B9",
        "no_deprimido": "#97B5D8",
        "deprimido": "#E6A283",
        "linea_corte": "#A9B6D8",
        "linea_mediana": "#ACA59D",
        "texto": "#EEEAE5",
        "texto_secundario": "#ACA59D",
        "cuadricula": "#36322F",
        "superficie": "#242120",
    },
}

# Mapas listos para color_discrete_map en plotly.express
MAPA_SEXO = lambda modo="claro": {"Boy": COLORES[modo]["chicos"], "Girl": COLORES[modo]["chicas"]}
MAPA_ESTADO = lambda modo="claro": {0: COLORES[modo]["no_deprimido"], 1: COLORES[modo]["deprimido"]}


def plantilla(modo="claro"):
    c = COLORES[modo]
    return go.layout.Template(layout=dict(
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=c["texto_secundario"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[c["no_deprimido"], c["deprimido"], c["chicos"], c["chicas"]],
        xaxis=dict(showgrid=False, linecolor=c["cuadricula"], zeroline=False, title_font_size=13),
        yaxis=dict(gridcolor=c["cuadricula"], zeroline=False, title_font_size=13),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color=c["texto"], size=13)),
        hoverlabel=dict(bgcolor=c["superficie"], bordercolor=c["cuadricula"],
                        font=dict(family="IBM Plex Sans, sans-serif", color=c["texto"], size=13)),
        margin=dict(l=48, r=12, t=36, b=44),
        separators=",.",
    ))


pio.templates["uninorte_claro"] = plantilla("claro")
pio.templates["uninorte_oscuro"] = plantilla("oscuro")
pio.templates.default = "uninorte_claro"
