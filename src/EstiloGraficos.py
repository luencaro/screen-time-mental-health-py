"""Paleta y estilo de gráficos del dashboard (4d Pizarra cálida) para matplotlib/seaborn.

Los colores se leen de components/plotly_theme.py, la misma fuente que usa el
dashboard, sin importar plotly: así el notebook funciona también en el entorno
Docker, que no lo tiene instalado.

Uso en el notebook:

    from EstiloGraficos import aplicar_estilo, PALETA_SEXO, PALETA_ESTADO, ...
    aplicar_estilo()
"""

import ast
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parent.parent
RUTA_TEMA = RAIZ / "components" / "plotly_theme.py"
CARPETA_FUENTES = RAIZ / "assets" / "fonts"


def _leer_colores(modo: str = "claro") -> dict:
    """Extrae el diccionario COLORES de plotly_theme.py sin ejecutar el módulo."""
    arbol = ast.parse(RUTA_TEMA.read_text(encoding="utf-8"))
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and any(getattr(t, "id", None) == "COLORES" for t in nodo.targets):
            return ast.literal_eval(nodo.value)[modo]
    raise ValueError(f"No se encontró COLORES en {RUTA_TEMA}")


COLORES = _leer_colores("claro")

CHICOS = COLORES["chicos"]
CHICAS = COLORES["chicas"]
NO_DEPRIMIDO = COLORES["no_deprimido"]
DEPRIMIDO = COLORES["deprimido"]
LINEA_CORTE = COLORES["linea_corte"]        # discontinua (BDI-II = 14)
LINEA_MEDIANA = COLORES["linea_mediana"]    # punteada
TEXTO = COLORES["texto"]
TEXTO_SECUNDARIO = COLORES["texto_secundario"]
CUADRICULA = COLORES["cuadricula"]
SUPERFICIE = COLORES["superficie"]
NEUTRO = COLORES["linea_corte"]             # una sola serie sin categoría (pizarra)

# Sexo y estado depresivo nunca comparten color
PALETA_SEXO = {"Boy": CHICOS, "Girl": CHICAS}
PALETA_ESTADO = {0: NO_DEPRIMIDO, 1: DEPRIMIDO}

# Divergente para correlaciones (negativa → 0 → positiva), igual que el heatmap del dashboard
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list("pizarra_divergente", [NO_DEPRIMIDO, SUPERFICIE, DEPRIMIDO])
# Secuencial de un solo tono para conteos (p. ej. matriz de confusión)
CMAP_SECUENCIAL = LinearSegmentedColormap.from_list("pizarra_secuencial", [SUPERFICIE, NEUTRO])


def _fuente_base() -> list[str]:
    """IBM Plex Sans si está instalada o en assets/fonts/; si no, DejaVu Sans."""
    if CARPETA_FUENTES.is_dir():
        for archivo in CARPETA_FUENTES.glob("IBMPlexSans*.ttf"):
            font_manager.fontManager.addfont(str(archivo))
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    return ["IBM Plex Sans", "DejaVu Sans"] if "IBM Plex Sans" in disponibles else ["DejaVu Sans"]


def aplicar_estilo() -> None:
    """Configura matplotlib con la guía del dashboard: fondo de superficie, sin cuadrícula
    vertical, cuadrícula horizontal fina, texto en tinta y paleta de categorías."""
    fuentes = _fuente_base()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fuentes,
        "font.size": 11,
        "text.color": TEXTO,
        "axes.labelcolor": TEXTO_SECUNDARIO,
        "axes.titlecolor": TEXTO,
        # DejaVu Sans no tiene seminegrita: se usa negrita si falta IBM Plex Sans
        "axes.titleweight": "semibold" if "IBM Plex Sans" in fuentes else "bold",
        "axes.edgecolor": CUADRICULA,
        "axes.facecolor": SUPERFICIE,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=[NO_DEPRIMIDO, DEPRIMIDO, CHICOS, CHICAS]),
        "grid.color": CUADRICULA,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SECUNDARIO,
        "ytick.color": TEXTO_SECUNDARIO,
        "figure.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "legend.frameon": False,
        "legend.labelcolor": TEXTO,
        "patch.edgecolor": SUPERFICIE,
    })
