"""Formato numérico en español: coma decimal y punto de miles."""

import math


def num(valor: float, decimales: int = 2) -> str:
    """Número con coma decimal y punto de miles (1234.5 → '1.234,50')."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    texto = f"{valor:,.{decimales}f}"
    texto = texto.replace(",", "_").replace(".", ",").replace("_", ".")
    return texto.replace("-", "−")


def entero(valor: float) -> str:
    """Entero con punto de miles (4810 → '4.810')."""
    return num(valor, 0)


def pct(proporcion: float, decimales: int = 1) -> str:
    """Proporción en porcentaje (0.1636 → '16,4 %')."""
    return f"{num(proporcion * 100, decimales)}\u00a0%"


def valor_p(p: float) -> str:
    """Valor p legible ('< 0,001' cuando es muy pequeño)."""
    if p < 0.001:
        return "< 0,001"
    return num(p, 3)


def texto_p(p: float) -> str:
    """Valor p con su operador para frases ('p < 0,001' o 'p = 0,023')."""
    return f"p {valor_p(p)}" if p < 0.001 else f"p = {valor_p(p)}"


def estadistico(valor: float) -> str:
    """Estadístico de prueba: entero con punto de miles, o con un decimal si lo tiene
    (U de Mann-Whitney puede terminar en ,5)."""
    return entero(valor) if float(valor).is_integer() else num(valor, 1)
