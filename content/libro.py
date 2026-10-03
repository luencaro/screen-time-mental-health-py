"""Lectura del contenido narrativo de book/*.md.

Extrae secciones por título de encabezado y convierte las citas MyST
{cite}`clave` en (Autor, año) a partir de book/references.bib.
"""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_LIBRO = RAIZ / "book"
RUTA_BIB = CARPETA_LIBRO / "references.bib"

_ENCABEZADO = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_NUMERACION = re.compile(r"^\d+(\.\d+)*\.?\s+")
_CITA = re.compile(r"\{cite(?::[a-z]+)?\}`([^`]+)`")

# Acentos LaTeX frecuentes en BibTeX
_ACENTOS = {'"': "̈", "'": "́", "`": "̀", "^": "̂", "~": "̃"}


def _normalizar(titulo: str) -> str:
    """Quita numeración ('2.2 ', '1. ') y pasa a minúsculas para comparar títulos."""
    return _NUMERACION.sub("", titulo).strip().lower()


@lru_cache(maxsize=None)
def _leer(archivo: str) -> str:
    return (CARPETA_LIBRO / archivo).read_text(encoding="utf-8")


def seccion(archivo: str, titulo: str, citas: bool = True) -> str:
    """Devuelve el cuerpo de la sección ``titulo`` (sin su encabezado).

    La sección termina en el siguiente encabezado de nivel igual o superior.
    """
    lineas = _leer(archivo).splitlines()
    objetivo = _normalizar(titulo)
    inicio, nivel = None, None
    for i, linea in enumerate(lineas):
        m = _ENCABEZADO.match(linea)
        if not m:
            continue
        if inicio is None and _normalizar(m.group(2)) == objetivo:
            inicio, nivel = i + 1, len(m.group(1))
        elif inicio is not None and len(m.group(1)) <= nivel:
            return _postprocesar("\n".join(lineas[inicio:i]), citas)
    if inicio is None:
        raise KeyError(f"No se encontró la sección '{titulo}' en book/{archivo}.")
    return _postprocesar("\n".join(lineas[inicio:]), citas)


def _postprocesar(texto: str, citas: bool) -> str:
    texto = texto.strip()
    return convertir_citas(texto) if citas else texto


def parrafos(texto: str) -> list[str]:
    """Divide un bloque Markdown en párrafos (separados por líneas en blanco)."""
    return [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]


def items_numerados(texto: str) -> list[str]:
    """Extrae los elementos de una lista numerada Markdown ('1. …')."""
    return [m.group(1).strip() for m in re.finditer(r"^\s*\d+\.\s+(.*)$", texto, re.MULTILINE)]


# ── Referencias ────────────────────────────────────────────────────────
def _latex_a_texto(valor: str) -> str:
    """Convierte acentos LaTeX ({\\"o}, \\'e) a Unicode y elimina llaves."""
    def reemplazar(m):
        return m.group(2) + _ACENTOS[m.group(1)]
    valor = re.sub(r"\{?\\([\"'`^~])\{?([A-Za-z])\}?\}?", reemplazar, valor)
    valor = re.sub(r"\s+", " ", valor.replace("{", "").replace("}", "")).strip()
    return unicodedata.normalize("NFC", valor)


def _campos(cuerpo: str) -> dict:
    """Extrae los campos 'nombre = {valor}' de una entrada BibTeX (llaves anidadas)."""
    campos, i = {}, 0
    patron = re.compile(r"(\w+)\s*=\s*([{\"])")
    while (m := patron.search(cuerpo, i)):
        nombre, apertura = m.group(1).lower(), m.group(2)
        j = m.end()
        if apertura == "{":
            profundidad = 1
            while profundidad and j < len(cuerpo):
                profundidad += {"{": 1, "}": -1}.get(cuerpo[j], 0)
                j += 1
            valor = cuerpo[m.end():j - 1]
        else:
            fin = cuerpo.index('"', j)
            valor, j = cuerpo[j:fin], fin + 1
        campos[nombre] = _latex_a_texto(valor)
        i = j
    return campos


@lru_cache(maxsize=1)
def referencias() -> dict:
    """Diccionario clave → {autores (apellidos), anio, titulo, revista, doi}."""
    texto = RUTA_BIB.read_text(encoding="utf-8") if RUTA_BIB.exists() else ""
    salida = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", texto):
        # El cuerpo llega hasta la siguiente entrada o el final del archivo
        siguiente = texto.find("\n@", m.end())
        cuerpo = texto[m.end(): siguiente if siguiente != -1 else len(texto)]
        campos = _campos(cuerpo)
        apellidos = [a.split(",")[0].strip() for a in campos.get("author", "").split(" and ") if a.strip()]
        salida[m.group(1)] = {
            "autores": apellidos,
            "anio": campos.get("year", "s. f."),
            "titulo": campos.get("title", ""),
            "revista": campos.get("journal", ""),
            "doi": campos.get("doi", ""),
        }
    return salida


def autor_anio(clave: str) -> str:
    """Formato autor-año sin paréntesis: 'Hökby et al., 2025'."""
    ref = referencias().get(clave)
    if not ref:
        return clave
    autores = ref["autores"]
    if len(autores) == 1:
        nombre = autores[0]
    elif len(autores) == 2:
        nombre = f"{autores[0]} y {autores[1]}"
    else:
        nombre = f"{autores[0]} et al."
    return f"{nombre}, {ref['anio']}"


def convertir_citas(texto: str) -> str:
    """Reemplaza {cite}`a,b` por (Autor, año; Autor, año).

    Si el texto ya nombra al autor justo antes ("Hökby et al. {cite}`…`"),
    se usa la forma narrativa y solo se añade el año: "Hökby et al. (2023)".
    """
    def reemplazar(m):
        claves = [c.strip() for c in m.group(1).split(",")]
        if len(claves) == 1:
            nombre, _, anio = autor_anio(claves[0]).rpartition(", ")
            if nombre and texto[:m.start()].rstrip().endswith(nombre):
                return f"({anio})"
        return "(" + "; ".join(autor_anio(c) for c in claves) + ")"
    return _CITA.sub(reemplazar, texto)
