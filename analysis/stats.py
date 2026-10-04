"""Cálculos estadísticos del EDA.

Funciones puras: reciben un DataFrame y devuelven un DataFrame (o un
diccionario pequeño de cifras) sin modificar la entrada ni dibujar nada.
"""

import numpy as np
import pandas as pd
from scipy import stats

from data.load_data import CORTE_BDI, ETIQUETAS_ESTADO, ETIQUETAS_SEXO


def magnitud(efecto: float) -> str:
    """Magnitud de un tamaño de efecto (|ρ|, |r| o V) según los umbrales de Cohen."""
    a = abs(efecto)
    if a < 0.1:
        return "despreciable"
    if a < 0.3:
        return "pequeño"
    if a < 0.5:
        return "moderado"
    return "grande"


# ── Resumen general ────────────────────────────────────────────────────
def resumen_muestra(df: pd.DataFrame) -> dict:
    """Cifras generales: tamaño, composición por sexo y prevalencia."""
    n = len(df)
    por_sexo = df["sex"].value_counts()
    prev_sexo = df.groupby("sex")["depressed"].mean()
    return {
        "n": n,
        "n_columnas": df.shape[1],
        "n_chicas": int(por_sexo.get("Girl", 0)),
        "n_chicos": int(por_sexo.get("Boy", 0)),
        "prop_chicas": por_sexo.get("Girl", 0) / n,
        "prop_chicos": por_sexo.get("Boy", 0) / n,
        "n_deprimidos": int(df["depressed"].sum()),
        "prevalencia": df["depressed"].mean(),
        "prevalencia_chicas": prev_sexo.get("Girl", np.nan),
        "prevalencia_chicos": prev_sexo.get("Boy", np.nan),
        "mediana_bdi": df["bdi_total"].median(),
        "media_bdi": df["bdi_total"].mean(),
    }


# ── Calidad de datos ───────────────────────────────────────────────────
def calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Por columna: tipo, nulos, valores únicos."""
    return pd.DataFrame({
        "variable": df.columns,
        "tipo": [str(t) for t in df.dtypes],
        "nulos": df.isna().sum().values,
        "unicos": df.nunique().values,
    })


def duplicados(df: pd.DataFrame, clave: str = "subject_id") -> dict:
    """Filas duplicadas completas y duplicados en la clave."""
    return {
        "filas_duplicadas": int(df.duplicated().sum()),
        "claves_duplicadas": int(df[clave].duplicated().sum()),
        "nulos_totales": int(df.isna().sum().sum()),
    }


def descriptivos(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Media, mediana, desviación, mínimo, P25, P75, máximo y asimetría."""
    filas = []
    for col in columnas:
        x = df[col]
        filas.append({
            "variable": col,
            "N": x.count(),
            "media": x.mean(),
            "mediana": x.median(),
            "desv": x.std(),
            "min": x.min(),
            "p25": x.quantile(0.25),
            "p75": x.quantile(0.75),
            "max": x.max(),
            "asimetria": x.skew(),
        })
    return pd.DataFrame(filas)


def descriptivos_por_grupo(df: pd.DataFrame, columna: str, grupo: str) -> pd.DataFrame:
    """Descriptivos de una variable dentro de cada categoría de ``grupo``."""
    g = df.groupby(grupo)[columna]
    tabla = pd.DataFrame({
        "n": g.size(),
        "media": g.mean(),
        "mediana": g.median(),
        "desv": g.std(),
        "asimetria": g.skew(),
    }).reset_index()
    return tabla.rename(columns={grupo: "grupo"})


# ── Variable objetivo ──────────────────────────────────────────────────
def distribucion_estado(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo y proporción de ``depressed``."""
    conteo = df["depressed"].value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame({
        "depressed": conteo.index,
        "etiqueta": [ETIQUETAS_ESTADO[i] for i in conteo.index],
        "n": conteo.values,
        "proporcion": conteo.values / conteo.sum(),
    })


def kde(valores: pd.Series, puntos: int = 200, limites: tuple | None = None) -> pd.DataFrame:
    """Densidad kernel gaussiana evaluada en una rejilla regular."""
    x = np.asarray(valores, dtype=float)
    inicio, fin = limites if limites else (x.min(), x.max())
    rejilla = np.linspace(inicio, fin, puntos)
    return pd.DataFrame({"x": rejilla, "densidad": stats.gaussian_kde(x)(rejilla)})


def granularidad_corte(df: pd.DataFrame, margen: int = 3) -> dict:
    """Cifras sobre la pérdida de información al binarizar ``bdi_total``."""
    bdi = df["bdi_total"]
    sobre = bdi[bdi >= CORTE_BDI]
    cerca = bdi.between(CORTE_BDI - margen, CORTE_BDI + margen - 1)
    return {
        "valores_distintos_sobre": int(sobre.nunique()),
        "min_sobre": int(sobre.min()),
        "max_sobre": int(sobre.max()),
        "margen": margen,
        "cerca_desde": CORTE_BDI - margen,
        "cerca_hasta": CORTE_BDI + margen - 1,
        "n_cerca": int(cerca.sum()),
        "prop_cerca": cerca.mean(),
        "prop_cero": (bdi == 0).mean(),
        "p90": bdi.quantile(0.90),
        "asimetria": bdi.skew(),
    }


# ── Sexo ───────────────────────────────────────────────────────────────
def distribucion_sexo(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo y proporción de chicas y chicos."""
    conteo = df["sex"].value_counts().reindex(list(ETIQUETAS_SEXO), fill_value=0)
    return pd.DataFrame({
        "sex": conteo.index,
        "etiqueta": [ETIQUETAS_SEXO[s] for s in conteo.index],
        "n": conteo.values,
        "proporcion": conteo.values / conteo.sum(),
    })


def depresion_por_sexo(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de cada estado dentro de cada sexo (formato largo)."""
    tabla = (
        df.groupby(["sex", "depressed"]).size().rename("n").reset_index()
    )
    tabla["proporcion"] = tabla["n"] / tabla.groupby("sex")["n"].transform("sum")
    tabla["etiqueta_sexo"] = tabla["sex"].map(ETIQUETAS_SEXO)
    tabla["etiqueta_estado"] = tabla["depressed"].map(ETIQUETAS_ESTADO)
    return tabla


def tabla_contingencia(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla sexo × estado con totales."""
    tabla = pd.crosstab(df["sex"], df["depressed"], margins=True, margins_name="Total")
    return tabla


def chi_cuadrado(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-cuadrado de independencia sexo × depressed, V de Cramér y razones."""
    observada = pd.crosstab(df["sex"], df["depressed"])
    chi2, p, gl, _ = stats.chi2_contingency(observada, correction=False)
    n = observada.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(observada.shape) - 1)))
    prev = observada[1] / observada.sum(axis=1)
    odds = observada[1] / observada[0]
    return pd.DataFrame([{
        "chi2": chi2,
        "gl": gl,
        "p": p,
        "v_cramer": v,
        "razon_prevalencias": prev["Girl"] / prev["Boy"],
        "odds_ratio": odds["Girl"] / odds["Boy"],
    }])


def mann_whitney(df: pd.DataFrame, columnas: list[str], grupo: str = "sex",
                 niveles: tuple = ("Boy", "Girl")) -> pd.DataFrame:
    """Mann-Whitney U entre dos grupos con correlación biserial de rangos.

    El tamaño de efecto r = 1 − 2U/(n1·n2) es positivo cuando el segundo
    nivel (por defecto chicas) tiende a valores más altos.
    """
    a_nivel, b_nivel = niveles
    filas = []
    for col in columnas:
        a = df.loc[df[grupo] == a_nivel, col]
        b = df.loc[df[grupo] == b_nivel, col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        filas.append({
            "variable": col,
            "mediana_a": a.median(),
            "mediana_b": b.median(),
            "u": u,
            "p": p,
            "r_biserial": 1 - 2 * u / (len(a) * len(b)),
        })
    return pd.DataFrame(filas)


# ── Correlaciones ──────────────────────────────────────────────────────
def matriz_spearman(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Matriz de correlación de Spearman."""
    return df[columnas].corr(method="spearman")


def spearman_pares(df: pd.DataFrame, x: list[str], y: list[str]) -> pd.DataFrame:
    """ρ de Spearman y valor p para cada combinación x × y."""
    filas = []
    for a in x:
        for b in y:
            if a == b:
                continue
            rho, p = stats.spearmanr(df[a], df[b])
            filas.append({"x": a, "y": b, "rho": rho, "p": p, "n": len(df)})
    return pd.DataFrame(filas)


def pares_redundantes(matriz: pd.DataFrame, umbral: float = 0.7) -> pd.DataFrame:
    """Pares distintos con |ρ| ≥ umbral (candidatos a redundancia)."""
    filas = []
    cols = list(matriz.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = matriz.loc[a, b]
            if abs(rho) >= umbral:
                filas.append({"x": a, "y": b, "rho": rho})
    return pd.DataFrame(filas, columns=["x", "y", "rho"])


def par_mas_fuerte(matriz: pd.DataFrame, excluir: list[str] = ()) -> dict:
    """Par distinto con mayor |ρ|, ignorando las columnas de ``excluir``."""
    cols = [c for c in matriz.columns if c not in excluir]
    mejor = {"x": None, "y": None, "rho": 0.0}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(matriz.loc[a, b]) > abs(mejor["rho"]):
                mejor = {"x": a, "y": b, "rho": matriz.loc[a, b]}
    return mejor


# ── Outliers ───────────────────────────────────────────────────────────
def limites_iqr(serie: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Límites inferior y superior de Tukey (Q1 − k·IQR, Q3 + k·IQR)."""
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def mascara_outliers(df: pd.DataFrame, columnas: list[str], k: float = 1.5) -> pd.DataFrame:
    """DataFrame booleano: True donde el valor queda fuera de los límites IQR."""
    return pd.DataFrame({
        col: ~df[col].between(*limites_iqr(df[col], k)) for col in columnas
    })


def outliers_iqr(df: pd.DataFrame, columnas: list[str], k: float = 1.5) -> pd.DataFrame:
    """Q1, Q3, IQR, límites y número/porcentaje de outliers por variable."""
    mascara = mascara_outliers(df, columnas, k)
    filas = []
    for col in columnas:
        q1, q3 = df[col].quantile([0.25, 0.75])
        inf, sup = limites_iqr(df[col], k)
        m = mascara[col]
        filas.append({
            "variable": col,
            "q1": q1, "q3": q3, "iqr": q3 - q1,
            "lim_inf": inf, "lim_sup": sup,
            "n_bajo": int((df[col] < inf).sum()),
            "n_alto": int((df[col] > sup).sum()),
            "n": int(m.sum()),
            "proporcion": m.mean(),
        })
    return pd.DataFrame(filas)


def outliers_por_estado(df: pd.DataFrame, columnas: list[str], k: float = 1.5) -> pd.DataFrame:
    """Proporción de outliers de cada variable dentro de cada estado depresivo.

    Incluye también la prevalencia de depresión entre los outliers.
    """
    mascara = mascara_outliers(df, columnas, k)
    filas = []
    for col in columnas:
        m = mascara[col]
        for estado in (0, 1):
            grupo = df["depressed"] == estado
            filas.append({
                "variable": col,
                "depressed": estado,
                "etiqueta_estado": ETIQUETAS_ESTADO[estado],
                "proporcion": m[grupo].mean(),
                "n": int((m & grupo).sum()),
                "prevalencia_en_outliers": df.loc[m, "depressed"].mean() if m.any() else np.nan,
            })
    return pd.DataFrame(filas)


def filas_con_outliers(df: pd.DataFrame, columnas: list[str], k: float = 1.5) -> dict:
    """Filas con al menos un outlier y prevalencia de depresión en ellas."""
    alguna = mascara_outliers(df, columnas, k).any(axis=1)
    return {
        "n": int(alguna.sum()),
        "proporcion": alguna.mean(),
        "prevalencia": df.loc[alguna, "depressed"].mean() if alguna.any() else np.nan,
        "prevalencia_resto": df.loc[~alguna, "depressed"].mean(),
    }
