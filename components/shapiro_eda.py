"""Resúmenes fijos del EDA que reutilizan los cálculos del dashboard."""

from functools import lru_cache

from analysis import stats
from components.formato import entero, estadistico, num, pct, texto_p
from data import load_data as datos


def _inicio(df) -> str:
    r = stats.resumen_muestra(df)
    return (
        f"Muestra de {entero(r['n'])} adolescentes. "
        f"Chicas {entero(r['n_chicas'])} ({pct(r['prop_chicas'])}), "
        f"chicos {entero(r['n_chicos'])} ({pct(r['prop_chicos'])}). "
        f"{entero(r['n_deprimidos'])} ({pct(r['prevalencia'])}) alcanzaron el corte de {datos.CORTE_BDI} del BDI-II."
    )


def _calidad(df) -> str:
    d = stats.duplicados(df)
    filas = [
        f"{entero(df.shape[0])} registros, {entero(df.shape[1])} columnas y "
        f"{entero(len(datos.VARIABLES_NUMERICAS))} variables numéricas de análisis.",
        f"Nulos totales {entero(d['nulos_totales'])}. Identificadores duplicados {entero(d['claves_duplicadas'])}. "
        f"Filas completas duplicadas {entero(d['filas_duplicadas'])}.",
        "Por columna, tipo, número de valores distintos, nulos y completitud (%).",
    ]
    filas.extend(
        f"{datos.ETIQUETAS[q.variable]} · {q.tipo}, {entero(q.unicos)}, {entero(q.nulos)}, {num(q.completitud, 1)} %."
        for q in stats.calidad(df).itertuples(index=False)
    )
    return "\n".join(filas)


def _objetivo(df) -> str:
    r = stats.resumen_muestra(df)
    bdi = stats.descriptivos(df, ["bdi_total"]).iloc[0]
    resumen = (
        f"Variable objetivo. BDI-II ≥ {datos.CORTE_BDI} · {entero(r['n_deprimidos'])} ({pct(r['prevalencia'])}). "
        f"BDI-II < {datos.CORTE_BDI} · {entero(r['n'] - r['n_deprimidos'])} ({pct(1 - r['prevalencia'])}). "
        f"Puntaje BDI-II en puntos. Mediana {num(r['mediana_bdi'])}, media {num(r['media_bdi'])}, "
        f"Q1 {num(bdi['p25'])}, Q3 {num(bdi['p75'])}, asimetría {num(bdi['asimetria'])}. "
        f"Rango observado {entero(bdi['min'])} a {entero(bdi['max'])} puntos."
    )
    if r["n_deprimidos"]:
        corte = stats.granularidad_corte(df)
        resumen += (
            f" Entre quienes alcanzaron el corte, el puntaje observado va de {entero(corte['min_sobre'])} "
            f"a {entero(corte['max_sobre'])} con {entero(corte['valores_distintos_sobre'])} valores distintos. "
            f"Entre {entero(corte['cerca_desde'])} y {entero(corte['cerca_hasta'])} puntos, "
            f"incluidos, hay {entero(corte['n_cerca'])} adolescentes ({pct(corte['prop_cerca'])})."
        )
    return resumen


def _sexo(df) -> str:
    tabla = stats.tabla_contingencia(df)
    niveles = ("Girl", "Boy")  # Mismo orden que Análisis por sexo.
    filas = [f"Estado según BDI-II ≥ {datos.CORTE_BDI}. Porcentajes dentro de cada grupo."]
    for nivel in niveles:
        if nivel in tabla.index:
            grupo = tabla.loc[nivel]
            positivos = grupo.get(1, 0)
            total = grupo["Total"]
            filas.append(f"{datos.ETIQUETAS_SEXO[nivel]} · {entero(positivos)} de {entero(total)} "
                         f"({pct(positivos / total)}) sobre el corte, {entero(total - positivos)} por debajo.")

    if set(niveles) <= set(tabla.index) and {0, 1} <= set(tabla.columns):
        chi = stats.chi_cuadrado(df).iloc[0]
        filas.append(
            f"Chi-cuadrado de independencia · χ² {num(chi['chi2'])}, gl {entero(chi['gl'])}, {texto_p(chi['p'])}, "
            f"V de Cramér {num(chi['v_cramer'], 3)}, razón de prevalencias chicas/chicos "
            f"{num(chi['razon_prevalencias'])}, odds ratio chicas/chicos {num(chi['odds_ratio'])}."
        )
    if set(niveles) <= set(tabla.index):
        filas.append("Mann–Whitney U bilateral. Medianas chicas/chicos, U, p y r rango-biserial. "
                     "Un r negativo señala rangos más altos en chicas.")
        filas.extend(
            f"{datos.ETIQUETAS[m.variable]} · {num(m.mediana_a)}/{num(m.mediana_b)}, U {estadistico(m.u)}, "
            f"{texto_p(m.p)}, r {num(m.r_biserial, 3)}."
            for m in stats.mann_whitney(df, datos.VARIABLES_NUMERICAS, niveles=niveles).sort_values("p").itertuples(index=False)
        )
    return "\n".join(filas)


def _descriptivos_grupo(grupos, nivel) -> str:
    if nivel in grupos.index:
        fila = grupos.loc[nivel]
        return "/".join(num(fila[columna]) for columna in ("media", "mediana", "desv", "asimetria"))
    return "—"


def _numericas(df) -> str:
    filas = [
        "Variables numéricas. Global y por grupo, en orden media/mediana/DE/asimetría.",
    ]
    for d in stats.descriptivos(df, datos.VARIABLES_NUMERICAS).itertuples(index=False):
        por_sexo = stats.descriptivos_por_grupo(df, d.variable, "sex").set_index("grupo")
        por_estado = stats.descriptivos_por_grupo(df, d.variable, "depressed").set_index("grupo")
        filas.append(
            f"{datos.ETIQUETAS[d.variable]} · global {num(d.media)}/{num(d.mediana)}/{num(d.desv)}/{num(d.asimetria)}, "
            f"rango {num(d.min)} a {num(d.max)}, Q1–Q3 {num(d.p25)} a {num(d.p75)}. "
            f"Chicas {_descriptivos_grupo(por_sexo, 'Girl')}, chicos {_descriptivos_grupo(por_sexo, 'Boy')}, "
            f"BDI-II < {datos.CORTE_BDI} {_descriptivos_grupo(por_estado, 0)}, "
            f"BDI-II ≥ {datos.CORTE_BDI} {_descriptivos_grupo(por_estado, 1)}."
        )
    return "\n".join(filas)


def _correlaciones(df) -> str:
    variables = datos.VARIABLES_NUMERICAS
    claves = {var: str(i) for i, var in enumerate(variables, start=1)}
    matriz = stats.matriz_spearman(df, variables)
    tablas = (
        stats.spearman_pares(df, ["screen_time_index"], datos.VARIABLES_SUENO),
        stats.spearman_pares(df, datos.VARIABLES_SUENO, ["bdi_total"]),
    )
    valores_p = {
        frozenset((fila.x, fila.y)): texto_p(fila.p)
        for tabla in tablas for fila in tabla.itertuples(index=False)
    }
    filas = [
        f"Matriz de Spearman, n = {entero(len(df))}. Cada par muestra ρ y p cuando aparece en las tablas.",
        ", ".join(f"{claves[var]} = {datos.ETIQUETAS[var]}" for var in variables) + ".",
    ]
    for i, x in enumerate(variables[:-1]):
        pares = []
        for y in variables[i + 1:]:
            p = valores_p.get(frozenset((x, y)))
            pares.append(f"{claves[x]}–{claves[y]} {num(matriz.loc[x, y], 3)}" + (f" ({p})" if p else ""))
        filas.append(". ".join(pares) + ".")
    return "\n".join(filas)


def _outliers(df) -> str:
    predictores = datos.VARIABLES_PANTALLA + datos.VARIABLES_SUENO
    resumen = stats.filas_con_outliers(df, predictores)
    por_estado = stats.outliers_por_estado(df, predictores).pivot(index="variable", columns="depressed", values="proporcion")
    filas = [
        f"{entero(resumen['n'])} filas ({pct(resumen['proporcion'])}) con al menos un atípico en un predictor. "
        f"Prevalencia sobre el corte del BDI-II en esas filas {pct(resumen['prevalencia'])} y en las demás "
        f"{pct(resumen['prevalencia_resto'])}.",
        "Regla de Tukey con 1,5 × IQR. Se indican cuartiles, límites, conteos bajo/alto y porcentaje de atípicos.",
    ]
    for r in stats.outliers_iqr(df, datos.VARIABLES_NUMERICAS).itertuples(index=False):
        detalle = (
            f"{datos.ETIQUETAS[r.variable]} · Q1 {num(r.q1)}, Q3 {num(r.q3)}, "
            f"límites {num(r.lim_inf)} a {num(r.lim_sup)}, conteos {entero(r.n_bajo)}/{entero(r.n_alto)} "
            f"({pct(r.proporcion)})."
        )
        if r.variable in por_estado.index:
            proporciones = por_estado.loc[r.variable]
            detalle += (f" Dentro de BDI-II < {datos.CORTE_BDI} · {pct(proporciones[0])}. "
                        f"Dentro de BDI-II ≥ {datos.CORTE_BDI} · {pct(proporciones[1])}.")
        filas.append(detalle)
    return "\n".join(filas)


_RESUMENES = {
    "inicio": _inicio,
    "calidad_datos": _calidad,
    "variable_objetivo": _objetivo,
    "analisis_sexo": _sexo,
    "analisis_numerico": _numericas,
    "correlaciones": _correlaciones,
    "outliers": _outliers,
}


@lru_cache(maxsize=len(_RESUMENES))
def _contexto_cacheado(seccion: str) -> str:
    df = datos.cargar_datos()
    if df.empty:
        raise datos.DatosNoDisponibles("Dataset vacío")
    return _RESUMENES[seccion](df)


def contexto_eda(seccion: str) -> str:
    """Devuelve únicamente los agregados de la sección indicada."""
    if seccion not in _RESUMENES:
        return ""
    try:
        return _contexto_cacheado(seccion)
    except (datos.DatosNoDisponibles, OSError, ValueError, KeyError):
        return "Las cifras agregadas de esta sección no están disponibles en este momento."
