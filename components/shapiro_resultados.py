"""Benchmarking y modelos finales desde las métricas agregadas del dashboard.

Los cargadores se eligen aquí. El chat no puede pedir archivos, filas de
participantes, predicciones individuales ni modelos serializados.
"""

from components.formato import entero, num, pct
from data import load_modelado as datos


def _leer_exportacion(cargar):
    """Una exportación ausente no impide responder con las que sí existen."""
    try:
        return cargar()
    except (datos.ResultadosNoDisponibles, OSError, ValueError):
        return None


def _ranking(tabla, columna: str, nombres: dict, ascendente: bool) -> str:
    """Proyecta únicamente identificador de modelo y puntuación agregada."""
    ordenada = tabla.dropna(subset=[columna]).sort_values(columna, ascending=ascendente)
    return "; ".join(
        f"{nombres.get(modelo, modelo)} {num(valor, 4)}"
        for modelo, valor in ordenada[["modelo", columna]].itertuples(index=False, name=None)
    )


def _resumen_benchmark(metrica, ascendente, cargar_final, cargar_cv, cargar_test):
    """Resumen de una tarea con el resultado de selección y sus comparaciones."""
    filas = []
    nombres = {}
    final = _leer_exportacion(cargar_final)
    if final and final.get("nombre"):
        filas.append(f"Modelo final seleccionado: {final['nombre']}.")
        if final.get("explicacion_eleccion"):
            filas.append(f"Motivo registrado de la selección: {final['explicacion_eleccion']}")

    cv = _leer_exportacion(cargar_cv)
    if cv is not None and {"metrica", "es_referencia", "modelo", "nombre", "media"} <= set(cv.columns):
        nombres = dict(zip(cv["modelo"], cv["nombre"]))
        comparables = cv[cv["metrica"].eq(metrica) & cv["es_referencia"].eq(False)]
        ranking_cv = _ranking(comparables, "media", nombres, ascendente)
        if ranking_cv:
            filas.append(f"Validación cruzada en entrenamiento, media de {metrica}, mejor primero: {ranking_cv}.")
        referencia = cv[cv["metrica"].eq(metrica) & cv["es_referencia"].eq(True)]
        ranking_referencia = _ranking(referencia, "media", nombres, ascendente)
        if ranking_referencia:
            filas.append(f"Referencia de CV del EDA, con una división diferente: {ranking_referencia}.")

    test = _leer_exportacion(cargar_test)
    if test is not None and {"metrica", "modelo", "valor"} <= set(test.columns):
        nombres.setdefault("benchmark", "Benchmark de referencia")
        ranking_test = _ranking(test[test["metrica"].eq(metrica)], "valor", nombres, ascendente)
        if ranking_test:
            filas.append(f"{metrica} en test reservado, estimaciones puntuales, mejor primero: {ranking_test}.")
    return filas


def _metricas_publicadas(final: dict, clave: str, metricas: tuple[str, ...]) -> str:
    resumen = final.get(clave, {})
    return ", ".join(
        f"{metrica} {num(resumen[metrica], 4)}" for metrica in metricas if metrica in resumen
    )


def contexto_resultados() -> str:
    """Criterios, modelos seleccionados y comparación CV/test por tarea."""
    clasificacion = _resumen_benchmark(
        "PR-AUC", False, datos.clasificacion_final,
        datos.clasificacion_benchmark_cv, datos.clasificacion_benchmark_test,
    )
    regresion = _resumen_benchmark(
        "RMSE", True, datos.regresion_final,
        datos.regresion_benchmark_cv, datos.regresion_benchmark_test,
    )

    partes = []
    if clasificacion:
        partes.append(
            "Clasificación del estado según el corte BDI-II. La selección usa PR-AUC (mayor es mejor), "
            "una métrica de precisión y recall útil cuando la clase positiva es minoritaria."
        )
        partes.extend(clasificacion)
    if regresion:
        partes.append("Regresión del puntaje BDI-II. Se compara el RMSE en puntos BDI-II (menor es mejor).")
        partes.extend(regresion)
    if partes:
        partes.append(
            "La selección usa validación cruzada en entrenamiento. El test reservado se usa después para "
            "evaluar generalización. Trata su ranking como descriptivo e interpreta las diferencias junto "
            "con los intervalos de incertidumbre. En la pestaña se pueden alternar ambas tareas "
            "y elegir qué modelos incluir en la comparación."
        )
    return "\n".join(partes)


def _resumen_clasificacion(clf: dict) -> list[str]:
    metricas = ("Precision", "Recall", "F1", "Accuracy", "Alertas (%)")
    base = _metricas_publicadas(clf, "metricas_test_umbral", ("PR-AUC", "ROC-AUC"))
    elegido = _metricas_publicadas(clf, "metricas_test_umbral", metricas)
    defecto = _metricas_publicadas(clf, "metricas_test_05", metricas)
    filas = []
    if base:
        filas.append(f"{clf['nombre']} en test reservado (n = {entero(clf.get('n_test'))}): {base}.")
    if elegido and clf.get("umbral") is not None:
        filas.append(f"Umbral elegido {num(clf['umbral'], 4)}: {elegido}.")
    if defecto and clf.get("umbral_defecto") is not None:
        filas.append(f"Umbral de referencia {num(clf['umbral_defecto'], 2)}: {defecto}.")
    if clf.get("criterio_umbral"):
        filas.append(f"Criterio del umbral: {clf['criterio_umbral']}.")

    tabla = _leer_exportacion(datos.clasificacion_matriz_confusion)
    if tabla is not None and {"real", "predicho", "n"} <= set(tabla.columns):
        conteos = tabla.set_index(["real", "predicho"])["n"]
        etiquetas = {(0, 0): "verdaderos negativos", (0, 1): "falsos positivos",
                     (1, 0): "falsos negativos", (1, 1): "verdaderos positivos"}
        if etiquetas.keys() <= set(conteos.index) and conteos.index.is_unique:
            celdas = [f"{entero(conteos.loc[par])} {etiqueta}" for par, etiqueta in etiquetas.items()]
            filas.append("Matriz de confusión en test con el umbral elegido: " + ", ".join(celdas) + ".")
    return filas


def _resumen_regresion(reg: dict) -> list[str]:
    metricas = _metricas_publicadas(reg, "metricas_test", ("RMSE", "MAE", "R²"))
    if not metricas:
        return []
    intervalos = reg.get("intervalos_test", {})
    ic = [
        f"{metrica} {num(intervalos[metrica][0], 3)} a {num(intervalos[metrica][1], 3)}"
        for metrica in ("RMSE", "R²") if metrica in intervalos and len(intervalos[metrica]) == 2
    ]
    filas = [
        f"{reg['nombre']} en test para el BDI-II: {metricas}. RMSE y MAE en puntos BDI-II.",
        "En el gráfico predicho frente a real, la distancia a la diagonal representa error.",
    ]
    if ic:
        filas.append("IC 95 % en test: " + ", ".join(ic) + ".")
    return filas


def _resumen_experimento() -> list[str]:
    tabla = _leer_exportacion(datos.regresion_vs_clasificacion)
    columnas = ["nombre", "pr_auc", "roc_auc", "recall", "precision", "f1", "alertas_pct"]
    if tabla is None or not set(columnas) <= set(tabla.columns):
        return []
    filas = [
        f"{nombre}: PR-AUC {num(pr, 3)}, ROC-AUC {num(roc, 3)}, recall {pct(recall)}, "
        f"precisión {pct(precision)}, F1 {num(f1, 3)} y alertas {pct(alertas / 100)}."
        for nombre, pr, roc, recall, precision, f1, alertas in tabla[columnas].itertuples(index=False, name=None)
    ]
    return ["Experimento de clasificación por los dos enfoques en el mismo test.", *filas] if filas else []


def contexto_modelos_finales() -> str:
    """Resultados finales, sin acceder a predicciones desglosadas por participante."""
    clf = _leer_exportacion(datos.clasificacion_final)
    reg = _leer_exportacion(datos.regresion_final)
    filas = []
    if clf and clf.get("nombre"):
        filas.extend(_resumen_clasificacion(clf))
    if reg and reg.get("nombre"):
        filas.extend(_resumen_regresion(reg))
    filas.extend(_resumen_experimento())
    return "\n".join(filas)
