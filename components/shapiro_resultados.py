"""Resumen del benchmarking a partir de las métricas agregadas del dashboard.

Los cargadores se eligen aquí. El chat no puede pedir archivos, filas de
participantes, predicciones individuales ni modelos serializados.
"""

from components.formato import num
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

    test = _leer_exportacion(cargar_test)
    if test is not None and {"metrica", "modelo", "valor"} <= set(test.columns):
        nombres.setdefault("benchmark", "Benchmark de referencia")
        ranking_test = _ranking(test[test["metrica"].eq(metrica)], "valor", nombres, ascendente)
        if ranking_test:
            filas.append(f"{metrica} en test reservado, estimaciones puntuales, mejor primero: {ranking_test}.")
    return filas, final


def _metricas_publicadas(final: dict, clave: str, metricas: tuple[str, ...]) -> str:
    resumen = final.get(clave, {})
    return ", ".join(
        f"{metrica} {num(resumen[metrica], 4)}" for metrica in metricas if metrica in resumen
    )


def contexto_resultados() -> str:
    """Criterios, modelos seleccionados y comparación CV/test por tarea."""
    clasificacion, clf = _resumen_benchmark(
        "PR-AUC", False, datos.clasificacion_final,
        datos.clasificacion_benchmark_cv, datos.clasificacion_benchmark_test,
    )
    regresion, reg = _resumen_benchmark(
        "RMSE", True, datos.regresion_final,
        datos.regresion_benchmark_cv, datos.regresion_benchmark_test,
    )

    if clf and clf.get("nombre"):
        metricas = _metricas_publicadas(clf, "metricas_test_umbral", ("PR-AUC", "ROC-AUC", "Precision", "Recall"))
        if metricas and clf.get("umbral") is not None:
            clasificacion.append(
                f"En Modelos finales, {clf['nombre']} en test con umbral {num(clf['umbral'], 4)}: {metricas}."
            )
    if reg and reg.get("nombre"):
        metricas = _metricas_publicadas(reg, "metricas_test", ("RMSE", "MAE", "R²"))
        if metricas:
            regresion.append(f"En Modelos finales, {reg['nombre']} en test: {metricas}. RMSE y MAE en puntos BDI-II.")

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
            "con los intervalos de incertidumbre. El benchmark sirve de referencia."
        )
    return "\n".join(partes)
