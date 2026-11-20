"""Contexto para Shapiro a partir de resúmenes agregados del dashboard."""

from components.formato import entero, num, pct
from data import load_modelado as datos


def _tres_importancias(tabla, modelo: str, metodo: str) -> str:
    seleccion = tabla.loc[tabla["modelo"].eq(modelo) & tabla["metodo"].eq(metodo), ["etiqueta", "valor"]].dropna()
    mayores = seleccion["valor"].abs().nlargest(3).index
    return ", ".join(
        f"{etiqueta} ({num(valor, 3)})" for etiqueta, valor in seleccion.loc[mayores].itertuples(index=False)
    )


def contexto_importancias() -> str:
    """Proyecta solo las variables más visibles de cada gráfico de importancia."""
    try:
        tabla = datos.importancias()
        graficos = (
            ("catboost", "shap_global", "CatBoost, media de |SHAP| en log-odds"),
            ("catboost", "prediction_values_change", "CatBoost, PredictionValuesChange en porcentaje"),
            ("elasticnet", "coeficiente", "ElasticNet, coeficiente en puntos BDI-II"),
            ("elasticnet", "permutacion", "ElasticNet, cambio de RMSE al permutar en puntos BDI-II"),
        )
        filas = []
        for modelo, metodo, titulo in graficos:
            variables = _tres_importancias(tabla, modelo, metodo)
            if variables:
                filas.append(f"{titulo}: {variables}.")
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError):
        filas = []
    if not filas:
        return "Los resúmenes exportados de importancia no están disponibles."
    return "\n".join([
        "Modelos finales. Hasta tres variables con las mayores magnitudes en cada gráfico de importancia.",
        *filas,
        "La media de |SHAP| mide magnitud en las predicciones. Los signos de los coeficientes indican la dirección "
        "de la asociación del modelo. El cambio se expresa por una desviación estándar para predictores numéricos. "
        "El coeficiente de sexo compara chicas con chicos. Los métodos tienen escalas distintas.",
    ])


def contexto_clustering() -> str:
    """Incluye la evaluación del k elegido y los resúmenes de cada perfil."""
    try:
        k = datos.clustering_meta()["k"]
        evaluacion = datos.clustering_seleccion_k().set_index("k").loc[k]
        columnas = [
            "nombre", "n", "prevalencia_depresion", "prevalencia_ic_inf", "prevalencia_ic_sup", "bdi_media",
        ]
        perfiles = datos.clustering_perfiles().sort_values("cluster")[columnas]
        filas = [
            f"{nombre}: n = {entero(n)}, prevalencia {pct(prev)} (IC 95 % de Wilson: {pct(inf)} a {pct(sup)}), "
            f"BDI-II medio {num(bdi)} puntos."
            for nombre, n, prev, inf, sup, bdi in perfiles.itertuples(index=False, name=None)
        ]
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError):
        return "Los resúmenes exportados de clustering no están disponibles."
    return "\n".join([
        f"Clustering K-Means de variables de pantalla y sueño estandarizadas. k elegido: {entero(k)}, "
        f"silueta {num(evaluacion['silueta'], 3)}, estabilidad ARI {num(evaluacion['ari_media'], 3)} "
        f"± {num(evaluacion['ari_desviacion'], 3)} entre réplicas bootstrap.",
        "El análisis buscó describir de tres a cinco perfiles. El criterio combina separación "
        "(silueta y Calinski–Harabasz) y estabilidad (ARI).",
        "La prevalencia es la proporción por perfil que alcanzó el corte del BDI-II y se calculó después de agrupar.",
        *filas,
    ])


def contexto_genero() -> str:
    """Compara los puntos publicados del modelo conjunto con cada modelo específico."""
    try:
        resumen = datos.genero_resumen()
        umbral_conjunto = num(resumen["umbral_conjunto"], 3)
        filas = [
            "Modelos por género. Métricas de CatBoost en el conjunto de prueba de cada grupo. "
            "Comparar conjunto con específico cambia tanto el modelo como el umbral de operación.",
        ]
        for grupo in ("chicos", "chicas"):
            info = resumen["grupos"][grupo]
            roc = info["roc_auc"]
            conjunto, propio = info["umbral_conjunto"], info["umbral_propio_metricas"]
            filas.append(
                f"{info['nombre']}, conjunto (umbral {umbral_conjunto}): ROC-AUC {num(roc['conjunto']['valor'], 3)}, "
                f"recall {pct(conjunto['recall'])}, precision {pct(conjunto['precision'])}. "
                f"Específico (umbral {num(info['umbral_propio'], 3)}): ROC-AUC {num(roc['especifico']['valor'], 3)}, "
                f"recall {pct(propio['recall'])}, precision {pct(propio['precision'])}."
            )
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError):
        return "Los resúmenes exportados de modelos por género no están disponibles."

    try:
        tabla = datos.genero_importancias()
        for grupo in ("chicos", "chicas"):
            variables = _tres_importancias(tabla[tabla["grupo"].eq(grupo)], "catboost", "shap_global")
            if variables:
                filas.append(f"Mayores medias de |SHAP| en {grupo} (log-odds): {variables}.")
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError):
        filas.append("La importancia por género no está disponible en la exportación.")
    return "\n".join(filas)
