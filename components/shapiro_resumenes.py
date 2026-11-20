"""Contexto para Shapiro a partir de resúmenes agregados del dashboard."""

from components.formato import entero, num, pct
from data import load_modelado as datos


def _importancias(tabla, modelo: str, metodo: str) -> str:
    seleccion = tabla.loc[tabla["modelo"].eq(modelo) & tabla["metodo"].eq(metodo), ["etiqueta", "valor"]].dropna()
    orden = seleccion["valor"].abs().sort_values(ascending=False).index
    return ", ".join(
        f"{etiqueta} ({num(valor, 3)})" for etiqueta, valor in seleccion.loc[orden].itertuples(index=False)
    )


def contexto_importancias() -> str:
    """Incluye las importancias agregadas de los gráficos publicados."""
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
            variables = _importancias(tabla, modelo, metodo)
            if variables:
                filas.append(f"{titulo}: {variables}.")
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError):
        filas = []
    if not filas:
        return "Los resúmenes exportados de importancia no están disponibles."
    return "\n".join([
        "Importancia de variables por magnitud dentro de cada método.",
        *filas,
        "La media de |SHAP| mide magnitud en las predicciones. Los signos de los coeficientes indican la dirección "
        "de la asociación del modelo. El cambio se expresa por una desviación estándar para predictores numéricos. "
        "El coeficiente de sexo compara chicas con chicos. Los métodos tienen escalas distintas.",
    ])


def contexto_clustering() -> str:
    """Incluye la evaluación del k elegido y los resúmenes de cada perfil."""
    try:
        meta = datos.clustering_meta()
        k = meta["k"]
        evaluacion = datos.clustering_seleccion_k().set_index("k").loc[k]
        columnas = [
            "nombre", "n", "prevalencia_depresion", "prevalencia_ic_inf", "prevalencia_ic_sup", "bdi_media",
            "bdi_mediana", "proporcion_chicas", "media_sleep_quality_index", "media_avg_sleep_hours",
            "media_midsleep_weekend_hours", "media_social_jetlag_hours", "media_screen_time_index",
        ]
        perfiles = datos.clustering_perfiles().sort_values("cluster")[columnas]
        filas = [
            f"{nombre}: n = {entero(n)}, prevalencia {pct(prev)} (IC 95 % de Wilson: {pct(inf)} a {pct(sup)}), "
            f"BDI medio {num(bdi)}, mediana {num(mediana, 1)}, chicas {pct(chicas)}. "
            f"Medias originales: índice calidad {num(calidad)}, sueño {num(sueno)} h, punto medio de fin de semana "
            f"{num(punto_medio)} h, jet lag social {num(jet_lag)} h, índice pantalla {num(pantalla)}."
            for (nombre, n, prev, inf, sup, bdi, mediana, chicas, calidad, sueno, punto_medio,
                 jet_lag, pantalla) in perfiles.itertuples(index=False, name=None)
        ]
        cp1, cp2 = meta["varianza_explicada"]
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError, TypeError):
        return "Los resúmenes exportados de clustering no están disponibles."
    return "\n".join([
        f"Clustering K-Means de variables de pantalla y sueño estandarizadas. k elegido: {entero(k)}, "
        f"silueta {num(evaluacion['silueta'], 3)}, estabilidad ARI {num(evaluacion['ari_media'], 3)} "
        f"± {num(evaluacion['ari_desviacion'], 3)} entre réplicas bootstrap.",
        "El análisis buscó describir de tres a cinco perfiles. El criterio combina separación "
        "(silueta y Calinski–Harabasz) y estabilidad (ARI).",
        f"El gráfico PCA es una proyección, con {pct(cp1)} de varianza en CP1 y {pct(cp2)} en CP2. "
        "Las cruces marcan centroides. En el gráfico de perfiles, cero es la media global. "
        "Un mayor índice de calidad significa peor calidad del sueño.",
        "La prevalencia es la proporción por perfil que alcanzó el corte del BDI-II y se calculó después de agrupar.",
        *filas,
    ])


def contexto_genero() -> str:
    """Compara los puntos publicados del modelo conjunto con cada modelo específico."""
    try:
        resumen = datos.genero_resumen()
        umbral_conjunto = num(resumen["umbral_conjunto"], 3)
        filas = [
            "Modelos por género. Métricas en el conjunto de prueba de cada grupo. "
            "En clasificación, comparar CatBoost conjunto con específico cambia el modelo y el umbral de operación.",
        ]
        for grupo in ("chicos", "chicas"):
            info = resumen["grupos"][grupo]
            roc, rmse, test = info["roc_auc"], info["rmse"], info["tamanos"]["test"]
            conjunto, propio = info["umbral_conjunto"], info["umbral_propio_metricas"]
            filas.append(
                f"{info['nombre']} (test n = {entero(test['filas'])}, prevalencia {pct(test['prevalencia'])}), "
                f"conjunto (umbral {umbral_conjunto}): ROC-AUC {num(roc['conjunto']['valor'], 3)}, "
                f"recall {pct(conjunto['recall'])}, precisión {pct(conjunto['precision'])}. Específico (umbral "
                f"{num(info['umbral_propio'], 3)}): ROC-AUC {num(roc['especifico']['valor'], 3)}, "
                f"recall {pct(propio['recall'])}, precisión {pct(propio['precision'])}. "
                f"RMSE de ElasticNet en puntos BDI-II (conjunto / específico): {num(rmse['conjunto']['valor'], 3)} / "
                f"{num(rmse['especifico']['valor'], 3)}."
            )
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError):
        return "Los resúmenes exportados de modelos por género no están disponibles."

    try:
        tabla = datos.genero_importancias()
        for grupo in ("chicos", "chicas"):
            variables = _importancias(tabla[tabla["grupo"].eq(grupo)], "catboost", "shap_global")
            if variables:
                filas.append(f"Medias de |SHAP| en {grupo} (log-odds): {variables}.")
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError):
        filas.append("La importancia por género no está disponible en la exportación.")
    return "\n".join(filas)


def contexto_prediccion() -> str:
    """Describe los controles con los límites publicados del formulario."""
    try:
        rangos = datos.rangos_prediccion()
        filas = []
        for variable in rangos["orden"]:
            info = rangos["variables"][variable]
            valores = info["valores"]
            minimo, maximo = (valores[0], valores[-1]) if valores else (info["p1"], info["p99"])
            filas.append(f"{info['etiqueta']}: {num(minimo)} a {num(maximo)} {info['unidad']}. {info['descripcion']}")
    except (datos.ResultadosNoDisponibles, OSError, ValueError, KeyError, TypeError):
        filas = []
    return "\n".join([
        "En Predicción, elige una categoría de sexo, ajusta los controles de ejemplo y pulsa Calcular. "
        "CatBoost muestra probabilidad estimada y una clasificación según el umbral elegido. ElasticNet "
        "muestra un puntaje BDI-II estimado, recortado a la escala del cuestionario. El simulador es académico.",
        *filas,
    ])
