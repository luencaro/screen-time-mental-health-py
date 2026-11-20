"""Documentación del proyecto seleccionada por sección para las consultas de Shapiro."""

from content import libro
from content.fichas import estudios

MAXIMO_DOCUMENTACION = 2600

# Los nombres de archivo y encabezado son fijos, con el mismo lector del dashboard.
FRAGMENTOS = {
    "inicio": (("intro.md", "Introducción"),),
    "objetivos": (("intro.md", "Objetivo General"), ("intro.md", "Objetivos específicos")),
    "metodologia": (("methodology.md", "Evaluación"),),
    "variable_objetivo": (("base_estadistica.md", "Variable objetivo y prevalencia"),),
    "analisis_sexo": (("base_estadistica.md", "Chi-cuadrado de independencia"),
                      ("base_estadistica.md", "Prueba U de Mann-Whitney")),
    "analisis_numerico": (("base_estadistica.md", "Estadística descriptiva"),),
    "correlaciones": (("base_estadistica.md", "Correlación"),),
    "outliers": (("base_estadistica.md", "Detección de valores atípicos"),),
    "benchmarking": (("modelado.md", "Comparación final y test"),),
    "modelos_finales": (("base_estadistica.md", "Regresión"), ("base_estadistica.md", "Clasificación")),
}

NOTAS = {
    "inicio": (
        "Inicio presenta la población estudiada, el contexto de la investigación y la "
        "distribución de la clasificación por BDI-II. La etiqueta «deprimido» designa al grupo que "
        "alcanzó el punto de corte del cuestionario aplicado en el estudio."
    ),
    "objetivos": (
        "Los objetivos 1 a 3 corresponden al análisis exploratorio. El primero se consulta en Variable "
        "objetivo, Variables numéricas y Calidad de datos, el segundo en Correlaciones y el tercero en "
        "Análisis por sexo. El cuarto se aborda en Benchmarking y Modelos por género. Para el quinto, "
        "consulta las importancias de Modelos finales y Modelos por género. El enfoque predictivo permite "
        "contrastar patrones del estudio original. Verificar mediación requiere un análisis específico."
    ),
    "metodologia": (
        "La fuente documentada es Screen Time vs Mental Health (ML-Ready), alojada en Kaggle. "
        "El índice de pantalla combina medidas de uso, y las horas de ocio son una estimación separada. "
        "Los cuatro dominios del sueño son calidad (SQI, un índice alto indica peor calidad), duración "
        "promedio (WASD), cronotipo aproximado con el punto medio del sueño del fin de semana y jet lag "
        "social, que resume la diferencia de horarios de sueño entre días de colegio y días libres.\n\n"
        "Los modelos conjuntos usan los cuatro dominios del sueño, el índice de pantalla y el sexo. Se "
        "reserva el 20 % para test, estratificado por sexo y estado según BDI-II. El escalado de las "
        "variables numéricas se ajusta en cada pliegue de entrenamiento y el sexo se codifica como una "
        "indicadora. Se comparan opciones de clasificación nativa, pesos de clase y SMOTENC. SMOTENC "
        "genera casos sintéticos de la clase minoritaria en el pliegue de entrenamiento; conserva el "
        "carácter categórico del sexo. Las variantes de regresión pueden usar el puntaje nativo o una "
        "transformación del objetivo. Su error se evalúa en puntos BDI-II."
    ),
    "marco_teorico": (
        "Marco teórico reúne investigaciones publicadas y la base estadística del dashboard. "
        "Atribuye los hallazgos a sus estudios de origen y distingue el diseño longitudinal, el transversal "
        "y el predictivo. La mediación con modelos de ecuaciones estructurales describe el estudio de "
        "Hökby et al., mientras que el dashboard estudia asociaciones y capacidad predictiva."
    ),
    "calidad_datos": (
        "La sección separa filas, columnas, valores nulos, claves repetidas y filas duplicadas completas. "
        "Una clave repetida significa que el identificador aparece más de una vez; una fila duplicada "
        "completa repite todos los campos. Completitud es el porcentaje de valores presentes dentro de "
        "cada columna. El porcentaje de valores únicos divide el número de valores distintos entre el total de filas. "
        "La tabla descriptiva muestra tamaño, media, mediana, desviación estándar, mínimo, Q1, Q3 y máximo."
    ),
    "variable_objetivo": (
        "El histograma muestra conteos de adolescentes por puntaje BDI-II, con una línea de densidad "
        "suavizada (KDE). La dona Estado según el corte resume los dos grupos. En Dispersión de BDI-II "
        "la línea punteada marca la mediana y la discontinua el corte de 14 puntos. La mediana resume "
        "el centro de la distribución; el corte define los grupos para el análisis."
    ),
    "analisis_sexo": (
        "El odds ratio (chicas / chicos) compara las odds de alcanzar el corte. Las odds son p/(1−p). "
        "La razón de prevalencias compara directamente las proporciones como p_chicas/p_chicos. Un valor "
        "mayor que uno indica una mayor prevalencia u odds en chicas, según la medida."
    ),
    "analisis_numerico": (
        "El histograma usa conteos si se elige Sin separar y porcentajes sobre el total de cada grupo "
        "al separar por sexo o estado según BDI-II. Compara las formas de las distribuciones con la "
        "misma escala. La caja abarca Q1–Q3, con la mediana como referencia. La asimetría positiva "
        "sugiere una cola más larga a la derecha; la negativa, a la izquierda."
    ),
    "correlaciones": (
        "La matriz muestra rho de Spearman para pares de variables. Las tablas separan las asociaciones "
        "de pantalla con cada dominio del sueño y las de sueño con BDI-II. El signo de rho indica la "
        "dirección, su magnitud la intensidad monotónica y el valor p la evidencia respecto a la "
        "hipótesis nula de correlación cero. Evalúa la magnitud para interpretar relevancia práctica."
    ),
    "outliers": (
        "La tabla aplica los límites a cada variable y cuenta los valores bajos y altos. El indicador "
        "Filas con algún outlier cuenta adolescentes que tienen al menos un predictor marcado. "
        "El gráfico por estado muestra la proporción de valores atípicos dentro de cada grupo de BDI-II "
        "y usa los predictores de pantalla y sueño. La regla detecta valores poco frecuentes; antes de "
        "excluirlos se debe comprobar si son errores de medición o valores plausibles."
    ),
    "modelos_finales": (
        "En la matriz de confusión las proporciones se calculan dentro de cada clase real. El umbral "
        "operativo se aprende con predicciones fuera de pliegue del entrenamiento para maximizar la "
        "precisión sujeto a un recall mínimo. En el experimento BDI predicho ≥ 14 se compara un segundo "
        "criterio de alerta sobre el mismo test. Los coeficientes de las variables numéricas de "
        "ElasticNet expresan cambios estimados del BDI-II por una desviación estándar del predictor. "
        "Un R² negativo en test indica más error cuadrático que una predicción constante con la media observada."
    ),
    "clustering": (
        "El gráfico de perfiles muestra medias estandarizadas por grupo. Cero representa la media "
        "global y la unidad es una desviación estándar. Un índice de calidad del sueño alto indica peor "
        "calidad. La silueta mide compacidad y separación; valores más altos indican una asignación más "
        "clara. El ARI resume la estabilidad de las asignaciones bajo remuestreo, ajustada por azar. "
        "En la proyección PCA los ejes son las dos primeras componentes, y las cruces marcan centroides. "
        "BDI-II y la proporción sobre el corte se comparan después de formar los grupos."
    ),
    "genero": (
        "Se compara el CatBoost conjunto con los modelos específicos de chicas y chicos en el test "
        "del grupo. ROC-AUC mide ordenamiento sin fijar un umbral. El gráfico de umbral compara la "
        "precisión y el recall entre el modelo conjunto con umbral único y el modelo específico con "
        "umbral propio. Los umbrales se obtienen en entrenamiento con el mismo criterio de precisión "
        "y recall mínimo. Un valor medio absoluto de SHAP más alto indica mayor contribución típica "
        "en la escala del modelo. La dirección se estudia con los valores SHAP con signo."
    ),
    "prediccion": (
        "El formulario admite un perfil de ejemplo con sexo, índice de pantalla y los cuatro dominios "
        "del sueño. Los controles continuos abarcan de los percentiles 1 a 99 del entrenamiento; para "
        "variables con pocos valores se usan opciones discretas. Tras pulsar Calcular, la clasificación "
        "compara la probabilidad con el umbral de tamizaje y la regresión estima el puntaje entre 0 y 63. "
        "Ese puntaje se etiqueta como mínimo (0–13), leve (14–19), moderado (20–28) o severo (29–63). "
        "Estas etiquetas interpretan el resultado simulado para fines del estudio."
    ),
    "acerca": (
        "Acerca de → Equipo y recursos reúne al equipo de la Universidad del Norte. La tarjeta "
        "Repositorio abre el código y los notebooks en GitHub. Jupyter Book abre la documentación "
        "del proyecto, incluidos los antecedentes, la metodología y el análisis exploratorio. Dataset "
        "abre la ficha Screen Time vs Mental Health (ML-Ready) en Kaggle."
    ),
}


def _fuentes(seccion: str):
    if seccion == "marco_teorico":
        try:
            for ficha in estudios():
                titulo = ficha.get("titulo") or ficha["cita"]
                yield f'{ficha["cita"]} — {titulo}. {ficha["diseno"]}. {ficha["hallazgo"]}'
        except (OSError, ValueError):
            pass

    for archivo, titulo in FRAGMENTOS.get(seccion, ()):
        try:
            fragmento = libro.seccion(archivo, titulo)
        except (OSError, KeyError, ValueError):
            continue
        if fragmento:
            yield f"{titulo}\n{fragmento}"


def contexto_documentacion(seccion: str) -> str:
    """Devuelve un extracto acotado y conserva los párrafos completos."""
    nota = NOTAS.get(seccion, "")
    partes = [nota] if nota else []
    restante = MAXIMO_DOCUMENTACION - len(nota)
    for fuente in _fuentes(seccion):
        for parrafo in libro.parrafos(fuente):
            separador = 2 if partes else 0
            if len(parrafo) + separador > restante:
                break
            partes.append(parrafo)
            restante -= len(parrafo) + separador
    return "\n\n".join(partes)
