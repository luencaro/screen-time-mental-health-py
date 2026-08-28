# Contenido

- Introducción
- Antecedentes
- Objetivos
- Metodología
- Análisis exploratorio de datos (EDA)
- Referencias

# Introducción

El uso de pantallas (redes sociales, videojuegos, streaming, mensajería) se ha vuelto una actividad central en la vida de los adolescentes, y un cuerpo creciente de investigación asocia un mayor tiempo de pantalla con síntomas depresivos, peor calidad de sueño y menor bienestar psicológico. Sin embargo, la relación no es simple ni unidireccional: el sueño parece actuar como mediador clave entre el tiempo de pantalla y la depresión, y el tamaño del efecto varía según el tipo de uso, la plataforma y el diseño del estudio. Este proyecto busca aprovechar un dataset de 4.810 adolescentes suecos (de entre 12 y 16 años) participantes en un estudio prospectivo realizado en centros escolares de Estocolmo para explorar y modelar esta relación con métodos de ciencia de datos.

# Antecedentes
El interés por los efectos del tiempo de pantalla sobre la salud mental adolescente ha crecido junto con el aumento de las tasas de depresión en esta población. La evidencia disponible sugiere que el sueño y el afrontamiento son dos vías explicativas clave, además de existir aproximaciones predictivas basadas en aprendizaje automático. Investigaciones como las de Hökby et al. {cite}`hokby2023longitudinal`, en la cual analizaron de manera longitudinal de 4793 adolescentes suecos (0, 3 y 12 meses) y encontraron que el tiempo de pantalla no predijo la depresión de forma directa (p = 0.469), pero sí de manera indirecta cuando interfirió con el afrontamiento centrado en el problema (interacción tiempo × pantalla × PFE: b = 0.009; p < 0.01), con un efecto máximo de 3.4 puntos en el BDI-II. El afrontamiento centrado en la emoción no mostró efectos significativos.

Lemke et al. {cite}`lemke2023associations`, en una muestra transversal de 8449 adolescentes suecos (12–16 años), reportaron que el 45.6% dormía menos de 8 horas en días de colegio (68.96% entre los deprimidos). La duración del sueño entre semana (OR = 0.773; p < 0.0001), la calidad del sueño (OR = 0.327; p < 0.0001) y un cronotipo tardío (OR = 1.126; p = 0.0017) predijeron la depresión (BDI-II > 13); un aumento de 30 minutos de sueño se asoció con un 10% menos de probabilidad de depresión. La duración del sueño en fin de semana no fue significativa.

Por otro lado, Saleem et al. {cite}`saleem2024exploring` realizaron una revisión narrativa sistemática hasta 2024 que incluyó 67 estudios de 4850 identificados, en población de 5 a 18 años. Encontraron una asociación positiva consistente entre el uso de redes sociales y los síntomas de depresión y ansiedad, aunque señalaron alta heterogeneidad metodológica entre los estudios.

Retomando con el marco de estudio de Hökby et al. {cite}`hokby2025adolescents`, tenemos que pusieron a prueba la teoría del desplazamiento pantalla-sueño en 4810 adolescentes suecos (12–16 años) mediante SEM multigrupo. El tiempo de pantalla deterioró el sueño a los 3 meses (β entre 0.14 y 0.30). En chicos, el efecto sobre la depresión a los 12 meses fue directo (β = 0.02; p < 0.038), sin mediación del sueño. En chicas, el efecto fue mediado por calidad del sueño (57%), duración (38%) y cronotipo (45%); el jetlag social no fue significativo.

Por ultimo, tenemos un enfoque predictivo del caso de estudio de tiempo en pantallas y salud mental. Chen et al. {cite}`chen2026predicting` compararon cinco algoritmos de aprendizaje automático para predecir el riesgo de depresión en 1226 adolescentes chinos (13–25 años). El modelo XGBoost obtuvo el mejor desempeño (AUC = 0.927). Según SHAP, los predictores más relevantes fueron el uso de somníferos, la indiferencia parental y el nivel educativo. En el análisis de regresión logística, el uso de somníferos (OR = 11.87), padres violentos (OR = 1.69) y presentar más de un trastorno del sueño (OR = 1.89) fueron factores de riesgo independientes.

En conjunto, la evidencia converge en el sueño (calidad, duración y cronotipo) como vía central y modificable entre el uso de pantallas y la depresión adolescente, con diferencias por género y matices según el afrontamiento disponible, aunque persisten limitaciones metodológicas —diseños mayormente transversales y heterogeneidad en la medición del uso de pantallas— que motivan el presente estudio.

# Objetivos
## Objetivo General
Analizar y modelar, mediante técnicas de análisis exploratorio de datos y aprendizaje automático, la relación entre el tiempo de pantalla, los cuatro dominios del sueño (calidad, duración, cronotipo y jet lag social) y la severidad de los síntomas depresivos (BDI-II) en una muestra de adolescentes suecos, con el fin de identificar los predictores de mayor peso en el riesgo depresivo y evaluar si un enfoque predictivo confirma los patrones de mediación reportados en el estudio original {cite}`hokby2025adolescents`.

## 2.2 Objetivos específicos
1. Realizar un análisis exploratorio de datos (EDA) para caracterizar la distribución del tiempo de pantalla, de los cuatro dominios de sueño (SQI, WASD, cronotipo, jet lag social) y del puntaje BDI-II (total), incluyendo la proporción de casos sobre el punto de corte clínico de depresión leve (>13 puntos).
2. Evaluar la asociación estadística entre el tiempo de pantalla y cada uno de los cuatro dominios de sueño por separado, en lugar de tratar "sueño" como una variable única, dado que el estudio base encontró tamaños de efecto distintos para cada dominio.
3. Analizar las diferencias por género en las variables de pantalla, sueño y depresión, replicando el hallazgo del estudio base {cite}`hokby2025adolescents` de que las chicas reportan puntajes de depresión considerablemente más altos y que la mediación por sueño solo resultó significativa en el subgrupo femenino.
4. Construir y comparar modelos de machine learning (p. ej. regresión logística, random forest, XGBoost) para predecir el nivel de riesgo depresivo (target definido en el dataset) a partir de variables de pantalla, sueño y demográficas. Adicionalmente se plantea el entrenamiento y evaluacion de modelos de forma separada por género para contrastar con el enfoque multigrupo del estudio original {cite}`hokby2025adolescents`.
5. Identificar mediante técnicas de importancia de variables cuáles predictores tienen mayor peso en el modelo, y contrastar explícitamente si el tiempo en pantalla y el sueño emerge como el predictores más relevantes.
