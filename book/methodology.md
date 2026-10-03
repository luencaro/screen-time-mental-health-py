# Metodología

A continuación se presenta la metodología que se seguirá en este proyecto.

## 1. Descripción del dataset

Se utilizará el dataset `screen_time_mental_health.csv`, adquirido desde Kaggle: [Screen Time vs Mental Health (ML-Ready)](https://www.kaggle.com/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready). El dataset está compuesto por 4,810 registros sin valores faltantes y 10 variables, cuya descripción se resume en la siguiente tabla. Cabe destacar que `subject_id` es únicamente un identificador y no se utilizará como predictor, y que la variable objetivo `depressed` presenta un desbalance de clases, con apenas un 16.4% de casos positivos.

| Variable | Tipo | Descripción |
|---|---|---|
| `subject_id` | Identificador | No se usa como predictor |
| `sex` | Categórica binaria | Boy / Girl (2,446 / 2,364) |
| `screen_time_index` | Continua (1–6) | Índice combinado de tiempo de pantalla |
| `est_leisure_screen_hours` | Continua | Horas estimadas de pantalla de ocio |
| `sleep_quality_index` | Continua | Calidad del sueño (SQI) |
| `avg_sleep_hours` | Continua | Duración promedio de sueño semanal (WASD) |
| `midsleep_weekend_hours` | Continua | Punto medio del sueño en fin de semana (proxy de cronotipo) |
| `social_jetlag_hours` | Continua | Jet lag social |
| `bdi_total` | Discreta (0–63) | Puntaje total BDI-II |
| `depressed` | Binaria | Target (0/1), corte clínico BDI-II > 13. Desbalanceada: 16.4% positivos |

## 2. Análisis exploratorio de datos (EDA)

El análisis exploratorio comenzará con un estudio descriptivo de cada variable, el cual se repetirá posteriormente de forma separada por género para identificar posibles diferencias entre chicos y chicas. Se analizará la distribución de las dos variables objetivo: `depressed`, una variable categórica dicotómica que indica si una persona presenta o no depresión, y `bdi_total`, una variable numérica correspondiente al Beck Depression Inventory, que mide la severidad de los síntomas depresivos. Finalmente, se examinará la matriz de correlación entre las variables y se realizarán pruebas estadísticas que permitan respaldar formalmente los patrones observados.

## 3. Preprocesamiento

El preprocesamiento iniciará con la división del dataset en conjuntos de entrenamiento y prueba, con el fin de evaluar los modelos sobre datos no vistos. Posteriormente, las variables continuas serán estandarizadas para que se encuentren en escalas comparables, lo cual es especialmente relevante para modelos de regresión. Por otro lado, dado el desbalance de clases en la variable objetivo, se abordará mediante la técnica SMOTE o mediante el parámetro `class_weight='balanced'`, aplicando cualquiera de estas estrategias únicamente sobre el conjunto de entrenamiento para evitar fugas de información hacia la evaluación.

## 4. Modelado

Se entrenarán diferentes algoritmos de regresión y clasificación {cite}`chen2026predicting`. Cada uno de ellos se ajustará bajo tres configuraciones distintas: un modelo conjunto que incluye `sex` como predictor, un modelo entrenado únicamente con chicos y un modelo entrenado únicamente con chicas, de modo que sea posible contrastar los resultados con lo reportado en {cite}`hokby2025adolescents`. Los hiperparámetros se optimizarán mediante `GridSearchCV` o `RandomizedSearchCV`, empleando validación cruzada con k=5 y tomando como métrica de optimización el RMSE para regresión y PR-AUC para clasificación, la cual resulta apropiada para la naturaleza de los puntajes y el desbalance de clase respectivamente.

## 5. Evaluación

Ademas de las metricas principales mencionadas anterioremente, el desempeño de los modelos se evaluará a partir de las métricas AUC-ROC, precisión, recall y F1, complementadas con la matriz de confusión y la curva Precision-Recall, que permiten entender el tipo de errores que comete cada modelo. 

## 6. Interpretabilidad

Para interpretar los modelos se utilizará la importancia de variables nativa de Random Forest y XGBoost. A partir de ella se evaluará si `sleep_quality_index` es efectivamente el predictor dominante de la depresión, y se comparará la jerarquía de variables entre los modelos específicos por género, con el fin de identificar si los factores asociados difieren entre chicos y chicas.

## 7. Herramientas

El proyecto se desarrollará en Python, haciendo uso de librerías como pandas para la manipulación de datos, scikit-learn para el modelado y la evaluación, y matplotlib y seaborn para la visualización, junto con otras herramientas complementarias según se requiera.