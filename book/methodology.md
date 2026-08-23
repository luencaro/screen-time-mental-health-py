# Metodología
A continuacion se presenta la metodologia a seguir en este proyecto. 

## 1. Descripción del dataset

Se utilizará el dataset `screen_time_mental_health.csv`, adquirido desde Kaggle: [Screen Time vs Mental Health (ML-Ready)](https://www.kaggle.com/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready). El dataset está compuesto por 4,810 registros sin valores faltantes y 10 variables:

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

## 2. EDA
- Estadística descriptiva por variable y por género.
- Distribución del target y de `bdi_total`.
- Matriz de correlación entre pantalla, sueño y `bdi_total`.
- Pruebas de diferencia por género.
- Visualizaciones: histogramas, boxplots, etc.

## 3. Preprocesamiento
- Split del data set.
- Estandarización de variables continuas.
- Manejo de desbalance: SMOTE o `class_weight='balanced'`, aplicado solo en entrenamiento.

## 4. Modelado
- **Algoritmos:** regresión logística, Random Forest, XGBoost {cite}`chen2026predicting`.
- **Configuraciones:** (a) modelo conjunto con `sex` como predictor, (b) modelo solo chicos, (c) modelo solo chicas — para contrastar con {cite}`hokby2025adolescents`.
- **Tuning:** `GridSearchCV`/`RandomizedSearchCV`, validación cruzada k=5, optimizando AUC-PR.

## 5. Evaluación
- AUC-ROC, precisión, recall, F1 (clase positiva).
- Matriz de confusión y curva Precision-Recall.
- Comparación contra baseline (clase mayoritaria).

## 6. Interpretabilidad
- Feature importance nativa (RF, XGBoost).
- SHAP — contrastar si `sleep_quality_index` es el predictor dominante, y comparar entre modelos por género.
- Coeficientes de regresión logística como referencia adicional.

## 7. Herramientas
- Python: pandas, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib/seaborn.

## 8. Consideraciones éticas
- Datos anonimizados de menores; sin intento de reidentificación.
- Diseño transversal en este proyecto (vs. longitudinal del estudio original): resultados interpretados como asociaciones predictivas, no causales.