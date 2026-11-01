# Metodología del modelado

En esta sección se explican los procedimientos comunes a todos los notebooks de modelado, tanto de clasificación (`depressed`) como de regresión (`bdi_total`). Cada notebook muestra los resultados de un modelo y remite aquí para la justificación de cada paso. El código común está en `src/Modelado.py` (clasificación y elementos compartidos) y en `src/ModeladoRegresion.py` (regresión).

(mod-datos)=
## 1. Datos, predictores y split

Se modela a todos los adolescentes juntos, con `sex` como un predictor más, porque el EDA mostró que es un factor de riesgo fuerte (24.4% de depresión en chicas frente a 8.6% en chicos). Se usan **6 predictores**: los cuatro dominios del sueño (`sleep_quality_index`, `avg_sleep_hours`, `midsleep_weekend_hours` y `social_jetlag_hours`), el índice de tiempo de pantalla (`screen_time_index`) y el sexo.

Se eliminan explícitamente de X:

- `subject_id`: es solo un identificador.
- `bdi_total` y `depressed`: son los targets. `depressed` se construye a partir de `bdi_total` (≥ 14), así que usar uno para predecir el otro sería fuga de información.
- `est_leisure_screen_hours`: es redundante con `screen_time_index` (Spearman r = 0.95).

A diferencia de los benchmarks del EDA, se incluye `social_jetlag_hours`. Aunque está correlacionada con `midsleep_weekend_hours` (r = 0.84), los modelos regularizados y los de árboles toleran bien esa colinealidad.

**Split.** Se separa un 20% de los datos como **test**, estratificando por la combinación **sexo × depresión** para que train y test conserven la prevalencia de depresión dentro de cada sexo. El split se guarda en `data/processed/modelado/split_indices.csv` y lo reutilizan todos los notebooks, de clasificación y de regresión, así que ambos enfoques se evalúan sobre los mismos adolescentes.

**Preprocesamiento.** Va dentro de un `Pipeline` junto con cada modelo: `StandardScaler` en las 5 numéricas y `OneHotEncoder(drop="first")` en `sex` (queda una sola columna binaria, `sex_Girl`). Así, en cada fold de la validación cruzada el escalado se ajusta solo con los datos de entrenamiento de ese fold. Los modelos de árboles no necesitan escalado, pero se mantiene el mismo preprocesamiento para todos, de modo que todos los modelos guardados reciben exactamente la misma entrada.

**Validación cruzada.** Clasificación: `StratifiedKFold(5, shuffle=True, random_state=42)`. Regresión: `KFold(5, shuffle=True, random_state=42)`. Como el train se ordena por `subject_id`, todos los notebooks de un mismo problema usan exactamente los mismos folds.

**Test.** Se usa **una sola vez**, en los notebooks de comparación. Ni el tuning, ni la elección de estrategia, variante, modelo o umbral, ni la corrección de smearing lo ven.

(mod-fases)=
## 2. Organización del modelado

| Problema | Notebook | Qué hace |
|---|---|---|
| Clasificación | `Modelado.ipynb` | Versión previa: benchmark conjunto de 7 modelos, todos con pesos de clase |
| Clasificación | `clf_<modelo>.ipynb` (fase 1) | Para cada modelo, compara las estrategias de manejo del desbalance |
| Clasificación | `clf_comparacion.ipynb` (fases 2 y 3) | Compara los ganadores, los evalúa en test y optimiza el umbral |
| Regresión | `reg_<modelo>.ipynb` (fase 1) | Para cada modelo, compara las variantes del target |
| Regresión | `reg_comparacion.ipynb` (fase 2) | Compara los ganadores, los evalúa en test y prueba el corte clínico |

(mod-clasificacion)=
## 3. Clasificación

(mod-metricas-clf)=
### Métricas

La métrica principal es la **PR-AUC** (average precision), más informativa que la ROC-AUC cuando la clase positiva es minoritaria (16.4%). Como referencia, la PR-AUC de un clasificador aleatorio es igual a la prevalencia (≈ 0.164). La PR-AUC y la ROC-AUC solo dependen de cómo se **ordenan** las probabilidades, no de un umbral.

Precision, recall, F1 y accuracy se calculan con el umbral por defecto de 0.5 sobre `predict_proba`, igual para todos los modelos. Con una prevalencia del 16.4%, predecir siempre «no deprimido» ya da una accuracy de 0.836, así que una accuracy cercana a ese valor no refleja capacidad predictiva. El umbral 0.5 no está optimizado: el de operación se elige en la fase 3, solo con el train.

(mod-desbalance)=
### Manejo del desbalance

En la fase 1, cada modelo se ajusta con tres estrategias:

| Estrategia | Qué hace |
|---|---|
| **a) Nativo** | Sin balanceo: el modelo ve la proporción real (16% de positivos). |
| **b) Pesos de clase** | Los errores en la clase minoritaria pesan más al entrenar. |
| **c) SMOTENC** | Genera deprimidos sintéticos a partir de sus vecinos deprimidos más cercanos, hasta igualar las clases. Sin pesos de clase. |

Cada librería aplica los pesos de clase con su propio parámetro:

| Modelo | Pesos de clase |
|---|---|
| Logística, Ridge, Lasso, SVC y LightGBM | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced_subsample"` (los pesos se recalculan en la muestra *bootstrap* de cada árbol) |
| XGBoost | `scale_pos_weight = n_neg / n_pos` del train (≈ 5.1) |
| CatBoost | `auto_class_weights="Balanced"` |
| KNN | No aplica: `KNeighborsClassifier` no admite `class_weight` |

**SMOTENC** se configura con `categorical_features=["sex"]`: interpola solo las numéricas y a cada caso sintético le asigna el sexo más frecuente entre sus vecinos. Va **antes** del preprocesamiento, dentro de un `Pipeline` de imbalanced-learn, y solo se aplica en `fit` (en los folds de entrenamiento). Por eso la validación y el test nunca contienen casos sintéticos.

Los pesos de clase y SMOTENC no cambian cómo se ordenan las probabilidades, sino su escala: desplazan las probabilidades hacia la clase positiva, lo que con el umbral 0.5 cambia mucho la precision y el recall, aunque la PR-AUC apenas cambie.

(mod-seleccion-clf)=
### Tuning y selección

Para cada estrategia se ejecuta un `RandomizedSearchCV` **solo sobre el train**, con los mismos folds y `scoring="average_precision"` (PR-AUC). Con `refit=True`, el mejor Pipeline queda reentrenado con todo el train.

| Modelos | Combinaciones por estrategia |
|---|---|
| Logística | Sin tuning (`cross_validate`) |
| Ridge y Lasso | 10 |
| KNN y SVC | 20 |
| Random Forest, XGBoost, LightGBM y CatBoost | 25 |

Se elige la estrategia con **mayor PR-AUC media de CV**. Como la PR-AUC no depende del umbral, un recall bajo con el umbral 0.5 no es un problema para la selección: el umbral de operación se optimiza en la fase 3, con probabilidades *out-of-fold* del train.

(mod-regresion)=
## 4. Regresión

El BDI-II es un entero entre 0 y 63 muy sesgado a la derecha (media 7.27, mediana 5, desviación estándar 7.70 y asimetría 1.89). La métrica principal es el **RMSE**, acompañado del MAE y el R². Todas las predicciones de todos los modelos se recortan al rango del BDI-II, $[0, 63]$, y todas las métricas se calculan en puntos BDI, nunca en una escala transformada.

(mod-variantes)=
### Variantes del target

Los predictores no se transforman (salvo el preprocesamiento); lo que cambia es la escala en la que se entrena el target:

| Variante | Qué se entrena | Cómo se predice | Modelos |
|---|---|---|---|
| **a) Nativo** | `bdi_total` sin transformar (pérdida cuadrática) | Predicción directa | Todos |
| **b) sqrt** | $\sqrt{y}$ | $\hat y = \frac{1}{n}\sum_i (\hat z + e_i)^2$ (smearing) | Todos salvo los de conteo |
| **c) log1p** | $\log(1+y)$ | $\hat y = \frac{1}{n}\sum_i \left(e^{\hat z + e_i} - 1\right)$ (smearing) | Todos salvo los de conteo |
| **d) Pérdida Poisson** | `bdi_total` sin transformar, con pérdida (deviance) Poisson | Predicción directa en la escala de la media | Solo árboles |

En la variante d), cada librería usa su propia pérdida Poisson: `RandomForestRegressor(criterion="poisson")`, XGBoost con `objective="count:poisson"`, LightGBM con `objective="poisson"` y CatBoost con `loss_function="Poisson"`. CatBoost se predice con `prediction_type="Exponent"` para obtener la media y no su logaritmo.

**Modelos de conteo: solo la variante nativa.** El GLM Poisson y la binomial negativa ya modelan conteos: suponen $\log \mu = x\beta$, con $\text{Var}(y) = \mu$ en el Poisson y $\text{Var}(y) = \mu + \alpha\mu^2$ en la binomial negativa. El enlace log ya hace el trabajo que se busca con las transformaciones (efectos multiplicativos, predicciones positivas y tolerancia a la cola derecha). Transformar su target rompería el modelo, porque $\sqrt{y}$ y $\log(1+y)$ no son conteos enteros. Además, su predicción $e^{x\beta}$ ya está en la escala de la media, sin sesgo de retransformación.

(mod-smearing)=
### Sesgo de retransformación y corrección de smearing

En las variantes sqrt y log1p, el modelo aprende la media **en la escala transformada**, $E[g(y)\mid x]$. Al deshacer la transformación se obtiene $g^{-1}(E[g(y)\mid x])$, que **no** es $E[y\mid x]$. Como las inversas ($z^2$ y $e^z - 1$) son convexas, por la desigualdad de Jensen

$$g^{-1}\big(E[g(y)\mid x]\big) \le E[y\mid x],$$

es decir, la predicción retransformada **subestima la media** del BDI, y más cuanto más dispersos son los residuos. Este es el **sesgo de retransformación**. Como el RMSE premia acertar la media condicional, ese sesgo es un error sistemático.

La **corrección de smearing** {cite}`duan1983smearing` no supone ninguna distribución para los errores. Usa los residuos de entrenamiento en la escala transformada, $e_i = g(y_i) - \hat z_i$, como muestra de los errores posibles, y promedia la retransformación sobre todos ellos:

$$\hat y = \frac{1}{n}\sum_{i=1}^{n} g^{-1}(\hat z + e_i)$$

La implementa `RegresorBDI`. En `fit` transforma $y$ y calcula los residuos **solo con los datos de entrenamiento de cada fold**; en `predict` invierte con la corrección y recorta a $[0, 63]$. Para $\sqrt{y}$, la inversa es $\max(z, 0)^2$: un valor negativo en la escala transformada corresponde a $y = 0$.

Hay dos consecuencias que se repiten en todos los modelos:

- **Corrección parcial en modelos flexibles.** En los árboles y en el boosting, los residuos de entrenamiento son más pequeños que los de datos nuevos, porque el modelo se ajusta a ellos. Por eso la corrección es parcial y queda un sesgo residual pequeño (de −0.1 a −0.2 puntos). En KNN con `weights="distance"` la predicción de entrenamiento reproduce cada valor, los residuos son 0 y el smearing no corrige nada.
- **Mejor RMSE, peor MAE.** Sin la corrección, la predicción retransformada se acerca a la mediana condicional, que en una distribución asimétrica está por debajo de la media y es lo que minimiza el MAE. El smearing lleva la predicción a la media, que es lo que minimiza el RMSE. Por eso, al aplicarlo, el RMSE y el R² mejoran y el MAE empeora.

**Efecto del smearing en los notebooks.** Para medir el sesgo, se toma la mejor variante transformada de cada modelo con sus hiperparámetros ya elegidos. En cada fold se entrena una vez y se predice el fold de validación **con y sin** la corrección. Con esas predicciones out-of-fold (solo train) se comparan la media predicha con la real y el RMSE, el MAE y el R² de ambas versiones.

(mod-seleccion-reg)=
### Tuning y selección

Para cada variante se ejecuta un `RandomizedSearchCV` **solo sobre el train**, con los mismos folds y `scoring="neg_root_mean_squared_error"`. Se reporta la media ± desviación estándar de RMSE, MAE y R² en esos folds, y el Pipeline ganador se reentrena con todo el train.

| Modelos | Combinaciones por variante |
|---|---|
| Lineal (OLS) y binomial negativa | Sin tuning (`cross_validate`) |
| Ridge, Lasso, ElasticNet y GLM Poisson | 10 |
| KNN y SVR | 20 |
| Random Forest, XGBoost, LightGBM y CatBoost | 25 (mismos rangos que en la clasificación) |

Se elige la opción con **menor RMSE medio de CV**. **Regla de empate:** si una opción más simple queda a menos de una desviación estándar de CV de la ganadora, se elige la más simple. Una diferencia menor que la variabilidad entre folds no es evidencia de que la opción compleja sea mejor, y la más simple es más fácil de interpretar. Los órdenes de simplicidad son:

- **Variantes (fase 1):** nativo < sqrt < log1p < Poisson.
- **Modelos (fase 2):** lineales (OLS, Ridge, Lasso, ElasticNet) < GLM Poisson < binomial negativa < KNN < SVR < Random Forest < boosting (XGBoost, LightGBM, CatBoost). Entre opciones igual de simples, gana la de menor RMSE.

La desviación estándar del RMSE entre folds es grande (≈ 0.48 puntos) porque los folds difieren en la dispersión del BDI, según cuántos adolescentes con puntajes muy altos les toquen. La del R², que se normaliza por esa dispersión, es mucho menor (≈ 0.02). Por eso la regla de empate es indulgente y suele favorecer la opción más simple.

(mod-hipotesis)=
### Pruebas de hipótesis

Los notebooks `reg_lineal`, `reg_poisson` y `reg_binomial_negativa` terminan con una sección de pruebas de hipótesis. Los modelos se ajustan con statsmodels sobre **todo el train**, con los predictores preprocesados con el mismo `ColumnTransformer` ajustado en el train, más la constante. Así, cada coeficiente numérico corresponde a una desviación estándar del predictor y el de `sex_Girl`, a pasar de chicos a chicas. Para cada prueba se reportan la hipótesis nula, el estadístico, el p-valor y la conclusión con α = 0.05.

**Tamaño muestral.** Con n ≈ 3,800 las pruebas tienen tanta potencia que detectan desviaciones muy pequeñas, sin importancia práctica. Por eso cada prueba se complementa con gráficos (Q-Q, residuos vs. ajustados, frecuencias observadas vs. predichas) y con tamaños de efecto (coeficientes, razones de tasas, R² ajustado, VIF, razón de dispersión), y la conclusión se basa en el conjunto y no solo en el p-valor.

**Razones de tasas.** En los modelos de conteo con enlace log, $e^{\beta}$ es la **razón de tasas**: el factor por el que se multiplica el BDI esperado por cada desviación estándar más del predictor (o al pasar de chicos a chicas), con las demás variables fijas. Una razón de 1.10 equivale a un BDI esperado un 10% mayor. Se reporta con su IC al 95%.

**Sobredispersión.** La prueba de Cameron-Trivedi {cite}`cameron1990regression` contrasta la equidispersión del Poisson con una regresión auxiliar. En la comparación entre la binomial negativa y el Poisson, alpha = 0 está en el borde del espacio de parámetros, así que el p-valor de la razón de verosimilitud se corrige dividiendo por 2 el de la χ²(1) {cite}`self1987asymptotic`.

(mod-comparacion)=
## 5. Comparación final y test

En los notebooks de comparación se cargan los ganadores de la fase 1 sin repetir el tuning y se agrega como **modelo de referencia fijo** el benchmark del EDA, ahora con los 6 predictores y sin tuning: la logística `LogisticRegression(class_weight="balanced", max_iter=1000)` en clasificación y `LinearRegression` con el target nativo en regresión. Sus métricas no coinciden exactamente con las del EDA, porque allí se usaban 4 predictores numéricos más el sexo y otra partición de los datos. El benchmark no compite: es la referencia contra la que se mide la ganancia.

En el test, cada modelo se evalúa **una sola vez**. Para saber si las diferencias entre modelos son reales se usan **intervalos bootstrap pareados al 95%**:

- **Réplicas:** se generan 1000 réplicas del test, remuestreando adolescentes con reemplazo.
- **Evaluación pareada:** en cada réplica se evalúan **todos los modelos sobre las mismas observaciones**.
- **Intervalos:** se obtienen los percentiles 2.5 y 97.5 de la métrica de cada modelo (PR-AUC y ROC-AUC en clasificación; RMSE y R² en regresión), y también los de la **diferencia pareada frente al benchmark**.
- **Lectura:** si el intervalo de la diferencia incluye el 0, no hay evidencia de que el modelo sea distinto del benchmark. La columna `significativa` lo marca.

El intervalo de la diferencia es más informativo que comparar los intervalos individuales: como los modelos se evalúan sobre las mismas réplicas, buena parte de la variabilidad se cancela. El mismo procedimiento, con PR-AUC y ROC-AUC, se usa para comparar el BDI predicho por la regresión con el clasificador final como puntuación de riesgo.

(mod-persistencia)=
## 6. Rendimiento y persistencia

XGBoost y LightGBM entrenan en la GPU si la máquina la tiene (en ese caso, la búsqueda usa un solo proceso), y todo funciona igual sin ella. CatBoost entrena en la GPU en clasificación, pero en regresión usa la CPU, porque con unas 4,000 filas la GPU es más lenta. Los modelos de scikit-learn y statsmodels usan la CPU, con la búsqueda paralelizada en todos los núcleos.

Cada modelo se guarda como un `.joblib` con el Pipeline completo (preprocesamiento + modelo), reentrenado con todo el train y configurado para CPU, de modo que se puede cargar en máquinas sin GPU, como la del dashboard. Los `.joblib` solo se cargan de forma fiable con las mismas versiones de librerías con que se guardaron; por eso están fijadas en `docker/environment.yml` y quedan registradas en los metadatos de cada modelo.
