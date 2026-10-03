# Base estadística

En esta sección se describen los procedimientos estadísticos usados en el proyecto: los estadísticos descriptivos, la detección de valores atípicos, las medidas de asociación, las pruebas de hipótesis y su regla de decisión, y las métricas con las que se evalúan los modelos de regresión y clasificación. En adelante, $x_1, \dots, x_n$ denota los valores observados de una variable en $n$ adolescentes.

## 1. Estadística descriptiva

La **media muestral** resume el centro de la distribución:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

La **mediana** es el valor que deja la mitad de las observaciones ordenadas a cada lado. Es menos sensible que la media a valores extremos, por lo que se reporta junto a ella en variables asimétricas como `bdi_total`. De forma general, el **percentil** $P_k$ deja el $k\,\%$ de las observaciones por debajo; los cuartiles son $Q_1 = P_{25}$, $Q_2 = P_{50}$ (la mediana) y $Q_3 = P_{75}$.

La **varianza muestral** y la **desviación estándar** miden la dispersión alrededor de la media; se divide entre $n-1$ para obtener un estimador insesgado de la varianza poblacional:

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2, \qquad s = \sqrt{s^2}
$$

El **rango intercuartílico** mide la dispersión de la mitad central de los datos:

$$
\mathrm{IQR} = Q_3 - Q_1
$$

La **estandarización** (*z-score*) expresa cada valor en desviaciones estándar respecto a la media y permite comparar variables con escalas distintas. Se usa para visualizar los atípicos de todas las variables en un mismo eje y, en el preprocesamiento, para escalar los predictores continuos:

$$
z_i = \frac{x_i - \bar{x}}{s}
$$

## 2. Detección de valores atípicos

Se aplica la regla de Tukey: una observación se considera atípica cuando queda fuera de los límites

$$
L_{\text{inf}} = Q_1 - 1{,}5\cdot\mathrm{IQR}, \qquad L_{\text{sup}} = Q_3 + 1{,}5\cdot\mathrm{IQR}
$$

## 3. Variable objetivo y prevalencia

El puntaje total del BDI-II va de 0 a 63. La variable binaria `depressed` se define con el corte clínico de depresión leve:

$$
y_i = \begin{cases} 1 & \text{si } \mathrm{BDI}_i \ge 14 \\ 0 & \text{si } \mathrm{BDI}_i < 14 \end{cases}
\qquad\qquad
\hat{p} = \frac{1}{n}\sum_{i=1}^{n} y_i
$$

La **prevalencia** $\hat{p}$ es la proporción de adolescentes con puntaje sobre el corte. Calculada por separado en cada grupo, permite comparar chicas y chicos.

## 4. Correlación

El **coeficiente de Pearson** mide la asociación lineal entre dos variables:

$$
r_{xy} = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i}(x_i - \bar{x})^2}\,\sqrt{\sum_{i}(y_i - \bar{y})^2}}
$$

El **coeficiente de Spearman** es el coeficiente de Pearson aplicado a los rangos de las observaciones (con rango promedio en caso de empates). Mide asociación monótona, no exige normalidad y es robusto ante valores atípicos, por eso es el que se usa en el EDA. Sin empates equivale a

$$
\rho = 1 - \frac{6\sum_{i} d_i^2}{n\,(n^2 - 1)}
$$

donde $d_i$ es la diferencia entre los rangos de $x_i$ y de $y_i$. Su significancia se contrasta con $H_0: \rho = 0$ mediante

$$
t = \rho\,\sqrt{\frac{n-2}{1-\rho^2}} \sim t_{n-2}
$$

Ambos coeficientes van de $-1$ a $1$. Para interpretar la magnitud se usan los umbrales de Cohen: $|\rho| < 0{,}1$ despreciable, $0{,}1$–$0{,}3$ pequeña, $0{,}3$–$0{,}5$ moderada y $\ge 0{,}5$ grande. Dos predictores con $|\rho| \ge 0{,}7$ se consideran redundantes, porque aportan información muy parecida a un modelo.

## 5. Pruebas de hipótesis y decisión mediante valor p

Toda prueba plantea una **hipótesis nula** $H_0$ (no hay diferencia o asociación) y una **alternativa** $H_1$. Con los datos se calcula un estadístico de prueba $T$ cuya distribución bajo $H_0$ es conocida, y el **valor p**: la probabilidad de obtener un estadístico al menos tan extremo como el observado si $H_0$ fuera cierta.

$$
p = P\left(|T| \ge |t_{\text{obs}}| \;\middle|\; H_0\right)
$$

La regla de decisión fija de antemano un **nivel de significancia** $\alpha = 0{,}05$:

- si $p < \alpha$, se rechaza $H_0$ y la diferencia o asociación se considera estadísticamente significativa;
- si $p \ge \alpha$, no se rechaza $H_0$ (lo que no demuestra que $H_0$ sea cierta).

Rechazar $H_0$ siendo cierta es un **error de tipo I**, cuya probabilidad es $\alpha$; no rechazarla siendo falsa es un **error de tipo II**. Con $n = 4.810$, diferencias muy pequeñas resultan significativas, por lo que cada prueba se acompaña de un **tamaño de efecto** que indica si la diferencia es relevante en la práctica.

### Chi-cuadrado de independencia

Contrasta si dos variables categóricas son independientes, en este caso `sex` y `depressed`. A partir de la tabla de contingencia con frecuencias observadas $O_{ij}$ se calculan las esperadas bajo independencia y el estadístico:

$$
E_{ij} = \frac{(\text{total fila } i)\,(\text{total columna } j)}{n}, \qquad
\chi^2 = \sum_{i}\sum_{j}\frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

con $(f-1)(c-1)$ grados de libertad ($1$ en una tabla $2\times 2$). Se usa sin corrección de continuidad de Yates, dado el tamaño de la muestra. El tamaño de efecto es la **V de Cramér**:

$$
V = \sqrt{\frac{\chi^2}{n\,\min(f-1,\;c-1)}}
$$

Para describir la diferencia entre grupos se reportan además la **razón de prevalencias** $\hat{p}_{\text{chicas}} / \hat{p}_{\text{chicos}}$ y el **odds ratio**:

$$
\mathrm{OR} = \frac{a/b}{c/d}
$$

donde $a$ y $b$ son las chicas deprimidas y no deprimidas, y $c$ y $d$ los chicos deprimidos y no deprimidos.

### Prueba U de Mann-Whitney

Compara la distribución de una variable numérica entre dos grupos independientes sin suponer normalidad. Se ordenan juntas las $n_1 + n_2$ observaciones, se suma el rango $R_1$ del primer grupo y se calcula

$$
U_1 = R_1 - \frac{n_1(n_1+1)}{2}, \qquad U_2 = n_1 n_2 - U_1
$$

Bajo $H_0$ (misma distribución en ambos grupos), $U_1$ tiene media $n_1 n_2 / 2$ y varianza $n_1 n_2 (n_1 + n_2 + 1)/12$; con muestras grandes el valor p se obtiene con la aproximación normal, corregida por empates. El tamaño de efecto es la **correlación biserial de rangos**:

$$
r = 1 - \frac{2\,U_1}{n_1\,n_2}
$$

Con chicos como primer grupo, $r > 0$ indica valores más altos en chicas. Su magnitud se interpreta con los mismos umbrales que $\rho$.

## 6. Métricas de evaluación

### Regresión

Con $\hat{y}_i$ la predicción del modelo y $\bar{y}$ la media observada:

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i}|y_i - \hat{y}_i|, \qquad
\mathrm{MSE} = \frac{1}{n}\sum_{i}(y_i - \hat{y}_i)^2, \qquad
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}
$$

$$
R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}{\sum_{i}(y_i - \bar{y})^2}
$$

El MAE y el RMSE se expresan en puntos del BDI-II; el RMSE penaliza más los errores grandes. $R^2$ es la proporción de la varianza de `bdi_total` explicada por el modelo.

### Clasificación

La **matriz de confusión** cruza la clase real con la predicha: verdaderos positivos ($VP$), falsos positivos ($FP$), falsos negativos ($FN$) y verdaderos negativos ($VN$). A partir de ella:

$$
\text{Accuracy} = \frac{VP + VN}{n}, \qquad
\text{Precisión} = \frac{VP}{VP + FP}, \qquad
\text{Recall} = \frac{VP}{VP + FN}
$$

$$
F_1 = 2\,\frac{\text{Precisión}\cdot\text{Recall}}{\text{Precisión} + \text{Recall}}
$$

El *recall* (sensibilidad) es la proporción de adolescentes sobre el corte que el modelo detecta; la precisión, la proporción de alertas que son correctas.

La **curva ROC** representa la tasa de verdaderos positivos ($\text{Recall}$) frente a la tasa de falsos positivos, $\mathrm{FPR} = FP/(FP + VN)$, al variar el umbral de decisión. Su área (**AUC-ROC**) equivale a la probabilidad de que el modelo asigne una puntuación más alta a un caso positivo que a uno negativo elegidos al azar; vale $0{,}5$ para un clasificador aleatorio.

La **curva precisión-recall** y su resumen, la **precisión promedio** (AUC-PR),

$$
\mathrm{AP} = \sum_{k}\left(R_k - R_{k-1}\right) P_k
$$

donde $P_k$ y $R_k$ son la precisión y el recall en el umbral $k$, son más informativas que la curva ROC cuando la clase positiva es minoritaria. Para un clasificador aleatorio, la AP es igual a la prevalencia.

Con clases desbalanceadas, el accuracy es engañoso: un modelo que predice siempre la clase mayoritaria obtiene un accuracy igual a $1 - \hat{p}$ sin detectar ningún caso. Por eso se priorizan el recall, el $F_1$ de la clase positiva y la AUC-PR, y se compara siempre contra ese modelo de referencia (*baseline*).
