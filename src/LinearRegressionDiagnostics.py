import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


def diagnosticar_regresion_lineal(
    X_train,
    y_train,
    variables_numericas,
    variable_categorica="sex",
):
    """Calcula los diagnosticos de los supuestos de OLS e indicadores de
    influencia para X_train e y_train.

    Analogo a diagnosticar_regresion_logistica, pero adaptado a los supuestos
    de la regresion lineal:
    - Multicolinealidad (VIF) en lugar de Box-Tidwell, ya que en OLS no
      aplica el test de linealidad del logit.
    - Linealidad de la relacion (Rainbow test).
    - Homocedasticidad de los residuos (Breusch-Pagan).
    - Normalidad de los residuos (Jarque-Bera y Shapiro-Wilk).
    - Independencia de los residuos (Durbin-Watson).
    - Observaciones influyentes (distancia de Cook y leverage).
    """

    columnas_modelo = list(variables_numericas) + [variable_categorica]
    X_modelo = pd.get_dummies(
        X_train[columnas_modelo],
        columns=[variable_categorica],
        drop_first=True,
        dtype=float,
    )
    X_modelo = sm.add_constant(X_modelo, has_constant="add")
    y_modelo = y_train.astype(float)

    modelo_ols = sm.OLS(y_modelo, X_modelo).fit()

    # --- Multicolinealidad: VIF ---
    resultado_vif = pd.DataFrame(
        {
            "Variable": X_modelo.columns,
            "VIF": [
                variance_inflation_factor(X_modelo.values, indice)
                for indice in range(X_modelo.shape[1])
            ],
        }
    )
    resultado_vif = resultado_vif[resultado_vif["Variable"] != "const"].reset_index(
        drop=True
    )
    resultado_vif["Conclusion"] = np.where(
        resultado_vif["VIF"] >= 5,
        "Posible multicolinealidad relevante",
        "Sin evidencia de multicolinealidad relevante",
    )

    # --- Linealidad: Rainbow test ---
    rainbow_estadistico, rainbow_pvalue = linear_rainbow(modelo_ols)

    # --- Homocedasticidad: Breusch-Pagan ---
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(
        modelo_ols.resid, X_modelo
    )

    # --- Normalidad de residuos: Jarque-Bera y Shapiro-Wilk ---
    jb_estadistico, jb_pvalue, _, _ = jarque_bera(modelo_ols.resid)
    shapiro_estadistico, shapiro_pvalue = shapiro(modelo_ols.resid)

    # --- Independencia de residuos: Durbin-Watson ---
    dw_estadistico = durbin_watson(modelo_ols.resid)

    resultado_supuestos = pd.DataFrame(
        {
            "Supuesto": [
                "Linealidad (Rainbow)",
                "Homocedasticidad (Breusch-Pagan)",
                "Normalidad (Jarque-Bera)",
                "Normalidad (Shapiro-Wilk)",
                "Independencia (Durbin-Watson)",
            ],
            "Estadistico": [
                rainbow_estadistico,
                bp_lm,
                jb_estadistico,
                shapiro_estadistico,
                dw_estadistico,
            ],
            "p-value": [
                rainbow_pvalue,
                bp_lm_pvalue,
                jb_pvalue,
                shapiro_pvalue,
                np.nan,
            ],
        }
    )
    resultado_supuestos["Conclusion"] = [
        "Compatible con linealidad"
        if rainbow_pvalue >= 0.05
        else "Posible incumplimiento de linealidad",
        "Compatible con homocedasticidad"
        if bp_lm_pvalue >= 0.05
        else "Evidencia de heterocedasticidad",
        "Compatible con normalidad"
        if jb_pvalue >= 0.05
        else "Posible incumplimiento de normalidad",
        "Compatible con normalidad"
        if shapiro_pvalue >= 0.05
        else "Posible incumplimiento de normalidad",
        "Sin autocorrelacion relevante"
        if 1.5 <= dw_estadistico <= 2.5
        else "Posible autocorrelacion en los residuos",
    ]

    # --- Observaciones influyentes: distancia de Cook y leverage ---
    influencia = modelo_ols.get_influence()
    distancia_cook, _ = influencia.cooks_distance
    leverage = influencia.hat_matrix_diag

    n_observaciones, n_parametros = X_modelo.shape
    umbral_cook = 4 / n_observaciones
    umbral_leverage = 2 * n_parametros / n_observaciones

    influencia_df = pd.DataFrame(
        {
            "Indice": X_modelo.index,
            "Distancia de Cook": distancia_cook,
            "Leverage": leverage,
        }
    )
    influencia_df["Cook alto"] = influencia_df["Distancia de Cook"] > umbral_cook
    influencia_df["Leverage alto"] = influencia_df["Leverage"] > umbral_leverage
    influencia_df["Cook y leverage altos"] = (
        influencia_df["Cook alto"] & influencia_df["Leverage alto"]
    )

    resumen_influencia = pd.DataFrame(
        {
            "Indicador": [
                "Umbral de distancia de Cook",
                "Umbral de leverage",
                "Observaciones con Cook alto",
                "Observaciones con leverage alto",
                "Observaciones con Cook y leverage altos",
                "Distancia de Cook maxima",
                "Leverage maximo",
            ],
            "Valor": [
                umbral_cook,
                umbral_leverage,
                influencia_df["Cook alto"].sum(),
                influencia_df["Leverage alto"].sum(),
                influencia_df["Cook y leverage altos"].sum(),
                influencia_df["Distancia de Cook"].max(),
                influencia_df["Leverage"].max(),
            ],
        }
    )

    observaciones_influyentes = influencia_df[
        influencia_df["Cook y leverage altos"]
    ].sort_values(
        by=["Distancia de Cook", "Leverage"],
        ascending=False,
    )

    return (
        resultado_vif,
        resultado_supuestos,
        resumen_influencia,
        observaciones_influyentes,
    )