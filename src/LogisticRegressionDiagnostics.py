import numpy as np
import pandas as pd
import statsmodels.api as sm


def diagnosticar_regresion_logistica(
    X_train,
    y_train,
    variables_numericas,
    variable_categorica="sex",
):
    """Calcula Box-Tidwell e indicadores de influencia para X_train e y_train."""

    columnas_modelo = list(variables_numericas) + [variable_categorica]
    X_modelo = pd.get_dummies(
        X_train[columnas_modelo],
        columns=[variable_categorica],
        drop_first=True,
        dtype=float,
    )
    X_modelo = sm.add_constant(X_modelo, has_constant="add")
    y_modelo = y_train.astype(float)

    X_box_tidwell = X_modelo.copy()
    for variable in variables_numericas:
        valores_positivos = X_train[variable].clip(lower=1e-6)
        X_box_tidwell[f"{variable}_log_interaccion"] = (
            valores_positivos * np.log(valores_positivos)
        )

    modelo_box_tidwell = sm.GLM(
        y_modelo,
        X_box_tidwell,
        family=sm.families.Binomial(),
    ).fit()

    interacciones = [
        f"{variable}_log_interaccion" for variable in variables_numericas
    ]
    resultado_box_tidwell = pd.DataFrame(
        {
            "Variable": variables_numericas,
            "p-value interaccion": [
                modelo_box_tidwell.pvalues[interaccion]
                for interaccion in interacciones
            ],
        }
    )
    resultado_box_tidwell["Conclusion"] = np.where(
        resultado_box_tidwell["p-value interaccion"] >= 0.05,
        "Compatible con linealidad del logit",
        "Posible incumplimiento de linealidad",
    )

    modelo_influencia = sm.GLM(
        y_modelo,
        X_modelo,
        family=sm.families.Binomial(),
    ).fit()
    influencia = modelo_influencia.get_influence()
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

    return resultado_box_tidwell, resumen_influencia, observaciones_influyentes
