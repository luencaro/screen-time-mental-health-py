import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


VARIABLES_NUMERICAS = [
    "sleep_quality_index",
    "avg_sleep_hours",
    "midsleep_weekend_hours",
    "screen_time_index",
]

VARIABLES_CATEGORICAS = ["sex"]
TARGET = "bdi_total"


def crear_pipeline():

    transformador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                StandardScaler(),
                VARIABLES_NUMERICAS,
            ),
            (
                "categoricas",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                VARIABLES_CATEGORICAS,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", transformador),
        ]
    )

    return pipeline