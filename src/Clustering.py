"""Clustering descriptivo con K-Means: perfiles de hábitos de sueño y tiempo de pantalla.

Lo usa notebooks/clustering.ipynb. Los clusters se forman solo con las 5 variables de sueño y
pantalla (estandarizadas); la depresión (`depressed`, `bdi_total`) y el sexo no intervienen y
se usan después para describir los perfiles (validación externa). Se usa el dataset completo,
sin split, porque no se predice nada.

Todo se guarda en data/processed/clustering/.
"""

import json
import platform
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import statsmodels
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))   # para importar data/load_data.py

from data.load_data import CORTE_BDI, ETIQUETAS, ETIQUETAS_ESTADO, ETIQUETAS_SEXO, cargar_datos
from EstiloGraficos import DEPRIMIDO, LINEA_CORTE, NEUTRO, PALETA_SEXO, TEXTO, TEXTO_SECUNDARIO
from Modelado import SEED

try:   # opcional: si no está instalado, las comparaciones por pares usan Mann-Whitney + Holm
    import scikit_posthocs
except ImportError:
    scikit_posthocs = None

RUTA_CLUSTERING = RAIZ / "data" / "processed" / "clustering"

# Variables con las que se forman los clusters (sueño y pantalla). bdi_total, depressed y sex quedan fuera.
VARIABLES_CLUSTER = [
    "sleep_quality_index",
    "avg_sleep_hours",
    "midsleep_weekend_hours",
    "social_jetlag_hours",
    "screen_time_index",
]

RANGO_K = range(2, 9)
N_INIT = 50
N_BOOTSTRAP = 50

# Paleta categórica de los clusters: lavanda, ocre, oliva, gris pizarra y terracota suave.
# Los gráficos (book) usan el modo claro; el oscuro queda definido para el dashboard.
PALETA_CLUSTERS = ["#B3A6DE", "#E3C77F", "#AFC59A", "#9AA3B8", "#C9A08E"]
PALETA_CLUSTERS_OSCURO = ["#A496D2", "#D4B86F", "#9FB68B", "#8A93A8", "#BA917F"]
# Borde de los centroides: pizarra (LINEA_CORTE; en modo oscuro, #A9B6D8)
BORDE_CENTROIDE = LINEA_CORTE


# ---------------------------------------------------------------------------
# Datos y pipeline
# ---------------------------------------------------------------------------

def cargar_dataset() -> pd.DataFrame:
    """Dataset completo validado con data/load_data.py (copia: cargar_datos() devuelve un objeto en caché)."""
    return cargar_datos().copy()


def nombre_variable(variable: str) -> str:
    """Nombre legible de una variable (data/load_data.py)."""
    return ETIQUETAS[variable]


def color_cluster(cluster: int) -> str:
    """Color de un cluster en la paleta categórica."""
    return PALETA_CLUSTERS[cluster % len(PALETA_CLUSTERS)]


def completar_nombres(nombres: dict, k: int) -> dict:
    """Nombre de cada cluster 0..k-1; los que falten en `nombres` quedan como "Cluster i"."""
    return {cluster: nombres.get(cluster, f"Cluster {cluster}") for cluster in range(k)}


def crear_pipeline_kmeans(k: int, seed: int = SEED) -> Pipeline:
    """Pipeline StandardScaler → KMeans(n_init=50). El escalado se ajusta con los mismos datos que se agrupan."""
    return Pipeline(steps=[
        ("escalador", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)),
    ])


def datos_estandarizados(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """X estandarizado con el escalador ya ajustado del pipeline."""
    return pipeline.named_steps["escalador"].transform(X)


# ---------------------------------------------------------------------------
# Elección del número de clusters
# ---------------------------------------------------------------------------

def estabilidad_bootstrap(X: pd.DataFrame, k: int, etiquetas: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                          seed: int = SEED) -> np.ndarray:
    """ARI entre las asignaciones del modelo ajustado en todos los datos y las de K-Means ajustados en
    submuestras bootstrap. Cada modelo bootstrap asigna a TODOS los adolescentes, así que se comparan
    las mismas observaciones. El random_state de cada réplica se deriva de SEED."""
    rng = np.random.default_rng(seed)
    aris = []
    for replica in range(n_bootstrap):
        indices = rng.integers(0, len(X), len(X))
        modelo = crear_pipeline_kmeans(k, seed=seed + replica + 1).fit(X.iloc[indices])
        aris.append(adjusted_rand_score(etiquetas, modelo.predict(X)))
    return np.array(aris)


def evaluar_k(X: pd.DataFrame, rango_k=RANGO_K, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> pd.DataFrame:
    """Para cada k: inercia, silueta, Davies-Bouldin, Calinski-Harabasz (en el espacio estandarizado)
    y estabilidad bootstrap (ARI media y desviación)."""
    filas = []
    for k in rango_k:
        pipeline = crear_pipeline_kmeans(k, seed).fit(X)
        etiquetas = pipeline.named_steps["kmeans"].labels_
        Z = datos_estandarizados(pipeline, X)
        aris = estabilidad_bootstrap(X, k, etiquetas, n_bootstrap=n_bootstrap, seed=seed)
        filas.append({
            "k": k,
            "inercia": pipeline.named_steps["kmeans"].inertia_,
            "silueta": silhouette_score(Z, etiquetas),
            "davies_bouldin": davies_bouldin_score(Z, etiquetas),
            "calinski_harabasz": calinski_harabasz_score(Z, etiquetas),
            "ari_media": aris.mean(),
            "ari_desviacion": aris.std(),
        })
        print(f"✓ k = {k}: silueta = {filas[-1]['silueta']:.3f}, ARI = {aris.mean():.3f} ± {aris.std():.3f}")
    return pd.DataFrame(filas)


def graficar_seleccion_k(tabla: pd.DataFrame, k_elegido: int | None = None):
    """Las 5 métricas por k. Mejor: codo en la inercia, silueta, Calinski-Harabasz y ARI altos, Davies-Bouldin bajo."""
    paneles = [
        ("inercia", "Inercia (método del codo)", "Inercia"),
        ("silueta", "Coeficiente de silueta (↑)", "Silueta"),
        ("davies_bouldin", "Índice de Davies-Bouldin (↓)", "Davies-Bouldin"),
        ("calinski_harabasz", "Índice de Calinski-Harabasz (↑)", "Calinski-Harabasz"),
        ("ari_media", "Estabilidad bootstrap (ARI, ↑)", "ARI (media ± DE)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (columna, titulo, etiqueta) in zip(axes.flat, paneles):
        if columna == "ari_media":
            ax.errorbar(tabla["k"], tabla["ari_media"], yerr=tabla["ari_desviacion"], fmt="o-", color=NEUTRO,
                        capsize=4, linewidth=2)
        else:
            ax.plot(tabla["k"], tabla[columna], "o-", color=NEUTRO, linewidth=2)
        if k_elegido is not None:
            ax.axvline(k_elegido, color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel(etiqueta)
        ax.set_xticks(list(tabla["k"]))
    axes.flat[-1].set_visible(False)
    sns.despine()
    fig.tight_layout()
    return fig


def ajustar_final(X: pd.DataFrame, k: int, seed: int = SEED) -> tuple[Pipeline, np.ndarray]:
    """Pipeline final con el k elegido, ajustado con todos los datos, y la asignación de cada adolescente."""
    pipeline = crear_pipeline_kmeans(k, seed).fit(X)
    return pipeline, pipeline.named_steps["kmeans"].labels_.copy()


# ---------------------------------------------------------------------------
# Descripción de los perfiles
# ---------------------------------------------------------------------------

def tabla_tamanos(etiquetas: np.ndarray, nombres: dict) -> pd.DataFrame:
    """Número de adolescentes y porcentaje de cada cluster."""
    conteo = pd.Series(etiquetas).value_counts().sort_index()
    return pd.DataFrame({
        "Nombre": [nombres[c] for c in conteo.index],
        "n": conteo.to_numpy(),
        "%": (conteo / conteo.sum() * 100).round(1).to_numpy(),
    }, index=pd.Index(conteo.index, name="Cluster"))


def tabla_medias_medianas(X: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Media y mediana (unidades originales) de cada variable por cluster, con nombres legibles."""
    agrupado = X.groupby(np.asarray(etiquetas))
    tabla = pd.concat({"Media": agrupado.mean(), "Mediana": agrupado.median()}, axis=1)
    tabla = tabla.swaplevel(axis=1)[[(v, e) for v in VARIABLES_CLUSTER for e in ("Media", "Mediana")]]
    tabla.columns = pd.MultiIndex.from_tuples([(nombre_variable(v), e) for v, e in tabla.columns])
    tabla.index.name = "Cluster"
    return tabla


def perfiles_estandarizados(pipeline: Pipeline, X: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Media de cada variable estandarizada (en desviaciones estándar respecto a la media global) por cluster."""
    Z = pd.DataFrame(datos_estandarizados(pipeline, X), columns=VARIABLES_CLUSTER)
    tabla = Z.groupby(np.asarray(etiquetas)).mean()
    tabla.columns = [nombre_variable(v) for v in VARIABLES_CLUSTER]
    tabla.index.name = "Cluster"
    return tabla


def graficar_perfiles(tabla_z: pd.DataFrame, nombres: dict):
    """Barras agrupadas: media estandarizada de cada variable (eje x) por cluster (color)."""
    k = len(tabla_z)
    posiciones = np.arange(tabla_z.shape[1])
    ancho = 0.8 / k
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (cluster, fila) in enumerate(tabla_z.iterrows()):
        ax.bar(posiciones + (i - (k - 1) / 2) * ancho, fila.to_numpy(), ancho,
               color=color_cluster(cluster), label=f"{cluster}: {nombres[cluster]}")
    ax.axhline(0, color=TEXTO, linewidth=1)
    ax.set_xticks(posiciones, tabla_z.columns, rotation=10)
    ax.set_ylabel("Media estandarizada (DE respecto a la media global)")
    ax.set_title("Perfil de cada cluster: media estandarizada de las variables", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    sns.despine()
    fig.tight_layout()
    return fig


def graficar_pca(pipeline: Pipeline, X: pd.DataFrame, etiquetas: np.ndarray, nombres: dict):
    """Proyección en las 2 primeras componentes principales (datos estandarizados), coloreada por cluster,
    con los centroides marcados y el % de varianza explicada en los ejes."""
    Z = datos_estandarizados(pipeline, X)
    pca = PCA(n_components=2).fit(Z)
    proyeccion = pca.transform(Z)
    centroides = pca.transform(pipeline.named_steps["kmeans"].cluster_centers_)
    varianza = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster in np.unique(etiquetas):
        mascara = etiquetas == cluster
        ax.scatter(proyeccion[mascara, 0], proyeccion[mascara, 1], s=10, alpha=0.35,
                   color=color_cluster(cluster), label=f"{cluster}: {nombres[cluster]}")
    for cluster, (x, y) in enumerate(centroides):
        ax.scatter(x, y, s=260, marker="X", color=color_cluster(cluster), edgecolor=BORDE_CENTROIDE, linewidth=1.5, zorder=3)
    ax.set_xlabel(f"Componente principal 1 ({varianza[0]:.1f}% de la varianza)")
    ax.set_ylabel(f"Componente principal 2 ({varianza[1]:.1f}% de la varianza)")
    ax.set_title("Proyección PCA de los clusters (X = centroide)", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=2)
    sns.despine()
    fig.tight_layout()
    return fig


def cargas_pca(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Cargas de las 2 primeras componentes principales (para leer los ejes del gráfico PCA)."""
    pca = PCA(n_components=2).fit(datos_estandarizados(pipeline, X))
    return pd.DataFrame(pca.components_.T, columns=["CP1", "CP2"],
                        index=pd.Index([nombre_variable(v) for v in VARIABLES_CLUSTER], name="Variable"))


# ---------------------------------------------------------------------------
# Validación externa
# ---------------------------------------------------------------------------

def prueba_chi_cuadrado(etiquetas: np.ndarray, variable: pd.Series) -> dict:
    """Chi-cuadrado de independencia entre el cluster y una variable categórica, con la V de Cramér."""
    contingencia = pd.crosstab(np.asarray(etiquetas), np.asarray(variable))
    chi2, p_valor, gl, _ = chi2_contingency(contingencia)
    n = contingencia.to_numpy().sum()
    v_cramer = np.sqrt(chi2 / (n * (min(contingencia.shape) - 1)))
    return {"Chi-cuadrado": chi2, "gl": int(gl), "p-valor": p_valor, "V de Cramér": v_cramer}


def prevalencia_por_cluster(df: pd.DataFrame, etiquetas: np.ndarray, nombres: dict) -> pd.DataFrame:
    """Prevalencia de depresión por cluster con su IC 95% de Wilson."""
    grupos = df["depressed"].groupby(np.asarray(etiquetas))
    deprimidos, n = grupos.sum(), grupos.count()
    ic_inf, ic_sup = proportion_confint(deprimidos, n, alpha=0.05, method="wilson")
    return pd.DataFrame({
        "Nombre": [nombres[c] for c in n.index], "n": n.to_numpy(), "Deprimidos": deprimidos.to_numpy(),
        "Prevalencia": (deprimidos / n).to_numpy(), "IC 2.5%": np.asarray(ic_inf), "IC 97.5%": np.asarray(ic_sup),
    }, index=pd.Index(n.index, name="Cluster"))


def graficar_prevalencia(tabla: pd.DataFrame, prevalencia_global: float):
    """Barras de la prevalencia de depresión por cluster con IC 95% (Wilson) y la prevalencia global como referencia."""
    error = [tabla["Prevalencia"] - tabla["IC 2.5%"], tabla["IC 97.5%"] - tabla["Prevalencia"]]
    etiquetas = [f"{c}: {nombre}" for c, nombre in tabla["Nombre"].items()]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(etiquetas, tabla["Prevalencia"], yerr=error, color=DEPRIMIDO, capsize=5)
    ax.axhline(prevalencia_global, color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1.5,
               label=f"Prevalencia global = {prevalencia_global:.3f}")
    ax.set_ylabel(f"Proporción de {ETIQUETAS_ESTADO[1].lower()}s")
    ax.set_title("Prevalencia de depresión por cluster (IC 95% de Wilson)", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    return fig


def resumen_bdi(df: pd.DataFrame, etiquetas: np.ndarray, nombres: dict) -> pd.DataFrame:
    """Media ± DE, mediana y rango intercuartílico del BDI-II por cluster."""
    grupos = df["bdi_total"].groupby(np.asarray(etiquetas))
    tabla = pd.DataFrame({
        "Media": grupos.mean(), "DE": grupos.std(), "Mediana": grupos.median(),
        "Q1": grupos.quantile(0.25), "Q3": grupos.quantile(0.75),
    })
    tabla.insert(0, "Nombre", [nombres[c] for c in tabla.index])
    tabla.index.name = "Cluster"
    return tabla


def prueba_kruskal(df: pd.DataFrame, etiquetas: np.ndarray) -> dict:
    """Kruskal-Wallis del BDI-II entre clusters, con épsilon cuadrado como tamaño de efecto: ε² = H (n + 1) / (n² − 1)."""
    etiquetas = np.asarray(etiquetas)
    muestras = [df["bdi_total"].to_numpy()[etiquetas == c] for c in np.unique(etiquetas)]
    h, p_valor = kruskal(*muestras)
    n = len(etiquetas)
    return {"H": h, "gl": len(muestras) - 1, "p-valor": p_valor, "Épsilon cuadrado": h * (n + 1) / (n ** 2 - 1)}


def comparaciones_por_pares(df: pd.DataFrame, etiquetas: np.ndarray, alfa: float = 0.05) -> pd.DataFrame:
    """Comparaciones del BDI-II por pares de clusters con corrección de Holm: prueba de Dunn si
    scikit-posthocs está instalado y, si no, Mann-Whitney bilateral."""
    etiquetas = np.asarray(etiquetas)
    clusters = list(np.unique(etiquetas))
    pares = list(combinations(clusters, 2))
    if scikit_posthocs is not None:
        datos = pd.DataFrame({"bdi_total": df["bdi_total"].to_numpy(), "cluster": etiquetas})
        matriz = scikit_posthocs.posthoc_dunn(datos, val_col="bdi_total", group_col="cluster", p_adjust="holm")
        metodo, estadisticos = "Dunn", [np.nan] * len(pares)
        p_ajustados = [matriz.loc[a, b] for a, b in pares]
        p_crudos = [np.nan] * len(pares)
    else:
        bdi = df["bdi_total"].to_numpy()
        resultados = [mannwhitneyu(bdi[etiquetas == a], bdi[etiquetas == b], alternative="two-sided") for a, b in pares]
        metodo = "Mann-Whitney"
        estadisticos = [r.statistic for r in resultados]
        p_crudos = [r.pvalue for r in resultados]
        p_ajustados = multipletests(p_crudos, alpha=alfa, method="holm")[1]
    tabla = pd.DataFrame({
        "Cluster A": [a for a, _ in pares], "Cluster B": [b for _, b in pares], "Método": metodo,
        "Estadístico U": estadisticos, "p-valor": p_crudos, "p-valor (Holm)": p_ajustados,
    })
    tabla["Significativa"] = tabla["p-valor (Holm)"] < alfa
    return tabla


def graficar_bdi(df: pd.DataFrame, etiquetas: np.ndarray, nombres: dict):
    """Boxplot del BDI-II por cluster con la línea del corte clínico (BDI-II = 14)."""
    clusters = sorted(np.unique(etiquetas))
    datos = pd.DataFrame({"bdi_total": df["bdi_total"].to_numpy(), "cluster": np.asarray(etiquetas)})
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.boxplot(data=datos, x="cluster", y="bdi_total", hue="cluster", order=clusters, hue_order=clusters,
                palette={c: color_cluster(c) for c in clusters}, legend=False, fliersize=2, ax=ax)
    ax.axhline(CORTE_BDI, color=LINEA_CORTE, linestyle="--", linewidth=1.5, label=f"Corte clínico (BDI-II = {CORTE_BDI})")
    ax.set_xticks(range(len(clusters)), [f"{c}: {nombres[c]}" for c in clusters], rotation=15)
    ax.set_xlabel("")
    ax.set_ylabel(ETIQUETAS["bdi_total"])
    ax.set_title("BDI-II por cluster", fontweight="bold")
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    return fig


def composicion_sexo(df: pd.DataFrame, etiquetas: np.ndarray, nombres: dict) -> pd.DataFrame:
    """Número de chicos y chicas por cluster y proporción de chicas."""
    tabla = pd.crosstab(np.asarray(etiquetas), df["sex"].to_numpy()).reindex(columns=list(PALETA_SEXO), fill_value=0)
    tabla.columns = [ETIQUETAS_SEXO[s] for s in tabla.columns]
    tabla["Proporción de chicas"] = tabla[ETIQUETAS_SEXO["Girl"]] / tabla.sum(axis=1)
    tabla.insert(0, "Nombre", [nombres[c] for c in tabla.index])
    tabla.index.name = "Cluster"
    return tabla


def graficar_sexo(tabla: pd.DataFrame, proporcion_global: float):
    """Barras apiladas al 100%: proporción de chicos y chicas en cada cluster, con la proporción global de chicas."""
    chicas = tabla["Proporción de chicas"]
    etiquetas = [f"{c}: {nombre}" for c, nombre in tabla["Nombre"].items()]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(etiquetas, 1 - chicas, color=PALETA_SEXO["Boy"], label=ETIQUETAS_SEXO["Boy"])
    ax.bar(etiquetas, chicas, bottom=1 - chicas, color=PALETA_SEXO["Girl"], label=ETIQUETAS_SEXO["Girl"])
    ax.axhline(1 - proporcion_global, color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1.5,
               label=f"Proporción global de chicas = {proporcion_global:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proporción")
    ax.set_title("Composición por sexo de cada cluster", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    sns.despine()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def tabla_perfiles(X: pd.DataFrame, df: pd.DataFrame, etiquetas: np.ndarray, nombres: dict) -> pd.DataFrame:
    """Una fila por cluster con todo lo que se guarda en perfiles.csv."""
    etiquetas = np.asarray(etiquetas)
    tamanos = tabla_tamanos(etiquetas, nombres)
    prevalencia = prevalencia_por_cluster(df, etiquetas, nombres)
    bdi = resumen_bdi(df, etiquetas, nombres)
    sexo = composicion_sexo(df, etiquetas, nombres)
    agrupado = X.groupby(etiquetas)
    perfiles = pd.DataFrame({"cluster": tamanos.index, "nombre": tamanos["Nombre"].to_numpy(),
                             "n": tamanos["n"].to_numpy(), "porcentaje": tamanos["%"].to_numpy()})
    for variable in VARIABLES_CLUSTER:
        perfiles[f"media_{variable}"] = agrupado[variable].mean().to_numpy()
        perfiles[f"mediana_{variable}"] = agrupado[variable].median().to_numpy()
    perfiles["prevalencia_depresion"] = prevalencia["Prevalencia"].to_numpy()
    perfiles["prevalencia_ic_inf"] = prevalencia["IC 2.5%"].to_numpy()
    perfiles["prevalencia_ic_sup"] = prevalencia["IC 97.5%"].to_numpy()
    perfiles["bdi_media"] = bdi["Media"].to_numpy()
    perfiles["bdi_mediana"] = bdi["Mediana"].to_numpy()
    perfiles["proporcion_chicas"] = sexo["Proporción de chicas"].to_numpy()
    return perfiles


def guardar_resultados(pipeline: Pipeline, tabla_k: pd.DataFrame, df: pd.DataFrame, etiquetas: np.ndarray,
                       perfiles: pd.DataFrame, k: int, ruta: Path = RUTA_CLUSTERING) -> dict:
    """Guarda en clustering/: kmeans.joblib, seleccion_k.csv, asignaciones.csv, perfiles.csv y metadatos.json."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ruta / "kmeans.joblib")
    tabla_k.to_csv(ruta / "seleccion_k.csv", index=False)
    pd.DataFrame({"subject_id": df["subject_id"].to_numpy(), "cluster": np.asarray(etiquetas)}).to_csv(
        ruta / "asignaciones.csv", index=False)
    perfiles.to_csv(ruta / "perfiles.csv", index=False)

    kmeans = pipeline.named_steps["kmeans"]
    metadatos = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "k_elegido": k,
        "nombres_clusters": dict(zip(perfiles["cluster"].astype(str), perfiles["nombre"])),
        "seed": SEED,
        "variables": VARIABLES_CLUSTER,
        "preprocesamiento": "StandardScaler (ajustado con el dataset completo)",
        "parametros_kmeans": {k_: (v.item() if isinstance(v, np.generic) else v) for k_, v in kmeans.get_params().items()},
        "rango_k_evaluado": [int(tabla_k["k"].min()), int(tabla_k["k"].max())],
        "n_bootstrap_estabilidad": N_BOOTSTRAP,
        "n_observaciones": int(len(df)),
        "versiones": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "joblib": joblib.__version__,
        },
    }
    with open(ruta / "metadatos.json", "w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    print(f"✓ Guardado en {ruta}: kmeans.joblib, seleccion_k.csv, asignaciones.csv, perfiles.csv y metadatos.json")
    return metadatos
