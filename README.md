# screen-time-mental-health-py
Dataset: [Screen Time vs Mental Health (ML-ready)](https://www.kaggle.com/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready)

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) o Docker Engine + Docker Compose (Linux)
- (Opcional) [Kaggle CLI](https://www.kaggle.com/docs/api) para descargar el dataset por comando

Todo lo demás (Python, conda, librerías) vive dentro del contenedor Docker

## Inicializar el proyecto (cualquier SO, cualquier editor)

1. **Clonar el repo**
   ```bash
   git clone <url-del-repo>
   cd screen-time-mental-health-py
   ```

2. **Descargar el dataset** (no viene versionado en el repo)
   ```bash
   kaggle datasets download -d kylefengkfeng209/screen-time-vs-mental-health-ml-ready -p data/raw --unzip
   ```
   O descárgalo manualmente desde Kaggle y descomprímelo en `data/raw/`.

3. **Levantar el contenedor**
   ```bash
   docker compose up --build -d
   ```
   La primera vez tarda unos minutos (construye la imagen).

4. **Verificar que esté arriba**
   ```bash
   docker compose ps
   ```

5. **Apagar cuando termines**
   ```bash
   docker compose down
   ```
   (Los archivos no se pierden — solo se detiene el contenedor.)

## Estructura del proyecto

```
screen-time-mental-health-py/
├── .devcontainer/            # configuración de Dev Containers para VS Code
├── .github/workflows/        # despliegue del Jupyter Book (deploy-book.yml)
├── docker/                   # Dockerfile + environment.yml (dependencias conda)
├── docker-compose.yml
├── myst.yml                  # configuración y tabla de contenidos del Jupyter Book (MyST)
├── requirements.txt          # dependencias del dashboard vía pip (fuera de Docker)
│
├── data/
│   ├── raw/                  # dataset original (no versionado)
│   ├── processed/            # datos limpios/transformados (no versionado)
│   └── load_data.py          # carga, validación de columnas y tipos, y caché del CSV
│
├── notebooks/
│   └── EDA.ipynb             # análisis exploratorio, pruebas inferenciales y modelos benchmark
├── src/                      # funciones reutilizables del notebook
│   ├── ClassificationPreprocessing.py    # pipeline de preprocesamiento para clasificación
│   ├── LinearRegressionPreprocessing.py  # pipeline de preprocesamiento para regresión
│   ├── LinearRegressionDiagnostics.py    # supuestos de la regresión lineal (VIF, BP, JB, DW, Cook)
│   ├── LogisticRegressionDiagnostics.py  # supuestos de la regresión logística (Box-Tidwell, Cook)
│   └── EstiloGraficos.py                 # paleta y estilo matplotlib/seaborn del dashboard
│
├── book/                     # Jupyter Book (documentación del proyecto)
│   ├── intro.md              # introducción, antecedentes y objetivos
│   ├── base_estadistica.md   # base estadística del estudio
│   ├── methodology.md        # metodología
│   ├── references.bib        # referencias bibliográficas
│   └── logo_uninorte.png
│
├── design/
│   └── guia_estilo_4d.md     # guía de estilo "4d Pizarra cálida" del dashboard
│
├── app.py                    # dashboard Dash: header, menú, selector de tema y routing
├── analysis/
│   ├── stats.py              # cálculos estadísticos (descriptivos, χ², Mann-Whitney, Spearman, IQR)
│   └── figures.py            # figuras Plotly; reciben el tema ("claro"/"oscuro")
├── content/
│   ├── libro.py              # lee secciones de book/*.md y convierte las citas {cite}
│   └── fichas.py             # resúmenes de los estudios y fase de cada objetivo
├── components/
│   ├── ui.py                 # card, KPI, badge, interpretación, tabla, encabezado, gráfico
│   ├── secciones.py          # registro de secciones del menú (bloque, número, descripción)
│   ├── formato.py            # formato numérico en español (coma decimal, punto de miles)
│   └── plotly_theme.py       # paleta de datos y plantillas Plotly claro/oscuro
├── tabs/                     # una sección del dashboard por archivo, cada una con layout()
│   ├── inicio.py, marco_teorico.py, objetivos.py, metodologia.py          # Proyecto
│   ├── calidad_datos.py, variable_objetivo.py, analisis_sexo.py,          # EDA
│   │   analisis_numerico.py, correlaciones.py, outliers.py
│   └── hallazgos.py, limitaciones.py                                      # Cierre
└── assets/                   # archivos estáticos que Dash carga automáticamente
    ├── style.css             # variables de diseño (colores, tipografía, espaciados; claro y oscuro)
    ├── componentes.css       # estilos de los componentes (usan solo las variables de style.css)
    ├── menu_hover.js         # abre los menús desplegables al pasar el cursor
    ├── mosaico.js            # rejilla tipo mosaico para las cards de texto
    ├── animacion_graficos.js # animación de entrada y transiciones de los gráficos
    └── logo_uninorte.png
```

## Dashboard (Dash)

Requiere Python 3.11 y el dataset en `data/raw/` (ver paso 2).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # abre http://127.0.0.1:8050
```

## Comandos útiles

| Acción | Comando |
|---|---|
| Levantar el entorno | `docker compose up --build -d` |
| Entrar a la terminal del contenedor | `docker compose exec jupyter bash` |
| Ver logs / token de Jupyter | `docker compose logs jupyter` |
| Apagar el entorno | `docker compose down` |
| Compilar el Jupyter Book | `cd book && jupyter book build` (dentro del contenedor) |
| Instalar un paquete nuevo | Agrégalo a `docker/environment.yml` → `docker compose up --build -d` (o "Rebuild Container" en VS Code) |