# screen-time-mental-health-py
Dataset: [Screen Time vs Mental Health (ML-ready)](https://www.kaggle.com/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready)

## Requisitos

- **Docker Desktop** ([descarga](https://www.docker.com/products/docker-desktop/)). En Windows, durante la instalación deja marcada la opción **"Use WSL 2"**; si lo pide, reinicia el equipo.
- Abrir Docker Desktop y esperar a que diga **"Engine running"** antes de seguir.

No hace falta instalar Python, conda ni ninguna librería: todo corre dentro de los contenedores.

## Cómo ejecutar el proyecto (Windows)

Los comandos se escriben en **PowerShell** (menú Inicio → "PowerShell"). Son los mismos en macOS/Linux.

1. **Obtener el proyecto**

   Con Git:
   ```powershell
   git clone <url-del-repo>
   cd screen-time-mental-health-py
   ```
   Sin Git: en GitHub, botón **Code → Download ZIP**, descomprimir y abrir PowerShell dentro de la carpeta descomprimida (en el Explorador: clic derecho en la carpeta → **"Abrir en Terminal"**).

2. **Descargar el dataset manualmente** (no viene incluido en el repo)
   1. Entrar a [Screen Time vs Mental Health (ML-ready)](https://www.kaggle.com/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready) e iniciar sesión en Kaggle.
   2. Clic en **Download** → **Download dataset as zip**.
   3. Descomprimir el `.zip` y copiar el archivo `screen_time_mental_health.csv` **directamente** dentro de la carpeta `data\raw\` del proyecto.

   Debe quedar exactamente así:
   ```
   screen-time-mental-health-py\data\raw\screen_time_mental_health.csv
   ```

3. **Levantar el proyecto**
   ```powershell
   docker compose up --build -d
   ```
   La primera vez tarda varios minutos (descarga e instala las dependencias).

4. **Abrir el dashboard**
   1. Verificar que los dos contenedores estén encendidos:
      ```powershell
      docker compose ps
      ```
      Deben aparecer `screen-time-dashboard` y `screen-time-jupyter` con estado **Up** (o **running**).
   2. Abrir un navegador (Chrome, Edge o Firefox) y entrar a:

      **<http://localhost:8050>**

   Desde la portada, selecciona **Explorar dashboard** para abrir el resumen del estudio. La cabecera da acceso a los análisis, el modelado y el selector de tema.

   **Shapiro** responde preguntas sobre la navegación, los gráficos y la metodología. Puedes abrirlo desde la cabecera, incluso si falta el CSV. Si el servidor tiene una clave de Gemini, Shapiro usa Gemini Flash-Lite. En los demás casos usa una guía local.

 
5. **(Opcional) Abrir JupyterLab** para ver `notebooks/EDA.ipynb`:
   ```powershell
   docker compose logs jupyter
   ```
   Copiar el enlace que empieza por `http://127.0.0.1:8888/lab?token=...` y pegarlo en el navegador.

   Los modelos de clasificación no vienen en el repo: para generarlos hay que ejecutar `notebooks/Modelado.ipynb` (quedan en `data/processed/modelado/`).

   **Modelado por fases** (después de `Modelado.ipynb`, que crea el split):
   1. Fase 1: ejecutar los 9 notebooks `notebooks/clf_<modelo>.ipynb`, en cualquier orden. Cada uno compara tres estrategias de desbalance (nativo, pesos de clase y SMOTENC, que trata `sex` como categórica) y guarda el ganador en `data/processed/modelado/fase1/`.
   2. Fases 2 y 3: ejecutar `notebooks/clf_comparacion.ipynb`. Compara los ganadores, elige el modelo y su umbral, y guarda el modelo final en `data/processed/modelado/final/`.

   **(Opcional) Entrenar con GPU NVIDIA.** Requiere una GPU NVIDIA con su driver y el [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) configurado para Docker. Se levanta con el override de GPU:
   ```powershell
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
   ```
   En VS Code: **Dev Containers: Reopen in Container** y elegir la configuración **"(GPU)"** (`.devcontainer/gpu/`).
   - **Qué usa GPU:** XGBoost, CatBoost y LightGBM. Con el override, el build instala automáticamente sus variantes CUDA de conda-forge.
   - **Qué usa CPU:** los modelos de scikit-learn (logísticas, Random Forest, KNN y SVC).
   - **Sin GPU:** todo funciona igual en CPU, sin cambiar nada.

6. **Apagar cuando termines**
   ```powershell
   docker compose down
   ```
   Los archivos no se pierden; solo se detienen los contenedores.

### Activar Shapiro con Gemini

1. Crea una clave en [Google AI Studio](https://aistudio.google.com/app/apikey) para un proyecto del nivel gratuito y confirma que Flash-Lite está disponible en ese proyecto. Mantén la facturación desactivada si quieres evitar cargos.
2. En la raíz del repositorio, abre o crea el archivo `.env` y agrega `GEMINI_API_KEY=<tu_clave>`. Git excluye este archivo.
3. Reconstruye y reinicia el servicio del dashboard.
   ```powershell
   docker compose up -d --build dashboard
   ```
4. Abre Shapiro desde la cabecera y prueba una pregunta seguida de una repregunta.
   Por ejemplo, «Explícame el benchmarking de modelos» y luego «¿Quién ganó y por qué?».

Con Gemini activo, se envían la pregunta, hasta cinco intercambios recientes y una guía de
todas las secciones. La guía prioriza la sección visible e identifica los filtros activos en
Variables numéricas y Benchmarking. Reúne documentación,
resultados agregados exportados y estadísticas exploratorias que se calculan localmente
con los mismos procedimientos del dashboard. Gemini recibe solo los resúmenes de esos datos.
Cuando falta el CSV, Shapiro puede seguir usando la documentación y las exportaciones disponibles.
Si la API falla, Shapiro muestra una respuesta de la guía local.
El modelo predeterminado es `gemini-3.5-flash-lite`. Si recibes HTTP 404, revisa en
Google AI Studio qué modelos ofrece el nivel gratuito de tu proyecto. Si `.env` indica
un modelo anterior, actualiza o elimina la línea `SHAPIRO_GEMINI_MODEL` y ejecuta
`docker compose up -d dashboard` para aplicar el cambio de configuración.

### Ver el bloque de Modelado en el dashboard

Los resultados del modelado no vienen en el repo. Para ver las pestañas de **Modelado** (benchmarking, modelos finales, género, clustering y predicción):

1. En JupyterLab, ejecutar los notebooks de modelado (`Modelado.ipynb`, `clf_*.ipynb`, `clf_comparacion.ipynb`, `reg_*.ipynb`, `reg_comparacion.ipynb`, `interpretabilidad.ipynb`, `genero.ipynb`) y `clustering.ipynb`.
2. Ejecutar `notebooks/exportar_dashboard.ipynb`: escribe en `data/processed/dashboard/` los CSV/JSON que lee el dashboard y los modelos portables de la pestaña de predicción, y verifica que reproducen los pipelines originales.
3. Reconstruir y levantar el dashboard (instala `catboost`, que usa la pestaña de predicción):
   ```powershell
   docker compose up --build -d
   ```

Mientras falten los resultados, las pestañas de Modelado muestran un aviso con estos pasos.

### Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `docker` no se reconoce como comando | Docker Desktop no está instalado o no está abierto. Abrirlo, esperar "Engine running" y abrir una PowerShell nueva. |
| `error during connect` / `cannot find the file specified` | Docker Desktop está instalado pero el motor no ha arrancado. Esperar a que diga "Engine running". |
| El dashboard muestra "Dataset no disponible" | El CSV no está en `data\raw\screen_time_mental_health.csv` (revisar el nombre y que no esté en una subcarpeta). Corregir y recargar la página. |
| `port is already allocated` (8050 u 8888) | Otro programa usa ese puerto. Cerrarlo, o ejecutar `docker compose down` si quedó otra copia del proyecto encendida. |
| La página no carga justo después de levantar | Esperar unos segundos; ver el estado con `docker compose logs dashboard`. |

## Estructura del proyecto

```
screen-time-mental-health-py/
├── .devcontainer/            # configuración de Dev Containers para VS Code
├── .github/workflows/        # despliegue del Jupyter Book (deploy-book.yml)
├── docker/                   # Dockerfile (Jupyter) + Dockerfile.dashboard + environment.yml
├── docker-compose.yml
├── docker-compose.gpu.yml    # override opcional: reserva la GPU NVIDIA para el servicio jupyter
├── myst.yml                  # configuración y tabla de contenidos del Jupyter Book (MyST)
├── requirements.txt          # dependencias del dashboard (las instala Dockerfile.dashboard)
│
├── data/
│   ├── raw/                  # dataset original (no versionado)
│   ├── processed/            # datos limpios/transformados (no versionado)
│   ├── load_data.py          # carga, validación de columnas y tipos, y caché del CSV
│   ├── load_modelado.py      # carga y caché de los resultados exportados (data/processed/dashboard/)
│   └── prediccion.py         # predicción con los modelos portables (catboost.cbm + JSON, sin scikit-learn)
│
├── notebooks/
│   ├── EDA.ipynb             # análisis exploratorio, pruebas inferenciales y modelos benchmark
│   ├── Modelado.ipynb        # versión previa: benchmark conjunto de clasificadores con pesos de clase
│   ├── clf_<modelo>.ipynb    # fase 1: estrategias de desbalance por modelo (9 notebooks)
│   ├── clf_comparacion.ipynb # fases 2 y 3: comparación, intervalos bootstrap y umbral; modelo final
│   └── exportar_dashboard.ipynb # exporta los resultados del modelado para el dashboard
├── src/                      # funciones reutilizables del notebook
│   ├── Modelado.py                       # modelos, desbalance, tuning, métricas, GPU, gráficas y guardado/carga
│   ├── ClassificationPreprocessing.py    # pipeline de preprocesamiento para clasificación
│   ├── LinearRegressionPreprocessing.py  # pipeline de preprocesamiento para regresión
│   ├── LinearRegressionDiagnostics.py    # supuestos de la regresión lineal (VIF, BP, JB, DW, Cook)
│   ├── LogisticRegressionDiagnostics.py  # supuestos de la regresión logística (Box-Tidwell, Cook)
│   ├── ExportarDashboard.py              # exportación de resultados y modelos portables para el dashboard
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
├── app.py                    # dashboard Dash, cabecera, selector de tema y rutas
├── analysis/
│   ├── stats.py              # cálculos estadísticos (descriptivos, χ², Mann-Whitney, Spearman, IQR)
│   ├── figures.py            # figuras Plotly; reciben el tema ("claro"/"oscuro")
│   └── figuras_modelado.py   # figuras Plotly del bloque de Modelado
├── content/
│   ├── libro.py              # lee secciones de book/*.md y convierte las citas {cite}
│   └── fichas.py             # resúmenes de los estudios y fase de cada objetivo
├── components/
│   ├── ui.py                 # card, KPI, badge, interpretación, tabla, encabezado, gráfico
│   ├── secciones.py          # registro de secciones del menú
│   ├── shapiro.py            # interfaz de la guía del dashboard
│   ├── shapiro_gemini.py     # guía editorial y conexión a Gemini desde el servidor
│   ├── shapiro_contexto.py   # contexto de todas las secciones del dashboard
│   ├── shapiro_documentacion.py # documentación seleccionada del proyecto
│   ├── shapiro_eda.py        # estadísticas exploratorias agregadas
│   ├── shapiro_resultados.py # resultados agregados de benchmarking y selección final
│   ├── shapiro_resumenes.py  # resúmenes de importancias, clustering y modelos por género
│   ├── enlaces.py            # enlaces públicos: repositorio, Jupyter Book y dataset
│   ├── formato.py            # formato numérico en español (coma decimal, punto de miles)
│   └── plotly_theme.py       # paleta de datos y plantillas Plotly claro/oscuro
├── tabs/                     # una sección del dashboard por archivo, cada una con layout()
│   ├── portada.py            # portada y accesos principales
│   ├── inicio.py, marco_teorico.py, objetivos.py, metodologia.py          # Proyecto
│   ├── calidad_datos.py, variable_objetivo.py, analisis_sexo.py,          # EDA
│   │   analisis_numerico.py, correlaciones.py, outliers.py
│   ├── benchmarking.py, modelos_finales.py, genero.py, clustering.py,     # Modelado
│   │   prediccion.py
│   └── acerca.py                                                          # Acerca de
└── assets/                   # archivos estáticos que Dash carga automáticamente
    ├── style.css             # variables de diseño (colores, tipografía, espaciados; claro y oscuro)
    ├── componentes.css       # estilos de los componentes compartidos
    ├── portada.css           # portada e ilustración
    ├── shapiro.css           # estilos de la conversación
    ├── shapiro.js            # conversación y guía local de respaldo
    ├── mosaico.js            # rejilla tipo mosaico para las cards de texto
    ├── menu_hover.js         # abre los menús al pasar el cursor
    ├── animacion_graficos.js # animación de entrada y transiciones de los gráficos
    ├── altura_cabecera.js    # mide el menú superior para los filtros fijos bajo él
    ├── iconos/               # iconos SVG de IBM Carbon (GitHub, LinkedIn, libro, datos, tema…)
    └── logo_uninorte.png
```

## Comandos útiles

| Acción | Comando |
|---|---|
| Levantar Jupyter y el dashboard | `docker compose up --build -d` |
| Ver el estado de los contenedores | `docker compose ps` |
| Ver logs del dashboard | `docker compose logs dashboard` |
| Ver logs / token de Jupyter | `docker compose logs jupyter` |
| Entrar a la terminal de Jupyter | `docker compose exec jupyter bash` |
| Reiniciar el dashboard tras cambiar código | `docker compose restart dashboard` |
| Apagar el entorno | `docker compose down` |
| Compilar el Jupyter Book | `cd book && jupyter book build` (dentro del contenedor) |
| Instalar un paquete nuevo | Notebook: `docker/environment.yml`; dashboard: `requirements.txt` → `docker compose up --build -d` |
