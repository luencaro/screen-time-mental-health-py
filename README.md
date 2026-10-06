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

 
5. **(Opcional) Abrir JupyterLab** para ver `notebooks/EDA.ipynb`:
   ```powershell
   docker compose logs jupyter
   ```
   Copiar el enlace que empieza por `http://127.0.0.1:8888/lab?token=...` y pegarlo en el navegador.

6. **Apagar cuando termines**
   ```powershell
   docker compose down
   ```
   Los archivos no se pierden; solo se detienen los contenedores.

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
├── myst.yml                  # configuración y tabla de contenidos del Jupyter Book (MyST)
├── requirements.txt          # dependencias del dashboard (las instala Dockerfile.dashboard)
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
│   └── hallazgos.py                                                       # Cierre
└── assets/                   # archivos estáticos que Dash carga automáticamente
    ├── style.css             # variables de diseño (colores, tipografía, espaciados; claro y oscuro)
    ├── componentes.css       # estilos de los componentes (usan solo las variables de style.css)
    ├── menu_hover.js         # abre los menús desplegables al pasar el cursor
    ├── mosaico.js            # rejilla tipo mosaico para las cards de texto
    ├── animacion_graficos.js # animación de entrada y transiciones de los gráficos
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
