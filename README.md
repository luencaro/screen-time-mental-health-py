# screen-time-mental-health-py

Clasificación de apnea obstructiva del sueño (OSA) pediátrica a partir de
señales EEG, usando Análisis Topológico de Datos (TDA).

Dataset: [Screen Time vs Mental Health (ML-ready)](https://www.kaggle.com/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready)

## Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac) o Docker Engine + Docker Compose (Linux)
- (Opcional) [Kaggle CLI](https://www.kaggle.com/docs/api) para descargar el dataset por comando

Todo lo demás (Python, conda, librerías) vive dentro del contenedor Docker —
no necesitas instalar nada de eso en tu sistema, sin importar el SO ni el editor que uses.

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
   La primera vez tarda unos minutos (construye la imagen). El `-d` lo corre en segundo plano.

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
├── .devcontainer/       # configuración de Dev Containers para VS Code
├── docker/              # Dockerfile + environment.yml (dependencias conda)
├── docker-compose.yml
├── data/
│   ├── raw/             # dataset original (no versionado)
│   └── processed/       # datos limpios/transformados (no versionado)
├── notebooks/           # notebooks de exploración y modelado
├── src/                 # funciones reutilizables
├── book/                # Jupyter Book (documentación del proyecto/tesis)
└── requirements.txt      # referencia rápida vía pip (opcional, fuera de Docker)
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
