FROM continuumio/miniconda3

WORKDIR /workspace

# Copiamos solo el archivo de dependencias primero para aprovechar
# el cache de Docker (si no cambia environment.yml, no se reinstala todo)
COPY docker/environment.yml .

# Durante el build no hay GPU visible, así que conda no detecta CUDA e instala las
# variantes CPU de xgboost/catboost/lightgbm. docker-compose.gpu.yml pasa este
# argumento (p. ej. "12.9") para que conda instale las variantes CUDA de conda-forge.
# Vacío (por defecto) = build solo CPU, como siempre.
ARG CONDA_OVERRIDE_CUDA=""
# xgboost y catboost toman la variante CUDA solos, pero lightgbm sigue eligiendo la CPU
# (conda-forge la prioriza), así que en la imagen GPU se pide la variante cuda* a mano.
RUN CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA}" conda env create -f environment.yml && \
    if [ -n "${CONDA_OVERRIDE_CUDA}" ]; then \
        CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA}" conda install -y -n screen-time-env -c conda-forge \
            "liblightgbm=4.7.0=cuda*" "lightgbm=4.7.0"; \
    fi && \
    conda clean -afy

# Activar el entorno en cada RUN/CMD posterior durante el build
SHELL ["conda", "run", "-n", "screen-time-env", "/bin/bash", "-c"]

# Hacer que cualquier terminal interactiva (docker compose exec ... bash)
# active screen-time-env automáticamente en vez de caer en "base"
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate screen-time-env" >> /root/.bashrc

EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "screen-time-env", \
     "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]