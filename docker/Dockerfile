FROM continuumio/miniconda3

WORKDIR /workspace

# Copiamos solo el archivo de dependencias primero para aprovechar
# el cache de Docker (si no cambia environment.yml, no se reinstala todo)
COPY docker/environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# Activar el entorno en cada RUN/CMD posterior durante el build
SHELL ["conda", "run", "-n", "screen-time-env", "/bin/bash", "-c"]

# Hacer que cualquier terminal interactiva (docker compose exec ... bash)
# active screen-time-env automáticamente en vez de caer en "base"
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate screen-time-env" >> /root/.bashrc

EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "screen-time-env", \
     "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]