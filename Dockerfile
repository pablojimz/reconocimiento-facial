FROM python:3.11-slim-bookworm AS dependencies

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 make \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY reconocimiento_facial ./reconocimiento_facial
RUN pip install '.[calibration]'

FROM dependencies AS download-models
RUN reconocimiento-facial models download --model all

FROM scratch AS model-files
COPY --from=download-models /app/models/ /

FROM dependencies AS app
COPY Makefile ./
COPY scripts ./scripts
COPY tests ./tests
COPY database ./database
RUN mkdir -p outputs && chmod a+rwx outputs

ENTRYPOINT ["reconocimiento-facial"]
