# syntax=docker/dockerfile:1
#
# Two images from one file:
#   api   slim service image (stub inspector); used by CI and for trying the API    (~400 MB)
#   full  adds CPU PyTorch, Qwen3-VL and YOLO support; model weights are downloaded at first
#         use into HF_HOME, which should be a mounted volume                         (~2.5 GB)
#
#   docker build --target api  -t vlm-inspect:api .
#   docker build --target full -t vlm-inspect:full .

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf
# libgl1/libglib: OpenCV runtime libraries (Ultralytics pulls in the non-headless OpenCV build).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 app \
 && mkdir -p /models/hf && chown app /models/hf
WORKDIR /app
COPY pyproject.toml README.md LICENSE alembic.ini ./
COPY migrations ./migrations
COPY src ./src

FROM base AS api
RUN pip install ".[api]"
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["sh", "-c", "alembic upgrade head && vlm-inspect serve"]

FROM base AS full
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[api,vlm,yolo]"
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["sh", "-c", "alembic upgrade head && vlm-inspect serve"]
