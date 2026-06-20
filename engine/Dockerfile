FROM python:3.12-slim AS engine

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/src/ ./src/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV JAX_PLATFORM_NAME=gpu

CMD ["python", "-m", "src.training"]