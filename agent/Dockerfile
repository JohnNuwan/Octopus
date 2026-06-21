FROM python:3.13-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY src/ ./src/

# Port du dashboard
EXPOSE 9093

# Santé
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:9093/api/health')" || exit 1

CMD ["uvicorn", "src.dashboard.app:app", "--host", "0.0.0.0", "--port", "9093"]