FROM python:3.12-slim AS invest

WORKDIR /app

RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Par défaut : mode dry-run (pas de connexion réelle)
CMD ["python", "-c", "import asyncio; from src.invest_manager import main; asyncio.run(main())"]