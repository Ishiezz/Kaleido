FROM python:3.12-slim

WORKDIR /app

# System deps for aiosqlite / asyncpg native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY ui/ ./ui/
COPY data/ ./data/

# Install the package and all runtime deps.
RUN pip install --no-cache-dir -e .

# Default environment — override via docker-compose env.
ENV KALEIDO_BACKEND=stub \
    KALEIDO_DATABASE_URL=sqlite+aiosqlite:///./kaleido.db \
    KALEIDO_FACETS_CSV_PATH=data/processed/facets_enriched.csv

EXPOSE 8000 8501
