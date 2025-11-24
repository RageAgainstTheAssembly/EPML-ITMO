FROM python:3.12-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app


RUN pip install --no-cache-dir poetry


COPY pyproject.toml poetry.lock ./


RUN poetry config virtualenvs.in-project true \
    && poetry install --no-root --no-interaction --no-ansi


COPY . .


CMD ["poetry", "run", "python", "-m", "wine_predictor.modeling.train"]
