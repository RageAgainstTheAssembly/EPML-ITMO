# Deployment

## Local setup (Poetry)

```bash
pip install --user poetry
poetry config virtualenvs.in-project true
poetry install --with docs
```

## Run the DVC pipeiline
```bash
poetry run dvc repro
```

## Run the MLFlow UI
```bash
poetry run mlflow ui \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root mlruns/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

## Run Docker
```bash
docker build -t epml-wine:dev .
docker run --rm epml-wine:dev
```

## Run local CLearML server
Download the suggested `docker-compose.yml` file from Github, in its directory run:
```bash
docker-compose up
```
Then, use the UI to set up credentials and copy the access key. In the project directory run:
```bash
poetry run clearml-init
```
and paste the access key when prompted.
