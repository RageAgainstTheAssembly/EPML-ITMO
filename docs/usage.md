# Usage examples

## Train a single run

```bash
poetry run python -m wine_predictor.modeling.train --model logreg
```

## Run an experiment grid
```bash
poetry run python -m wine_predictor.experiments
```

## Run the DVC pipeline
```bash
poetry run dvc repro
```

## Run the ClearML pipeline
```bash
poetry run python -m wine_predictor.pipelines.clearml_pipeline --model logreg
```
