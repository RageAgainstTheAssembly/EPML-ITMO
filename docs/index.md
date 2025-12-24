# EPML ITMO — Wine Predictor

This repository contains a project project for ML/DL Engineering Practices course:
- Poetry dependency management
- DVC pipeline (train → evaluate → notify)
- MLflow experiment tracking
- ClearML tracking + pipeline
- OmegaConf configuration
- and more

## Quickstart

```bash
git clone https://github.com/RageAgainstTheAssembly/EPML-ITMO
cd EPML-ITMO
poetry install --with docs
poetry run dvc repro
poetry run python -m wine_predictor.reporting.generate_experiment_report
poetry run mkdocs serve
```
