
# EPML HW 1

This is just a brief README, you can find the full report in `docs/report.md`

Perfectly reproducible on my machine, hoping that's a good sign

- Cookiecutter for the overall template
- Poetry for dependencies, convenient setup with pyproject
- Pre-commit hooks and linting (Black, isort, Ruff, MyPy, Bandit)
- `wine_predictor` as a package
- Jupyter notebook solution using that package
- Docker reproducibility


## Quick Start

### 1. Install Poetry and dependencies

```bash
git clone https://github.com/RageAgainstTheAssembly/EPML-ITMO
cd EPML_ITMO

pip install --user poetry
poetry config virtualenvs.in-project true

poetry install
```

### 2. Code quality checks

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

---

## Baseline Model

Impelented in the `wine_predictor` package.

To train and evaluate the model:

```bash
poetry run python -m wine_predictor.modeling.train
```

This will:

* Load dataset
* Split into train/test
* Train a basic logistic regression model
* Print metrics

## Jupyter Notebooks

To run:

```bash
poetry add --group dev jupyterlab ipykernel
poetry run python -m ipykernel install --user --name epml-wine --display-name "Python (epml_itmo)"
poetry run jupyter lab
```

Use the kernel **“Python (epml_itmo)”** and open:

* `notebooks/solution.ipynb`

## Docker

Build and run the container:

```bash
docker build -t epml-wine:dev .
docker run --rm epml-wine:dev
```

The container runs the same baseline training (`wine_predictor.modeling.train`)


## Full Report

* `docs/report.md`
