# HW 4

## 1. Overview

The point of HW 4 is setting up a proper ML pipeline with DVC and OmegaConf for our existing project.
Also, from previous stages we inherit:

- Cookiecutter for the overall template
- Poetry for dependencies, convenient setup with pyproject
- Pre-commit hooks and linting (Black, isort, Ruff, MyPy, Bandit)
- `wine_predictor` as a package
- Jupyter notebook solution using that package
- Docker reproducibility
- Git usage: starting out with pushing changes to `develop` branch, testing docker on `feature/docker`, finally pushing to `main`
- DVC-based data and model versioning
- MLFlow experiment tracking and logging

In this HW we add:
- A multi stage (train->evaluate->notify) ML pipeline
- Structured composable configuration with OmegaConf

#### A few notes on the specifics:
- We use Wine Quality prediction as an example
- The actual ML pipeline is just a baseline solution, for now we don't really care about metrics
- Running the Docker container means running the training and printing the metrics by default
- A few folders from the template (e.g. `data/processed`) were kept even though they are not currently used in case we need them in the future
- We don't actually make and plots or figures, but keep those folders for the reason mentioned above
- Detailed tool configs can be found in the corresponding files (mainly `pyproject.toml`)
- Why DVC? Seemed versatile and simple enough. I like the parallels between DVC and Git.
- Why MLFlow? I've already used W&B and TensorBoard. MLFlow was something I've been wanting to try for a while.
- Aside from MLFlow functionality, we track DVC pipeline progress entirely in console. We also apply the same logic to completion notifications, using `wine_predictor/pipelines/notify.py` as an example. Its logic could be replaced with a call to the Telegram API for a proper message-based notification system.


## 2. Project structure


![Tree](./figures/tree.png)

## 3. Dependency management

### 3.1 Poetry

Poetry is used as the dependency manager

This is a barebones example of using poetry to install our dependencies:

```bash
pip install --user poetry

poetry config virtualenvs.in-project true

poetry install
```

This creates `.venv/` inside the project and installs everything from `pyproject.toml` / `poetry.lock`.

### 3.2 pyproject.toml

It contains:
* Project metadata
* Python dependencies
* Dev dependencies (linters, pre-commit hooks, jupyter)
* Tool configuration

Installing from scratch:

```bash
git clone https://github.com/RageAgainstTheAssembly/EPML-ITMO
cd EPML_ITMO
poetry install
```

---

## 4. Formatters, linters, pre-commit hooks


* Black – code formatter
* isort – import sorter
* Ruff – fast linter + auto-fixes
* MyPy – static type checker
* Bandit – security scanner


Installing and running hooks:

```bash
poetry run pre-commit install

poetry run pre-commit run --all-files
```

![Hooks](./figures/hooks.png)


## 5. Actual ML solution

### 5.1 Implementation
Won't be going into detail - it's essentially a basic SKLearn pipeline with a Logistic Regression classifier. Training a great ML model is not the point of this homework

### 5.3 Running from a notebook

Solution in `notebooks/solution.ipynb` uses the package code to run training and inference.


## 6. Jupyter setup

We use Jupyter to showcase usage and register a dedicated kernel:

```bash
poetry add --group dev jupyterlab ipykernel

poetry run python -m ipykernel install --user --name epml-wine --display-name "Python (epml_itmo)"

poetry run jupyter lab
```

## 7. Docker


We provide a simple `Dockerfile` in the project root

To build and run:

```bash
docker build -t epml-wine:dev .

docker run --rm epml-wine:dev
```
This should train the model and print its metrics to console.
Note that the docker image works as a demo, we didn't change it to use DVC itself.

![Build](./figures/docker_build.png)
![Build](./figures/docker_run.png)


## 8. Versioning tools
DVC was chosen because it's very versatile and also more distinct than Git LFS, which I already have some experience with.
We use DVC for both data and model versioning, while also tracking hyperparams and metrics.

## 9. Data versioning
We use a local remote to keep track of our only dataset - WineQT.csv


## 10. MLFlow setup and integration
We use MLFlow to track experiments for reasons outlined above.

### 10.1 Setup
MLFlow is added as a project dependency via Poetry:
```bash
poetry add mlflow
```
For convenience, we isolate all MLFlow helpers and utils in `wine_predictor/mlflow_utils.py`

### 10.2 Database and artifacts
MLFlow is configured to run on top of a local SQLite DB and a local artifact directory:
Tracking DB URI:
sqlite:///mlruns/mlflow.db
Artifact location:
mlruns/artifacts

`configure_mlflow` handles basic configuration:
```python
# wine_predictor/mlflow_utils.py

DEFAULT_EXPERIMENT_NAME = "wine_quality_hw3"
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"

def configure_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
```
All runs are logged under a single MLFlow experiment.

### 10.3 Authentication
Since we're running a local-only MLFlow setup and UI is bound to localhost, we don't really have any traditional authentication:
```python
poetry run mlflow ui \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root mlruns/artifacts \
  --host 127.0.0.1 \
  --port 5000
```
By default, this means the UI can only be accessed from the local machine, so access comes down to local users' rights configuration. If we really wanted to emulate some kind of access restriction, we could do
```bash
chmod -R go-rwx mlruns
```
to make it so that only the current OS user can read/write tracking data.


### 10.4 Code integration
We use:

- `@training_run(...)` – decorator that wraps a training function into an MLflow run

- `mlflow_run(...)` – context manager for the decorator

`@training_run`:

- Configures MLFlow (URI + experiment name)

- Starts and stops runs

- Logs params, metrics, artifacts, model and tags

The decorator is defined in `wine_predictor/mlflow_utils.py` and is used in `wine_predictor/modeling/train.py`. The updated training code uses the new decorators and builds a trainining pipeline with the provided params.

### 10.5 Launching a single run
To launch a single run with `params.yaml`:
```bash
poetry run python -m wine_predictor.modeling.train
```

### 10.6 Launching a grid of experiments
To launch an entire grid of training runs:
```bash
poetry run python -m wine_predictor.experiments
```
All the params that define the grid can be found in `wine_predictor/experiments.py`

## 11. MLFlow usage
To run the UI:
```bash
poetry run mlflow ui \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root mlruns/artifacts \
  --host 127.0.0.1 \
  --port 5000
```
Once you've selected the experiment, you should see of all your logged runs:
![MLFlow UI list](./figures/mlflow_list.png)
Built-in functionality allows you to search and filter by metrics, values, time created, state. You can also sort and open additional columns with all of your tracked params.  For example, you could:
Filter runs by tags:

- `tags.hw = "3"`
- `tags.experiment = "grid"`
- `tags.algorithm = "random_forest"`

Sort by:
`metrics.accuracy DESC`

Select several runs and click `Compare` to see:

- parallel coordinate plots
- metric vs parameter plots (e.g. accuracy vs C)
- individual run details (artifacts, params, tags)

MLFlow also makes it easy to visualise key metrics:
![MLFlow UI visual](./figures/mlflow_vis.png)
Clicking on a specific run will allow you to see its details and params, as well as information on run artifacts:
![MLFLow UI single](./figures/mlflow_single.png)



## 12. DVC ML pipeline and OmegaConf configuration

### 12.1 Why DVC?
- Fairly lightweight and simple
- Already integrated for data versioning in previous stages of this project
- Supports all the essential features like stages, dependency tracking, caching, parallel execution

### 12.2 DVC pipeline setup

In practice, we implement the pipeline by changing `dvc.yaml` from a single `train` stage to `train`->`evaluate`->`notify`:
```yaml
stages:
  train:
    cmd: poetry run python -m wine_predictor.modeling.train --model logreg
    deps:
      - data/external/WineQT.csv
      - wine_predictor/dataset.py
      - wine_predictor/features.py
      - wine_predictor/modeling/train.py
      - wine_predictor/config.py
      - configs/train/base.yaml
      - configs/model/logreg.yaml
    outs:
      - models/baseline_model.joblib
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: poetry run python -m wine_predictor.pipelines.evaluate
    deps:
      - models/baseline_model.joblib
      - wine_predictor/pipelines/evaluate.py
      - wine_predictor/dataset.py
      - wine_predictor/features.py
      - wine_predictor/config.py
    metrics:
      - reports/metrics_detailed.json:
          cache: false

  notify:
    cmd: poetry run python -m wine_predictor.pipelines.notify
    deps:
      - reports/metrics_detailed.json
      - wine_predictor/pipelines/notify.py
    always_changed: true
```
We set `always_changed: true` for `notify` stage to make sure it actually runs every time, even when we're reproducing existing results.

This configuration results in a fairly simple DAG:
![DVC DAG](./figures/dvc_dag.png)

### 12.3 DVC pipeline usage
In order to run the pipeline with all its stages, run
```bash
poetry run dvc repro
```
This will run `train`, `evaluate` and `notify` sequentially. Initiating another run without changes will lead to the following behaviour:
![DVC Cache](./figures/dvc_cache.png)
Caching will ensure unchanged stages (`train`, `evaluate`) do not run again. Instead, only the pipeline completion notification stage `notify` will run to show results.

DVC also supports running multiple jobs in parallel
```bash
poetry run dvc repro -j 2
```
although our pipeline is not suited to parallel execution, as it lacks independent stages.

The console output of the various stages is used in conjunction with the `notify` stage to keep track of execution, errors and results.

### 12.4 Why OmegaConf?
- Easy to use with YAML files
- Supports the necessary features
- Pretty lightweight and easy to add to our existing code

### 12.5 OmegaConf usage
- Loading base training configuration from `configs/train/base.yaml`
- Loading algorithm-specific configurations from `configs/model/*.yaml`
- Providing these configs to the training code in a composable way

Configs are stored in a dedicated `configs/` directory:
```text
configs/
├── train
│   └── base.yaml
└── model
    ├── logreg.yaml
    ├── random_forest.yaml
    └── gradient_boosting.yaml

```
`wine_predictor/config.py` integrates OmegaConf and performs extra validation.

We also add some validation on top to make sure configs are checked for broken values.

Overall, we're using the following hierarchy: base -> model-type -> per-run overrides.


## 13. Reproducing everything


```bash
# 1. Clone
git clone https://github.com/RageAgainstTheAssembly/EPML-ITMO
cd EPML-ITMO
git checkout main_hw3

# 2. Poetry
pip install --user poetry
poetry config virtualenvs.in-project true

# 3. Dependencies
poetry install

# 4. Install pre-commit hooks and run them
poetry run pre-commit install
poetry run pre-commit run --all-files

# 5. Get data and model artifacts from DVC remote (this assumes you have access to the remote)
poetry run dvc remote modify local_remote url <insert your remote path>
poetry run dvc pull

# 6. Run the pipeline
poetry run dvc repro

# 7. See results in MLFlow UI
poetry run mlflow ui \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root mlruns/artifacts \
  --host 127.0.0.1 \
  --port 5000
```
