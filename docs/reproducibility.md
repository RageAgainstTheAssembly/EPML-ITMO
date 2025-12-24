# Reproducibility

## 1. Clone and install

```bash
git clone https://github.com/RageAgainstTheAssembly/EPML-ITMO
cd EPML-ITMO
poetry install --with docs
poetry run dvc remote modify local_remote url <insert your remote path>
poetry run dvc pull
```

## 2. Run DVC pipeline
```bash
poetry run dvc repro
```

## 3. Generate report
```bash
poetry run generate-experiment-report
```

## 4. Build docs
```bash
poetry run mkdocs build
```
