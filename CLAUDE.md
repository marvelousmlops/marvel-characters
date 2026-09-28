# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

End-to-end MLOps course project on Databricks (Marvelous MLOps). A LightGBM classifier predicts whether a Marvel character is `Alive` from the Kaggle Marvel Characters dataset (`data/marvel_characters_dataset.csv`). The `marvel_characters` package (in `src/`) is built as a wheel and deployed with a Databricks Asset Bundle.

## Environment

- Target runtime is **Databricks serverless environment version 3** (`client: "3"` in `resources/*.yml`), which pairs with Python 3.12 and `databricks-connect` 16.x. The exact pins in `pyproject.toml` (numpy 1.26.4, pyarrow 17, etc.) are chosen to match it.
- Do not accept `databricks environments setup-local` / VS Code extension changes that target a newer environment (e.g. env 5): they inject a `[tool.uv] constraint-dependencies` block and `databricks-connect~=18` that conflict with the pins and make `uv lock` unsatisfiable. Keep dependencies aligned with env 3. Also reject a `[dependency-groups] dev` block it may add: uv installs that group on every sync (including CI's `--extra test`), which puts `databricks-connect` next to `pyspark` and breaks test collection.
- Local dev is on Windows. `.venv` holds only the `dev` extra: never install the `test` extra into it (`pyspark` and `databricks-connect` conflict, and `uv sync --extra test` drops `dev`); run tests with `uv run --isolated --extra test pytest -m "not ci_exclude"`. A running notebook kernel locks `.venv` files, so `uv sync` fails until it is closed.

## Commands

```bash
uv sync --extra dev          # local dev env (includes databricks-connect)
uv sync --extra test         # what CI installs (pytest, pyspark)
uv run pre-commit run --all-files   # ruff lint+format, yaml/toml/json checks
uv run pytest -m "not ci_exclude"   # test suite as run in CI
uv run pytest tests/marvel_characters/test_data_processor.py::TestDataProcessor::test_preprocess  # single test
uv build                     # builds the wheel (also run by `databricks bundle deploy`)

databricks bundle validate --profile <PROFILE>
databricks bundle deploy -t dev --profile <PROFILE>
databricks bundle run deployment -t dev --profile <PROFILE>
```

pytest config adds coverage flags automatically. Tests use a local SparkSession (falls back to `MagicMock` if Spark can't start) and mock Spark/config, so they don't need a workspace.

Pre-commit and ruff exclude `notebooks/` and `scripts/`; ruff enforces docstrings (`D`) and type annotations (`ANN`) in `src/` and `tests/`, line length 120.

## Architecture

**Config**: `project_config_marvel.yml` holds per-env (`dev`/`stage`/`prod`) Unity Catalog catalog/schema (`mlops_<env>.marvel_characters`), MLflow experiment names, LightGBM params, feature lists and target. `ProjectConfig.from_yaml(path, env)` flattens the chosen env's catalog/schema into the model. Every script and notebook starts from this.

**Pipeline** (the `deployment` job in `resources/model_deployment.yml`, tasks chained in order):
1. `scripts/process_data.py` → `DataProcessor.preprocess()` / `split_data()` / `save_to_catalog()` writes `train_set` and `test_set` Delta tables.
2. `scripts/train_register_custom_model.py` → `BasicModel` (sklearn `Pipeline` with a category encoder + `LGBMClassifier`) trains, logs to MLflow, and compares against the current `latest-model` alias via `model_improved()`. If better, it registers `marvel_character_model_basic`, then `MarvelModelWrapper` (pyfunc that maps predictions to labels via `adjust_predictions`) registers `marvel_character_model_custom` with the package wheel as `code_paths`. It sets the job task value `model_updated` (1/0) and `model_version`.
3. Two `condition_task` gates guard `scripts/deploy_model.py`: `model_updated` (new model better?) and `endpoint_enabled` (the per-target bundle variable `deploy_endpoint` == "true"). `deploy_model` reads `model_version` from task values and calls `ModelServing.deploy_or_update_serving_endpoint()`. `deploy_endpoint` is "true" by default and "false" for `stage` (free tier: 2 endpoints, used by dev + prod), so the stage run trains/registers but skips the endpoint; flip it back on a paid workspace.

**Monitoring** (`resources/bundle_monitoring.yml` → `scripts/refresh_monitor.py` → `monitoring.create_or_refresh_monitoring`): parses the serving endpoint's inference table `custom_model_payload`, appends to `model_monitoring`, and creates/refreshes a Lakehouse quality monitor on it.

**Conventions that span files**:
- Models are addressed by the `latest-model` alias (set on registration, read by `model_improved()` and `ModelServing`); when no `latest-model` exists yet (first run in a fresh environment), `model_improved()` returns `True` so the first model gets registered.
- Scripts receive `--root_path ${workspace.root_path}` and `--env ${bundle.target}` and read config/data from `{root_path}/files/...`; the custom model's wheel path is `{root_path}/artifacts/.internal/marvel_characters-<version>-py3-none-any.whl`, with the version from `version.txt`.
- `scripts/deploy_model.py` deploys to one endpoint per environment, `marvel-characters-model-serving-<env>`, created on first run and owned by the job's run-as identity (so no cross-identity `CAN_MANAGE` grants). New endpoints do not yet enable inference tables, which monitoring (`custom_model_payload`) relies on.
- The workspace is a free (Community Edition) workspace capped at **2 serving endpoints** in total. Creating a third returns `429 RESOURCE_EXHAUSTED ... limit for endpoints for free usage`, which the SDK retries silently until `TimeoutError: Timed out after 0:05:00`. When `deploy_model` times out, check endpoint count first; any new endpoint (per-env, A/B, app) competes for those 2 slots.
- `notebooks/lectureN.*.py` are Databricks source-format notebooks (`# COMMAND ----------` cells) that `%pip install -e ..` the package and demo each lecture step interactively; they are not part of the deployed job.
- pyfunc wrappers must normalize artifact paths in `load_context` (`context.artifacts[...].replace("\\", "/")`): MLflow stores them with `os.path.join`, so models logged from Windows contain `artifacts\.` and fail to load on the Linux serving container. The custom model's fix ships inside the registered wheel, so re-register after changing it; `lecture6.deploy_model_serving_endpoint.py` has a pre-deploy check that verifies the packaged wheel.
- Use explicit `int64`/`float64` dtypes in pandas code that feeds Delta tables, never `astype(int)`: with numpy 1.x that is int32 on Windows and int64 on Linux, so tables written locally and by jobs end up with conflicting `INT`/`BIGINT` schemas (`DELTA_MERGE_INCOMPATIBLE_DATATYPE`).
- Call serving endpoints with `w.config.authenticate()` headers, not the short-lived PATs some notebooks mint at startup (they expire before long training runs finish, giving 403 "Invalid access token").

**Bundle targets** (`databricks.yml`): `dev` (default, development mode, per-user root path), `stage` (`stage_` name prefix), `prod` (production mode); schedules are `PAUSED` in all targets. CD (`.github/workflows/cd.yml`) runs on push to `main`: job `deploy-stage`, then `deploy-prod` (`needs: deploy-stage`, paused by the `prod` GitHub environment's required reviewers), which also tags the release with `version.txt` — bump it in every PR. Each GitHub environment (`stage`, `prod`; there is deliberately no `dev`) holds its own service principal's credentials (`stage_spn`, `prod_spn`), and each SP is granted only its own catalog.

The README's script list (`01.process_data.py`, etc.) is outdated; the actual scripts are the four in `scripts/`. The README's "Fixes and changes log" section records past bugs and fixes; add new entries there (problem, error, fix). Its "Roadmap / future to-dos" section lists planned work (end-user app on Render, hyperparameter tuning, beginner template).
