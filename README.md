<h1 align="center">
Marvelous MLOps Free End-to-end MLOps with Databricks Course

## Set up your environment
In this course, we use Databricks serverless [version 3](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/three)

In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv sync --extra dev
```



# Data
Using the [**Marvel Characters Dataset**](https://www.kaggle.com/datasets/mohitbansal31s/marvel-characters?resource=download) from Kaggle.

This dataset contains detailed information about Marvel characters (e.g., name, powers, physical attributes, alignment, etc.).
It is used to build classification and feature engineering models for various MLOps tasks, such as predicting character attributes or status.

# Scripts

- `01.process_data.py`: Loads and preprocesses the Marvel dataset, splits into train/test, and saves to the catalog.
- `02.train_register_fe_model.py`: Performs feature engineering and trains the Marvel character model.
- `03.deploy_model.py`: Deploys the trained Marvel model to a Databricks model serving endpoint.
- `04.post_commit_status.py`: Posts status updates for Marvel integration tests to GitHub.
- `05.refresh_monitor.py`: Refreshes monitoring tables and dashboards for Marvel model serving.

# Fixes and changes log

Problems found while running this project locally on Windows, what they caused, and how they were fixed.

## 1. Notebooks could not get a Spark session locally (`SparkSession` → `DatabricksSession`)

- **Problem:** the notebooks were written for the Databricks workspace and create Spark with `SparkSession.builder.getOrCreate()`. Run locally from VS Code, there is no Spark cluster behind that call, so the notebooks could not read the Unity Catalog tables (`train_set`, `test_set`).
- **Error:** the Spark cells did not run locally (no session to the workspace).
- **Fix:** switched to Databricks Connect, which runs the Spark code on the workspace's serverless compute:
  ```python
  from databricks.connect import DatabricksSession

  spark = DatabricksSession.builder.profile("dbc-07d12638-ef42").serverless(True).getOrCreate()
  ```
  Applied in `lecture2.marvel_data_preprocessing.py`, both `lecture4.*` notebooks and `lecture6.ab_testing.py`. `databricks-connect` 16.x (the `dev` extra) matches serverless environment 3.
- **Follow-up bug:** the new import in `lecture4.train_register_custom_model.py` was written as `from Databricks.connect import ...`. Python package names are case-sensitive, so it failed with `ModuleNotFoundError: No module named 'Databricks'`. Fixed to lowercase `databricks`.
- **Not yet converted:** `lecture6.deploy_model_serving_endpoint.py` and the first part of `lecture10.marvel_create_monitoring_table.py` still use `SparkSession`.

## 2. Custom model fails to load on the serving endpoint (Windows path bug)

- **Problem:** when a pyfunc model is logged with `artifacts=...`, MLflow (3.1.1, `mlflow/pyfunc/model.py`) records the artifact path with `os.path.join`. Logged from Windows, the `MLmodel` file stores `path: artifacts\.`. The Linux serving container joins that onto `/model/`, producing a path that does not exist.
- **Error:** the endpoint image builds, but the model never loads and the deployment ends in `DEPLOYMENT_FAILED` after ~10 minutes of retries:
  `OSError: No such file or directory: '/model/artifacts\.'`
- **Fix:**
  - `src/marvel_characters/models/custom_model.py`: `MarvelModelWrapper.load_context` normalizes the path with `.replace("\\", "/")` before `mlflow.sklearn.load_model`. Forward slashes also work on Windows, so local loading is unaffected.
  - Rebuilt the wheel (`uv build`) and re-registered the custom model. Each registered version carries the wheel it was logged with, so older versions keep the bug.
  - `notebooks/lecture6.deploy_model_serving_endpoint.py`: new **pre-deploy check** cell before the deploy cell. It loads the `latest-model` version, runs a prediction, then opens the wheel packaged inside the model and asserts the fix is present. A local prediction alone cannot catch this bug, because Windows accepts backslash paths.
- **Note:** models registered by the Databricks `deployment` job (Linux) are not affected. Prefer registering models meant for serving from Databricks.

## 3. A/B testing notebook (`notebooks/lecture6.ab_testing.py`)

| Problem | Error / effect | Fix |
|---|---|---|
| Model B's "different" parameters were identical to the config values used by model A | The A/B test compared a model with itself | Model B now uses `learning_rate=0.05, n_estimators=300, max_depth=3` |
| The inline A/B wrapper had the same Windows path bug as #2 | `OSError: No such file or directory: '/model/artifacts\...'` on the endpoint | `.replace("\\", "/")` on both artifact paths in `load_context`. The wrapper is defined in the notebook, so the fix is pickled with the model; no wheel rebuild needed |
| The endpoint was called with a token created at the top of the notebook with `lifetime_seconds=1200` | Training plus endpoint start-up took longer than 20 minutes: `403 {"error_code":403,"message":"Invalid access token."}` | `call_endpoint` uses `w.config.host` and `w.config.authenticate()`, which returns a fresh auth header on every call |
| `WorkspaceClient()` had no profile; the local `DEFAULT` profile is empty and `dev-databricks` points at another workspace | The client could resolve to the wrong workspace, or fail with `cannot configure default credentials` | `WorkspaceClient(profile="dbc-07d12638-ef42")`, the same workspace as the Spark session; the endpoint-creation cell reuses that client |

## 4. Environment names: `acc`/`prd` to `stage`/`prod`

- **Problem:** the bundle targets were renamed to `dev`/`stage`/`prod`, but the jobs pass `--env ${bundle.target}` to the scripts, and the config and its validation still used `acc`/`prd`.
- **Error:** every `stage`/`prod` job would fail on its first task with `ValueError: Invalid environment: prod. Expected 'prd', 'acc', or 'dev'`, and CD would fail on an unknown bundle target.
- **Fix:** renamed consistently in `databricks.yml` (targets and `stage_` name prefix), `project_config_marvel.yml` (`stage`/`prod` sections, catalogs `mlops_stage`/`mlops_prod`), `src/marvel_characters/config.py` (validation), `.github/workflows/cd.yml` (matrix `[stage, prod]`, tag on `prod`) and `CLAUDE.md`.
- **Still required outside the code:** Unity Catalog catalogs `mlops_stage` and `mlops_prod`, and GitHub environments named exactly `stage` and `prod`, each with the variable `DATABRICKS_HOST` and the secrets `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` (service principal OAuth).

## 5. CI/CD release tag never created

- **Problem:** `.github/workflows/cd.yml` and `ci.yml` used `echo "VERSION=$(cat version.txt)"`, which only prints text; it does not set a variable.
- **Error:** none, which is the problem: `$VERSION` was empty, so `git tag $VERSION` just listed tags and `git push origin $VERSION` pushed the branch. `prod` releases were silently never tagged.
- **Fix:** `VERSION=$(cat version.txt)` followed by `git tag "$VERSION"` / `git push origin "$VERSION"`. Because the tag is now really created, **bump `version.txt` before each merge to `main`**. The CI tag step fails early on a pull request that reuses an existing version.

## 6. `preprocessing` job task fails writing `train_set` (INT vs BIGINT)

- **Problem:** `DataProcessor.preprocess()` used `.astype("int")` / `.astype(int)` for `Teams` and `Alive`. With numpy 1.x that is **int32 on Windows** but **int64 on Linux**. The tables were first written from a laptop (lecture 2 via Databricks Connect), so Delta stored those columns as `INT`; the `deployment` job then ran `process_data.py` on Linux and produced `BIGINT`. `mode("overwrite")` replaces the data but keeps the table schema, and Delta refuses to change the type implicitly.
- **Error:** `databricks bundle run deployment` failed at the `preprocessing` task, skipping all downstream tasks:
  `[DELTA_FAILED_TO_MERGE_FIELDS] Failed to merge fields 'Teams' and 'Teams'` caused by `[DELTA_MERGE_INCOMPATIBLE_DATATYPE] Failed to merge incompatible data types IntegerType and LongType`.
- **Fix:** in `src/marvel_characters/data_processor.py`:
  - `Teams` and `Alive` use `.astype("int64")`, so the schema is the same whichever OS writes the tables.
  - `save_to_catalog` writes with `.option("overwriteSchema", "true")`, so a full overwrite also replaces column types left by earlier writes (needed once to move the existing `INT` columns to `BIGINT`).
- Redeploy (`databricks bundle deploy`) before re-running the job; the job uses the wheel uploaded at deploy time.

## 7. CI/CD setup: service principals, GitHub environments, stage → prod order

- **Problem:** CD had never run: no GitHub environments, no secrets, no service principals. Also `cd.yml` deployed with a matrix `[stage, prod]`, which starts **both deployments in parallel**. The prod approval gate could be approved before stage finished, or even after stage failed.
- **Error / risk:** CD would fail at authentication; once credentials existed, untested changes could reach prod before stage had deployed successfully.
- **Fix:**
  - **Databricks:** one service principal per environment, `stage_spn` and `prod_spn`, each with an OAuth secret. Each can use only its own catalog: `USE_CATALOG` on `mlops_stage` / `mlops_prod`, and `USE_SCHEMA, CREATE_TABLE, CREATE_MODEL, SELECT, MODIFY, EXECUTE` on `<catalog>.marvel_characters`.
  - **GitHub → Settings → Environments:** `stage` and `prod`, each with secrets `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` (that environment's service principal) and variable `DATABRICKS_HOST`. `prod` has **required reviewers**. There is no `dev` environment: no workflow uses one, because dev is deployed from a laptop with a personal login.
  - **`.github/workflows/cd.yml`:** two jobs instead of a matrix. `deploy-stage` runs first; `deploy-prod` has `needs: deploy-stage`, so it is skipped if stage fails and then waits for approval. Only the prod job gets `contents: write`, for the release tag. A `concurrency` group stops two CD runs deploying over each other. The unused step that wrote the secret into `~/.databrickscfg` was removed; the CLI authenticates from the `DATABRICKS_*` environment variables.
- **Flow:** feature branch → pull request (CI: lint + tests) → merge to `main` → CD deploys stage → approve → CD deploys prod and tags `version.txt`.
- **Problems on the first run, and fixes:**
  - The PR was opened against the course's original repo (`marvelousmlops/marvel-characters`) because GitHub defaults a fork's PR base to the parent repo. Closed it and opened it on the fork; `gh repo set-default kelvin-meyet/mlops-marvel-characters` avoids this.
  - CI did not run on the fork at first: GitHub disables workflows on forks until **Actions → "I understand my workflows, go ahead and enable them"** is clicked.
  - `deploy-stage` failed with `default auth: cannot configure default credentials ... client_id=***, client_secret=***`. With `--debug` the real reason is `POST /oidc/v1/token → "error": "invalid_client"`: the service principal secrets had been generated with a narrow scope. The CLI requests the `all-apis` scope, so the secrets were regenerated with **All APIs** scope for both `stage_spn` and `prod_spn`, then updated in the GitHub environments.
  - Result: `deploy-stage` ✓, then approval, then `deploy-prod` ✓, and tag `0.1.0` pushed.
  - While fixing the secret scopes, `stage_spn` and `prod_spn` were **recreated**, which gives them new Application IDs. Unity Catalog grants belong to the old IDs, so the new service principals had **no data access**; CD still passed (it only deploys), but the stage/prod jobs would have failed on their first task. Fixed by granting the same privileges to the new IDs and removing the stale grants. **Whenever a service principal is recreated, re-apply its grants** (`databricks grants get catalog mlops_stage` shows who has access).
- **Follow-up:** the stage/prod `deployment` jobs could not safely reach `deploy_model`, because every environment deployed to one endpoint owned by a person. Fixed in entry 8.

## 8. One serving endpoint per environment

- **Problem:** `scripts/deploy_model.py` built `marvel-characters-model-serving-{env}` and then overwrote it with the fixed name `marvel-character-model-serving`, so dev, stage and prod all deployed to the same endpoint.
- **Error / risk:** whichever job ran last decided what was served (a stage or dev model could replace the prod one). The endpoint was owned by a person, so the prod job, running as `prod_spn`, would fail at `deploy_model` with a permission error unless given `CAN_MANAGE` on it.
- **Fix:** removed the override, so each environment uses its own endpoint: `marvel-characters-model-serving-dev`, `-stage`, `-prod`. `ModelServing.deploy_or_update_serving_endpoint()` creates the endpoint on the first run, and the identity that creates it (you in dev, `stage_spn` / `prod_spn` via CD) owns it, so no `CAN_MANAGE` grant is needed. `notebooks/lecture6.deploy_model_serving_endpoint.py` now targets `-dev`. `version.txt` bumped to `0.1.1`.
- **After deploying:** the old shared endpoint `marvel-character-model-serving` is no longer updated by any job; delete it once `-dev` exists.
- **Still open:** `ModelServing` does not enable inference tables when it creates an endpoint, so new endpoints do not log requests to `custom_model_payload`, and the monitoring job has no data for them.
- **Problem found on the first dev run (free workspace limit):** the dev job's `deploy_model` task failed twice with `TimeoutError: Timed out after 0:05:00` in `ModelServing.deploy_or_update_serving_endpoint()` → `serving_endpoints.create`. Sending the same request directly showed the real response: `429 RESOURCE_EXHAUSTED: "You've hit the limit for endpoints for free usage."` (`currentUsage: 2, maxLimit: 2, limitReason: COMMUNITY_EDITION`). The SDK treats 429 as retryable, so it retried silently for 5 minutes and hid the message. This workspace allows **at most 2 serving endpoints**, fewer than dev + stage + prod need. **Fix for dev:** deleted the two old endpoints (`marvel-character-model-serving`, `marvel-characters-ab-testing`; registered models are unaffected) and re-ran the dev job: all tasks succeeded and `marvel-characters-model-serving-dev` (created by the job, serving `marvel_character_model_custom` v7) became `READY` in about 8 minutes and answered a test request. With one slot left, prod can have an endpoint but stage cannot.

## 9. Per-environment endpoint switch (`deploy_endpoint`): stage skips the endpoint

- **Problem:** the free workspace allows only 2 serving endpoints (entry 8), but dev, stage and prod each tried to create their own. Whichever of stage/prod ran after the other two would fail at `deploy_model` with the hidden 429 timeout.
- **Fix, in simple terms:** a yes/no setting per environment, "should this environment create a serving endpoint?" (dev: yes, stage: **no**, prod: yes). The job checks it right before `deploy_model`; when it is "no", `deploy_model` is **skipped** and the run still succeeds. The code is identical in every environment; only the setting differs.
- **How it is wired:**
  - `databricks.yml`: bundle variable `deploy_endpoint` (default `"true"`); target `stage` sets `deploy_endpoint: "false"`.
  - `resources/model_deployment.yml`: new `condition_task` `endpoint_enabled` (`${var.deploy_endpoint}` == `"true"`) after `model_updated`; `deploy_model` depends on its `true` outcome.
  - Pipeline: `preprocessing → train_model → model_updated? → endpoint_enabled? → deploy_model`.
- **Trade-off:** stage still proves preprocessing, training, the model comparison and registration work as `stage_spn` on `mlops_stage` (where most bugs show up), but no longer tests serving; prod is now the first place serving is exercised outside dev. On a paid workspace, delete the `deploy_endpoint: "false"` line and stage gets its own endpoint again.
- `version.txt` bumped to `0.1.2`.

## 10. First run in a fresh environment crashed at `train_model`

- **Problem:** `BasicModel.model_improved()` compares the new model with the registered `latest-model`, but in a brand-new environment (the first stage run, and later the first prod run) no model is registered yet.
- **Error:** the stage job's `train_model` task failed (and its retry too), skipping everything after it:
  `RestException: RESOURCE_DOES_NOT_EXIST: Routine or Model 'mlops_stage.marvel_characters.marvel_character_model_basic' does not exist.`
  `preprocessing` succeeded as `stage_spn`, which confirmed the re-applied data permissions work.
- **Fix:** `src/marvel_characters/models/basic_model.py`: if looking up `latest-model` fails with `RESOURCE_DOES_NOT_EXIST`, `model_improved()` returns `True` (nothing to beat, so the first model is registered). Any other error (e.g. `PERMISSION_DENIED`) still fails the job.
- **Tests:** new `tests/marvel_characters/test_basic_model.py` covers both cases; the first test fails on the old code. It stubs the `delta` module, which exists only on Databricks. `version.txt` bumped to `0.1.3`.
- **Verified (v0.1.3):** the stage job ran end to end as `stage_spn`: `preprocessing` ✓, `train_model` ✓ (registered `mlops_stage.marvel_characters.marvel_character_model_basic` and `..._custom` as v1 with `latest-model`), `model_updated` = true, `endpoint_enabled` = **false**, `deploy_model` **skipped**, run result SUCCESS. This also confirms the entry 9 switch works in a real run. The first prod run then succeeded as `prod_spn`: registered `mlops_prod...marvel_character_model_custom` v1, `endpoint_enabled` = true, and `deploy_model` created **`marvel-characters-model-serving-prod`** (owned by `prod_spn`, READY in about 7 minutes, test request answered). Endpoints in use: dev + prod = 2 of 2. The narrow per-catalog grants from entry 7 were enough for every task; no `ALL PRIVILEGES` needed. `ALL PRIVILEGES` had been added to `stage_spn` on `mlops_stage`; it was removed again (least privilege: it covers the whole catalog, future schemas included, and would let a leaked secret modify or drop anything there). Note: revoking `ALL_PRIVILEGES` also removed the separate `USE_CATALOG` grant, so it had to be re-added; check with `databricks grants get catalog mlops_stage` after any revoke.

## 11. Notebook to query the prod endpoint (`notebooks/query_prod_endpoint.py`)

- **What it does:** shows the endpoint's status and served model version, sends 3 hand-made characters, then sends 20 real test-set rows and compares predictions with the true `Alive` label. `ENDPOINT_NAME` at the top switches endpoints (e.g. `-dev`). Works locally (profile) and in the workspace; authenticates with `w.config.authenticate()` on every call (see entry 3).
- **Problem hit while building it:** reading `mlops_prod.marvel_characters.test_set` as a person failed with `[INSUFFICIENT_PERMISSIONS] User does not have SELECT on Table 'mlops_prod.marvel_characters.test_set'`. The prod tables are created and owned by `prod_spn`, and people have no read access to prod data; that is the intended lock-down, not a bug.
- **Fix:** the notebook reads sample rows from **dev's** `test_set` (`DATA_ENV = "dev"`), which is built from the same CSV with the same random split, and sends them to the **prod** endpoint. If people ever need to read prod data, grant `SELECT` on the prod schema to a named group deliberately rather than widening the service principal.

## Local environment notes (Windows)

- **Do not install the `test` extra into `.venv`.** `pyspark` (test) and `databricks-connect` (dev) both provide the `pyspark` module and break each other, and `uv sync --extra test` removes the dev extra. Run tests in a throwaway environment:
  ```
  uv run --isolated --extra test pytest -m "not ci_exclude"
  ```
- **No `[dependency-groups] dev` in `pyproject.toml`.** The VS Code Databricks extension added `[dependency-groups] dev = ["databricks-connect~=16.4.0"]`. uv installs the `dev` group on every `uv sync` by default, including CI's `uv sync --extra test`, so `databricks-connect` and `pyspark` were installed together and test collection failed with `ImportError: cannot import name '_with_origin' from 'pyspark.errors.utils'`. The group was removed; `databricks-connect` comes from the `dev` extra. An older local uv (0.5.x) also rewrites `uv.lock` in an older format (drops `upload-time`), so do not commit a relock that only changes formatting.
- **Close notebooks before `uv sync`.** A running Jupyter kernel locks `debugpy` files; the sync then stops halfway with `Access is denied (os error 5)` and can leave `ipykernel` uninstalled. Close the kernel and re-run `uv sync --extra dev`.
- **Keep the Databricks CLI current, and installed only once.** An old Chocolatey install (v0.240.0) sat ahead of the winget install (v1.18.0) on `PATH`. `databricks bundle validate`/`deploy` download Terraform and verify it with a HashiCorp signing key built into the CLI; the key in v0.240.0 has expired, so they failed with `Error: error downloading Terraform: unable to verify checksums signature: openpgp: key expired`. Fixed by `choco uninstall databricks-cli` (admin PowerShell); check with `databricks --version`. CD had the same exposure: `.github/workflows/cd.yml` pinned `databricks/setup-cli` and the CLI to v0.246.0, so both were bumped to v1.18.0 (action pinned by commit SHA `6a2e75f`).
- **Use a PowerShell terminal in VS Code.** The Databricks extension's "Run as file with Databricks Connect" sends PowerShell syntax (`& "...python.exe" ...`); in Git Bash it fails with `syntax error near unexpected token '&'`.

# Roadmap / future to-dos

Planned work, not started yet:

1. **End-user app.** Build a small web app that calls the model (a UI with a form for the 10 features that shows "alive"/"dead"), and deploy it on **Render**. Decide between a FastAPI backend + simple frontend, or an all-Python UI (Streamlit, Gradio or Dash). The app would call the Databricks serving endpoint with a token stored as a Render secret, not in code.
2. **Hyperparameter tuning.** The LightGBM parameters are fixed in `project_config_marvel.yml`; there is no tuning. Add a tuning step (e.g. Optuna, logged to MLflow) and possibly revisit the dataset / features.
3. **Beginner template.** A barebones, easy-to-follow template in a separate folder that a novice can copy for their own end-to-end project (data → training → MLflow registry → serving → CI/CD), with the lessons from this log built in.
4. **Offline serving check in stage.** Stage skips the endpoint (entry 9), so add a `validate_model` task there that loads the newly registered model with `mlflow.models.predict(...)` in a fresh, isolated environment on Linux (roughly what the serving container does). This would catch serving-only bugs like the Windows artifact path (entry 2) without using an endpoint slot.
