# Databricks notebook source
# Query the prod serving endpoint with sample data.
# Runs locally (VS Code + Databricks Connect) or in the Databricks workspace.

import numpy as np
import pandas as pd
import requests
from databricks.sdk import WorkspaceClient

from marvel_characters.utils import is_databricks

# COMMAND ----------
# Settings: change ENDPOINT_NAME to query another endpoint (e.g. marvel-characters-model-serving-dev)
ENDPOINT_NAME = "marvel-characters-model-serving-prod"
PROFILE = "dbc-07d12638-ef42"  # local runs only; inside Databricks the notebook's own login is used

w = WorkspaceClient() if is_databricks() else WorkspaceClient(profile=PROFILE)
print(f"Workspace: {w.config.host}")

# COMMAND ----------
# Endpoint status: is it ready, and which model version is it serving?
endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint: {endpoint.name}")
print(f"Ready:    {endpoint.state.ready.value}")
for entity in endpoint.config.served_entities:
    print(f"Serving:  {entity.entity_name} version {entity.entity_version}")

# COMMAND ----------
# Helper that sends records to the endpoint and returns the predictions


def query_endpoint(records: list[dict]) -> list[str]:
    """Send feature records to the serving endpoint and return one prediction per record."""
    response = requests.post(
        f"{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers=w.config.authenticate(),  # fresh auth header on every call, never expires mid-run
        json={"dataframe_records": records},
        timeout=120,  # the first call after an idle period can be slow (scale-to-zero)
    )
    response.raise_for_status()
    return response.json()["predictions"]["Survival prediction"]


# COMMAND ----------
# 1) Hand-made sample characters. Every request needs all 10 features; use None for unknown values.
sample_characters = [
    {
        "Height": 183.0,
        "Weight": 86.0,
        "Universe": "Earth-616",
        "Identity": "Public",
        "Gender": "Male",
        "Marital_Status": "Single",
        "Teams": 1,
        "Origin": "Human",
        "Magic": 0,
        "Mutant": 0,
    },
    {
        "Height": 170.0,
        "Weight": 60.0,
        "Universe": "Earth-616",
        "Identity": "Secret",
        "Gender": "Female",
        "Marital_Status": "Married",
        "Teams": 1,
        "Origin": "Mutant",
        "Magic": 0,
        "Mutant": 1,
    },
    {
        "Height": None,
        "Weight": None,
        "Universe": "Other",
        "Identity": "Unknown",
        "Gender": "Other",
        "Marital_Status": "Unknown",
        "Teams": 0,
        "Origin": "Unknown",
        "Magic": 1,
        "Mutant": 0,
    },
]

predictions = query_endpoint(sample_characters)
results = pd.DataFrame(sample_characters).assign(prediction=predictions)
print(results.to_string(index=False))

# COMMAND ----------
# 2) Real rows from a test set, compared with the true label (Alive: 1 = alive, 0 = dead).
# The rows come from DEV's test_set but are sent to the PROD endpoint. Prod tables are created and owned by
# prod_spn, and people have no read access to prod data by design. Dev's test_set is built from the same CSV
# with the same random split, so the rows are identical.
from databricks.connect import DatabricksSession  # noqa: E402

from marvel_characters.config import ProjectConfig  # noqa: E402

DATA_ENV = "dev"  # which catalog to read sample rows from (mlops_<DATA_ENV>)
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env=DATA_ENV)
spark = (
    DatabricksSession.builder.getOrCreate()
    if is_databricks()
    else DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
)

N_ROWS = 20
test_set = (
    spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(N_ROWS).toPandas()
)
features = config.num_features + config.cat_features
records = test_set[features].replace({np.nan: None}).to_dict(orient="records")

comparison = test_set[features].copy()
comparison["actual"] = test_set[config.target].map({1: "alive", 0: "dead"})
comparison["prediction"] = query_endpoint(records)
accuracy = (comparison["actual"] == comparison["prediction"]).mean()

print(comparison[["Universe", "Gender", "Origin", "actual", "prediction"]].to_string(index=False))
print(f"\nMatches on these {N_ROWS} rows: {accuracy:.0%}")
