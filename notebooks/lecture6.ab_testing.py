# Databricks notebook source
# MAGIC %pip install marvelousmlops-marvel-characters-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python  

# COMMAND ----------

import hashlib 
import os
import time

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from dotenv import load_dotenv
from mlflow.models import infer_signature
#from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel
from marvel_characters.utils import is_databricks

# COMMAND ----------

# Set up Databricks or local MLflow tracking
#spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.profile("dbc-07d12638-ef42").serverless(True).getOrCreate()

w = WorkspaceClient(profile="dbc-07d12638-ef42")  # same workspace as the Spark session above

os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
# Define tags (customize as needed)
tags = Tags(git_sha="dev", branch="ab-testing")

# COMMAND ----------
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Train model A
basic_model_a = BasicModel(config=config, tags=tags, spark=spark)
basic_model_a.load_data()
basic_model_a.prepare_features()
basic_model_a.train()
basic_model_a.log_model()
basic_model_a.register_model()
model_A_uri = f"models:/{basic_model_a.model_name}@latest-model"

# COMMAND ----------
# Train model B (with different hyperparameters or features)
basic_model_b = BasicModel(config=config, tags=tags, spark=spark)
# Model A uses the config params (learning_rate 0.01, n_estimators 1000, max_depth 6);
# B must differ, otherwise the test compares a model with itself
basic_model_b.parameters = {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 3}
basic_model_b.model_name = f"{catalog_name}.{schema_name}.marvel_character_model_basic_B"
basic_model_b.load_data()
basic_model_b.prepare_features()
basic_model_b.train()
basic_model_b.log_model()
basic_model_b.register_model()
model_B_uri = f"models:/{basic_model_b.model_name}@latest-model"

# COMMAND ----------
# Define A/B test wrapper
class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # MLflow records artifact paths with os.path.join, so models logged on Windows
        # contain backslashes that break loading on Linux serving containers.
        self.model_a = mlflow.sklearn.load_model(
            context.artifacts["sklearn-pipeline-model-A"].replace("\\", "/")
        )
        self.model_b = mlflow.sklearn.load_model(
            context.artifacts["sklearn-pipeline-model-B"].replace("\\", "/")
        )

    def predict(self, context, model_input):
        # Use PageID (or another unique identifier) for splitting
        page_id = str(model_input["Id"].values[0])
        hashed_id = hashlib.md5(page_id.encode(encoding="UTF-8")).hexdigest()
        if int(hashed_id, 16) % 2:
            predictions = self.model_a.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model A"}
        else:
            predictions = self.model_b.predict(model_input.drop(["Id"], axis=1))
            return {"Prediction": predictions[0], "model": "Model B"}

# COMMAND ----------
# Prepare data
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
X_train = train_set[config.num_features + config.cat_features + ["Id"]]
X_test = test_set[config.num_features + config.cat_features + ["Id"]]

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/marvel-characters-ab-testing")
model_name = f"{catalog_name}.{schema_name}.marvel_character_model_pyfunc_ab_test"
wrapped_model = MarvelModelWrapper()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 1, "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-marvel-character-model-ab",
        artifacts={
            "sklearn-pipeline-model-A": model_A_uri,
            "sklearn-pipeline-model-B": model_B_uri},
        signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-marvel-character-model-ab", name=model_name
)

# COMMAND ----------
# Model serving setup
workspace = w  # reuse the profile-pinned client so the endpoint lands in the same workspace
endpoint_name = "marvel-characters-ab-testing"
entity_version = model_version.version

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=entity_version,
    )
]

workspace.serving_endpoints.create(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=served_entities,
    ),
)

# COMMAND ----------
# Create sample request body
sampled_records = train_set[config.num_features + config.cat_features + ["Id"]].sample(n=1000, replace=True)

import numpy as np
sampled_records = sampled_records.replace({np.nan: None}).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

print(train_set.dtypes)
print(dataframe_records[0])

# COMMAND ----------
# Call the endpoint with one sample record
def call_endpoint(record):
    """Calls the model serving endpoint with a given input record."""
    serving_endpoint = f"{w.config.host}/serving-endpoints/marvel-characters-ab-testing/invocations"

    # Fresh auth header per call: the DBR_TOKEN minted at the top expires after 20 minutes,
    # which training both models and waiting for the endpoint usually exceeds
    response = requests.post(
        serving_endpoint,
        headers=w.config.authenticate(),
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
# COMMAND ----------
