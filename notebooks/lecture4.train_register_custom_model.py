# Databricks notebook source
# run below command in a separate cell if you are using databricks
# %pip install mlchapter_project01-0.1.0-py3-none-any.whl
# %restart_python

import mlflow
from pyspark.sql import SparkSession

from mlchapter_project01.config import ProjectConfig, Tags
from mlchapter_project01.models.custom_model import mlchapterModelWrapper
from importlib.metadata import version
from dotenv import load_dotenv
from mlflow import MlflowClient
import os

# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config_mlchapter.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "main"})
mlchapter_project01_v = version("mlchapter_project01")

code_paths=[f"../dist/mlchapter_project01-{mlchapter_project01_v}-py3-none-any.whl"]

# COMMAND ----------
client = MlflowClient()
wrapped_model_version = client.get_model_version_by_alias(
    name=f"{config.catalog_name}.{config.schema_name}.marvel_character_model_basic",
    alias="latest-model")
# Initialize model with the config path

# COMMAND ----------
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
X_test = test_set[config.num_features + config.cat_features]

# COMMAND ----------
pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.mlchapter_project01_model_custom"
wrapper = mlchapterModelWrapper()
wrapper.log_register_model(wrapped_model_uri=f"models:/{wrapped_model_version.model_id}",
                           pyfunc_model_name=pyfunc_model_name,
                           experiment_name=config.experiment_name_custom,
                           input_example=X_test[0:1],
                           tags=tags,
                           code_paths=code_paths)
# use this in databricks code_paths=["../src/mlchapter_project01"]
# COMMAND ----------
# unwrap and predict
# use below in databricks
# mlflow.log_artifact(f"mlchapter_project01-{mlchapter_project01_v}-py3-none-any.whl", "dist")
loaded_pufunc_model = mlflow.pyfunc.load_model(f"models:/{pyfunc_model_name}@latest-model")

unwraped_model = loaded_pufunc_model.unwrap_python_model()

# COMMAND ----------
unwraped_model.predict(context=None, model_input=X_test[0:1])
# COMMAND ----------
# another predict function with uri

loaded_pufunc_model.predict(X_test[0:1])
# COMMAND ----------
