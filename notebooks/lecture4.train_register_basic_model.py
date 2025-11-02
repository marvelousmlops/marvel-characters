# Databricks notebook source

import json

import mlflow
from dotenv import load_dotenv
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel
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


config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "main"})

# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config,
                         tags=tags,
                         spark=spark)

# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
basic_model.train()

# COMMAND ----------
basic_model.log_model()

# COMMAND ----------
logged_model = mlflow.get_logged_model(basic_model.model_info.model_id)
model = mlflow.sklearn.load_model(f"models:/{basic_model.model_info.model_id}")

# COMMAND ----------
logged_model_dict = logged_model.to_dictionary()
logged_model_dict["metrics"] = [x.__dict__ for x in logged_model_dict["metrics"]]
with open("../demo_artifacts/logged_model.json", "w") as json_file:
    json.dump(logged_model_dict, json_file, indent=4)
# COMMAND ----------
logged_model.params
# COMMAND ----------
logged_model.metrics

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/marvel-characters-basic"], filter_string="tags.git_sha='abcd12345'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------
run = mlflow.get_run(basic_model.run_id)

# COMMAND ----------
inputs = run.inputs.dataset_inputs
training_input = next((x for x in inputs if len(x.tags) > 0 and x.tags[0].value == 'training'), None)
training_source = mlflow.data.get_source(training_input)
training_source.load()
# COMMAND ----------
testing_input = next((x for x in inputs if len(x.tags) > 0 and x.tags[0].value == 'testing'), None)
testing_source = mlflow.data.get_source(testing_input)
testing_source.load()

# COMMAND ----------
basic_model.register_model()

# COMMAND ----------
# only searching by name is supported
v = mlflow.search_model_versions(
    filter_string=f"name='{basic_model.model_name}'")
print(v[0].__dict__)

# COMMAND ----------
# not supported
v = mlflow.search_model_versions(
    filter_string="tags.git_sha='abcd12345'")

# COMMAND ----------
