# Databricks notebook source
import json
import os

import mlflow
from dotenv import load_dotenv


# Set up Databricks or local MLflow tracking
def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------
mlflow.get_tracking_uri()

# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

mlflow.get_tracking_uri()
# COMMAND ----------
experiment = mlflow.set_experiment(experiment_name="/Shared/marvel-demo")
mlflow.set_experiment_tags({"repository_name": "marvelousmlops/marvel-characters"})

print(experiment)
# COMMAND ----------
# dump class attributes in a json file for visualization
os.makedirs("../demo_artifacts", exist_ok=True)
with open("../demo_artifacts/mlflow_experiment.json", "w") as json_file:
    json.dump(experiment.__dict__, json_file, indent=4)

# COMMAND ----------
# get experiment by id
mlflow.get_experiment(experiment.experiment_id)
# COMMAND ----------
# search for experiment
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='marvelousmlops/marvel-characters'"
)
print(experiments)

# COMMAND ----------
# start a run
mlflow.start_run()

# COMMAND ----------
# get active run
print(mlflow.active_run().__dict__)

# COMMAND ----------
mlflow.end_run()
print(mlflow.active_run() is None)

# COMMAND ----------
# start a run
with mlflow.start_run(
    run_name="marvel-demo-run",
    tags={"git_sha": "1234567890abcd"},
    description="marvel character prediction demo run",
) as run:
    run_id = run.info.run_id
    mlflow.log_params({"type": "marvel_demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})
# COMMAND ----------
print(mlflow.active_run() is None)

# COMMAND ----------
run_info = mlflow.get_run(run_id=run_id).to_dictionary()
print(run_info)

# COMMAND ----------
with open("../demo_artifacts/run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
print(run_info["data"]["metrics"])

# COMMAND ----------
print(run_info["data"]["params"])

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/marvel-demo"],
    filter_string="tags.git_sha='1234567890abcd'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
mlflow.start_run(run_id=run_id)

# COMMAND ----------
# this will fail: not allowed to overwrite value
mlflow.log_param("type", "marvel_demo2")
# COMMAND ----------
mlflow.log_param(key="purpose", value="get_certified")
mlflow.end_run()

# COMMAND ----------
# start another run and log other things
mlflow.start_run(run_name="marvel-demo-run-extra",
                 tags={"git_sha": "1234567890abcd"},
                       description="marvel demo run with extra artifacts",)
mlflow.log_metric(key="metric3", value=3.0)
# dynamically log metric (trainings epochs)
for i in range(0,3):
    mlflow.log_metric(key="metric1", value=3.0+i/2, step=i)
mlflow.log_artifact("../demo_artifacts/mlflow_meme.jpeg")
mlflow.log_text("hello, MLflow!", "hello.txt")
mlflow.log_dict({"k": "v"}, "dict_example.json")
mlflow.log_artifacts("../demo_artifacts", artifact_path="demo_artifacts")

# COMMAND ----------
# log figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0, 1], [2, 3])

mlflow.log_figure(fig, "figure.png")

# log image dynamically
# COMMAND ----------
import numpy as np

for i in range(0,3):
    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    mlflow.log_image(image, key="demo_image", step=i)

mlflow.end_run()

# COMMAND ----------
# other ways
from time import time

time_hour_ago = int(time() - 3600) * 1000

runs = mlflow.search_runs(
    search_all_experiments=True, #or experiment_ids=[], or experiment_names=[]
    order_by=["start_time DESC"],
    filter_string="status='FINISHED' AND "
                  f"start_time>{time_hour_ago} AND "
                  "run_name LIKE '%marvel-demo-run%' AND "
                  "metrics.metric3>0 AND "
                  "tags.mlflow.source.type!='JOB'"
)
# COMMAND ----------
runs

# COMMAND ----------
# load objects
artifact_uri = runs.artifact_uri[0]
mlflow.artifacts.load_dict(f"{artifact_uri}/dict_example.json")
# nested runs

# COMMAND ----------
mlflow.artifacts.load_image(f"{artifact_uri}/figure.png")
# COMMAND ----------
# download artifacts
mlflow.artifacts.download_artifacts(
    artifact_uri=f"{artifact_uri}/demo_artifacts",
    dst_path="../downloaded_artifacts")

# COMMAND ----------
# nested runs: useful for hyperparameter tuning
with mlflow.start_run(run_name="marvel_top_level_run") as run:
    for i in range(1,5):
        with mlflow.start_run(run_name=f"marvel_subrun_{str(i)}", nested=True) as subrun:
            mlflow.log_metrics({"m1": 5.1+i,
                                "m2": 2*i,
                                "m3": 3+1.5*i})

# COMMAND ----------
