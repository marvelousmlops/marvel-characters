# Databricks notebook source
# %pip install marvel_characters-0.1.0-py3-none-any.whl
# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import os
from pyspark.dbutils import DBUtils

from marvel_characters.config import ProjectConfig
from marvel_characters.utils import is_databricks

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set") \
                        .withColumn("Id", col("Id").cast("string")) \
                        .toPandas()

# COMMAND ----------
if is_databricks():
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
    os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
else:
    from dotenv import load_dotenv
    load_dotenv()
    # DBR_TOKEN and DBR_HOST should be set in your .env file
    assert os.environ.get("DBR_TOKEN"), "DBR_TOKEN must be set in your environment or .env file."
    assert os.environ.get("DBR_HOST"), "DBR_HOST must be set in your environment or .env file."
    profile = os.environ["PROFILE"]
# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()

# Required columns for inference
required_columns = [
    "Height",
    "Weight",
    "Universe",
    "Identity",
    "Gender",
    "Marital_Status",
    "Teams",
    "Origin",
    "Magic",
    "Mutant"]


# COMMAND ----------

def send_request_https(dataframe_record):
    """
    Sends a request to the model serving endpoint using HTTPS.
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/marvel-characters-model-serving/invocations"
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": dataframe_record},
    )
    return response.status_code, response.text

def send_request_workspace(dataframe_record):
    """
    Sends a request to the model serving endpoint using workspace client.
    """    
    response = workspace.serving_endpoints.query(
        name="marvel-characters-model-serving",
        dataframe_records=dataframe_record
    )
    return response

# COMMAND ----------

# Sample records for testing
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Test the endpoint
for i in range(len(dataframe_records)):
    status_code, response_text = send_request_https(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
    

# COMMAND ----------

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from marvel_characters.config import ProjectConfig
from marvel_characters.monitoring import create_or_refresh_monitoring

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
