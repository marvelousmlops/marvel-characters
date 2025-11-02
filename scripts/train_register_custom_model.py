import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from importlib.metadata import version

from marvel_characters.config import ProjectConfig, Tags
from marvel_characters.models.basic_model import BasicModel
from marvel_characters.models.custom_model import MarvelModelWrapper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
parser.add_argument("--job_run_id", type=str, required=True, help="run id of the run of the databricks job")
parser.add_argument("--branch", type=str, required=True, help="branch of the project")

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config_marvel.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize Marvel custom model
basic_model = BasicModel(config=config, tags=tags, spark=spark)
logger.info("Marvel BasicModel initialized.")

# Load Marvel data
basic_model.load_data()
logger.info("Marvel data loaded.")

# Prepare features
basic_model.prepare_features()

# Train the Marvel model
basic_model.train()
logger.info("Marvel model training completed.")
 
# Train the Marvel model
basic_model.log_model()

# Evaluate Marvel model
model_improved = basic_model.model_improved()
logger.info("Marvel model evaluation completed, model improved: %s", model_improved)

if model_improved:
    # Register the model
    basic_model.register_model()
    marvel_characters_v = version("marvel_characters")

    pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.marvel_character_model_custom"
    code_paths=[f"{root_path}/artifacts/.internal/marvel_characters-{marvel_characters_v}-py3-none-any.whl"]

    wrapper = MarvelModelWrapper()
    latest_version = wrapper.log_register_model(wrapped_model_uri=f"{basic_model.model_info.model_uri}",
                            pyfunc_model_name=pyfunc_model_name,
                            experiment_name=config.experiment_name_custom,
                            input_example=basic_model.X_test[0:1],
                            tags=tags,
                            code_paths=code_paths)

    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
