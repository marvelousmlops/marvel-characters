import argparse
import yaml
from loguru import logger
from pyspark.sql import SparkSession
import pandas as pd

from mlchapter_project01.config import ProjectConfig
from mlchapter_project01.data_processor import DataProcessor

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


args = parser.parse_args()
config_path = f"{args.root_path}/files/project_config_mlchapter.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the mlchapter project01 dataset
spark = SparkSession.builder.getOrCreate()

# Example: Adjust the path and loading logic as per your mlchapter dataset location
filepath = f"{args.root_path}/files/data/mlchapter_project01_dataset.csv"

# Load the data
df = pd.read_csv(filepath)

# If you have mlchapter-specific synthetic/test data generation, use them here.
# Otherwise, just use the loaded mlchapter dataset as is.
logger.info("mlchapter data loaded for processing.")

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
