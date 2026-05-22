# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the Marvel characters dataset
spark = SparkSession.builder.getOrCreate()

filepath = "../data/marvel_characters_dataset.csv"

# Load the data
df = pd.read_csv(filepath)

# Display basic info about the dataset
logger.info(f"Dataset shape: {df.shape}")
logger.info(f"Columns: {list(df.columns)}")
logger.info(f"Target column '{config.target}' distribution:")
logger.info(df[config.target].value_counts())

# COMMAND ----------
# Load the Marvel characters dataset

data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

logger.info(f"Data preprocessing completed.")

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
# COMMAND ---------- 