"""Data preprocessing module for Marvel characters."""

import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from marvel_characters.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing Marvel character DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the Marvel character DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        target = self.config.target

        self.df.rename(columns={"Height (m)": "Height"}, inplace=True)
        self.df.rename(columns={"Weight (kg)": "Weight"}, inplace=True)

        # Universe
        self.df["Universe"] = self.df["Universe"].fillna("Unknown")
        counts = self.df["Universe"].value_counts()
        small_universes = counts[counts < 50].index
        self.df["Universe"] = self.df["Universe"].replace(small_universes, "Other")

        # Teams
        # Explicit int64: with numpy 1.x, "int" is int32 on Windows but int64 on Linux, which
        # makes locally written and job-written Delta tables disagree (INT vs BIGINT).
        self.df["Teams"] = self.df["Teams"].notna().astype("int64")

        # Origin
        self.df["Origin"] = self.df["Origin"].fillna("Unknown")

        # Identity
        self.df["Identity"] = self.df["Identity"].fillna("Unknown")
        self.df = self.df[self.df["Identity"].isin(["Public", "Secret", "Unknown"])]

        # Gender
        self.df["Gender"] = self.df["Gender"].fillna("Unknown")
        self.df["Gender"] = self.df["Gender"].where(self.df["Gender"].isin(["Male", "Female"]), other="Other")

        # Marital status
        self.df.rename(columns={"Marital Status": "Marital_Status"}, inplace=True)
        self.df["Marital_Status"] = self.df["Marital_Status"].fillna("Unknown")
        self.df["Marital_Status"] = self.df["Marital_Status"].replace("Widow", "Widowed")
        self.df = self.df[self.df["Marital_Status"].isin(["Single", "Married", "Widowed", "Engaged", "Unknown"])]

        # Magic
        self.df["Magic"] = self.df["Origin"].str.lower().apply(lambda x: int("magic" in x))

        # Mutant
        self.df["Mutant"] = self.df["Origin"].str.lower().apply(lambda x: int("mutate" in x or "mutant" in x))

        # Normalize origin
        def normalize_origin(x: str) -> str:
            x_lower = str(x).lower()
            if "human" in x_lower:
                return "Human"
            elif "mutate" in x_lower or "mutant" in x_lower:
                return "Mutant"
            elif "asgardian" in x_lower:
                return "Asgardian"
            elif "alien" in x_lower:
                return "Alien"
            elif "symbiote" in x_lower:
                return "Symbiote"
            elif "robot" in x_lower:
                return "Robot"
            elif "cosmic being" in x_lower:
                return "Cosmic Being"
            else:
                return "Other"

        self.df["Origin"] = self.df["Origin"].apply(normalize_origin)

        self.df = self.df[self.df["Alive"].isin(["Alive", "Dead"])]
        self.df["Alive"] = (self.df["Alive"] == "Alive").astype("int64")

        self.df = self.df[num_features + cat_features + [target] + ["PageID"]]

        for col in cat_features:
            self.df[col] = self.df[col].astype("category")
        # Rename PageID to Id for consistency
        if "PageID" in self.df.columns:
            self.df = self.df.rename(columns={"PageID": "Id"})
            self.df["Id"] = self.df["Id"].astype("str")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # overwriteSchema: a full overwrite should also replace column types left by earlier writes
        train_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic Marvel character data matching input DataFrame distributions with optional drift.

    Creates artificial dataset replicating statistical patterns from source columns including numeric,
    categorical, and datetime types. Supports intentional data drift for specific features when enabled.

    :param df: Source DataFrame containing original data distributions
    :param drift: Flag to activate synthetic data drift injection
    :param num_rows: Number of synthetic records to generate
    :return: DataFrame containing generated synthetic data
    """
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Id":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if column in {"Height", "Weight"}:  # Handle physical attributes
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)
                # Ensure positive values for physical attributes
                synthetic_data[column] = np.maximum(0.1, synthetic_data[column])
            else:
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to appropriate types
    float_columns = {"Height", "Weight"}
    for col in float_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.float64)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["Id"] = [str(timestamp_base + i) for i in range(num_rows)]

    if drift:
        # Skew the physical attributes to introduce drift
        drift_features = ["Height", "Weight"]
        for feature in drift_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

        # Introduce bias in categorical features
        if "Gender" in synthetic_data.columns:
            synthetic_data["Gender"] = np.random.choice(["Male", "Female"], num_rows, p=[0.7, 0.3])

    return synthetic_data


def generate_test_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 100) -> pd.DataFrame:
    """Generate test data matching input DataFrame distributions with optional drift."""
    return generate_synthetic_data(df, drift, num_rows)
