<h1 align="center">
mlschool MLOps Free End-to-end MLOps with Databricks Course

## Set up your environment
In this course, we use Databricks serverless [version 3](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/three)

In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv sync --extra dev
```



# Data
Using the [**mlchapter project01 Dataset**](https://www.kaggle.com/datasets/mohitbansal31s/mlchapter-project01?resource=download) from Kaggle.

This dataset contains detailed information about mlchapter project01 (e.g., name, powers, physical attributes, alignment, etc.).
It is used to build classification and feature engineering models for various MLOps tasks, such as predicting character attributes or status.

# Scripts

- `01.process_data.py`: Loads and preprocesses the Marvel dataset, splits into train/test, and saves to the catalog.
- `02.train_register_fe_model.py`: Performs feature engineering and trains the mlchapter project01 model.
- `03.deploy_model.py`: Deploys the trained Marvel model to a Databricks model serving endpoint.
- `04.post_commit_status.py`: Posts status updates for Marvel integration tests to GitHub.
- `05.refresh_monitor.py`: Refreshes monitoring tables and dashboards for Marvel model serving.

# some fixex to run the code
updated code in notebooks\lecture6.deploy_model_serving_endpoint.py to use the basic model instead of custom as custom was giving some error.
done changes in data_processor.py to fix error in job 
To run the DAB job .. command - databricks bundle run deployment instead of databricks bundle run
change in Lecture 6 file change to 	MAGIC %pip install mlchapter_project01-0.1.0-py3-none-any.whl	
