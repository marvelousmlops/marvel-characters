from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env

from marvel_characters.config import Tags


def adjust_predictions(predictions: np.ndarray | list[int]) -> dict[str, list[str]]:
    """Adjust predictions to human-readable format."""
    return {"Survival prediction": ["alive" if pred == 1 else "dead" for pred in predictions]}


class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for LightGBM model."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the LightGBM model."""
        # MLflow records artifact paths with os.path.join, so models logged on Windows
        # contain backslashes that break loading on Linux serving containers.
        self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline"].replace("\\", "/"))

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        """Predict the survival of a character."""
        predictions = self.model.predict(model_input)
        return adjust_predictions(predictions)

    def log_register_model(
        self,
        wrapped_model_uri: str,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],
        input_example: pd.DataFrame,
    ) -> None:
        """Log and register the model.

        :param wrapped_model_uri: URI of the wrapped model
        :param pyfunc_model_name: Name of the PyFunc model
        :param experiment_name: Name of the experiment
        :param tags: Tags for the model
        :param code_paths: List of code paths
        :param input_example: Input example for the model
        """
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"code/{whl_name}")
            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            signature = infer_signature(model_input=input_example, model_output={"Survival prediction": ["alive"]})
            model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={"lightgbm-pipeline": wrapped_model_uri},
                signature=signature,
                code_paths=code_paths,
                conda_env=conda_env,
            )
        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=pyfunc_model_name,
            tags=tags.to_dict(),
        )
        latest_version = registered_model.version
        client.set_registered_model_alias(
            name=pyfunc_model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
