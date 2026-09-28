"""Tests for BasicModel.model_improved."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import RestException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# `delta` ships with the Databricks runtime, not with the CI test environment; basic_model only needs its import.
if "delta.tables" not in sys.modules:
    delta_module = types.ModuleType("delta")
    delta_tables_module = types.ModuleType("delta.tables")
    delta_tables_module.DeltaTable = MagicMock()
    delta_module.tables = delta_tables_module
    sys.modules["delta"] = delta_module
    sys.modules["delta.tables"] = delta_tables_module

from marvel_characters.models.basic_model import BasicModel  # noqa: E402


def _model() -> BasicModel:
    """Create a BasicModel without running __init__ (which needs Spark and config)."""
    model = BasicModel.__new__(BasicModel)
    model.model_name = "mlops_stage.marvel_characters.marvel_character_model_basic"
    model.metrics = {"f1_score": 0.7}
    return model


def test_model_improved_when_no_model_registered() -> None:
    """First run in a fresh environment: no registered model means the new one counts as improved."""
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RestException(
        {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Routine or Model does not exist."}
    )
    with patch("marvel_characters.models.basic_model.MlflowClient", return_value=client):
        assert _model().model_improved() is True


def test_model_improved_reraises_other_errors() -> None:
    """Errors other than 'does not exist' (e.g. permissions) must still fail the job."""
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = RestException(
        {"error_code": "PERMISSION_DENIED", "message": "User does not have permission."}
    )
    with (
        patch("marvel_characters.models.basic_model.MlflowClient", return_value=client),
        pytest.raises(RestException),
    ):
        _model().model_improved()
