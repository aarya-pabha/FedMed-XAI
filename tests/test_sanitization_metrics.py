import torch
import numpy as np
import pytest
from src.federated.client_test import FlowerClient
from unittest.mock import MagicMock

def test_sanitization_metrics_logging(monkeypatch):
    # Setup client with dry_run and a fake run_id
    # We use a dummy data path to avoid loading real data
    client = FlowerClient(center_id=0, dry_run=True, run_id="test_run", data_path="tests")
    
    # Inject NaN into model
    with torch.no_grad():
        found = False
        for name, param in client.model.named_parameters():
            if "weight" in name:
                param[0, 0, 0, 0] = float('nan')
                found = True
                break
        assert found, "Could not find a weight parameter to inject NaN"
    
    # Mock MlflowClient
    mock_mlflow_client_instance = MagicMock()
    mock_mlflow_client_class = MagicMock(return_value=mock_mlflow_client_instance)
    monkeypatch.setattr("src.federated.client_test.MlflowClient", mock_mlflow_client_class)
    
    # Trigger get_parameters
    client.get_parameters(config={})
    
    # Verify metric logged
    mock_mlflow_client_instance.log_metric.assert_called()
    args, kwargs = mock_mlflow_client_instance.log_metric.call_args
    assert args[0] == "test_run"
    assert args[1] == "client_0_sanitization_percentage"
    assert args[2] > 0
