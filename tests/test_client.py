import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch
from src.federated.client import FlowerClient

@pytest.fixture
def mock_dependencies():
    with patch("src.federated.client.load_partition") as mock_load, \
         patch("src.federated.client.get_model") as mock_get_model, \
         patch("src.federated.client.wrap_model_for_dp") as mock_wrap, \
         patch("src.federated.client.mlflow") as mock_mlflow:
        
        # Mock model
        mock_model = MagicMock(spec=torch.nn.Module)
        param = torch.nn.Parameter(torch.randn(1, 1))
        mock_model.parameters.return_value = [param]
        mock_model.state_dict.return_value = {"layer1.weight": param}
        mock_model.to.return_value = mock_model
        # Mock model call for evaluate
        mock_model.return_value = torch.randn(1, 1, 8, 8, 8)
        
        mock_get_model.return_value = mock_model
        
        # Mock DataLoaders
        mock_loader = MagicMock()
        mock_loader.__len__.return_value = 1
        mock_loader.dataset = [0] * 10
        # Mock iteration
        mock_loader.__iter__.return_value = iter([
            (torch.randn(1, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8))
        ])
        mock_load.return_value = mock_loader
        
        # Mock Opacus wrap
        mock_private_model = MagicMock()
        mock_private_model.parameters.return_value = [param]
        mock_private_model.return_value = torch.randn(1, 1, 8, 8, 8)
        
        mock_private_optimizer = MagicMock()
        mock_private_optimizer.privacy_engine.get_epsilon.return_value = 0.1
        
        mock_private_loader = [
            (torch.randn(1, 1, 8, 8, 8), torch.randn(1, 1, 8, 8, 8))
        ]
        mock_wrap.return_value = (mock_private_model, mock_private_optimizer, mock_private_loader, None)
        
        yield {
            "load": mock_load,
            "get_model": mock_get_model,
            "wrap": mock_wrap,
            "mlflow": mock_mlflow,
            "model": mock_model,
            "private_optimizer": mock_private_optimizer,
            "loader": mock_loader
        }

def test_client_get_parameters(mock_dependencies):
    client = FlowerClient(center_id=0)
    params = client.get_parameters(config={})
    assert isinstance(params, list)
    assert len(params) > 0
    assert isinstance(params[0], np.ndarray)

def test_client_set_parameters(mock_dependencies):
    client = FlowerClient(center_id=0)
    params = [np.array([[1.0]])]
    client.set_parameters(params)
    assert mock_dependencies["model"].load_state_dict.called

def test_client_fit(mock_dependencies):
    # Use real model for 'fit' to ensure grad_fn exists
    from src.models.unet_pp import get_model
    real_model = get_model()
    
    # We want to keep the mocks for other things, but use a real model for the fit loop
    with patch("src.federated.client.get_model", return_value=real_model), \
         patch("src.federated.client.wrap_model_for_dp") as mock_wrap, \
         patch("src.federated.client.FlowerClient._upload_shard"), \
         patch("src.federated.client.FlowerClient._sync_firestore"), \
         patch("src.federated.client.FlowerClient._log_medplum"), \
         patch("src.federated.client.FlowerClient._pull_global_shard") as mock_pull:
        
        mock_pull.return_value = [val.detach().cpu().numpy() for _, val in real_model.state_dict().items()]
        
        client = FlowerClient(center_id=0)
        
        # Create real dummy data
        dummy_input = torch.randn(1, 1, 32, 32, 32)
        dummy_target = torch.randn(1, 2, 32, 32, 32)
        loader = DataLoader(TensorDataset(dummy_input, dummy_target), batch_size=1)        
        # We need batch structure to match: batch["image"], batch["label"]
        class BatchLoader:
            def __init__(self, loader): self.loader = loader
            def __iter__(self):
                for img, label in self.loader:
                    yield (img, label)
            def __len__(self): return len(self.loader)
            @property
            def dataset(self): return [0]

        # Setup mock_wrap to return real objects but controlled
        optimizer = torch.optim.Adam(real_model.parameters(), lr=1e-3)
        private_optimizer = MagicMock()
        private_optimizer.privacy_engine.get_epsilon.return_value = 0.5
        # Ensure private_optimizer.zero_grad and step work or are mocked
        
        mock_wrap.return_value = (real_model, private_optimizer, BatchLoader(loader), None)

        # Set parameters (real shape)
        params = [val.detach().cpu().numpy() for _, val in real_model.state_dict().items()]

        new_params, num_examples, metrics = client.fit(params, config={"lr": 0.001})

        assert num_examples == 1
        assert "loss" in metrics
        assert "epsilon" in metrics

def test_client_evaluate(mock_dependencies):
    client = FlowerClient(center_id=0)
    params = [np.array([[1.0]])]
    
    # Mock evaluate specific data
    mock_dependencies["loader"].dataset = [0] * 5
    
    loss, num_examples, metrics = client.evaluate(params, config={})
    
    assert isinstance(loss, float)
    assert num_examples == 5
    assert "dice" in metrics
