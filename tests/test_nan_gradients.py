import unittest
import torch
import numpy as np
from src.federated.client_test import FlowerClient
from unittest.mock import patch, MagicMock

class TestNaNGradients(unittest.TestCase):
    @patch("src.federated.client_test.FlowerClient._upload_shard")
    @patch("src.federated.client_test.FlowerClient._sync_firestore")
    @patch("src.federated.client_test.FlowerClient._log_medplum")
    def test_nan_gradients_do_not_corrupt_weights(self, mock_log, mock_sync, mock_upload):
        # Initialize client in dry-run mode for speed
        client = FlowerClient(center_id=0, dry_run=True, batch_size=1)
        
        # Store initial weights to compare later
        initial_params = [p.clone() for p in client.model.parameters()]
        
        # Create a custom autograd function that returns a normal loss but NaN gradients
        class PoisonGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, i):
                return i
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * float('nan')

        original_loss = client.loss_functions
        def poison_loss(outputs, labels):
            loss = original_loss(outputs, labels)
            return PoisonGradient.apply(loss)
            
        client.loss_functions = poison_loss
        
        # Run a single fit loop
        # If the safeguard is missing, the NaN loss will cause optimizer.step() to corrupt weights
        client.fit(client.get_parameters(config={}), config={"round": 1, "lr": 1e-3})
        
        # Restore
        client.loss_functions = original_loss
            
        # Check if any weights became NaN
        has_nan_weights = False
        for p in client.model.parameters():
            if torch.isnan(p).any():
                has_nan_weights = True
                break
                
        # The test should pass if weights are NOT NaN (meaning the safeguard worked)
        self.assertFalse(has_nan_weights, "Model weights were corrupted by NaN gradients!")

    @patch("src.federated.client_test.FlowerClient._upload_shard")
    @patch("src.federated.client_test.FlowerClient._sync_firestore")
    @patch("src.federated.client_test.FlowerClient._log_medplum")
    def test_avg_loss_skips_nan_batches_correctly(self, mock_log, mock_sync, mock_upload):
        client = FlowerClient(center_id=0, dry_run=True, batch_size=1)
        
        # We mock loss function to return NaN for first batch, and 0.5 for second
        class MockLoss:
            def __init__(self):
                self.calls = 0
            def __call__(self, outputs, labels):
                self.calls += 1
                if self.calls == 1:
                    return torch.tensor(float('nan'), requires_grad=True)
                return torch.tensor(0.5, requires_grad=True)

        original_loss = client.loss_functions
        client.loss_functions = MockLoss()
        
        # Mock dataloader to have 2 batches using a real DataLoader
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = [(torch.randn(1,32,32,32), torch.randn(1,32,32,32)) for _ in range(2)]
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        client.train_loader = torch.utils.data.DataLoader(SimpleDataset(), batch_size=1)
        
        # We need accumulation_steps to be 1 for test
        client.accumulation_steps = 1

        _, _, metrics = client.fit(client.get_parameters(config={}), config={"round": 1, "lr": 1e-3})
        
        client.loss_functions = original_loss
        
        # If arithmetic is correct, total_loss=0.5, successful_batches=1, avg_loss=0.5
        # If bug exists, total_loss=0.5, len=2, avg_loss=0.25
        self.assertEqual(metrics["loss"], 0.5, "avg_loss arithmetic bug: divided by total loader length instead of successful batches")

if __name__ == "__main__":
    unittest.main()
