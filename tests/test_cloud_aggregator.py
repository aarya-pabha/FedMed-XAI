import unittest
from unittest.mock import patch, MagicMock
import io
import torch

# Import the module to be tested
import src.cloud.aggregator.main as aggregator_main

class TestAggregator(unittest.TestCase):
    @patch('src.cloud.aggregator.main.storage.Client')
    @patch('src.cloud.aggregator.main.firestore.Client')
    @patch('src.cloud.aggregator.main.st_load')
    @patch('src.cloud.aggregator.main.st_save')
    def test_aggregate_shards_success(self, mock_st_save, mock_st_load, mock_firestore_client, mock_storage_client):
        # 1. Setup Firestore Mock
        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        
        mock_transaction = MagicMock()
        mock_db.transaction.return_value = mock_transaction
        
        mock_client1 = MagicMock(); mock_client1.id = "client_1"
        mock_client2 = MagicMock(); mock_client2.id = "client_2"
        mock_client3 = MagicMock(); mock_client3.id = "client_3"
        mock_uploaded_docs = [mock_client1, mock_client2, mock_client3]

        # Patch the transactional decorator before it's used
        with patch('google.cloud.firestore.transactional', lambda x: x):
            mock_shard_ref = MagicMock()
            mock_db.collection().document().collection().document.return_value = mock_shard_ref
            
            # Mock shard_ref.get() to return snapshot with no status
            mock_snapshot = MagicMock()
            mock_snapshot.exists = True
            mock_snapshot.get.return_value = None
            mock_shard_ref.get.return_value = mock_snapshot
            
            # Mock shard_ref.collection('uploads').get()
            mock_shard_ref.collection().get.return_value = mock_uploaded_docs

            # 2. Setup Storage Mock
            mock_s_client = MagicMock()
            mock_storage_client.return_value = mock_s_client
            mock_bucket_in = MagicMock()
            mock_bucket_out = MagicMock()
            mock_s_client.bucket.side_effect = lambda name: mock_bucket_in if "in" in name else mock_bucket_out
            
            mock_blob = MagicMock()
            mock_blob.exists.return_value = True
            mock_bucket_in.blob.return_value = mock_blob

            # 3. Setup Safetensors Mock (Dictionaries of tensors)
            mock_st_load.side_effect = [
                {"w": torch.tensor([1.0])}, 
                {"w": torch.tensor([2.0])}, 
                {"w": torch.tensor([3.0])}
            ]
            mock_st_save.return_value = b"fake_safetensors_bytes"

            # 4. Trigger Event
            cloud_event = MagicMock()
            cloud_event.data = {
                "value": {
                    "name": "projects/p/databases/d/documents/rounds/1/shards/0/uploads/client_3"
                }
            }

            # 5. Execute
            result = aggregator_main.aggregate_shards(cloud_event)

            # 6. Assert
            self.assertTrue(result)
            self.assertEqual(mock_st_save.call_count, 1)

    @patch('src.cloud.aggregator.main.firestore.Client')
    def test_aggregate_shards_waiting(self, mock_firestore_client):
        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        
        # Mock only 2 clients
        mock_uploaded_docs = [MagicMock(), MagicMock()]
        
        with patch('google.cloud.firestore.transactional', lambda x: x):
            mock_shard_ref = MagicMock()
            mock_db.collection().document().collection().document.return_value = mock_shard_ref
            mock_snapshot = MagicMock()
            mock_snapshot.exists = True
            mock_snapshot.get.return_value = None
            mock_shard_ref.get.return_value = mock_snapshot
            mock_shard_ref.collection().get.return_value = mock_uploaded_docs

            cloud_event = MagicMock()
            cloud_event.data = {
                "value": {
                    "name": "projects/p/databases/d/documents/rounds/1/shards/0/uploads/client_2"
                }
            }
            
            result = aggregator_main.aggregate_shards(cloud_event)
            self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
