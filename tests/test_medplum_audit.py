# tests/test_medplum_audit.py
import unittest
from unittest.mock import patch, MagicMock
from src.utils.medplum_audit import log_metrics_to_medplum

class TestMedplumAudit(unittest.TestCase):
    @patch('src.utils.medplum_audit.requests.post')
    def test_log_metrics_success(self, mock_post):
        # 1. Mock Token Response
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.json.return_value = {"access_token": "fake_token"}
        
        # 2. Mock Observation Response
        mock_obs_resp = MagicMock()
        mock_obs_resp.status_code = 201
        
        # Configure mock_post to return different responses for sequential calls
        mock_post.side_effect = [mock_token_resp, mock_obs_resp]
        
        result = log_metrics_to_medplum(
            client_id="test_client",
            client_secret="test_secret",
            project_id="test_project",
            round_num=1,
            loss=0.5,
            dice=0.85,
            epsilon=2.5
        )
        self.assertTrue(result)
        self.assertEqual(mock_post.call_count, 2) # 1 for token, 1 for observation

    @patch('src.utils.medplum_audit.requests.post')
    def test_log_metrics_auth_failure(self, mock_post):
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 401
        mock_token_resp.text = "Unauthorized"
        mock_post.return_value = mock_token_resp
        
        result = log_metrics_to_medplum(
            client_id="test_client",
            client_secret="test_secret",
            project_id="test_project",
            round_num=1,
            loss=0.5,
            dice=0.85,
            epsilon=2.5
        )
        self.assertFalse(result)
        self.assertEqual(mock_post.call_count, 1)

if __name__ == '__main__':
    unittest.main()
