import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add scripts to path
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

try:
    from setup_medplum_rbac import MedplumBootstrapper
except ImportError:
    MedplumBootstrapper = None

class TestMedplumRBAC(unittest.TestCase):
    def test_import(self):
        """Step 1: Verify script exists (will fail first)"""
        self.assertIsNotNone(MedplumBootstrapper, "setup_medplum_rbac.py should be importable")

    @patch('requests.post')
    def test_login(self, mock_post):
        """Step 2: Test login logic"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"accessToken": "fake_token"}
        mock_post.return_value = mock_response

        bootstrapper = MedplumBootstrapper("http://localhost:8103")
        token = bootstrapper.login("admin@example.com", "password")
        
        self.assertEqual(token, "fake_token")
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
