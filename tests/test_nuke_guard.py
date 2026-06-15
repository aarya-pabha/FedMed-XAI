import subprocess
import os
import pytest

def test_nuke_requires_force():
    # Use the python executable from .venv311
    python_exe = os.path.join(".venv311", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = os.path.join(".venv311", "bin", "python")
        
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run without --force - should fail (after implementation)
    # For now, it will likely succeed because the guard isn't there yet.
    # We use a dummy project ID to avoid hitting real Firestore if it doesn't fail fast.
    env["GCP_PROJECT"] = "test-project"
    
    # We pass a flag that doesn't exist yet to see if it handles it
    result = subprocess.run([python_exe, "scripts/nuke_firestore.py"], capture_output=True, text=True, env=env)
    
    # If the script hasn't been modified yet, this test will fail here (returncode 0)
    assert result.returncode != 0, "Script should fail without --force flag"
    assert "[ERROR]" in result.stdout or "[ERROR]" in result.stderr or "force" in result.stdout.lower()
