import os
import subprocess
import time
import pytest
import signal

def test_pid_lock_collision():
    # Ensure no stale lock exists
    if os.path.exists("orchestrator.pid"):
        os.remove("orchestrator.pid")
        
    # Start first orchestrator. We use --num-rounds 1 and --no-mlflow to keep it light.
    # We also pass a dummy GCP_PROJECT to ensure it doesn't try to connect to real infra if not needed.
    env = os.environ.copy()
    env["GCP_PROJECT"] = "test-project"
    
    # Use the python executable from .venv311
    python_exe = os.path.join(".venv311", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        # Fallback for Linux/Mac if needed, though we are on win32
        python_exe = os.path.join(".venv311", "bin", "python")

    # Set PYTHONPATH to '.' so it can find 'src'
    env["PYTHONPATH"] = "."
    env["PYTHONIOENCODING"] = "utf-8"
    
    # We use a subprocess that we can terminate.
    proc1 = subprocess.Popen(
        [python_exe, "tests/repro_pid_lock.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    # Wait for PID file to appear
    timeout = 10
    while not os.path.exists("orchestrator.pid") and timeout > 0:
        time.sleep(1)
        timeout -= 1
    
    try:
        assert os.path.exists("orchestrator.pid"), "PID file should have been created by proc1"
        
        # Start second orchestrator - this should fail immediately
        proc2 = subprocess.run(
            [python_exe, "tests/repro_pid_lock.py"],
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        assert proc2.returncode != 0, "Second orchestrator should have failed with non-zero exit code"
        assert "[ERROR] Another orchestrator" in proc2.stdout or "[ERROR] Another orchestrator" in proc2.stderr
        
    finally:
        # Cleanup
        proc1.terminate()
        try:
            proc1.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc1.kill()
            
        if os.path.exists("orchestrator.pid"):
            os.remove("orchestrator.pid")
