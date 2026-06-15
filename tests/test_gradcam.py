import os
import subprocess
import pytest

def test_gradcam_script_generates_image():
    # Ensure no old output exists
    output_path = "results/gradcam_brain_extraction.png"
    if os.path.exists(output_path):
        os.remove(output_path)
        
    # We pass --dry-run so the script uses random data and a fresh model
    # instead of requiring the real downloaded dataset and trained weights.
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    python_exe = os.path.join(".venv311", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = os.path.join(".venv311", "bin", "python")

    result = subprocess.run(
        [python_exe, "scripts/generate_gradcam.py", "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    assert result.returncode == 0, f"Script failed with output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert os.path.exists(output_path), "Grad-CAM image was not generated"
