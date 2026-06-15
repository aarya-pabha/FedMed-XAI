import os
import subprocess
import pytest

def test_clinical_report_generation(monkeypatch):
    output_img = "results/privacy_vs_accuracy.png"
    output_md = "results/Phase_6_Clinical_Report.md"
    
    for f in [output_img, output_md]:
        if os.path.exists(f):
            os.remove(f)
            
    # Use dummy run ID so it doesn't query real MLflow unless specified
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    python_exe = os.path.join(".venv311", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = os.path.join(".venv311", "bin", "python")

    # Run with --mock flag to avoid network calls during unit testing
    result = subprocess.run(
        [python_exe, "scripts/generate_clinical_report.py", "--mock"],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    assert result.returncode == 0, f"Script failed with output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert os.path.exists(output_img), "Plot image was not generated"
    assert os.path.exists(output_md), "Markdown report was not generated"
