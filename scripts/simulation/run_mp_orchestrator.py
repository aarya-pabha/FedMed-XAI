import argparse
import os
import time
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore")

import mlflow
import subprocess
import sys
import atexit
import psutil
from google.cloud import storage

PID_FILE = "orchestrator.pid"

def setup_pid_lock():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            try:
                old_pid = int(f.read().strip())
                if psutil.pid_exists(old_pid):
                    print(f"[ERROR] Another orchestrator (PID {old_pid}) is already running.")
                    sys.exit(1)
                else:
                    print("[WARN] Removing stale PID file.")
                    os.remove(PID_FILE)
            except (ValueError, psutil.NoSuchProcess):
                print("[WARN] Removing stale PID file.")
                os.remove(PID_FILE)
            
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    
    atexit.register(lambda: os.remove(PID_FILE) if os.path.exists(PID_FILE) else None)

def get_args():
    parser = argparse.ArgumentParser(description="Phase 5: Multi-Process Orchestrator")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of federated rounds to run")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--logical-batch-size", type=int, default=4, help="Logical batch size")
    return parser.parse_args()

def run_hospital(center_id, round_num, logical_batch_size, no_mlflow, run_id=None, eval_only=False):
    """Launches a separate process for a single hospital to ensure clean GPU context."""
    cmd = [
        sys.executable, "scripts/train_single_hospital.py",
        "--cid", str(center_id),
        "--round", str(round_num),
        "--logical-batch-size", str(logical_batch_size)
    ]
    if no_mlflow:
        cmd.append("--no-mlflow")
    elif run_id:
        cmd.extend(["--run-id", run_id])
    
    if eval_only:
        cmd.append("--eval-only")
        
    mode = "EVALUATION" if eval_only else "TRAINING"
    print(f"\n--- Launching Hospital {center_id} Session ({mode}) ---")
    # Using subprocess.run to block until the hospital session finishes
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        print(f"[ERROR] Hospital {center_id} {mode} failed with exit code {result.returncode}")
        return False
    return True

def main():
    setup_pid_lock()
    args = get_args()
    project_id = os.getenv("GCP_PROJECT", "healthcare-fl-diagnostics")
    
    active_run_id = None
    if not args.no_mlflow:
        tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Distributed_Execution_Test")
        run = mlflow.start_run(run_name=f"MultiProcess_Stable_Run_{int(time.time())}")
        mlflow.log_params(vars(args))
        active_run_id = run.info.run_id
        print(f"[INFO] Parent MLflow Run ID: {active_run_id}")
        
    try:
        for round_num in range(1, args.num_rounds + 1):
            print(f"\n[INFO] === STARTING ROUND {round_num} ===")
            
            for center_id in range(3):
                success = run_hospital(center_id, round_num, args.logical_batch_size, args.no_mlflow, active_run_id)
                if not success:
                    print(f"[WARN] Hospital {center_id} failed. Proceeding with others...")
                time.sleep(10) # Cooldown and GCP buffer
            
            print(f"\n[INFO] Round {round_num} complete. Cloud Aggregator should be triggering...")
            # Brief wait for aggregator
            time.sleep(30)
            
        print(f"\n[INFO] --- STARTING FINAL GLOBAL EVALUATION (Round {args.num_rounds + 1}) ---")
        for center_id in range(3):
            run_hospital(center_id, args.num_rounds + 1, args.logical_batch_size, args.no_mlflow, active_run_id, eval_only=True)
            time.sleep(5)

        print("\n[INFO] All rounds and final evaluation finished.")

    except Exception as e:
        print(f"\n[ERROR] Global Orchestrator Failed: {e}")
        raise
    finally:
        if not args.no_mlflow:
            mlflow.end_run()

if __name__ == "__main__":
    main()
