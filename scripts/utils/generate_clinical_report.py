import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

def generate_report(run_id=None, mock=False):
    os.makedirs("results", exist_ok=True)
    
    rounds = list(range(1, 6))
    dice_scores = []
    epsilons = []
    
    if mock:
        print("[INFO] Using mock data for unit tests...")
        dice_scores = [0.1, 0.4, 0.6, 0.7, 0.75]
        epsilons = [0.5, 1.2, 2.1, 3.0, 4.1]
    else:
        if not run_id:
            print("[ERROR] run_id is required unless using --mock")
            return
            
        print(f"[INFO] Fetching data from MLflow Run: {run_id}")
        client = MlflowClient(tracking_uri="https://mlflow-server-789083630799.us-central1.run.app")
        
        for r in rounds:
            round_dices = []
            round_eps = []
            for c in range(3):
                try:
                    # Support new Dual-Dice format (prefers global_dice) or legacy client_c_dice
                    dice_key = f"client_{c}_global_dice"
                    dice_hist = client.get_metric_history(run_id, dice_key)
                    
                    if not dice_hist:
                        dice_key = f"client_{c}_dice"
                        dice_hist = client.get_metric_history(run_id, dice_key)

                    eps_hist = client.get_metric_history(run_id, f"client_{c}_cumulative_epsilon")
                    
                    if len(dice_hist) >= r:
                        round_dices.append(dice_hist[r-1].value)
                    if len(eps_hist) >= r:
                        round_eps.append(eps_hist[r-1].value)
                except Exception as e:
                    print(f"[WARN] Could not fetch data for client {c} round {r}: {e}")
            
            if round_dices:
                dice_scores.append(np.mean(round_dices))
            else:
                dice_scores.append(0.0)
                
            if round_eps:
                epsilons.append(max(round_eps)) # Take worst-case (max) privacy budget
            else:
                epsilons.append(0.0)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Federated Round')
    ax1.set_ylabel('Global Dice Score (Accuracy)', color=color)
    ax1.plot(rounds, dice_scores, marker='o', color=color, linewidth=2, label="Dice")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Epsilon (Privacy Budget)', color=color)  
    ax2.plot(rounds, epsilons, marker='s', color=color, linewidth=2, linestyle='--', label="Epsilon")
    ax2.tick_params(axis='y', labelcolor=color)
    # HIPAA limit is 10.0
    ax2.axhline(y=10.0, color='r', linestyle=':', label="HIPAA Legal Limit (eps=10)")
    ax2.set_ylim(0, 12.0)
    
    fig.tight_layout()
    plt.title("Federated Learning: Diagnostic Accuracy vs. Privacy Budget")
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    img_path = "results/privacy_vs_accuracy.png"
    plt.savefig(img_path, dpi=300)
    print(f"[SUCCESS] Saved plot to {img_path}")
    
    # Generate Markdown Report
    md_path = "results/Phase_6_Clinical_Report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6 Clinical Audit Report\n\n")
        f.write("## 1. Accuracy vs Privacy Trade-off\n")
        f.write("The following chart demonstrates the cumulative privacy expenditure against model convergence.\n\n")
        f.write("![Privacy vs Accuracy](privacy_vs_accuracy.png)\n\n")
        f.write("## 2. Final Metrics\n")
        f.write(f"- **Final Global Dice Score**: {dice_scores[-1]:.4f}\n")
        f.write(f"- **Final Cumulative Epsilon**: {epsilons[-1]:.4f}\n")
        f.write("- **Status**: " + ("[COMPLIANT]" if epsilons[-1] < 10.0 else "[NON-COMPLIANT]") + "\n")
        
    print(f"[SUCCESS] Saved report to {md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, help="MLflow Run ID to extract data from")
    parser.add_argument("--mock", action="store_true", help="Generate mock data for testing")
    args = parser.parse_args()
    generate_report(args.run_id, args.mock)
