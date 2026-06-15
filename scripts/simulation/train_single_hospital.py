import argparse
import os
import torch
import time
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore")

from src.federated.client import FlowerClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--logical-batch-size", type=int, default=4)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true", help="Perform evaluation only (no training)")
    args = parser.parse_args()

    print(f"🏥 [Standalone] Hospital {args.cid} Round {args.round} starting...")

    # Initialize client
    client = FlowerClient(
        center_id=args.cid,
        dry_run=False,
        batch_size=16, # High-intensity scale for L4
        logical_batch_size=args.logical_batch_size,
        run_id=args.run_id # Pass run_id for centralized logging
    )

    # 1. Pull Global Model (This is the 'Federated Brain' for the current round)
    global_params = client._pull_global_shard(args.round)

    # 2. Global Evaluation (Initial performance of Federated Brain)
    print(f"🌍 [Standalone] Hospital {args.cid} starting GLOBAL evaluation...")
    client.evaluate(global_params, {"metric_prefix": "global"})

    if args.eval_only:
        print(f"✅ [Standalone] Hospital {args.cid} Final Evaluation complete.")
        return

    # 3. Local Training
    config = {"round": args.round, "lr": 1e-4}
    _, _, fit_metrics = client.fit(global_params, config)
    
    # Extract just-trained local parameters
    target = client.model._module if hasattr(client.model, "_module") else client.model
    local_params = [val.detach().cpu().numpy() for _, val in target.state_dict().items()]

    # 4. Local Evaluation (Refined performance of Clinical Node)
    print(f"🏠 [Standalone] Hospital {args.cid} starting LOCAL evaluation...")
    _, _, val_metrics = client.evaluate(local_params, {"metric_prefix": "local"})
    dice_score = val_metrics['dice']

    # 5. Peak VRAM Diagnostic
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"📊 [Diagnostic] Peak VRAM: {peak_vram:.2f} MB")

    # 5. Log consolidated results
    print(f"✅ [Standalone] Hospital {args.cid} complete. Loss: {fit_metrics['loss']:.4f}, Dice: {dice_score:.4f}")

if __name__ == "__main__":
    main()
