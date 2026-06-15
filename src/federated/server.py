import argparse
import flwr as fl
import mlflow
import os
import torch
from src.federated.client import FlowerClient
from src.models.unet_pp import get_model

# Set technical intensity to high for server output
os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1" # Bypass PyTorch Dynamo multiprocessing bug
def get_args():
    parser = argparse.ArgumentParser(description="Healthcare FL: Local Simulation Entrypoint")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of federated rounds")
    parser.add_argument("--dry-run", action="store_true", help="Use tiny model and mock data for fast verification")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for clients")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for local training")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable remote MLflow logging")
    parser.add_argument("--reset-nodes", action="store_true", help="Clear persistent privacy states (ray_tmp/*.pkl)")
    return parser.parse_args()

def reset_simulation_state():
    """
    Clears persistent DP states to start a fresh study.
    """
    tmp_dir = os.path.join(os.getcwd(), "ray_tmp")
    if os.path.exists(tmp_dir):
        print(f"🧹 Resetting simulation state: Clearing {tmp_dir}...")
        # Only delete the .pkl files to avoid breaking Ray's own internal sockets
        for f in os.listdir(tmp_dir):
            if f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except Exception as e:
                    print(f"⚠️ Could not remove {f}: {e}")
    print("✅ Reset complete.")

def main():
    args = get_args()
    if args.reset_nodes:
        reset_simulation_state()

def get_initial_parameters(dry_run: bool):
    """
    Generate initial global parameters from a fresh model instance.
    """
    model = get_model(dry_run=dry_run)
    return fl.common.ndarrays_to_parameters(
        [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    )

def check_env_resources():
    """
    Architectural check for Colab/Low-Resource compatibility.
    """
    cpus = os.cpu_count() or 2
    # Cap actors to ensure server + ray head have breathing room
    num_actors = max(1, cpus - 1)
    print(f"🖥️ Environment Resources: Detected {cpus} CPUs. Allocating {num_actors} Ray Actors.")
    return cpus, num_actors

def main():
    args = get_args()
    cpus, num_actors = check_env_resources()
    
    # cap min clients to detected actors for stability in simulation
    min_clients = min(3, num_actors)
    
    print(f"--- 🏥 Healthcare FL: {'Dry-Run ' if args.dry_run else ''}Simulation Entrypoint ---")
    
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        return {
            "round": server_round,
            "lr": 1e-3,
        }

    # 1. Setup MLflow Tracking (Optional)
    if not args.no_mlflow:
        tracking_uri = "https://mlflow-server-789083630799.us-central1.run.app"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Federated_IXI_Simulation")

    # 2. Define client_fn closure
    def client_fn(cid: str) -> fl.client.Client:
        center_id = int(cid)
        return FlowerClient(
            center_id=center_id,
            target_epsilon=10.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        ).to_client()

    # 3. Define Federated Learning Strategy (FedAvg with Comprehensive MLflow Logging)
    class MLflowStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            weights, metrics = super().aggregate_fit(server_round, results, failures)
            
            # Log failure count
            if not args.no_mlflow and mlflow.active_run():
                mlflow.log_metric("fit_failures", len(failures), step=server_round)
            
            # Log per-client epsilon
            epsilons = []
            for _, fit_res in results:
                cid = fit_res.metrics.get("cid", "unknown")
                eps = fit_res.metrics.get("epsilon")
                if eps is not None:
                    epsilons.append(eps)
                    if not args.no_mlflow and mlflow.active_run():
                        mlflow.log_metric(f"client_{cid}_epsilon", eps, step=server_round)
            
            if epsilons and not args.no_mlflow and mlflow.active_run():
                mlflow.log_metric("max_epsilon", max(epsilons), step=server_round)
                
            return weights, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            
            if not args.no_mlflow and mlflow.active_run():
                mlflow.log_metric("eval_failures", len(failures), step=server_round)
                
                # Log per-client eval performance
                for _, eval_res in results:
                    # In evaluate, we don't always have cid in metrics unless added by client
                    # We can use CID if the client provided it
                    cid = eval_res.metrics.get("cid", "unknown")
                    c_loss = eval_res.loss
                    c_dice = eval_res.metrics.get("dice")
                    
                    mlflow.log_metric(f"client_{cid}_loss", c_loss, step=server_round)
                    if c_dice is not None:
                        mlflow.log_metric(f"client_{cid}_dice", c_dice, step=server_round)

                # Log global aggregates
                if loss is not None:
                    print(f"📈 Round {server_round} aggregated loss: {loss}")
                    mlflow.log_metric("distributed_loss", loss, step=server_round)
                    if metrics and "dice" in metrics:
                        mlflow.log_metric("distributed_dice", metrics["dice"], step=server_round)
            
            return loss, metrics

    strategy = MLflowStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=get_initial_parameters(args.dry_run),
        on_fit_config_fn=fit_config,
    )

    # 4. Start Simulation
    run_name = "FL_DryRun" if args.dry_run else "FL_FullSimulation"
    
    def run_sim():
        print(f"🚀 Simulation Config: Rounds={args.num_rounds}, DryRun={args.dry_run}")
        print("🔧 Initializing Ray/Flower simulation...")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=min_clients,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": 1,
                "num_gpus": 0.0,
            },
            ray_init_args={
                "include_dashboard": False,
                "num_cpus": cpus,
                "_temp_dir": os.path.join(os.getcwd(), "ray_tmp")
            }
        )
        print("✅ Simulation complete.")

    if not args.no_mlflow:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(vars(args))
            run_sim()
    else:
        print("⚠️ MLflow logging DISABLED.")
        run_sim()

if __name__ == "__main__":
    main()
