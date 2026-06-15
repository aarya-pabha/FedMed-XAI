import torch
import flwr as fl
import mlflow
import numpy as np
import os
import pickle
import io
import time
from typing import Dict, List, Tuple, Optional
from google.cloud import storage, firestore
from safetensors.numpy import save as st_save, load as st_load
from opacus import PrivacyEngine
from src.models.unet_pp import get_model
from src.utils.data import load_partition
from src.utils.privacy import wrap_model_for_dp
from src.utils.medplum_audit import log_metrics_to_medplum
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from mlflow.tracking import MlflowClient

class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, 
        center_id: int, 
        data_path: str = "data/fed_ixi",
        target_epsilon: float = 10.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        batch_size: int = 1, # Physical batch size
        logical_batch_size: int = 4, # Logical batch size for DP accumulation
        dry_run: bool = False,
        run_id: Optional[str] = None
    ):
        self.center_id = center_id
        self.run_id = run_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(dry_run=dry_run).to(self.device)
        self.physical_batch_size = batch_size
        self.logical_batch_size = logical_batch_size
        self.accumulation_steps = max(1, self.logical_batch_size // self.physical_batch_size)
        
        # Load datasets with physical batch size to avoid OOM
        self.train_loader = load_partition(center_id, batch_size=self.physical_batch_size, train=True, data_path=data_path, dry_run=dry_run)
        self.test_loader = load_partition(center_id, batch_size=self.physical_batch_size, train=False, data_path=data_path, dry_run=dry_run)
        
        # GCP & Medplum IDs
        self.project_id = os.getenv("GCP_PROJECT", "healthcare-fl-diagnostics")
        self.medplum_client_id = os.getenv("MEDPLUM_CLIENT_ID")
        self.medplum_client_secret = os.getenv("MEDPLUM_CLIENT_SECRET")
        self.medplum_project_id = os.getenv("MEDPLUM_PROJECT_ID")

        # Fallback to JSON file if env vars missing
        if not all([self.medplum_client_id, self.medplum_client_secret, self.medplum_project_id]):
            creds_path = "medplum_client_credentials.json"
            if os.path.exists(creds_path):
                try:
                    import json
                    with open(creds_path, "r") as f:
                        creds = json.load(f)
                        self.medplum_client_id = self.medplum_client_id or creds.get("clientId")
                        self.medplum_client_secret = self.medplum_client_secret or creds.get("clientSecret")
                        self.medplum_project_id = self.medplum_project_id or creds.get("projectId")
                        print(f"✅ Loaded Medplum credentials from {creds_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load Medplum credentials from JSON: {e}")
        
        # Privacy settings
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        
        # Persistent Privacy Engine for cumulative accounting across rounds
        self.privacy_engine = PrivacyEngine()
        
        # True DP Persistence: Load previous RDP state if it exists
        os.makedirs("ray_tmp", exist_ok=True)
        self.dp_state_file = f"ray_tmp/dp_state_client_{self.center_id}.pkl"
        if os.path.exists(self.dp_state_file):
            try:
                with open(self.dp_state_file, 'rb') as f:
                    accountant_state = pickle.load(f)
                    self.privacy_engine.accountant.load_state_dict(accountant_state)
            except Exception as e:
                print(f"⚠️ Failed to load privacy state for client {self.center_id}: {e}")
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.loss_functions = DiceLoss(sigmoid=True)

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        # If model is wrapped with GradSampleModule (Opacus), we access ._module
        target = self.model._module if hasattr(self.model, "_module") else self.model
        
        # Stability Fix: Sanitize weights before they leave the clinical node
        params = []
        total_malformed = 0
        total_elements = 0
        
        for name, val in target.state_dict().items():
            arr = val.detach().cpu().numpy()
            total_elements += arr.size
            
            # 1. Replace NaN/Inf with 0.0 (safest fallback for DP-SGD)
            if np.isnan(arr).any() or np.isinf(arr).any():
                count = np.isnan(arr).sum() + np.isinf(arr).sum()
                total_malformed += count
                print(f"⚠️ [Stability] {name}: Sanitized {count}/{arr.size} malformed values.")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Removed destructive hard clip (arr = np.clip(arr, -10.0, 10.0))
            params.append(arr)
        
        # Log global sanitization metric
        if total_elements > 0:
            san_pct = (total_malformed / total_elements) * 100
            if self.run_id:
                try:
                    ml_client = MlflowClient(tracking_uri="https://mlflow-server-789083630799.us-central1.run.app")
                    ml_client.log_metric(self.run_id, f"client_{self.center_id}_sanitization_percentage", san_pct)
                    print(f"📊 [Stability] Logged sanitization percentage: {san_pct:.4f}%")
                except Exception as e:
                    print(f"⚠️ MLflow logging failed for sanitization metric: {e}")
            
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        target = self.model._module if hasattr(self.model, "_module") else self.model
        params_dict = zip(target.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        target.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        self.set_parameters(parameters)
        self.model.train() # Opacus requires training mode for make_private
        
        # E=1 constraint hardcoded for simulation stability under DP
        epochs = 1
        lr = float(config.get("lr", 5e-7))
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.0)
        
        # Wrap for DP (using persistent engine)
        self.model, private_optimizer, private_train_loader, _ = wrap_model_for_dp(
            model=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            epochs=epochs,
            max_grad_norm=self.max_grad_norm,
            privacy_engine=self.privacy_engine
        )
        
        self.model.train()
        total_loss = 0.0
        successful_batches = 0
        private_optimizer.zero_grad()
        
        for _ in range(epochs):
            for batch_idx, batch in enumerate(private_train_loader):
                if batch_idx % 10 == 0:
                    print(f"  [Client {self.center_id}] Processing batch {batch_idx}/{len(private_train_loader)}...")
                images = batch[0].to(self.device) # FedIXI return (img, label)
                labels = batch[1].to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                
                # Scale loss by accumulation steps to average the gradients properly
                loss = self.loss_functions(outputs, labels)
                
                # --- DIAGNOSTICS INJECTION ---
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n🚨 DIAGNOSTICS TRIGGERED FOR BATCH {batch_idx}")
                    print(f"  -> Images Shape: {images.shape}, Min: {images.min().item():.4f}, Max: {images.max().item():.4f}, Has NaN: {torch.isnan(images).any().item()}")
                    print(f"  -> Labels Shape: {labels.shape}, Min: {labels.min().item():.4f}, Max: {labels.max().item():.4f}, Has NaN: {torch.isnan(labels).any().item()}")
                    print(f"  -> Outputs Shape: {outputs.shape}, Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}, Has NaN: {torch.isnan(outputs).any().item()}")
                    print(f"  -> Loss Value: {loss.item()}")
                    print(f"🚨 ----------------------------------\n")
                # -----------------------------

                scaled_loss = loss / self.accumulation_steps
                
                # Safety check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Batch produced {loss.item()}. Skipping backward pass to preserve stability.")
                    continue
                    
                scaled_loss.backward()
                total_loss += loss.item()
                successful_batches += 1
                
                # Step optimizer only after accumulating enough gradients
                if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(private_train_loader)):
                    # Safety check for NaN/Inf gradients before step
                    has_nan = False
                    for p in self.model.parameters():
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            has_nan = True
                            break
                    
                    if has_nan:
                        print(f"⚠️ Batch produced NaN/Inf gradients. Skipping optimizer step to preserve stability.")
                        private_optimizer.zero_grad()
                        continue
                    
                    # Opacus safety: only step if we actually called backward() at least once in this accumulation window
                    # Otherwise, grad_sample won't be initialized and step() will throw ValueError
                    has_grad_sample = False
                    for p in self.model.parameters():
                        if hasattr(p, "grad_sample") and p.grad_sample is not None:
                            has_grad_sample = True
                            break
                    
                    if not has_grad_sample:
                        print(f"⚠️ No valid gradients accumulated. Skipping optimizer step.")
                        continue
                        
                    private_optimizer.step()
                    private_optimizer.zero_grad()

        avg_loss = total_loss / max(1, successful_batches)
        
        # Get CUMULATIVE privacy account
        epsilon = self.privacy_engine.get_epsilon(self.target_delta)
        
        # True DP Persistence: Save RDP state for next round
        with open(self.dp_state_file, 'wb') as f:
            pickle.dump(self.privacy_engine.accountant.state_dict(), f)
        
        # MLflow Logging using low-level Client (Process-Safe)
        if self.run_id:
            try:
                ml_client = MlflowClient(tracking_uri="https://mlflow-server-789083630799.us-central1.run.app")
                ml_client.log_metric(self.run_id, f"client_{self.center_id}_train_loss", avg_loss)
                ml_client.log_metric(self.run_id, f"client_{self.center_id}_cumulative_epsilon", epsilon)
            except Exception as e:
                print(f"⚠️ MLflow logging failed: {e}")
            
        round_id = int(config.get("round", 1))
        params_to_upload = self.get_parameters(config={})
        
        # Cloud Orchestration
        self._upload_shard(round_id, params_to_upload)
        self._sync_firestore(round_id)
        self._log_medplum(round_id, avg_loss, epsilon)
        
        return [], len(private_train_loader.dataset), {
            "loss": avg_loss,
            "epsilon": epsilon,
            "cid": self.center_id
        }

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        self.set_parameters(parameters)
        self.model.eval()
        
        prefix = config.get("metric_prefix", "val") # Default to val, but can be 'global' or 'local'
        
        total_loss = 0.0
        self.dice_metric.reset()
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                loss = self.loss_functions(outputs, labels)
                total_loss += loss.item()
                
                # Dice Calculation (assumes binary for DiceLoss(sigmoid=True))
                preds = (torch.sigmoid(outputs) > 0.5).float()
                self.dice_metric(y_pred=preds, y=labels)

        avg_loss = total_loss / len(self.test_loader)
        dice_score = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        # MLflow Logging using low-level Client (Process-Safe)
        if self.run_id:
            try:
                ml_client = MlflowClient(tracking_uri="https://mlflow-server-789083630799.us-central1.run.app")
                ml_client.log_metric(self.run_id, f"client_{self.center_id}_{prefix}_loss", avg_loss)
                ml_client.log_metric(self.run_id, f"client_{self.center_id}_{prefix}_dice", dice_score)
            except Exception as e:
                print(f"⚠️ MLflow logging failed: {e}")

        return float(avg_loss), len(self.test_loader.dataset), {
            "dice": dice_score,
            "cid": self.center_id
        }

    def _upload_shard(self, round_id: int, parameters: List[np.ndarray]) -> None:
        """Serializes and uploads a parameter shard to GCS."""
        storage_client = storage.Client(project=self.project_id)
        bucket_in = storage_client.bucket(f"{self.project_id}-fl-shards-in")
        
        # Serialize to in-memory buffer using safetensors
        params_dict = {f"arr_{i}": arr for i, arr in enumerate(parameters)}
        raw_bytes = st_save(params_dict)
        byte_stream = io.BytesIO(raw_bytes)
        
        blob_name = f"round_{round_id}/client_{self.center_id}/shard_0.safetensors"
        bucket_in.blob(blob_name).upload_from_file(byte_stream)
        print(f"✅ Uploaded shard to {blob_name}")

    def _sync_firestore(self, round_id: int) -> None:
        """Signals Firestore that a shard has been uploaded."""
        db = firestore.Client(project=self.project_id)
        shard_ref = db.collection('rounds').document(str(round_id)).collection('shards').document("0")
        shard_ref.collection('uploads').document(str(self.center_id)).set({
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"✅ Signaled Firestore for round {round_id}, client {self.center_id}")

    def _log_medplum(self, round_id: int, loss: float, epsilon: float) -> None:
        """Logs metrics to Medplum via FHIR Observation."""
        if self.medplum_client_id:
            success = log_metrics_to_medplum(
                client_id=self.medplum_client_id,
                client_secret=self.medplum_client_secret,
                project_id=self.medplum_project_id,
                round_num=round_id,
                loss=loss,
                dice=0.0, # Dice not available in fit()
                epsilon=epsilon
            )
            if success:
                print(f"✅ Logged metrics to Medplum for round {round_id}")
            else:
                print(f"⚠️ Failed to log metrics to Medplum for round {round_id}")

    def _pull_global_shard(self, round_id: int) -> List[np.ndarray]:
        """Polls GCS for the aggregated global shard and returns parameters."""
        storage_client = storage.Client(project=self.project_id)
        bucket_out = storage_client.bucket(f"{self.project_id}-fl-shards-out")
        global_blob_name = f"round_{round_id}/global/shard_0.safetensors"
        global_blob = bucket_out.blob(global_blob_name)
        
        print(f"🕒 Waiting for global shard {global_blob_name}...")
        timeout = 600
        start_time = time.time()
        while not global_blob.exists():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out waiting for global shard {global_blob_name} after {timeout}s")
            time.sleep(10)
            
        print("📥 Global shard found. Downloading...")
        global_stream = io.BytesIO()
        global_blob.download_to_file(global_stream)
        global_stream.seek(0)
        
        # Deserialize using safetensors
        raw_bytes = global_stream.read()
        params_dict = st_load(raw_bytes)
        parameters = [params_dict[f"arr_{i}"] for i in range(len(params_dict))]
        
        return parameters
