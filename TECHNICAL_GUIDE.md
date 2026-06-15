# Technical Guide: Privacy-Preserving Clinical Intelligence

This guide provides the rigorous mathematical and architectural specifications for the Federated Medical Image Diagnostics platform.

## 1. Differential Privacy Mathematical Framework

To protect sensitive radiological volumes, we implement **Local Differential Privacy (LDP)** via the DP-SGD algorithm.

### 1.1. The $(\epsilon, \delta)$ Guarantee
The system guarantees that the participation of any single patient's data in the training loop cannot be reliably inferred by an adversary observing the model weights.
$$P[\mathcal{A}(D) \in S] \le e^{\epsilon} P[\mathcal{A}(D') \in S] + \delta$$
- **Final Configuration:** $\epsilon \le 10.0$, $\delta = 10^{-5}$.
- **Sensitivity Bounding:** We enforce a `max_grad_norm = 1.0`. Every per-sample gradient $g_i$ is clipped: $\bar{g}_i = g_i / \max(1, \frac{||g_i||_2}{C})$.

### 1.2. Gaussian Noise Injection
Noise is sampled from $\mathcal{N}(0, \sigma^2 C^2)$ and added to the aggregated batch gradient. We utilize the Opacus RDP (Rényi Differential Privacy) accountant to track cumulative budget expenditure across $T$ rounds.
$$w_{noisy} = w_{clipped} + \mathcal{N}(0, \sigma^2 C^2)$$

## 2. Serverless GradsSharding Topology

Traditional Federated Averaging (FedAvg) fails in serverless environments when model parameters exceed memory (OOM). We implement **GradsSharding** to parallelize aggregation.

### 2.1. Gradient Partitioning
The global parameter vector $W \in \mathbb{R}^{|\theta|}$ is partitioned into $M$ contiguous shards $s$:
$$W_k^{(t)} = [w_{k,1}^{(t)}, w_{k,2}^{(t)}, \dots, w_{k,M}^{(t)}]$$

### 2.2. Distributed Aggregation
Each shard $i$ is processed by an independent instance of a Google Cloud Function. The function performs an element-wise weighted average:
$$w_{global, i}^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{N} w_{k,i}^{(t)}$$
Where $n_k$ is the local sample count and $N = \sum n_k$. This ensures $O(1)$ memory complexity relative to model size $M$.

## 3. Architecture: 32-Channel Wide U-Net++

We utilize a custom `U-Net++` (MONAI) optimized for the **Fed-IXI** dataset's specific clinical characteristics.

### 3.1. Technical Specifications
- **Wide Feature Maps:** $(32, 32, 64, 128, 256, 32)$. This increased capacity compensates for the high signal-to-noise ratio requirements of DP-SGD.
- **Group Normalization:** Replaced BatchNorm with **GroupNorm (8 groups)**. This is mandatory for DP, as BatchNorm's cross-sample statistics leak privacy and break per-sample gradient independence.
- **Anisotropy Stride Optimization:** The U-Net++ architecture is configured with asymmetric kernel sizes to handle the non-cubic voxel dimensions ($1.73 \times 0.73 \times 1.15$ mm) of the IXI MRI scans.

## 4. Compliance Control Plane

### 4.1. FHIR-native Audit Logging (Medplum)
Every federated transaction is logged as a FHIR `Observation` resource:
- **Identifier:** `urn:uuid:[round_id]`
- **Value:** Combined `{loss, dice, epsilon_spent}`.
- **Access Control:** Hospital nodes utilize `ClientApplication` credentials with **AccessPolicies** restricted to the `Observation` create scope.

### 4.2. Infrastructure Hardening (GCP)
- **GCS:** Public Access Prevention (PAP) + Object Versioning.
- **Firestore:** Transactional locking ensures atomic aggregation, preventing race conditions if two hospitals upload shards simultaneously.
- **De-identification:** Mandatory scrubbing of NIfTI `aux_file` and `descrip` headers via `nibabel` before data leaves the clinical site simulator.

---
*Developed for Regulatory-Grade Healthcare AI by Aarya Pabha.*
