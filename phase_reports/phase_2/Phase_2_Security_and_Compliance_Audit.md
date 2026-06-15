# Phase 2 Security & Compliance Audit

**Date:** 2026-05-26
**Auditor Persona:** HIPAA-Compliant Software Architect
**Scope:** Phase 2 (Local Simulation, Opacus Differential Privacy, NIfTI Data Handling)

## 1. Executive Summary

Phase 2 successfully demonstrated the mathematical feasibility of training a 3D medical imaging model (MONAI DynUNet) under Local Differential Privacy (LDP) constraints using the Opacus framework. However, the architectural transition from theoretical simulation to a production-ready, HIPAA-compliant system has revealed critical engineering vulnerabilities and compliance gaps that must be remediated prior to Phase 3 (Medplum Orchestration) and Phase 4 (Cloud Deployment).

The current implementation achieves a baseline of privacy but relies on insecure persistence mechanisms and mathematically flawed budget allocation loops.

## 2. HIPAA Safeguard Mapping

| Safeguard Type | Requirement | Current Posture | Assessment |
| :--- | :--- | :--- | :--- |
| **Technical** | Access Control (164.312(a)(1)) | Static IAM key (`colab-client-key.json`) present in workspace. | **FAIL** (See Finding 2.4) |
| **Technical** | Audit Controls (164.312(b)) | GCS Data Access Logs enabled. | **PASS** (Infrastructure Layer) |
| **Technical** | Integrity (164.312(c)(1)) | NIfTI files tracked via DVC. State tracked via insecure serialization. | **FAIL** (See Finding 2.1) |
| **Technical** | Transmission Security (164.312(e)(1))| Gradient transfer over TLS 1.3 (implied via GCP SDK). | **PASS** |
| **Administrative** | Data De-identification | NIfTI text headers scrubbed. | **PARTIAL** (See Finding 2.3) |

---

## 3. Vulnerability & Engineering Debt Report

### 3.1 [CRITICAL] Insecure Deserialization of Privacy State (CWE-502)
*   **Location:** `src/federated/client.py` (Lines 36-41, 91-92)
*   **Description:** The "True DP Persistence" mechanism utilizes Python's native `pickle` module to save and load the Opacus RDP (Renyi Differential Privacy) accountant state (`ray_tmp/dp_state_client_{cid}.pkl`). `pickle` is inherently unsafe; it can execute arbitrary Python code during deserialization. In a federated environment, a compromised or malicious client node could craft a poisoned `.pkl` file leading to Remote Code Execution (RCE) on the aggregator or sibling simulation nodes.
*   **Remediation:** 
    1.  Immediately replace `pickle` with `safetensors` or standard `json`. 
    2.  Extract the raw dictionary from `privacy_engine.accountant.state_dict()`, serialize it securely, and reconstruct it upon loading.

### 3.2 [HIGH] Differential Privacy Budget Depletion Loop
*   **Location:** `src/federated/client.py` (Line 79), `src/utils/privacy.py` (Line 23)
*   **Description:** The `fit` function calls `make_private_with_epsilon(target_epsilon=10.0)` in *every* federated round. Opacus calculates the noise multiplier required to hit $\epsilon=10.0$ for that specific call based on the *current* accountant history. Because we are correctly persisting the history across rounds, Round 2 sees that the budget is already partially spent, triggering the warning: `You're calling make_private_with_epsilon with non-zero privacy budget already spent.` By Round 3 or 4, Opacus will either crash or inject infinite noise because the cumulative $\epsilon$ has reached 10.0.
*   **Remediation:**
    1.  Switch from `make_private_with_epsilon` to the standard `make_private` method.
    2.  Calculate the fixed `noise_multiplier` mathematically *once* before the simulation begins based on the total number of intended rounds and epochs, and pass that static multiplier to `make_private` in every round.

### 3.3 [MEDIUM] Incomplete PHI Scrubbing Scope (Pixel-Level Leakage)
*   **Location:** `scripts/scrub_nifti_headers.py`
*   **Description:** The current scrubbing script effectively removes explicit text-based Protected Health Information (PHI) from the NIfTI headers (`descrip`, `aux_file`). However, 3D MRI scans of the head often include highly detailed facial structures (nose, eyes, contours) that can be reconstructed to identify the patient, constituting pixel-level PHI.
*   **Remediation:**
    1.  Integrate a "defacing" utility (e.g., `pydeface` or FSL's `mri_deface`) into the data ingestion pipeline to physically remove facial features from the `get_fdata()` arrays before DVC tracking.

### 3.4 [HIGH] Static IAM Credential Exposure
*   **Location:** Project Root (`colab-client-key.json`)
*   **Description:** The presence of a long-lived Service Account JSON key in the workspace violates the Principle of Least Privilege and Zero-Trust architecture. If accidentally committed to version control or exfiltrated from the VM, it grants persistent access to the GCP environment.
*   **Remediation:**
    1.  Delete the JSON key immediately.
    2.  On the GCE VM, rely exclusively on the attached Compute Engine Default Service Account (`789083630799-compute@developer.gserviceaccount.com`), which authenticates transparently without files.

---

## 4. Conclusion & Next Steps

Phase 2 must be considered "Conditionally Approved". The mathematical foundation is sound, but the implementation is brittle. 

Before proceeding to the Medplum orchestration in Phase 3, the engineering team **MUST**:
1. Refactor the `pickle` logic to `json` for DP persistence.
2. Fix the Opacus `make_private` budget loop to prevent infinite noise injection.
3. Remove the static JSON credential.

*(End of Report)*