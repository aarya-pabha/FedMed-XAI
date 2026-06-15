# Phase 5 Architectural & HIPAA Compliance Audit

**Date:** May 30, 2026
**Auditor:** Principal Software Architect / HIPAA Compliance Lead
**Scope:** Distributed Integration & Execution (GradsSharding, Client Orchestration, FHIR Audit Logging)

## 1. Executive Summary
Phase 5 successfully bridges the local federated training nodes with the Phase 4 serverless infrastructure. The implementation of the Medplum FHIR audit logger and the GradsSharding client logic establishes a robust, distributed control plane. However, the mechanism used to serialize gradient tensors introduces a critical Remote Code Execution (RCE) vulnerability that violates both general cloud security postures and HIPAA integrity mandates.

## 2. Audit Findings

### Task 1: Medplum FHIR Audit Logger
*   **Status:** **PASS**
*   **Analysis:** Implemented `src/utils/medplum_audit.py` to post metrics as FHIR `Observation` resources.
*   **Strengths:** Successfully leverages OAuth2 `client_credentials` flow. Hardcodes `verify=True` for all HTTPS requests to prevent MITM attacks (satisfying previous Phase 3 remediations).
*   **HIPAA Mapping:** Audit Controls (§ 164.312(b)), Transmission Security (§ 164.312(e)(1)).

### Task 2 & 3: GradsSharding Client & Orchestration
*   **Status:** **FAIL (Critical Risk)**
*   **Analysis:** The `FlowerClient.fit` method slices parameters and uploads them to GCS, triggering the Cloud Function via Firestore state updates. The polling loop for the global shard is correctly bounded.
*   **Vulnerability Identified [CRITICAL]:** The client uses `torch.save(parameters, byte_stream)` to serialize the model weights before uploading to GCS. PyTorch's native `save` function relies on Python's `pickle` module. 
    *   *Context:* Even though the client attempts to use `torch.load(..., weights_only=True)` when pulling the global model, recent disclosures (CVE-2025-32434, affecting PyTorch < 2.6.0) demonstrate that `weights_only=True` can be bypassed to achieve RCE. A compromised node or Man-in-the-Middle attacker could replace a shard in the `-in` or `-out` bucket with a maliciously crafted pickle payload, achieving remote code execution on the serverless aggregator or peer hospitals.
*   **HIPAA Mapping:** **FAIL** on Integrity (§ 164.312(c)(1)) and Protection from Malicious Software.

---

## 3. Remediation Matrix

| Finding ID | Severity | Component | Recommended Action |
| :--- | :--- | :--- | :--- |
| **VULN-501** | **Critical** | `src/federated/client.py` & `src/cloud/aggregator/main.py` | Replace all instances of `torch.save()` and `torch.load()` with the `safetensors` library. Safetensors prevents arbitrary code execution by storing pure numerical data with a JSON header, completely eliminating the pickle attack surface. |
| **VULN-502** | **Low** | `src/federated/client.py` | In `_log_medplum`, the `dice` metric is hardcoded to `0.0` during the `fit` phase because validation metrics aren't computed there. This creates misleading FHIR records. Update the logging strategy to omit the Dice component if it evaluates to exactly 0.0 during training. |

## 4. Final Verdict
**BLOCKED.** The reliance on `pickle`-based serialization (via `torch.save`) in a distributed, multi-institution environment poses an unacceptable RCE risk. The client and aggregator code must be migrated to `safetensors` before Phase 5 can be considered complete.