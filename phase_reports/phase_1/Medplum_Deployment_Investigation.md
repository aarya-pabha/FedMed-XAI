# ✦ Medplum Deployment Investigation Report

**Date:** May 17, 2026
**Subject:** Failure Analysis of Official Medplum Docker Image on Cloud Run
**Status:** BLOCKED (Official Image) -> Transitioning to Custom Build

## 1. Objective
To deploy a self-hosted, open-source Medplum FHIR server on Google Cloud Run to serve as the HIPAA-compliant orchestration layer for federated learning.

## 2. Exhaustive Attempt Log

### Attempt 1: Standard Cloud Run Deploy
*   **Strategy:** Deploy `docker.io/medplum/medplum-server:latest` with pure environment variables.
*   **Result:** **FAILED**. 
*   **Error:** `Cannot find module './packages/server/dist/otel/instrumentation.js'`.
*   **Root Cause:** The official image uses a relative path entrypoint that fails when the working directory is managed by Cloud Run's runtime.

### Attempt 2: Local Config Injection (Dockerfile)
*   **Strategy:** Build a wrapper Dockerfile that copies a `medplum.config.json` into the container.
*   **Result:** **FAILED**. 
*   **Error:** Same module error + `invalid tar header` during Cloud Build.
*   **Root Cause:** Cloud Build encountered archive corruption when attempting to layer onto the official vendor image.

### Attempt 3: Secret Manager Mapping
*   **Strategy:** Map the configuration JSON from GCP Secret Manager to the container's filesystem at runtime.
*   **Result:** **FAILED**. 
*   **Error:** `MODULE_NOT_FOUND`.
*   **Root Cause:** Mapping a file to the container root or `/usr/src/medplum` obscured the existing vendor files, breaking the Node.js module loader.

### Attempt 4 & 5: Entrypoint Overrides (Absolute Paths)
*   **Strategy:** Explicitly tell the container to start using absolute paths: `node /usr/src/medplum/packages/server/dist/index.js`.
*   **Result:** **FAILED**. 
*   **Error:** `Error: Cannot find module '/usr/src/medplum/packages/server/dist/index.js'`.
*   **Root Cause:** Despite documentation suggesting this path, the internal filesystem of the official image appears to be structured differently or utilizes symlinks that Cloud Run's security sandbox does not resolve.

---

## 3. Conclusion & Technical Justification
The official Medplum server image is **not "Serverless-Native."** It is optimized for GKE/Kubernetes where persistent volumes and specific working directory controls are easier to enforce. For a "Scale-Ready" project on Cloud Run, the official image is a **"Black Box" dependency failure.**

### Why we need a Custom Build:
1.  **Path Integrity:** By building from source, we control the exact `dist` output and can guarantee that `node index.js` works without complex pre-loads.
2.  **Audit Security:** We can bake the HIPAA-required GCS/S3 interoperability drivers directly into the build.
3.  **Zero-Ops Scale:** A custom image optimized for Cloud Run will scale faster and consume less memory than the vendor's multi-purpose "everything" image.

---

## 4. Custom Source-to-Image Build Log (Attempts 8 - 16)

To bypass the official image issues, we attempted to build Medplum directly from source using Google Cloud Build.

### The Monorepo Challenges:
1.  **Missing Dependencies (`Attempt 8-11`)**: Medplum is a TurboRepo monorepo. Building the `@medplum/server` package requires building `@medplum/core`, `@medplum/definitions`, `@medplum/fhir-router`, and more. Attempting to copy only specific `package.json` files failed because internal build scripts required the full monorepo context.
2.  **Missing Configs (`Attempt 12-14`)**: Tools like `api-extractor` and `tsc` failed because they relied on root-level configuration files (`tsconfig.json`, `tsdoc.json`, `api-extractor.json`) that were not initially copied into the Docker builder stage.
3.  **The `.gcloudignore` Bug (`Attempt 15`)**: Even after copying the entire source directory, the build failed because a global `.gcloudignore` rule (`data/`) accidentally excluded the critical `packages/server/src/migrations/data` folder from the Cloud Build upload.
4.  **Successful Build (`Attempt 16`)**: After anchoring the `.gcloudignore` rules (e.g., `/data/`) and copying the entire source tree, the Cloud Build **succeeded**, producing a custom image.

### The Final Runtime Crash (The ESM Symlink Issue):
Despite a successful build, deploying the custom image to Cloud Run resulted in the same `MODULE_NOT_FOUND` error for `/usr/src/medplum/packages/server/dist/index.js`.
*   **The Probe:** We ran a remote probe (`probe-custom-image.yaml`) which confirmed that `index.js` **did** exist in the container.
*   **Root Cause:** Medplum compiles to an ESM (ECMAScript Module) format. The compiled `dist/index.js` still contains relative imports/symlinks pointing back to other packages (like `core` and `fhirtypes`) in the monorepo workspace. Because our Stage 2 Dockerfile only copied the `dist` folders and `node_modules` to save space, those internal symlinks were broken at runtime. Node.js threw `MODULE_NOT_FOUND` because a dependency of `index.js` was missing, not `index.js` itself.

---

## 5. Final Verdict & Technical Debt

Reverse-engineering Medplum's proprietary `scripts/build-docker-server.sh` (which generates specific `.tar.gz` archives to preserve exact symlink structures) into a Cloud Run compatible Dockerfile is excessively complex and brittle. 

**Decision:**
*   **Databases (Cloud SQL & Redis) are LIVE and correctly provisioned.**
*   **Medplum Compute is accepted as Technical Debt.**
*   For Phase 3 (when FHIR is actually needed), we will run the official Medplum Docker image via a local Docker Compose setup that connects to our live Cloud SQL instance. This maintains data centralization and HIPAA compliance without the serverless deployment overhead.
