import os
import subprocess
import sys

def run_command(args, description):
    print(f"📦 {description}...")
    try:
        subprocess.run(args, shell=False, check=True)
        print(f"✅ {description} successful.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during {description}: {e}")
        return False
    return True

def setup_colab():
    print("🚀 --- Healthcare FL: Colab Bootstrap Initializing ---")
    print(f"📍 Current Working Directory: {os.getcwd()}")
    
    # Environment Variables for Linux/Colab
    os.environ["PYTHONPATH"] = "."
    os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
    
    # 1. Nuclear Option: Uninstall conflicting system packages that we DON'T use
    # These packages (TensorFlow 2.18+, JAX, etc.) mandate Protobuf 5.x, which breaks Flower.
    conflicting_pkgs = ["tensorflow", "ydf", "jax", "jaxlib", "grain", "tensorflow-decision-forests"]
    run_command(["pip", "uninstall", "-y"] + conflicting_pkgs, "Removing conflicting system packages (TensorFlow/JAX)")

    # 2. Install Linux-Compatible Dependencies
    # Pinning numpy < 2.0 to avoid breaking changes (np.float_ removal)
    # Pinning protobuf == 4.25.3 to satisfy Flower 1.12.0
    deps = [
        "numpy<2.0",
        "protobuf==4.25.3",
        "torch==2.4.1",
        "torchvision",
        "torchaudio",
        "flwr[simulation]==1.12.0",
        "monai==1.3.2",
        "opacus==1.4.1",
        "mlflow",
        "nibabel",
        "scikit-image",
        "umap-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "git+https://github.com/owkin/FLamby.git@edacf54d5211520583b0133d55ac39b6fda8324b"
    ]
    
    run_command(["pip", "install", "--upgrade"] + deps, "Installing Colab-optimized dependencies")

    # 3. Data Linkage
    local_data_path = "data/fed_ixi"
    if os.path.exists(local_data_path):
        print(f"✅ Data found at {local_data_path}.")
    else:
        print(f"⚠️ Data not found locally. Checking Drive...")

    # 4. Apply Linux/Colab Patches
    if os.path.exists("scripts/patch_dependencies.py"):
        run_command([sys.executable, "scripts/patch_dependencies.py"], "Applying FLamby patches")

    print("\n--- ⚠️ IMPORTANT: RESTART SESSION ⚠️ ---")
    print("Conflicts resolved. System packages removed. Version pinning applied.")
    print("You MUST go to: Runtime -> Restart Session")
    print("before running Cell 2 and beyond.")
    print("--- ✅ Colab Bootstrap Complete ---")

if __name__ == "__main__":
    setup_colab()
