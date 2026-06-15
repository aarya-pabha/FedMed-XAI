import nibabel as nib
import os
from pathlib import Path
from tqdm import tqdm

def scrub_nifti_file(file_path):
    """
    Scrubs potential PHI from NIfTI headers and removes all extensions.
    """
    try:
        img = nib.load(file_path)
        header = img.header
        
        # 1. Clear text fields that often leak PHI
        header['descrip'] = b''
        header['aux_file'] = b''
        header['db_name'] = b''
        header['intent_name'] = b''
        
        # 2. Remove NIfTI extensions (can contain arbitrary metadata/PHI)
        header.extensions.clear()
        
        # 3. Save the scrubbed file (overwrites original)
        # Note: We use the same image data, just modified header
        scrubbed_img = nib.Nifti1Image(img.get_fdata(), img.affine, header)
        nib.save(scrubbed_img, file_path)
        return True
    except Exception as e:
        print(f"Error scrubbing {file_path}: {e}")
        return False

def main():
    data_root = Path("data/fed_ixi")
    image_dir = data_root / "image"
    label_dir = data_root / "label"
    
    all_files = list(image_dir.glob("*.nii.gz")) + list(label_dir.glob("*.nii.gz"))
    
    print(f"--- HIPAA Scrubbing: {len(all_files)} files ---")
    
    success_count = 0
    for f in tqdm(all_files):
        if scrub_nifti_file(f):
            success_count += 1
            
    print(f"--- SUCCESS: Scrubbed {success_count}/{len(all_files)} files ---")
    print("Action Required: Run 'dvc add data/fed_ixi' and 'dvc push' to update cloud provenance.")

if __name__ == "__main__":
    main()
