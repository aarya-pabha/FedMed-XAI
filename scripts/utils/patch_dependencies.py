import os
import flamby
from pathlib import Path

def patch_flamby_ixi():
    """
    Patches FLamby's Fed-IXI dataset for:
    1. MONAI 1.3+ compatibility (AddChannel -> EnsureChannelFirst)
    2. Flat directory structure support (removes hardcoded Mendeley IDs)
    3. Direct disk loading (removes .zip requirement)
    """
    ixi_dataset_path = Path(flamby.__path__[0]) / "datasets" / "fed_ixi" / "dataset.py"
    
    if not ixi_dataset_path.exists():
        print(f"Error: Could not find {ixi_dataset_path}")
        return

    content = ixi_dataset_path.read_text()

    # 1. Update Imports
    if "import nibabel as nib" not in content:
        content = "import nibabel as nib\n" + content
    
    if "AddChannel," in content:
        content = content.replace("AddChannel,", "EnsureChannelFirst,")
        print("Patched: Imports (AddChannel -> EnsureChannelFirst)")

    # 2. Update Transform usage
    if "AddChannel()," in content:
        content = content.replace("AddChannel()", 'EnsureChannelFirst(channel_dim="no_channel")')
        print("Patched: Transform usage")

    # 3. Flatten paths & subjects extraction
    # Handle the fact that dataset.py might already be partially patched
    bad_split_logic = 'self.images_sets = ["train" for _ in self.subjects]'
    if bad_split_logic in content:
        correct_split_logic = """        # Correctly restore split from metadata
        from flamby.datasets.fed_ixi.utils import _get_id_from_filename
        self.filenames = [filename.name for filename in self.images_paths]
        self.subject_ids = tuple(map(_get_id_from_filename, self.filenames))
        self.images_sets = [self.metadata.loc[sid, "Split"] for sid in self.subject_ids]"""
        content = content.replace(bad_split_logic, correct_split_logic)
        print("Patched: Directory structure logic (fixed train/test split)")
    else:
        old_path_logic = 'self.parent_dir_name = os.path.join(self.parent_folder, "IXI_sample")'
        if old_path_logic in content:
            new_logic = """        self.parent_dir_name = ""
        self.subjects_dir = self.root_folder

        image_folder = self.root_folder / "image"
        label_folder = self.root_folder / "label"
        
        self.images_paths = sorted(list(image_folder.glob("*.nii.gz")))
        self.labels_paths = sorted(list(label_folder.glob("*.nii.gz")))

        self.subjects = [p.name.replace("_image.nii.gz", "") for p in self.images_paths]
        self.images_centers = [_extract_center_name_from_filename(s) for s in self.subjects]
        
        # Correctly restore split from metadata
        from flamby.datasets.fed_ixi.utils import _get_id_from_filename
        self.filenames = [filename.name for filename in self.images_paths]
        self.subject_ids = tuple(map(_get_id_from_filename, self.filenames))
        self.images_sets = [self.metadata.loc[sid, "Split"] for sid in self.subject_ids]
        
        self.demographics = None # Patched for flat structure"""
            
            start_marker = 'self.parent_dir_name = os.path.join(self.parent_folder, "IXI_sample")'
            end_marker = 'self.filenames = [filename.name for filename in self.images_paths]'
            
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                # Use a clean replacement without extra trailing spaces
                content = content[:start_idx] + new_logic + "\n" + content[end_idx:]
                print("Patched: Directory structure logic")

    # 4. Patch __getitem__ to load from disk instead of ZIP
    old_getitem_call = """        header_img, img, label, center_name = _load_nifti_image_and_label_by_id(
            zip_file=self.zip_file, patient_id=patient_id, modality=self.modality
        )"""
    if old_getitem_call in content:
        new_getitem_call = """        img_path = self.images_paths[item]
        label_path = self.labels_paths[item]
        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()"""
        content = content.replace(old_getitem_call, new_getitem_call)
        print("Patched: __getitem__ disk loading")

    ixi_dataset_path.write_text(content)
    print(f"Successfully patched {ixi_dataset_path}")

if __name__ == "__main__":
    patch_flamby_ixi()
