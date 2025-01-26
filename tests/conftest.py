import os
from pathlib import Path

TEST_DIR = Path(__file__).parent
DATA_DIR = os.path.join(TEST_DIR, "data")


def load_data(type: str = "hifi") -> dict:
    """Loads sample dataset paths.

    Args:
        type (str): Type of data to load. Defaults to "hifi".

    Returns:
        dict: Sample data paths.
    """
    if type == "hifi":
        data_path = os.path.join(DATA_DIR, type)
        data = {
            "nifti": os.path.join(data_path, "hifi_splenium_4vox.nii"),
            "mif": os.path.join(data_path, "hifi_splenium_4vox.mif"),
            "bval": os.path.join(data_path, "hifi_splenium_4vox.bval"),
            "bvec": os.path.join(data_path, "hifi_splenium_4vox.bvec"),
            "json": os.path.join(data_path, "hifi_splenium_4vox.json"),
            "mask": os.path.join(data_path, "brain_mask.nii"),
            "mean_b0": os.path.join(data_path, "mean_b0.mif"),
            "no_json": os.path.join(data_path, "hifi_splenium_4vox_no_json.nii"),
            "no_bval": os.path.join(data_path, "hifi_splenium_4vox_no_bval.nii"),
            "no_bvec": os.path.join(data_path, "hifi_splenium_4vox_no_bvec.nii"),
            "no_sidecar": os.path.join(data_path, "hifi_splenium_4vox_no_sidecar.nii"),
        }
    return data
