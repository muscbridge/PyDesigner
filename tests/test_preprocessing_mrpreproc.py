import os
from pathlib import Path
from pydesigner.preprocessing import mrinfoutil
import pytest

TEST_DIR = Path(__file__).parent
PATH_DWI = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.nii")
PATH_BVEC = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.bvec")
PATH_BVAL = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.bval")
PATH_JSON = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.json")
PATH_MIF = os.path.join(TEST_DIR, "data", "hifi_splenium_mrgrid.mif")

