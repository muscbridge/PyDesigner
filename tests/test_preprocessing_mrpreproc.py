import pytest
from conftest import load_data

from pydesigner.preprocessing import mrpreproc

DATA = load_data(type="hifi")
PATH_DWI = DATA["nifti"]
PATH_BVEC = DATA["bvec"]
PATH_BVAL = DATA["bval"]
PATH_JSON = DATA["json"]
PATH_MIF = DATA["mif"]


def test_miftonii_error_path():
    """Test whether function `miftonii` raises error on invalid path"""
    with pytest.raises(IOError):
        mrpreproc.miftonii("nonexistentfile", "output.nii")
