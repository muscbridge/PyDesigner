import os
import nibabel as nib
import pytest
from conftest import load_data

from pydesigner.preprocessing import mrpreproc, mrinfoutil

DATA = load_data(type="hifi")
PATH_DWI = DATA["nifti"]
PATH_DWI_NO_JSON = DATA["no_json"]
PATH_DWI_NO_BVEC = DATA["no_bvec"]
PATH_DWI_NO_BVAL = DATA["no_bval"]
PATH_BVEC = DATA["bvec"]
PATH_BVAL = DATA["bval"]
PATH_JSON = DATA["json"]
PATH_MIF = DATA["mif"]


def test_miftonii_error_path():
    """Test whether function `miftonii` raises error on invalid path"""
    with pytest.raises(IOError) as exc:
        mrpreproc.miftonii("nonexistentfile", "output.nii")
    assert "Input path does not exist" in str(exc.value)


def test_miftonii_missing_input(tmp_path):
    """Test whether function `miftonii` raises error when input file is missing"""
    input_mif = tmp_path / "input.nii"
    output_nii = tmp_path / "output.nii"
    with pytest.raises(IOError) as exc:
        mrpreproc.miftonii(input_mif, output_nii)
    assert "Input path does not exist" in str(exc.value)


def test_miftonii_error_directory():
    """Test whether function `miftonii` raises error on nonexistent directory"""
    with pytest.raises(OSError) as exc:
        mrpreproc.miftonii(PATH_MIF, "nonexistent/output.nii")
    assert "Specifed directory for output file nonexistent does not exist" in str(exc.value)


def test_miftonii_error_input(tmp_path):
    """Test whether function `miftonii` raises error on invalid output file"""
    output_nii = tmp_path / "output.tar"
    with pytest.raises(IOError) as exc:
        mrpreproc.miftonii(PATH_MIF, output_nii)
    assert "Output specified does not possess the .nii extension" in str(exc.value)


def test_miftonii_success(tmp_path):
    """Test whether function `miftonii` successfully converts a valid MIF file to NIfTI"""
    output_nii = tmp_path / "output.nii"
    mrpreproc.miftonii(PATH_MIF, output_nii)
    assert os.path.exists(output_nii)


def test_miftonii_output_correctness_image(tmp_path):
    """Test whether function `miftonii` generates a valid NIfTI file"""
    output_nii = tmp_path / "output.nii"
    output_bval = tmp_path / "output.bval"
    output_bvec = tmp_path / "output.bvec"
    mrpreproc.miftonii(PATH_MIF, output_nii)
    assert os.path.exists(output_nii)
    assert os.path.exists(output_bval)
    assert os.path.exists(output_bvec)
    assert(os.path.splitext(output_nii))[-1] == ".nii"
    img = nib.load(output_nii)
    assert type(img).__name__ == "Nifti1Image"
    assert img.shape == (2, 2, 2, 337)


def test_niitomif_error_path():
    """Test whether function `miftonii` raises error on invalid path"""
    with pytest.raises(IOError) as exc:
        mrpreproc.niitomif("nonexistentfile", "output.mif")
    assert "Input path does not exist" in str(exc.value)


def test_niitomif_missing_input(tmp_path):
    """Test whether function `niitomif` raises error when input file is missing"""
    input_nii = tmp_path / "input.nii"
    output_mif = tmp_path / "output.mif"
    with pytest.raises(IOError) as exc:
        mrpreproc.miftonii(input_nii, output_mif)
    assert "Input path does not exist" in str(exc.value)


def test_niitomif_error_directory():
    """Test whether function `niitomif` raises error on nonexistent directory"""
    with pytest.raises(OSError) as exc:
        mrpreproc.niitomif(PATH_DWI, "nonexistent/output.mif")
    assert "Specifed directory for output file nonexistent does not exist" in str(exc.value)


def test_niitomif_error_output(tmp_path):
    """Test whether function `niitomif` raises error on invalid output file"""
    output_mif = tmp_path / "output.tar"
    with pytest.raises(IOError) as exc:
        mrpreproc.niitomif(PATH_DWI, output_mif)
    assert "Output specified does not possess the .mif extension" in str(exc.value)


def test_niitomif_failure_bval(tmp_path):
    """Test whether function `niitomif` fails when there is no sidecar file"""
    output_mif = tmp_path / "output.mif"
    with pytest.raises(OSError) as exc:
         mrpreproc.niitomif(PATH_DWI_NO_BVAL, output_mif)
    assert f"Unable to locate BVAL file {os.path.splitext(PATH_DWI_NO_BVAL)[0]}.bval" in str(exc.value)


def test_niitomif_failure_bvec(tmp_path):
    """Test whether function `niitomif` fails when there is no sidecar file"""
    output_mif = tmp_path / "output.mif"
    with pytest.raises(OSError) as exc:
         mrpreproc.niitomif(PATH_DWI_NO_BVEC, output_mif)
    assert f"Unable to locate BVEC file {os.path.splitext(PATH_DWI_NO_BVEC)[0]}.bvec" in str(exc.value)


def test_niitomif_failure_json(tmp_path):
    """Test whether function `niitomif` fails when there is no sidecar file"""
    output_mif = tmp_path / "output.mif"
    with pytest.raises(OSError) as exc:
         mrpreproc.niitomif(PATH_DWI_NO_JSON, output_mif)
    assert f"Unable to locate JSON file {os.path.splitext(PATH_DWI_NO_JSON)[0]}.json" in str(exc.value)

def test_niitomif_success(tmp_path):
    """Test whether function `niitomif` successfully converts a valid NifTI file to MIF"""
    output_mif = tmp_path / "output.mif"
    mrpreproc.niitomif(PATH_DWI, output_mif)
    assert os.path.exists(output_mif)


def test_niitomif_output_correctness_image(tmp_path):
    output_mif = tmp_path / "output.mif"
    mrpreproc.niitomif(PATH_DWI, output_mif)
    assert mrinfoutil.format(output_mif) == "MRtrix"
    mrinfoutil.size(output_mif) == (2, 2, 2, 337)

