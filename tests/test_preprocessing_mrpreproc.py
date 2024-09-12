import os
import nibabel as nib
import pytest
from unittest.mock import patch, MagicMock
import subprocess
from conftest import load_data

from pydesigner.preprocessing import mrpreproc, mrinfoutil
from pydesigner.system.errors import FileExtensionError, MRTrixError

DATA = load_data(type="hifi")
PATH_DWI = DATA["nifti"]
PATH_DWI_NO_JSON = DATA["no_json"]
PATH_DWI_NO_BVEC = DATA["no_bvec"]
PATH_DWI_NO_BVAL = DATA["no_bval"]
PATH_BVEC = DATA["bvec"]
PATH_BVAL = DATA["bval"]
PATH_JSON = DATA["json"]
PATH_MIF = DATA["mif"]

def test_miftonii_output_failure(tmp_path):
    """Test whether function `miftonii` fails when return code is non-zero"""
    output_nii = str(tmp_path / "output.nii")
    output_bval = str(tmp_path / "output.bval")
    output_bvec = str(tmp_path / "output.bvec")
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="stderr"
        )
        with pytest.raises(MRTrixError) as exc:
            mrpreproc.miftonii(PATH_MIF, output_nii)
        assert f"Conversion from .mif to .nii failed" in str(exc.value)
        assert "stderr" in str(exc.value)


@pytest.mark.parametrize(
    "nthreads, force, verbose",
    [
        (None, None, None),
        (1, None, None),
        (None, True, None),
        (None, False, None),
        (None, None, True),
        (None, None, False),
    ]
)
def test_miftonii_output_success(tmp_path, nthreads, force, verbose):
    """Test whether function `miftonii` generates a valid NIfTI file"""
    output_nii = str(tmp_path / "output.nii")
    output_bval = str(tmp_path / "output.bval")
    output_bvec = str(tmp_path / "output.bvec")
    mrpreproc.miftonii(PATH_MIF, output_nii, nthreads=nthreads, force=force, verbose=verbose)
    assert os.path.exists(output_nii)
    assert os.path.exists(output_bval)
    assert os.path.exists(output_bvec)
    assert os.path.splitext(output_nii)[-1] == ".nii"
    img = nib.load(output_nii)
    assert type(img).__name__ == "Nifti1Image"
    assert img.shape == (2, 2, 2, 337)


def test_niitomif_failure_bval(tmp_path):
    """Test whether function `niitomif` fails when there is no sidecar file"""
    output_mif = str(tmp_path / "output.mif")
    with pytest.raises(OSError) as exc:
         mrpreproc.niitomif(PATH_DWI_NO_BVAL, output_mif)
    assert f"Input file path ({os.path.splitext(PATH_DWI_NO_BVAL)[0]}.bval) does not exist" in str(exc.value)


def test_niitomif_failure_bvec(tmp_path):
    """Test whether function `niitomif` fails when there is no sidecar file"""
    output_mif = str(tmp_path / "output.mif")
    with pytest.raises(OSError) as exc:
         mrpreproc.niitomif(PATH_DWI_NO_BVEC, output_mif)
    assert f"Input file path ({os.path.splitext(PATH_DWI_NO_BVEC)[0]}.bvec) does not exist" in str(exc.value)


def test_niitomif_failure_json(tmp_path):
    """Test whether function `niitomif` fails when there is no sidecar file"""
    output_mif = str(tmp_path / "output.mif")
    with pytest.raises(OSError) as exc:
         mrpreproc.niitomif(PATH_DWI_NO_JSON, output_mif)
    assert f"Input file path ({os.path.splitext(PATH_DWI_NO_JSON)[0]}.json) does not exist" in str(exc.value)


def test_niitomif_output_failure(tmp_path):
    """Test whether function `niitomif` fails when return code is non-zero"""
    output_mif = str(tmp_path / "output.mif")
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="stderr"
        )
        with pytest.raises(MRTrixError) as exc:
            mrpreproc.niitomif(PATH_DWI, output_mif)
        assert f"Conversion from .nii to .mif failed" in str(exc.value)
        assert "stderr" in str(exc.value)


@pytest.mark.parametrize(
    "nthreads, force, verbose",
    [
        (None, None, None),
        (1, None, None),
        (None, True, None),
        (None, False, None),
        (None, None, True),
        (None, None, False),
    ]
)
def test_niitomif_success(tmp_path, nthreads, force, verbose):
    """Test whether function `niitomif` successfully converts a valid NifTI file to MIF"""
    output_mif = str(tmp_path / "output.mif")
    mrpreproc.niitomif(PATH_DWI, output_mif, nthreads=nthreads, force=force, verbose=verbose)
    assert os.path.exists(output_mif)
    assert mrinfoutil.format(output_mif) == "MRtrix"
    mrinfoutil.size(output_mif) == (2, 2, 2, 337)
    assert "mrconvert" in mrinfoutil.commandhistory(output_mif)[-1]


def test_stride_match_output_failure(tmp_path):
    """Test whether function `stride_match` fails when return code is non-zero"""
    output_nii = str(tmp_path / "output.nii")
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="stderr"
        )
        with pytest.raises(MRTrixError) as exc:
            mrpreproc.stride_match(PATH_DWI, PATH_DWI_NO_BVAL, output_nii)
        assert f"Stride matching failed" in str(exc.value)
        assert "stderr" in str(exc.value)


@pytest.mark.parametrize(
    "nthreads, force, verbose",
    [
        (None, None, None),
        (1, None, None),
        (None, True, None),
        (None, False, None),
        (None, None, True),
        (None, None, False),
    ]
)
def test_stride_match_output_success(tmp_path, nthreads, force, verbose):
    """Test whether function `stride_match` fails when return code is non-zero"""
    output_mif = str(tmp_path / "output.mif")
    mrpreproc.stride_match(PATH_DWI, PATH_DWI_NO_BVAL, output_mif, nthreads=nthreads, force=force, verbose=verbose)
    assert os.path.exists(output_mif)
    assert mrinfoutil.format(output_mif) == "MRtrix"
    mrinfoutil.size(output_mif) == (2, 2, 2, 337)
    assert mrinfoutil.strides(output_mif) == (1, 2, 3, 4)
    assert "mrconvert" in mrinfoutil.commandhistory(output_mif)[-1]


def test_denoise_noisemap_invalid(tmp_path):
    """Test whether function `denoise` fails when noisemap is invalid"""
    output_mif = str(tmp_path / "output.mif")
    with pytest.raises(TypeError) as exc:
        mrpreproc.denoise(PATH_MIF, output_mif, noisemap="invalid")
    assert "Please specify whether noisemap generation is True or False" in str(exc.value)


def test_denoise_output_failure(tmp_path):
    """Test whether function `stride_match` fails when return code is non-zero"""
    output_mif = str(tmp_path / "output.mif")
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="stderr"
        )
        with pytest.raises(MRTrixError) as exc:
            mrpreproc.denoise(PATH_MIF, output_mif, noisemap=True, extent="1,1,1")
        assert f"Dwidenoise failed" in str(exc.value)
        assert "stderr" in str(exc.value)


@pytest.mark.parametrize(
    "nthreads, force, verbose",
    [
        (None, None, None),
        (1, None, None),
        (None, True, None),
        (None, False, None),
        (None, None, True),
        (None, None, False),
    ]
)
def test_denoise_output_success(tmp_path, nthreads, force, verbose):
    """Test whether function `denoise` successfully denoises a DWI dataset"""
    output_mif = str(tmp_path / "output.mif")
    noisemap_nii = str(tmp_path / "noisemap.nii")
    mrpreproc.denoise(PATH_MIF, output_mif, noisemap=True, extent="1,1,1", nthreads=nthreads, force=force, verbose=verbose)
    assert os.path.exists(output_mif)
    assert os.path.exists(noisemap_nii)
    assert mrinfoutil.format(output_mif) == "MRtrix"
    mrinfoutil.size(output_mif) == (2, 2, 2, 337)
    assert "dwidenoise" in mrinfoutil.commandhistory(output_mif)[-1]


def test_degibbs_output_failure(tmp_path):
    """Test whether function `degibbs` fails when return code is non-zero"""
    output_mif = str(tmp_path / "output.mif")
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="stderr"
        )
        with pytest.raises(MRTrixError) as exc:
            mrpreproc.degibbs(PATH_MIF, output_mif)
        assert f"Mrdegibbs failed" in str(exc.value)
        assert "stderr" in str(exc.value)

@pytest.mark.parametrize(
    "nthreads, force, verbose",
    [
        (None, None, None),
        (1, None, None),
        (None, True, None),
        (None, False, None),
        (None, None, True),
        (None, None, False),
    ]
)
def test_degibbs_output_success(tmp_path, nthreads, force, verbose):
    """Test whether function `degibbs` fails when return code is non-zero"""
    output_mif = str(tmp_path / "output.mif")
    mrpreproc.degibbs(PATH_MIF, output_mif, nthreads=nthreads, force=force, verbose=verbose)
    assert os.path.exists(output_mif)
    assert mrinfoutil.format(output_mif) == "MRtrix"
    mrinfoutil.size(output_mif) == (2, 2, 2, 337)
    assert mrinfoutil.strides(output_mif) == (1, 2, 3, 4)
    assert "mrdegibbs" in mrinfoutil.commandhistory(output_mif)[-1]


def test_undistort_output_failure_conversion(tmp_path):
    """Test whether function `undistort` fails at gradient conversion when return code is non-zero"""
    output_mif = str(tmp_path / "output.mif")
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(
            returncode=1, stderr="stderr"
        )
        with pytest.raises(MRTrixError) as exc:
            mrpreproc.undistort(PATH_MIF, output_mif)
        assert f"Extraction of FSL BVEC and BVAL gradients failed" in str(exc.value)
        assert "stderr" in str(exc.value)


def test_undistort_output_failure_all(tmp_path):
    """Test whether function `undistort` fails when final return code is non-zero"""
    output_mif = str(tmp_path / "output.mif")
    call_count = 0
    original_run = subprocess.run
    def subprocess_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 6:
            return MagicMock(returncode=1, stderr="stderr")
        else:
            with patch("subprocess.run", original_run):
                return original_run(*args, **kwargs)
    with patch("subprocess.run", side_effect=subprocess_side_effect) as mock_subprocess:
        with pytest.raises(MRTrixError) as exc:
            mrpreproc.undistort(PATH_MIF, output_mif, epib0=0)
        assert f"Dwifslpreproc failed" in str(exc.value)
        assert "stderr" in str(exc.value)

# def test_undistort_output_success(tmp_path):
#     """Test whether function `undistort` successfully undistorts a DWI dataset"""
#     output_mif = str(tmp_path / "output.mif")
#     mrpreproc.undistort(PATH_MIF, output_mif, epib0=0)
#     assert os.path.exists(output_mif)
#     assert mrinfoutil.format(output_mif) == "MRtrix"
#     mrinfoutil.size(output_mif) == (2, 2, 2, 337)
#     assert mrinfoutil.strides(output_mif) == (1, 2, 3, 4)
#     assert "dwifslpreproc" in mrinfoutil.commandhistory(output_mif)[-1]



# def test_brainmask_output_failure(tmp_path):
#     """Test whether function `undistort` fails when return code is non-zero"""
#     output_nii = str(tmp_path / "output.nii")
#     with patch("subprocess.run") as mock_subprocess:
#         mock_subprocess.return_value = MagicMock(
#             returncode=1, stderr="stderr"
#         )
#         with pytest.raises(MRTrixError) as exc:
#             mrpreproc.brainmask(PATH_MIF, output_nii)
#         assert f"Unable to compute brain mask from B0" in str(exc.value)
#         assert "stderr" in str(exc.value)
