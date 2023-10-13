import os
from pathlib import Path

import pytest

from pydesigner.preprocessing import mrinfoutil

TEST_DIR = Path(__file__).parent
PATH_DWI = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.nii")
PATH_BVEC = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.bvec")
PATH_BVAL = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.bval")
PATH_JSON = os.path.join(TEST_DIR, "data", "hifi_splenium_4vox.json")
PATH_MIF = os.path.join(TEST_DIR, "data", "hifi_splenium_mrgrid.mif")


def test_getconsole_error_exists():
    """Tests whether function raises OSError when input is not found"""
    with pytest.raises(OSError):
        mrinfoutil.getconsole("nonexistentfile", "--size")


def test_getconsole_error_flag_non_string():
    """Tests whether function raises TypeError when flag is not a string"""
    with pytest.raises(TypeError):
        mrinfoutil.getconsole(PATH_DWI, 420)


def test_getconsole_invalid_flag():
    """Tests whether function raises ValueError when flag is not valid"""
    with pytest.raises(OSError):
        mrinfoutil.getconsole(PATH_DWI, "--foo")


def test_getconsole_valid_flag():
    """Test normal function of getconsole"""
    assert mrinfoutil.getconsole(PATH_DWI, "--format") == "NIfTI-1.1"


def test_console_dtype():
    """Test whether function returns string type"""
    assert isinstance(mrinfoutil.getconsole(PATH_DWI, "--format"), str)


def test_format():
    """Test whether function returns correct format"""
    assert mrinfoutil.format(PATH_DWI) == "NIfTI-1.1"


def test_format_dtype():
    """Test whether function returns string type"""
    assert isinstance(mrinfoutil.format(PATH_DWI), str)


def test_ndim():
    """Test whether function returns correct number of dimensions"""
    assert mrinfoutil.ndim(PATH_DWI) == 4


def test_ndim_dtype():
    """Test whether function returns int type"""
    assert isinstance(mrinfoutil.ndim(PATH_DWI), int)


def test_size():
    """Test whether function returns correct size"""
    assert mrinfoutil.size(PATH_DWI) == (2, 2, 2, 337)


def test_size_dtype():
    """Test whether function returns tuple type"""
    assert isinstance(mrinfoutil.size(PATH_DWI), tuple)


def test_spacing():
    """Test whether function returns correct spacing"""
    assert any(t in (3.0, 3.0, 3.0) for t in mrinfoutil.spacing(PATH_DWI)) is True


def test_spacing_dtype():
    """Test whether function returns tuple type"""
    assert isinstance(mrinfoutil.spacing(PATH_DWI), tuple)


def test_datatype():
    """Test whether function returns correct datatype"""
    assert mrinfoutil.datatype(PATH_DWI) == "Float32LE"


def test_datatype_dtype():
    """Test whether function returns string type"""
    assert isinstance(mrinfoutil.datatype(PATH_DWI), str)


def test_strides():
    """Test whether function returns correct strides"""
    assert mrinfoutil.strides(PATH_DWI) == (1, 2, 3, 4)


def test_strides_dtype():
    """Test whether function returns tuple type"""
    assert isinstance(mrinfoutil.strides(PATH_DWI), tuple)


def test_offset():
    """Test whether function returns correct offset"""
    assert mrinfoutil.offset(PATH_DWI) == 0


def test_offset_dtype():
    """Test whether function returns int type"""
    assert isinstance(mrinfoutil.offset(PATH_DWI), float)


def test_multiplier():
    """Test whether function returns correct multiplier"""
    assert mrinfoutil.multiplier(PATH_DWI) == 1.0


def test_multiplier_dtype():
    """Test whether function returns float type"""
    assert isinstance(mrinfoutil.multiplier(PATH_DWI), float)


def test_transform():
    """Test whether function returns correct transform"""
    result = (
        ["0.994345310289832", "2.73896740248844", "08", "-0.106195121171422", "-7.16012287139893"],
        ["-0.0163522207646128", "0.988073569252652", "-0.153111814252808", "-15.8305568695068"],
        ["0.104928588957842", "0.153982537140822", "0.982486319790561", "-10.7536220550537"],
        ["0", "0", "0", "1"],
    )
    assert mrinfoutil.transform(PATH_DWI) == result


def test_transform_dtype():
    """Test whether function returns tuple type"""
    assert isinstance(mrinfoutil.transform(PATH_DWI), tuple)


def test_commandhistory_invalid():
    """Test whether function raises OSError when input is invalid"""
    with pytest.raises(OSError):
        mrinfoutil.commandhistory(PATH_DWI)


def test_commandhistory_valid():
    """Test normal function of commandhistory"""
    result = [
        "variable",
        "mrcat -axis 3 /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwi0.mif /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwi1.mif /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwi2.mif /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/working.mif",
        "dwidenoise -noise /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/noisemap.nii -extent 5,5,5 /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/working.mif /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/1_dwi_denoised.mif",
        "mrdegibbs /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/working.mif /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/2_dwi_degibbs.mif",
        "/usr/local/mrtrix3/bin/dwifslpreproc -se_epi /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/B0_EPI.mif -eddy_options --repol --data_is_shelled -rpe_header -eddyqc_all /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/metrics_qc/eddy /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/working.mif /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/3_dwi_undistorted.mif",
        "mrconvert -force -quiet -fslgrad /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwism.bvec /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwism.bval -json_import /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwism.json -strides 1,2,3,4 /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwism.nii /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/4_dwi_smoothed.mif",
        "mrconvert -force -quiet -fslgrad /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwirc.bvec /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwirc.bval -json_import /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwirc.json -strides 1,2,3,4 /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwirc.nii /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/5_dwi_rician.mif",
        "mrconvert -fslgrad /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwi_preprocessed.bvec /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwi_preprocessed.bval -json_import /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwi_preprocessed.json /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/dwi_preprocessed.nii /media/sid/Secondary/Datasets/IAM_HiFI/out/pydesigner/working.mif",
        "mrgrid /Users/siddhiman/Datasets/IAM_HiFI/out/pydesigner/working.mif regrid -size 1,1,1 /Users/siddhiman/Repos/PyDesigner/tests/data/hifi_splenium_mrgrid.mif",
    ]
    assert mrinfoutil.commandhistory(PATH_MIF) == result


def test_commandhistory_dtype():
    """Test whether function returns list type"""
    assert isinstance(mrinfoutil.commandhistory(PATH_MIF), list)
