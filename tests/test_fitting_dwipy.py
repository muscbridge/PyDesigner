import numpy as np
import pytest
from conftest import load_data

from pydesigner.fitting.dwipy import DWI

DATA = load_data(type="hifi")
PATH_DWI = DATA["nifti"]
PATH_BVEC = DATA["bvec"]
PATH_BVAL = DATA["bval"]
PATH_JSON = DATA["json"]
PATH_MIF = DATA["mif"]


def test_dwi_image_path_nonexistent():
    """Tests whether function raises OSError when input is not found"""
    with pytest.raises(OSError):
        DWI("foo")


def test_dwi_bvec_path_invalid():
    """Tests whether function raises TypeError when bvec file input is invalid"""
    with pytest.raises(TypeError):
        DWI(PATH_DWI, bvecPath=10)


def test_dwi_bvec_path_nonexistent():
    """Tests whether function raises OSError when bvec file is not found"""
    with pytest.raises(OSError):
        DWI(PATH_DWI, bvecPath="foo")


def test_dwi_bval_path_invalid():
    """Tests whether function raises TypeError when bval file input is invalid"""
    with pytest.raises(TypeError):
        DWI(PATH_DWI, bvalPath=10)


def test_dwi_bval_path_nonexistent():
    """Tests whether function raises OSError when bval file is not found"""
    with pytest.raises(OSError):
        DWI(PATH_DWI, bvalPath="foo")


def test_dwi_mask_path_nonexistent(capsys):
    """Tests whether function raises OSError when mask file is not found"""
    DWI(PATH_DWI, mask="foo")
    captured = capsys.readouterr()
    assert "No brain mask supplied" in captured.out


def test_dwi_path_nosidecar():
    """Tests whether function raises OSError when sidecar files are not found"""
    with pytest.raises(OSError):
        DWI(DATA["no_sidecar"])


def test_dwi_nthreads_nonint():
    """Tests whether function raises TypeError when nthreads is not an int"""
    with pytest.raises(TypeError):
        DWI(PATH_DWI, nthreads="foo")


def test_dwi_nthreads_negative_int():
    """Tests whether function raises ValueError when nthreads is negative"""
    with pytest.raises(ValueError):
        DWI(PATH_DWI, nthreads=-5)


def test_dwi_paths_valid(capsys):
    """Tests whether function responds normally when all paths are valid"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    captured = capsys.readouterr()
    print(captured.out)
    assert dwi is not None
    assert "Image hifi_splenium_4vox.nii loaded successfully" in captured.out
    assert "Processing with" in captured.out
    assert "workers..." in captured.out


def test_dwi_get_bvals():
    """Tests whether function returns correct bvals"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    bvals = dwi.getBvals()
    assert bvals.dtype == np.float64
    assert len(bvals) == 337
    assert 0 in bvals
    assert 1 in bvals
    assert 2 in bvals
    assert 8 in bvals


def test_dwi_get_bvecs():
    """Tests whether function returns correct bvecs"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    bvecs = dwi.getBvecs()
    assert bvecs.dtype == np.float64
    assert bvecs.shape == (337, 3)


def test_dwi_max_bval():
    """Tests whether function returns correct max bval"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    dwi.maxBval() == float
    assert dwi.maxBval() == 8


def test_dwi_max_dti_bval():
    """Tests whether function returns correct max DTI bval"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    dwi.maxDTIBval() == float
    assert dwi.maxDTIBval() == 1


def test_dwi_max_dki_bval():
    """Tests whether function returns correct max DKI bval"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    dwi.maxDKIBval() == float
    assert dwi.maxDKIBval() == 2


def test_max_fbi_bval():
    """Tests whether function returns correct max FBI bval"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    dwi.maxFBIBval() == float
    assert dwi.maxFBIBval() == 8


def test_dwi_idx_b0():
    """Tests whether function returns correct index of b0"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    idx = dwi.idxb0()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 21


def test_dwi_idx_dti():
    """Tests whether function returns correct index of DTI b-values"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    idx = dwi.idxdti()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 51


def test_dwi_idx_dki():
    """Tests whether function returns correct index of DKI b-values"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    idx = dwi.idxdki()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 81


def test_idx_fbi():
    """Tests whether function returns correct index of FBI b-values"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    idx = dwi.idxfbi()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 256


def test_dwi_n_dirs():
    """Tests whether function returns correct number of directions"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    assert dwi.getndirs() == 30


def test_dwi_tensor_type():
    """Tests whether function returns correct tensor type"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    tensor = dwi.tensorType()
    assert isinstance(tensor, list)
    assert "dti" in tensor
    assert "dki" in tensor
    assert "fbi" in tensor
    assert "fbwm" in tensor


def test_dwi_is_dti():
    """Tests whether function returns correct boolean for DTI dataset"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    assert dwi.isdti() is True


def test_dwi_is_dki():
    """Tests whether function returns correct boolean for DKI dataset"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    assert dwi.isdki() is True


def test_dwi_is_fbi():
    """Tests whether function returns correct boolean for FBI dataset"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    assert dwi.isfbi() is True


def test_dwi_is_fbwm():
    """Tests whether function returns correct boolean for FBWM dataset"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    assert dwi.isfbwm() is True


def test_dwi_tensor_order_invalid_order():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    with pytest.raises(ValueError):
        cnt, ind = dwi.createTensorOrder(5)


def test_dwi_tensor_order_valid_order():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    cnt, ind = dwi.createTensorOrder(2)
    assert len(cnt) == 6
    assert np.shape(ind) == (6, 2)


def test_dwi_tensor_order_auto_detect():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    cnt, ind = dwi.createTensorOrder()
    assert len(cnt) == 15
    assert np.shape(ind) == (15, 4)


def test_fibonacci_sphere_invalid_samples():
    """Tests whether function returns correct response from invalid samples type"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    with pytest.raises(TypeError):
        dwi.fibonacciSphere(samples=5.2)


def test_fibonacci_sphere():
    """Tests whether function returns correct response"""
    dwi = DWI(PATH_DWI, bvecPath=PATH_BVEC, bvalPath=PATH_BVAL, mask=DATA["mask"])
    sphere = dwi.fibonacciSphere(samples=5)
    assert sphere.dtype == np.float64
    assert np.shape(sphere) == (5, 3)
