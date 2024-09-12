import os.path as op
import nibabel as nib
import pytest
import numpy as np
from conftest import load_data

from pydesigner.fitting.dwipy import DWI
from pydesigner.fitting.dwidirs import dirs30

DATA = load_data(type="hifi")
PATH_DWI = DATA["nifti"]
PATH_BVEC = DATA["bvec"]
PATH_BVAL = DATA["bval"]
PATH_MASK = DATA["mask"]

def test_dwi_init_image_nonexistent(tmp_path):
    input_nii = str(tmp_path / "nonexistent.nii")
    with pytest.raises(FileNotFoundError) as exc:
        dwi = DWI(input_nii, PATH_BVEC, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert f"Input image ({input_nii}) not found" in str(exc.value)


def test_dwi_init_bvec_invalid(tmp_path):
    with pytest.raises(TypeError) as exc:
        dwi = DWI(PATH_DWI, 50, PATH_BVAL, PATH_MASK)
    assert "Input file path (input=50) is not a string type." in str(exc.value)


def test_dwi_init_bvec_nonexistent(tmp_path):
    input_bvec = str(tmp_path / "nonexistent.bvec")
    with pytest.raises(FileNotFoundError) as exc:
        dwi = DWI(PATH_DWI, input_bvec, PATH_BVAL, PATH_MASK)
    assert f"Input file path ({input_bvec}) does not exist." in str(exc.value)


def test_dwi_init_bval_invalid(tmp_path):
    with pytest.raises(TypeError) as exc:
        dwi = DWI(PATH_DWI, PATH_BVEC, 50, PATH_MASK)
    assert "Input file path (input=50) is not a string type" in str(exc.value)


def test_dwi_init_bval_nonexistent(tmp_path):
    input_bval = str(tmp_path / "nonexistent.bval")
    with pytest.raises(OSError) as exc:
        dwi = DWI(PATH_DWI, PATH_BVEC, input_bval, PATH_MASK)
    assert f"Input file path ({input_bval}) does not exist." in str(exc.value)


def test_dwi_init_mask_nonexistent(tmp_path, capsys):
    input_mask = str(tmp_path / "nonexistent.nii")
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, input_mask)
    captured = capsys.readouterr()
    assert "No brain mask supplied" in captured.out


def test_dwi_init_success(capsys):
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    captured = capsys.readouterr()
    assert dwi.hdr.header.get_data_shape() == (2, 2, 2, 337)
    assert np.shape(dwi.grad) == (337, 4)
    assert f"Image {op.basename(PATH_DWI)} loaded successfully" in captured.out


def test_dwi_get_bvals():
    """Tests whether function returns correct bvals"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    bvals = dwi.getBvals()
    assert bvals.dtype == np.float64
    assert np.shape(bvals) == (337,)
    assert 0 in bvals
    assert 1 in bvals
    assert 2 in bvals
    assert 8 in bvals


def test_dwi_get_bvecs():
    """Tests whether function returns correct bvecs"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    bvecs = dwi.getBvecs()
    assert bvecs.dtype == np.float64
    assert bvecs.shape == (337, 3)


def test_dwi_max_bval():
    """Tests whether function returns correct max bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxBval()
    assert isinstance(val, int)
    assert val == 8


def test_dwi_max_dti_bval():
    """Tests whether function returns correct max DTI bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxDTIBval()
    assert isinstance(val, int)
    assert val == 1


def test_dwi_max_dki_bval():
    """Tests whether function returns correct max DKI bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxDKIBval()
    assert isinstance(val, int)
    assert val == 2


def test_max_fbi_bval():
    """Tests whether function returns correct max FBI bval"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.maxFBIBval()
    assert isinstance(val, int)
    assert val == 8


def test_dwi_idx_b0():
    """Tests whether function returns correct index of b0"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxb0()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 21


def test_dwi_idx_dti():
    """Tests whether function returns correct index of DTI b-values"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxdti()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 51


def test_dwi_idx_dki():
    """Tests whether function returns correct index of DKI b-values"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxdki()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 81


def test_idx_fbi():
    """Tests whether function returns correct index of FBI b-values"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    idx = dwi.idxfbi()
    assert idx.dtype == bool
    assert len(idx) == 337
    assert sum(idx) == 256

def test_dwi_n_dirs():
    """Tests whether function returns correct number of directions"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.getndirs() == 30


def test_dwi_tensor_type():
    """Tests whether function returns correct tensor type"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    tensor = dwi.tensorType()
    assert isinstance(tensor, list)
    assert "dti" in tensor
    assert "dki" in tensor
    assert "fbi" in tensor
    assert "fbwm" in tensor


def test_dwi_is_dti():
    """Tests whether function returns correct boolean for DTI dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isdti() is True


def test_dwi_is_dki():
    """Tests whether function returns correct boolean for DKI dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isdki() is True


def test_dwi_is_fbi():
    """Tests whether function returns correct boolean for FBI dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isfbi() is True


def test_dwi_is_fbwm():
    """Tests whether function returns correct boolean for FBWM dataset"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    assert dwi.isfbwm() is True


def test_dwi_tensor_order_invalid_order():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    with pytest.raises(ValueError):
        cnt, ind = dwi.createTensorOrder(5)


def test_dwi_tensor_order_valid_order():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    cnt, ind = dwi.createTensorOrder(2)
    assert len(cnt) == 6
    assert np.shape(ind) == (6, 2)


def test_dwi_tensor_order_auto_detect():
    """Tests whether function returns correct tensor order"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    cnt, ind = dwi.createTensorOrder()
    assert len(cnt) == 15
    assert np.shape(ind) == (15, 4)


def test_dwi_fibonacci_sphere_invalid_samples():
    """Tests whether function returns correct response from invalid samples type"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    with pytest.raises(TypeError):
        dwi.fibonacciSphere(samples=5.2)


def test_dwi_fibonacci_sphere_success():
    """Tests whether function returns correct response"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    sphere = dwi.fibonacciSphere(samples=5)
    assert sphere.dtype == np.float64
    assert np.shape(sphere) == (5, 3)


def test_dwi_radial_sampling():
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    samples = 128
    dirs = dwi.radialSampling(dirs30, samples)
    assert dirs.dtype == np.float64
    assert np.shape(dirs) == (samples - 1, 3)


def test_dwi_constraints_invalid():
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    with pytest.raises(ValueError) as exc:
        val = dwi.createConstraints([1,2,3])
    assert "Invalid contraints" in str(exc.value)


@pytest.mark.parametrize(
    "constraints",
    ([0, 0, 0],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 0])
)
def test_dwi_constraints_success(constraints):
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    val = dwi.createConstraints(constraints)
    if sum(constraints) == 0:
        shape = (0, 22)
    elif sum(constraints) == 1:
        shape = (30, 22)
    elif sum(constraints) == 2:
        shape = (60, 22)
    elif sum(constraints) == 3:
        shape = (90, 22)
    assert val.dtype == np.float64
    assert np.shape(val) == shape


def test_dwi_fit_constrained(capsys):
    """Tests whether constrained fitting works normally"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    dwi.fit([0, 1, 0])
    captured = capsys.readouterr()
    assert hasattr(dwi, "dt")
    assert np.shape(dwi.dt) == (21, 5)
    assert "Constrained Tensor Fit" in captured.err


def test_dwi_fit_unconstrained(capsys):
    """Tests whether unconstrained fitting works normally"""
    dwi = DWI(PATH_DWI, PATH_BVEC, PATH_BVAL, PATH_MASK)
    dwi.fit()
    captured = capsys.readouterr()
    assert hasattr(dwi, "dt")
    assert np.shape(dwi.dt) == (21, 5)
    assert "Unconstrained Tensor Fit" in captured.err
