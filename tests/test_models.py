from pydesigner.system.models import modelmrtrix, input_path_validator, output_path_validator
from pydesigner.system.errors import FileExtensionError
from pydantic import ValidationError
import pytest
from conftest import load_data

DATA = load_data(type="hifi")
PATH_DWI = DATA["nifti"]

def test_modelmrtrix_input_invalid():
    with pytest.raises(TypeError) as exc:
        model = modelmrtrix(input=50)
    assert "Input file path (input=50) is not a string type" in str(exc.value)


def test_modelmrtrix_input_nonexistent(tmp_path):
    input_nii = str(tmp_path / "input.nii")
    with pytest.raises(FileNotFoundError) as exc:
        model = modelmrtrix(input=input_nii)
    assert f"Input file path ({input_nii}) does not exist" in str(exc.value) 


def test_modelmrtrix_input_basedir():
    with pytest.raises(OSError) as exc:
        model = modelmrtrix(input="nonexistent/input.nii")
    assert "Pleasure ensure that the input parent directory exists" in str(exc.value)


def test_modelmrtrix_input_success():
    model = modelmrtrix(input=PATH_DWI)
    assert model.input == PATH_DWI


def test_modelmrtrix_output_invalid():
    with pytest.raises(TypeError) as exc:
        model = modelmrtrix(output=50)
    assert "Output file path (output=50) is not a string type" in str(exc.value)


def test_modelmrtrix_output_basedir():
    with pytest.raises(OSError) as exc:
        model = modelmrtrix(output="nonexistent/output.nii")
    assert "Pleasure ensure that the output parent directory exists" in str(exc.value)


def test_modelmrtrix_output_success(tmp_path):
    output_nii = str(tmp_path / "output.nii")
    model = modelmrtrix(output=output_nii)
    assert model.output == output_nii
    

def test_modelmrtrix_verbose_fail():
    with pytest.raises(ValidationError) as exc:
        model = modelmrtrix(verbose="foo")
    assert "Input should be a valid boolean" in str(exc.value)


def test_modelmrtrix_verbose_success():
    model = modelmrtrix(verbose=True)
    print(model)
    assert model.verbose == True 


def test_modelmrtrix_force_fail():
    with pytest.raises(ValidationError) as exc:
        model = modelmrtrix(force="foo")
    assert "Input should be a valid boolean" in str(exc.value)


def test_modelmrtrix_nthreads_invalid():
    with pytest.raises(TypeError) as exc:
        model = modelmrtrix(nthreads="foo")
    assert "Please provide a positive integer" in str(exc.value)


def test_modelmrtrix_nthreads_negative():
    with pytest.raises(ValueError) as exc:
        model = modelmrtrix(nthreads=-1)
    assert "nthreads needs to be a valid positive integer" in str(exc.value)


def test_input_validator_path_invalid():
    with pytest.raises(TypeError) as exc:
        path = input_path_validator(path=10)
    assert "Please enter path as a string" in str(exc.value)


def test_input_validator_path_nonexistent(tmp_path):
    input_path = str(tmp_path / "input.nii")
    with pytest.raises(FileNotFoundError) as exc:
        path = input_path_validator(path=input_path)
    assert f"Input file ({input_path}) does not exist" in str(exc.value)


def test_input_validator_ctype_invalid():
    with pytest.raises(TypeError) as exc:
        path = input_path_validator(path=PATH_DWI, ctype=20)
    assert "ctype variable (ctype=20) needs to be a valid string" in str(exc.value)


def test_input_validator_ctype_check_fail():
    with pytest.raises(FileExtensionError) as exc:
        path = input_path_validator(path=PATH_DWI, ctype=".tar")
    assert f"Input file ({PATH_DWI}) does not posses the required .tar extension" in str(exc.value)


def test_input_validator_success():
    path = input_path_validator(path=PATH_DWI)
    assert path == PATH_DWI


def test_output_validator_path_invalid():
    with pytest.raises(TypeError) as exc:
        path = output_path_validator(path=10)
    assert "Please enter path as a string" in str(exc.value)


def test_output_validator_basepath():
    with pytest.raises(OSError) as exc:
        path = output_path_validator(path="nonexistent/output.nii")
    assert "Pleasure ensure that the output parent directory exists" in str(exc.value)


def test_output_validator_ctype_invalid():
    with pytest.raises(TypeError) as exc:
        path = output_path_validator(path=PATH_DWI, ctype=20)
    assert "ctype variable (ctype=20) needs to be a valid string" in str(exc.value)


def test_output_validator_ctype_check_fail():
    with pytest.raises(FileExtensionError) as exc:
        path = output_path_validator(path=PATH_DWI, ctype=".tar")
    assert f"Input file ({PATH_DWI}) does not posses the required .tar extension" in str(exc.value)


def test_output_validator_success(tmp_path):
    path = output_path_validator(path=PATH_DWI, ctype=".nii")
    assert path == PATH_DWI
