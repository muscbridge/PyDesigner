import os.path as op
from typing import Optional

from pydantic import BaseModel, ValidationInfo, field_validator


class modelmrtrix(BaseModel):
    """Define model validation for MRtrix3 CLI"""

    file_input: str = None
    file_output: Optional[str] = None
    nthreads: Optional[bool] = None
    force: Optional[bool] = True
    verbose: Optional[bool] = False

    @field_validator("file_input")
    @classmethod
    def file_input_valid(cls, var: str, info: ValidationInfo):
        """Validate that input file is correct type and exists."""
        if not isinstance(var, str):
            raise TypeError(
                f"Input file path {info.field_name}: {var} is not a string " f"type. Type entered is {type(var)}."
            )
        if not op.exists(var):
            raise FileNotFoundError(f"Input file {info.field_name}: {var} does not exist.")

    @field_validator("file_output")
    @classmethod
    def file_output_valid(cls, var: str, info: ValidationInfo):
        """Validate output filename and parent folder."""
        if not isinstance(var, str):
            raise TypeError(
                f"Output file path {info.field_name}: {var} is not a string " f"type. Type entered is {type(var)}."
            )
        if not op.exists(op.dirname(var)):
            raise FileNotFoundError(
                f"Parent directory {op.dirname(var)} for input file {var} does "
                "not exists. Please ensure that the parent folder exists."
            )

    @field_validator("nthreads")
    @classmethod
    def correct_nthreads(cls, var: int, info: ValidationInfo) -> int:
        """Check that nthreads is a postive integer."""
        if not isinstance(var, int):
            raise TypeError(
                f"{info.field_name} is not provided as a {type(var)}. Please " "provide a positive integer."
            )
        if var <= 0:
            raise ValueError(f"{info.field_name} needs to be a valid integer.")
