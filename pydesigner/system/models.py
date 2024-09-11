import os.path as op
import typing as t

from pydantic import BaseModel, StrictBool, ValidationInfo, field_validator

from .errors import FileExtensionError


class modelmrtrix(BaseModel):
    """Define model validator for common MRtrix3 CLI options."""

    input: t.Optional[t.Union[str, t.Any]] = None
    output: t.Optional[t.Union[str, t.Any]] = None
    nthreads: t.Optional[t.Union[int, t.Any]] = None
    verbose: t.Optional[StrictBool] = None
    force: t.Optional[StrictBool] = None

    @field_validator("input", mode="before")
    @classmethod
    def input_valid(cls, var: str, info: ValidationInfo) -> str:
        """Validate that input file is correct type and exists."""
        if not isinstance(var, str):
            msg = f"Input file path ({info.field_name}={var}) is not a string type. Type entered is {type(var)}."
            raise TypeError(msg)
        if not op.exists(op.dirname(var)):
            msg = f"Specified directory ({op.dirname(var)}) for output file "
            msg += f"({op.basename(var)}) does not exist. "
            msg += "Pleasure ensure that the input parent directory exists."
            raise OSError(msg)
        if not op.exists(var):
            msg = f"Input file path ({var}) does not exist."
            raise FileNotFoundError(msg)
        return var

    @field_validator("output", mode="before")
    @classmethod
    def output_valid(cls, var: str, info: ValidationInfo) -> str:
        """Validate output filename and parent folder."""
        if not isinstance(var, str):
            msg = f"Output file path ({info.field_name}={var}) is not a string type. "
            msg += f"Type entered is {type(var)}."
            raise TypeError(msg)
        if not op.exists(op.dirname(var)):
            msg = f"Specified directory ({op.dirname(var)}) for output file "
            msg += f"({op.basename(var)}) does not exist. "
            msg += "Pleasure ensure that the output parent directory exists."
            raise OSError(msg)
        return var

    @field_validator("nthreads", mode="before")
    @classmethod
    def correct_nthreads(cls, var: int, info: ValidationInfo) -> int:
        """Check that nthreads is a postive integer."""
        if var is not None:
            if not isinstance(var, int):
                msg = f"{info.field_name} is provided as a {type(var)}. "
                msg += "Please provide a positive integer."
                raise TypeError(msg)
            if var <= 0:
                raise ValueError(f"{info.field_name} needs to be a valid positive integer.")
            return int(var)
        else:
            return var


def input_path_validator(path: str, ctype: str = None):
    """Validates a provided input path

    Parameters
    ----------
    path : str
        Path to input file
    ctype : str
        File extension to enfore on input path

    Returns
    -------
        str
            Sanitized path

    See Also
    --------
    output_path_validator
    """
    opts = modelmrtrix(input=path)
    if ctype is not None:
        if not isinstance(ctype, str):
            msg = f"ctype variable (ctype={ctype}) needs to be a valid string."
            raise (TypeError(msg))
    if ctype:
        if op.splitext(op.basename(opts.input))[-1] != ctype:
            msg = f"Input file ({path}) does not posses the required {ctype} extension."
            raise FileExtensionError(msg)
    return str(path)


def output_path_validator(path: str, ctype: str = None):
    """Validates a provided input path
    Parameters
    ----------
    path : str
        Path to output file
    ctype : str
        File extension to enfore on output path

    Returns
    -------
        str
            Sanitized path

    See Also
    --------
    input_path_validator
    """
    if ctype is not None:
        if not isinstance(ctype, str):
            msg = f"ctype variable (ctype={ctype}) needs to be a valid string."
            raise (TypeError(msg))
    if ctype:
        if op.splitext(op.basename(path))[-1] != ctype:
            msg = f"Output file ({path}) does not posses the required {ctype} extension."
            raise FileExtensionError(msg)
    return str(path)
