#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Adds utilities for preparing the data for eddy and analysis
"""

import os #mkdir
import os.path as op # dirname, basename, join, splitext
import sys # exit
import json # decode
from enum import Enum
import nibabel as nib # various utilities for reading Nifti images
import subprocess
import re # regex substitution

def fix_bval(bvalfile):
    """Converts all whitespace into newlines in the file

    Parameters
    ----------
    bvalfile :obj: `str`
        The .bval to ensure is the correct format for mrtrix

    Returns
    -------
    None, overwrites bval
    """

    if not op.exists(bvalfile):
        raise Exception('File '+ bvalfile + ' does not exist.')

    with open(bvalfile, 'r') as f:
        data = f.read()

    # replace whitespace with lines
    data = re.sub(r'\s+', '\n', data)

    # write to file
    with open(bvalfile, 'w') as f:
        f.write(data)
