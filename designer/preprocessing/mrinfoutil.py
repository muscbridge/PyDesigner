#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Utilities for extracting information on various input files using
MRtrix3's mrinfo tool. All values are returned in basic Python data
types.
"""

import os.path as op
import subprocess
import re

def getconsole(path, flag):
    """
    Fetches the console output of MRtrix3's mrinfo with specified
    flag

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory
    flag: :obj: `str`
        Flag to pass onto mrinfo

    Returns
    -------
    str
        MRtrix3's mrinfo conmsol output
    """
    if not op.exists(path):
        raise OSError('Input path does not exist. Please ensure that the '
                    'folder or file specified exists.')
    if not isinstance(flag, str):
        raise Exception('Input flag is not a string')
    arg = ['mrinfo', flag]
    arg.append(path)
    completion = subprocess.run(arg, stdout=subprocess.PIPE)
    if completion.returncode != 0:
        raise IOError('Input {} is not currently supported by '
                      'PyDesigner.'.format(path))
    console = str(completion.stdout).split('\\n')[0]
    console = console.split('b')[-1]
    console = console.replace("'", "")
    return console

def format(path):
    """
    Returns the file format of input DWI

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    str
        Image file format
    """
    type = getconsole(path, '-format')
    return type

def ndim(path):
    """
    Returns the number of image dimensions of input DWI

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    int
        Number of dimensions in image
    """
    num = getconsole(path, '-ndim')
    return int(num)

def size(path):
    """
    Returns the size of input DWI image along each axis

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    tuple of int
        Number of voxels in [X, Y, Z, B-value]
    """
    num = getconsole(path, '-size').split()
    num = tuple(map(int, num))
    return num

def spacing(path):
    """
    Returns the voxel spacing along each of input DWI's dimensions

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    tuple of int
        Number of spacing between voxels [X, Y, Z, B-value]
    """
    num = getconsole(path, '-spacing').split()
    num = tuple(map(float, num))
    return num

def datatype(path):
    """
    Returns the data type used for storing input DWI

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    str
        MRtrix3 datatype
    """
    return getconsole(path, '-datatype')

def strides(path):
    """
    Returns data strides of input DWI

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    tuple of ints
        MRtrix3's strides
    """
    num = getconsole(path, '-strides').split()
    num = tuple(map(int, num))
    return num

def offset(path):
    """
    Returns the input DWI's intensity offset

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    float
        Image intensity offset
    """
    num = getconsole(path, '-offset')
    num = float(num)
    return num

def multiplier(path):
    """
    Returns the input DWI's intensity multiplier

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    float
        Image intensity multiplier
    """
    num = getconsole(path, '-multiplier')
    num = float(num)
    return num

def transform(path):
    """
    Returns the input DWI's 4x4 voxel to image transformation matrix

    Parameters
    ----------
    path: :obj: `str` 
        Path to input image or directory

    Returns
    -------
    tuple of float
        Image transformation matrix
    """
    if not op.exists(path):
        raise OSError('Input path does not exist. Please ensure that the '
                      'folder or file specified exists.')
    arg = ['mrinfo', '-transform']
    arg.append(path)
    completion = subprocess.run(arg, stdout=subprocess.PIPE)
    if completion.returncode != 0:
        raise IOError('Input {} is not currently supported by '
                      'PyDesigner.'.format(path))
    console = str(completion.stdout).split('\\n')
    num = [re.findall(r"[-+]?\d*\.\d+|\d+", s) for s in console]
    num = [s for s in num if s != []]
    return tuple(num)

def commandhistory(path):
    """
    Returns a list of command history (manipulations or transformations)
    performed on MRtrix file format .mif

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    list of str
        command history of input file
    """
    if not op.exists(path):
        raise OSError('Input path does not exist. Please ensure that the '
                      'folder or file specified exists.')
    ftype = format(path)
    if ftype != 'MRtrix':
        raise IOError('This function only works with MRtrix (.mif) '
                      'formatted filetypes. Please ensure that the input '
                      'filetype meets this requirement')
    arg = ['mrinfo', '-property', 'command_history']
    arg.append(path)
    completion = subprocess.run(arg, stdout=subprocess.PIPE)
    if completion.returncode != 0:
        raise IOError('Input {} is not currently supported by '
                      'PyDesigner.'.format(path))
    # Remove new line delimiter
    console = str(completion.stdout).split('\\n')
    # Remove 'b'
    console[0] = console[0][1:]
    # Remove quotes
    console = [s.replace("'", "") for s in console]
    # Condense empty strings
    console = [s.replace('"', '') for s in console]
    # Remove empty strings form list
    console = list(filter(None, console))
    # Remove MRtrix3 version
    console = [re.sub(r'\([^)]*\)', '', s) for s in console]
    # Remove whitespace to the right of string
    console = [s.rstrip() for s in console]
    return list(console)

def dwscheme(path):
    """
    Returns a list of input DWI's diffusion weighting scheme

    Parameters
    ----------
    path: :obj: `str` 
        Path to input image or directory

    Returns
    -------
    list of float
        diffusion weighing scheme
    """
    if not op.exists(path):
        raise OSError('Input path does not exist. Please ensure that the '
                      'folder or file specified exists.')
    ftype = format(path)
    if ftype != 'MRtrix':
        raise IOError('This function only works with MRtrix (.mif) '
                      'formatted filetypes. Please ensure that the input '
                      'filetype meets this requirement')
    arg = ['mrinfo', '-dwgrad']
    arg.append(path)
    completion = subprocess.run(arg, stdout=subprocess.PIPE)
    if completion.returncode != 0:
        raise IOError('Input {} is not currently supported by '
                      'PyDesigner.'.format(path))
    # Remove new line delimiter
    console = str(completion.stdout).split('\\n')
    # Remove 'b'
    console[0] = console[0][1:]
    # Remove quotes
    console = [s.replace("'", "") for s in console]
    # Condense empty strings
    console = [s.replace('"', '') for s in console]
    # Remove empty strings form list
    console = list(filter(None, console))
    # Convert list of strings to float
    dw_scheme = []
    for idx_a, line in enumerate(console):
        nums = []
        for idx_b, num in enumerate(line.split()):
            nums.append(float(num))
        dw_scheme.append(nums)
    return dw_scheme

def pescheme(path):
    """
    Returns a list of phase encoding scheme. If len(pescheme) > 1,
    the .mif DWI contains more than one directons

    Parameters
    ----------
    path: :obj: `str`
        Path to input image or directory

    Returns
    -------
    nPE: int
        Number of PE directions
    PE: int or list of int
        Phase encoding direction(s)
    """
    if not op.exists(path):
        raise OSError('Input path does not exist. Please ensure that the '
                      'folder or file specified exists.')
    ftype = format(path)
    if ftype != 'MRtrix':
        raise IOError('This function only works with MRtrix (.mif) '
                      'formatted filetypes. Please ensure that the input '
                      'filetype meets this requirement')
    arg = ['mrinfo', '-petable']
    arg.append(path)
    completion = subprocess.run(arg, stdout=subprocess.PIPE)
    if completion.returncode != 0:
        raise IOError('Input {} is not currently supported by '
                      'PyDesigner.'.format(path))
    # Remove new line delimiter
    console = str(completion.stdout).split('\\n')
    # Remove 'b'
    console[0] = console[0][1:]
    # Remove quotes
    console = [s.replace("'", "") for s in console]
    # Condense empty strings
    console = [s.replace('"', '') for s in console]
    # Remove empty strings form list
    console.remove('')
    # Convert list of strings to float
    pe_scheme = []
    for idx_a, line in enumerate(console):
        nums = []
        for idx_b, num in enumerate(line.split()):
            nums.append(float(num))
        pe_scheme.append(nums)
    return pe_scheme
