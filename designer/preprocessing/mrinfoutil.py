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
    Fetches the console output of mrinfo

    Parameters
    ----------
    path:   string
            path to input image or directory
    flag:   string
            flag to pass onto mrinfo

    Returns
    -------
    String, information for flag
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
    Returns the file format of DWI at path

    Parameters
    ----------
    path:   string
            path to input image or directory

    Returns
    -------
    String indicating format
    """
    type = getconsole(path, '-format')
    return type

def ndim(path):
    """
    Returns the number of image dimensions

    Parameters
    ----------
    path:   string
            path to input image or directory

    Returns
    -------
    Int, number of dimensions in image
    """
    num = getconsole(path, '-ndim')
    return int(num)

def size(path):
    """
    Returns the size of image along each axis

    Parameters
    ----------
    path:  string
            path to input image or directory

    Returns
    -------
    Int tuple, number of voxels in [X, Y, Z, B-value]
    """
    num = getconsole(path, '-size').split()
    num = tuple(map(int, num))
    return num

def spacing(path):
    """
    Returns the voxel spacing along each image dimension

    Parameters
    ----------
    path:  string
            path to input image or directory

    Returns
    -------
    Int tuple, number of spacing between voxels [X, Y, Z, B-value]
    """
    num = getconsole(path, '-spacing').split()
    num = tuple(map(float, num))
    return num

def datatype(path):
    """
    Returns the data type used for image storage

    Parameters
    ----------
    path:   string
            path to input image or directory

    Returns
    -------
    Str, mrtrix3 datatypes
    """
    return getconsole(path, '-datatype')

def strides(path):
    """
    Returns data strides

    Parameters
    ----------
    path:   string
            path to input image or directory

    Returns
    -------
    Int tuple, mrtrix3 strides
    """
    num = getconsole(path, '-strides').split()
    num = tuple(map(int, num))
    return num

def offset(path):
    """
    Returns the image intensity offset

    Parameters
    ----------
    path:   string
            path to input image or directory

    Returns
    -------
    Flaot, image intensity offset
    """
    num = getconsole(path, '-offset')
    num = float(num)
    return num

def multiplier(path):
    """
    Returns the image intensity multiplier

    Parameters
    ----------
    path:   string
            path to input image or directory

    Returns
    -------
    Float, image intensity multiplier
    """
    num = getconsole(path, '-multiplier')
    num = float(num)
    return num

def transform(path):
    """
    Returns the 4-by-4 voxel to image transformation matrix

    Parameters
    ----------
    path:   string
            path to input image or directory

    Returns
    -------
    Tuple float list
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
   path:   string
           path to input image or directory

   Returns
   -------
   Tuple
   """
    if not op.exists(path):
        raise OSError('Input path does not exist. Please ensure that the '
                      'folder or file specified exists.')
    arg = ['mrinfo', '-property', 'command_history']
    arg.append(path)
    completion = subprocess.run(arg, stdout=subprocess.PIPE)
    if completion.returncode != 0:
        raise IOError('Input {} is not currently supported by '
                      'PyDesigner.'.format(path))
    # Remove new line delimiter
    console = str(completion.stdout).split('\\n')
    # Remove 'b'
    console = [s.split('b')[-1] for s in console]
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
    return tuple(console)
