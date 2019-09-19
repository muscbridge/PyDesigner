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

def prealign(d4img, outname, force=False, resume=False):
    """Takes a 4D nifti and aligns them with affine registration

    Parameters
    ----------
    d4img :obj: `str`
        The filename of the 4D image to align to itself
    outname :obj: `str`
        The filename of the final prealigned file
    force :bool:
        Whether to force overwrites of files. Default: True
    resume :bool:
        Whether to allow resuming from existing files

    Returns
    -------
    None
    """

    # Attempt to load file
    try:
        img = nib.load(d4img)
    except FileNotFoundError:
        raise Exception('Cannot find file '+d4img)
    except nibabel.filebasedimages.ImageFileError:
        raise Exception('Cannot read file '+d4img)

    fsize = img.header.get_data_shape()
    ndims = len(fsize)
    if ndims < 3:
        # Error: we can't handle slices
        raise Exception('File '+d4img+'has less than three dimensions')
    elif ndims == 3:
        # We can't do anything to align this
        return
    elif ndims > 4:
        raise Exception('File '+d4img+'has '+ndims+' dimensions, '
                        'expected 4.')

    # Check for existence of output
    if op.exists(outname):
        print('exists')
        if resume:
            # already done, skip
            return
        elif not force:
            # Error: we can neither resume nor overwrite
            raise Exception('Performing prealignment would cause an '
                            'overwrite. Please delete '+outname+
                            ', use --force to overwrite, or use --resume '
                            'to continue with the existing file.')

    # Make our outpath clear
    outpath = op.dirname(outname)
    tmpdir = op.join(outpath, 'TMP_PREALIGN')

    # Make temporary workspace
    os.mkdir(tmpdir)
    
    # Split up the files
    split_args = ['fslsplit', d4img, op.join(tmpdir, 'vol_'), '-t']
    completion = subprocess.run(split_args)
    if completion.returncode != 0:
        raise Exception('fslsplit failed during prealign with return code '
                        +completion.returncode+
                        'Contact developers for assistance at '
                        'https://github.com/m-ama/PyDesigner/issues')

    # Begin aligning volumes to first
    splitfiles = os.listdir(tmpdir)
    alignvol = 'vol_0000.nii.gz'
    for f in splitfiles:
        if f == alignvol:
            continue
        else:
            flirt_args = ['flirt', '-in', op.join(tmpdir,f), '-ref', 
                    op.join(tmpdir,alignvol), '-out', op.join(tmpdir,f)]
            completion = subprocess.run(flirt_args)
            if completion.returncode != 0:
                raise Exception('flirt alignment failed during prealign '
                                'with return code '
                                +completion.returncode+
                                'Contact developers for assistance at '
                                'https://github.com/m-ama/PyDesigner/issues')

    # Merge volumes into one

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
