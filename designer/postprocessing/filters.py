#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
This module contains filter(s) for postprocessing DTI/DKI maps
"""
#---------------------------------------------------------------------
# Package Management
#---------------------------------------------------------------------
import os.path as op
import numpy as np
from scipy.ndimage import median_filter, generate_binary_structure
import nibabel as nib

#---------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------
def readnii(input):
    """
    Reads nifti files and returns header and numpy data array

    Parameters
    ----------
    input: path to nifti fule

    Returns
    -------
    img:    numpy array
    hdr:    image header
    """
    hdr = nib.load(input)
    img = np.array(hdr.dataobj)
    return hdr, img

def writenii(hdr, img, output):
    """
    Write nupy array to nifti file

    Parameters
    ----------
    hdr:    image header
    img:    numpy array
    output: path to save file as
    """
    struct = nib.Nifti1Image(img, hdr.affine, hdr.header)
    nib.save(struct, output)

def median(input, output, mask=None):
    """
    Applies median filtering to input nifti file

    Parameters
    ----------
    input:  path to input nifti file
    output: path to output nifti file
    mask:   path to brainmask nifti file
            default: None

    Returns
    -------
    written to drive
    """
    if not op.exists(input):
        raise IOError('Input file {} does not exist.'.format(input))
    hdr, img = readnii(input)
    if mask is not None:
        if not op.exists(mask):
            raise IOError('Input mask {} does not '
                          'exist.'.format(input))
        maskhdr, mask = readnii(mask)
    else:
        mask = np.ones((img.shape[0], img.shape[1], img.shape[2]),
                        order='F')
    mask.astype(bool)
    conn = generate_binary_structure(rank=3, connectivity=1)
    if np.ndim(img) == 4:
        for i in range(img.shape[-1]):
            img[:, :, :, i] = median_filter(img[:, :, :, i],
                                                footprint=conn,
                                                mode='constant',
                                                cval=float('nan')) \
                                                    * mask
    elif np.ndim(img) == 3:
        img = median_filter(img, 
                            footprint=conn,
                            mode='constant', cval=float('nan')) \
                                * mask
    else:
        raise Exception('Input nifti image needs to be either 3D or '
                        '4D. Please check the file provided.')
    writenii(hdr, img, output)
