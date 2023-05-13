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
    input : str
        Path to nifti file

    Returns
    -------
    hdr : class
        Nibabel class object
    img : ndarray
        3D or 4D array containing the loaded nifti image
    
    """
    hdr = nib.load(input)
    img = np.array(hdr.dataobj)
    return hdr, img

def writenii(hdr, img, output):
    """
    Write nupy array to nifti file

    Parameters
    ----------
    hdr : class
        Nibabel class object
    img : ndarray
        3D or 4D array containing the image array
    output : str
        Path to save file as

    None; writes out file
    """
    struct = nib.Nifti1Image(img, hdr.affine, hdr.header)
    nib.save(struct, output)

def median(input, output, mask=None):
    """
    Applies median filtering to input nifti file

    Parameters
    ----------
    input : str
        Path to input nifti file
    output : str
        Path to output nifti file
    mask : str, optional
        Path to brainmask nifti file (Default: None)

    Returns
    -------
    None; writes out file
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
