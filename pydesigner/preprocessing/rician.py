#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import scipy as sc
from scipy import ndimage
import os.path as op

def rician_img_correct(dwiname, noisemapname, outpath=None):
    """
    Performs Rician correction on a dataset with a noisemap

    Parameters
    ----------
    dwiname : str
        Filename of image to be corrected
    noisemapname : str
        Filename of noisemap to use for correction
    outpath : str
        Path to put resulting file

    Returns
    -------
    None; writes out file

    See Also
    --------
    rician_correct(dwi, noise) is wrapped by this function
    """

    print('Running Rician correction...')

    # load files
    dwiimg = nib.load(dwiname)
    noiseimg = nib.load(noisemapname)

    # run the correction
    corrected = rician_correct(dwiimg.get_fdata(), noiseimg.get_fdata())

    # create Nifti-1 in memory
    newimg = nib.Nifti1Image(corrected, dwiimg.affine, dwiimg.header)

    # determine the output name
    path = op.dirname(outpath)
    print('Path: ' + path)
    [name, ext] = op.splitext(op.basename(outpath))
    print('Name: ' + name)

    if not name:
        name = 'rdwi.nii'
    else:
        name += '.nii'
    out = op.join(path, name)
    print('Full: ' + out)

    nib.save(newimg, out)

    return

def rician_correct(dwi, noisemap):
    """
    Smooths a DWI dataset

    Parameters
    ----------
    dwi : (X x Y x Z x N) img_like object
        Image to be corrected, where N is the number of volumes in
        the DWI acquisition.
    csfmask : (X x Y x Z x N) img_like object
        The noise map from dwidenoise
    width : float, optional
        The full width half max in voxels to be smoothed. Default: 1.25
       
    Returns
    -------
    corrected : (X x Y x Z x N) array_like or img_like object
        The rician-corrected version of dwi
    """

    # Replace NaN with 0
    minZero = 1e-8
    # dwi
    nanidx = np.isnan(dwi)
    dwi[nanidx] = minZero
    # noise
    nanidx = np.isnan(noisemap)
    noisemap[nanidx] = minZero

    sqr_noise = np.square(noisemap)
    sqr_data = np.square(dwi)
    difference = np.zeros(sqr_data.shape)
    for i in range(len(sqr_data[1,1,1,:])):
        difference[:,:,:,i] = sqr_noise - sqr_data[:,:,:,i]
    result = np.sqrt(np.absolute(difference))

    return result
