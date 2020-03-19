#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import scipy as sc
from scipy import ndimage

def smooth_image(dwiname, csfname=None, outname='dwism.nii', width=1.2):
    """
    Smooths a DWI dataset

    Parameters
    ----------
    dwiname : str
        Filename of image to be smoothed
    csfname : str
        Filename of CSF mask
    outname : str
        Filename to be written out
    width : float
        The full width half max in voxels to be smoothed. Default: 1.25

    Returns
    -------
    None; writes out file

    See Also
    --------
    smooth(dwi, csfmask=None, width=1.25) is wrapped by this function
    """

    print('Running smoothing at FWHM = {}...'.format(width))

    dwiimg = nib.load(dwiname)

    if csfname is not None:
        csfimg = nib.load(csfname)
        smoothed = smooth(dwiimg, csfmask=csfimg, width=width)
    else:
        smoothed = smooth(dwiimg, width=width)

    newimg = nib.Nifti1Image(smoothed, dwiimg.affine, dwiimg.header)
    nib.save(newimg, outname)

    return

def smooth(dwi, csfmask=None, width=1.25):
    """
    Smooths a DWI dataset

    Parameters
    ----------
    dwi : (X x Y x Z x N) img_like object
        Image to be smoothed, where N is the number of volumes in
        the DWI acquisition.
    csfmask : (S) img_like object
        The mask of CSF that will be applied to each volume in DWI
    width : float, optional
        The full width half max in voxels to be smoothed. Default: 1.25
       
    Returns
    -------
    smoothed : (X x Y x Z x N) array_like or img_like object
        The smoothed version of dwi

    Notes
    -----
    This is done mainly to reduce the Gibbs ringing. It might be
    recommended to only smooth the high SNR (or b-valued) data in order
    not to alter the Rice distribution of the low SNR data. This is
    important to maintain the high accuracy of WLLS. If a CSF mask is 
    given as an additional argument, CSF infiltration in microstructural
    signal is avoided during smoothing.
    """

    if dwi.ndim != 4:
        raise ValueError('Input dwi dataset is not 4-D')

    dwidata = dwi.get_fdata()
    if csfmask:
        csfdata = csfmask.get_fdata()

    if csfmask is not None:
        for i in range(dwi.shape[-1]):
            newvol = dwidata[:,:,:,i]
            newvol[csfdata > 0] = np.nan
            dwidata[:,:,:,i] = newvol

    smoothed = dwidata
    for i in range(dwi.shape[-1]):
        for z in range(dwi.shape[-2]):
            currslice = dwidata[:,:,z,i]
            smoothed[:,:,z,i] = nansmooth(currslice, width)

    return smoothed

def nansmooth(imgslice, fwhm):
    """
    Smooths an image slice while ignoring NaNs

    Parameters
    ----------
    imgslice : (X x Y) img_like or array_like object
        2D image to be smoothed
    fwhm : float
        The full width half max to be used for smoothing

    Returns
    -------
    smoothslice : (X x Y) array_like object
        2D smoothed image

    Notes
    -----
    This is done because a masked dataset will contain NaNs. In typical
    operations and filtering, the NaNs will propagate instead of being
    ignored (which is the desired behavior). During runtime, divide by 0
    warnings are suppressed due to the high probability of its occuring.
    The operation to avoid this is as follows:
    """

    # Scipy requires standard deviation rather than FWHM
    stddev = fwhm / np.sqrt(8 * np.log(2))

    # Auxilary matrices
    V = imgslice.copy()
    V[np.isnan(imgslice)] = 0
    VV = sc.ndimage.gaussian_filter(V, sigma=stddev)

    W = 0*imgslice.copy()+1
    W[np.isnan(imgslice)] = 0
    WW = sc.ndimage.gaussian_filter(W, sigma=stddev)

    # Temporarily ignore divide by 0 errors while doing the math
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothedslice = VV / WW

    return smoothedslice
