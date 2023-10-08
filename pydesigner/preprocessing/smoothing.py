#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np

# import scipy as sc
from scipy.ndimage import gaussian_filter


def smooth_image(dwiname, csfname=None, outname="dwism.nii", width=1.25, size=5):
    """
    Smooths a DWI dataset

    Parameters
    ----------
    dwiname : str
        Filename of image to be smoothed
    csfname : str, optional
        Filename of CSF mask
    outname : str
        Filename to be written out
    width : float, optional
        The full width half max in voxels to be smoothed. Default: 1.25
    size : int
        The size of 2D Gaussian kernel [size, size]. Default: 5

    Returns
    -------
    None; writes out file

    See Also
    --------
    smooth(dwi, csfmask=None, width=1.25) is wrapped by this function
    """
    if csfname is None:
        print("Running smoothing at FWHM = {}...".format(width))
    else:
        print("Running CSF-excluded smoothing at FWHM = {}...".format(width))

    dwiimg = nib.load(dwiname)

    if csfname is not None:
        csfimg = nib.load(csfname)
        smoothed = smooth(dwiimg, csfmask=csfimg, width=width)
    else:
        smoothed = smooth(dwiimg, width=width, size=size)

    newimg = nib.Nifti1Image(smoothed, dwiimg.affine, dwiimg.header)
    nib.save(newimg, outname)

    return


def smooth(dwi, csfmask=None, width=1.25, size=5):
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
    size : int, optional
        The size of 2D Gaussian kernel [size, size]. Default: 5

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
        raise ValueError("Input dwi dataset is not 4-D")

    dwidata = dwi.get_fdata()
    if csfmask:
        csfdata = csfmask.get_fdata().astype("bool")

    if csfmask is None:
        smoothed = dwidata.copy()
        for i in range(dwi.shape[-1]):
            for z in range(dwi.shape[-2]):
                currslice = dwidata[:, :, z, i]
                smoothed[:, :, z, i] = nansmooth(currslice, width, size=size)
    else:
        bgmask = np.isnan(dwidata)
        smoothed = dwidata.copy()
        for i in range(dwi.shape[-1]):
            for z in range(dwi.shape[-2]):
                currslice = dwidata[:, :, z, i]
                currcsf = csfdata[:, :, z]
                wmgm = currslice.copy()
                wmgm[currcsf] = np.nan
                wmgm_ = nansmooth(wmgm, width, size=size)
                csf = currslice.copy()
                csf[np.logical_not(currcsf)] = np.nan
                csf_ = nansmooth(csf, width, size=size)
                total = np.nansum(np.dstack((wmgm_, csf_)), 2)
                smoothed[:, :, z, i] = total
        smoothed[bgmask] = np.nan
    return smoothed


def nansmooth(imgslice, fwhm, size=5):
    """Smooths an image slice while ignoring NaNs

    Parameters
    ----------
    imgslice : (X x Y) img_like or array_like object
        2D image to be smoothed
    fwhm : float
        The full width half max to be used for smoothing
    size : int, optional
        The size of 2D Gaussian kernel [size, size]. Default: 5

    Returns
    -------
    gauss : (X x Y) array_like object
        2D smoothed image

    Notes
    -----
    Intensity is only shifted between not-nan pixels and is hence
    conserved. The intensity redistribution with respect to each
    single point is done by the weights of available pixels according
    to a gaussian distribution. All nans in imgslice, stay nans in gauss.
    This approach is to spead the intesity of each point by a gaussian
    filter. The intensity, which is mapped to nan-pixels is reshifted
    back to the origin. If this maskes sense, depends on the
    application. I have a closed area surronded by nans and want to
    preseve the total intensity + avoid distortions at the boundaries.

    Solution adapted from https://stackoverflow.com/a/61481246
    """
    # Scipy requires standard deviation rather than FWHM
    stddev = fwhm / np.sqrt(8 * np.log(2))

    # Scipy required truncate instead of size
    truncate = (((size - 1) / 2) - 0.5) / stddev

    nan_msk = np.isnan(imgslice)

    loss = np.zeros(imgslice.shape)
    loss[nan_msk] = 1
    loss = gaussian_filter(loss, sigma=stddev, mode="constant", truncate=truncate)

    gauss = imgslice.copy()
    gauss[nan_msk] = 0
    gauss = gaussian_filter(gauss, sigma=stddev, mode="constant", truncate=truncate)
    gauss[nan_msk] = np.nan

    gauss += loss * imgslice

    return gauss
