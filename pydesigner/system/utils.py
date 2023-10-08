#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np


def vectorize(img, mask) -> np.ndarray[float]:
    """
    Returns vectorized image based on brain mask, requires no input
    parameters
    If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
    If the input is 3D or 4D, vectorize it using a mask
    Classification: Function

    Parameters
    ----------
    img : ndarray
        1D, 2D, 3D or 4D image array to vectorize
    mask : ndarray
        3D image array for masking

    Returns
    -------
    vec : N X number_of_voxels vector or array, where N is the number
        of DWI volumes

    Examples
    --------
    vec = vectorize(img) if there's no mask
    vec = vectorize(img, mask) if there's a mask
    """
    if mask is None:
        mask = np.ones((img.shape[0], img.shape[1], img.shape[2]), order="F", dtype=img.dtype)
    mask = mask.astype(bool)
    if img.ndim == 1:
        n = img.shape[0]
        s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), order="F", dtype=img.dtype)
        s[mask] = img
    if img.ndim == 2:
        n = img.shape[0]
        s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], n), order="F", dtype=img.dtype)
        for i in range(0, n):
            s[mask, i] = img[i, :]
    if img.ndim == 3:
        maskind = np.ma.array(img, mask=np.logical_not(mask), dtype=img.dtype, order="F")
        s = np.ma.compressed(maskind)
    if img.ndim == 4:
        s = np.zeros((img.shape[-1], np.sum(mask).astype(int)), order="F", dtype=img.dtype)
        for i in range(0, img.shape[-1]):
            tmp = img[:, :, :, i]
            # Compressed returns non-masked area, so invert the mask first
            maskind = np.ma.array(tmp, mask=np.logical_not(mask))
            s[i, :] = np.ma.compressed(maskind)
    return np.squeeze(s)


def writeNii(map, hdr, outDir, range=None) -> None:
    """
    Write clipped NifTi images

    Parameters
    ----------
    map : ndarray(dtype=float)
        3D array to write
    header : class
        Nibabel class header file containing NifTi properties
    outDir : str
        Output file name
    range : array_like
        [1 x 2] vector specifying range to clip, inclusive of value
        in range, e.g. range = [0, 1] for FA map

    Returns
    -------
    None; writes out file

    Examples
    --------
    writeNii(matrix, header, output_directory, [0, 2])

    See Also
    clipImage(img, range) : this function is wrapped around
    """
    if range is None:
        clipped_img = nib.Nifti1Image(map, hdr.affine, hdr.header)
    else:
        clipped_img = clipImage(map, range)
        clipped_img = nib.Nifti1Image(clipped_img, hdr.affine, hdr.header)
    nib.save(clipped_img, outDir)


def clipImage(img, range) -> np.ndarray[float]:
    """
    Clips input matrix within desired range. Min and max values are
    inclusive of range
    Classification: Function

    Parameters
    ----------
    img : ndarray(dtype=float)
        Input 3D image array
    range : array_like
        [1 x 2] vector specifying range to clip

    Returns
    -------
    clippedImage:   clipped image; same size as img

    Examples
    --------
    clippedImage = clipImage(image, [0 3])
    Clips input matrix in the range 0 to 3
    """
    img[img > range[1]] = range[1]
    img[img < range[0]] = range[0]
    return img


def highprecisionexp(array, maxp=1e32) -> np.ndarray[float]:
    """
    Prevents overflow warning with numpy.exp by assigning overflows
    to a maxumum precision value
    Classification: Function

    Parameters
    ----------
    array : ndarray
        Array or scalar of number to run np.exp on
    maxp : float, optional
        Maximum preicison to assign if overflow (Default: 1e32)

    Returns
    -------
    exponent or max-precision

    Examples
    --------
    a = highprecisionexp(array)
    """
    np.seterr(all="ignore")
    defaultErrorState = np.geterr()
    np.seterr(over="raise", invalid="raise")
    try:
        ans = np.exp(array)
    except:
        ans = np.full(array.shape, maxp)
    np.seterr(**defaultErrorState)
    return ans


def highprecisionpower(x1, x2, maxp=1e32) -> np.ndarray[float]:
    """
    Prevents overflow warning with numpy.powerr by assigning overflows
    to a maxumum precision value
    Classification: Function

    Parameters
    ----------
    x1 : array_like
        The bases
    x2 : array_like
        The exponents
    maxp : float, optional
        Maximum preicison to assign if overflow (Default: 1e32)

    Returns
    -------
    x1 raised to x2 power, or max precision defined by maxp

    Examples
    --------
    a = highprecisionexp(array)
    """
    np.seterr(all="ignore")
    defaultErrorState = np.geterr()
    np.seterr(over="raise", invalid="raise")
    try:
        ans = np.power(x1, x2)
    except:
        ans = np.full(x1.shape, maxp)
    np.seterr(**defaultErrorState)
    return ans
