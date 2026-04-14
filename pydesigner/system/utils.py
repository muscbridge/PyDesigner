import nibabel as nib
import numpy as np

import nibabel as nib
import numpy as np

def vectorize(img, mask) -> np.ndarray[float]:
    """Vectorize or unpatch an image using a 3D mask.

    Behavior
    --------
    - 1D input + 3D mask  -> reconstruct 3D image
    - 2D input + 3D mask  -> reconstruct 4D image
    - 3D input            -> vectorize to 1D using mask
    - 4D input            -> vectorize to 2D using mask

    Parameters
    ----------
    img : ndarray
        1D, 2D, 3D or 4D image array.
    mask : ndarray or None
        3D boolean/image mask. Required for 1D/2D reconstruction.
        Optional for 3D/4D vectorization.

    Returns
    -------
    ndarray
        Vectorized or reconstructed image.
    """
    img = np.asarray(img)

    # ------------------------------------------------------------
    # Reconstruct 1D -> 3D
    # ------------------------------------------------------------
    if img.ndim == 1:
        if mask is None:
            raise ValueError(
                "vectorize(): mask is required when reconstructing a 1D array into 3D."
            )
        mask = np.asarray(mask).astype(bool)

        s = np.zeros(mask.shape, order="F", dtype=img.dtype)
        s[mask] = img
        return np.squeeze(s)

    # ------------------------------------------------------------
    # Reconstruct 2D -> 4D
    # Expect img shape = (N, nvox)
    # ------------------------------------------------------------
    if img.ndim == 2:
        if mask is None:
            raise ValueError(
                "vectorize(): mask is required when reconstructing a 2D array into 4D."
            )
        mask = np.asarray(mask).astype(bool)

        n = img.shape[0]
        s = np.zeros(mask.shape + (n,), order="F", dtype=img.dtype)
        for i in range(n):
            s[..., i][mask] = img[i, :]
        return np.squeeze(s)

    # ------------------------------------------------------------
    # For 3D / 4D vectorization, create default full mask if needed
    # ------------------------------------------------------------
    if mask is None:
        mask = np.ones(img.shape[:3], order="F", dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)

    # ------------------------------------------------------------
    # Vectorize 3D -> 1D
    # ------------------------------------------------------------
    if img.ndim == 3:
        maskind = np.ma.array(img, mask=np.logical_not(mask), dtype=img.dtype, order="F")
        s = np.ma.compressed(maskind)
        return np.squeeze(s)

    # ------------------------------------------------------------
    # Vectorize 4D -> 2D
    # Output shape = (N, nvox)
    # ------------------------------------------------------------
    if img.ndim == 4:
        s = np.zeros((img.shape[-1], int(np.sum(mask))), order="F", dtype=img.dtype)
        for i in range(img.shape[-1]):
            tmp = img[:, :, :, i]
            maskind = np.ma.array(tmp, mask=np.logical_not(mask))
            s[i, :] = np.ma.compressed(maskind)
        return np.squeeze(s)

    raise ValueError(f"vectorize(): unsupported input with shape {img.shape}")


# def vectorize(img, mask) -> np.ndarray[float]:
#     """Returns vectorized image based on brain mask, requires no input
#     parameters
#     If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
#     If the input is 3D or 4D, vectorize it using a mask
#     Classification: Function

#     Parameters
#     ----------
#     img : ndarray
#         1D, 2D, 3D or 4D image array to vectorize
#     mask : ndarray
#         3D image array for masking

#     Returns
#     -------
#     vec : N X number_of_voxels vector or array, where N is the number
#         of DWI volumes

#     Examples
#     --------
#     vec = vectorize(img) if there's no mask
#     vec = vectorize(img, mask) if there's a mask
#     """
#     if mask is None:
#         mask = np.ones((img.shape[0], img.shape[1], img.shape[2]), order="F", dtype=img.dtype)
#     mask = mask.astype(bool)
#     if img.ndim == 1:
#         n = img.shape[0]
#         s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), order="F", dtype=img.dtype)
#         s[mask] = img
#     if img.ndim == 2:
#         n = img.shape[0]
#         s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], n), order="F", dtype=img.dtype)
#         for i in range(0, n):
#             s[mask, i] = img[i, :]
#     if img.ndim == 3:
#         maskind = np.ma.array(img, mask=np.logical_not(mask), dtype=img.dtype, order="F")
#         s = np.ma.compressed(maskind)
#     if img.ndim == 4:
#         s = np.zeros((img.shape[-1], np.sum(mask).astype(int)), order="F", dtype=img.dtype)
#         for i in range(0, img.shape[-1]):
#             tmp = img[:, :, :, i]
#             # Compressed returns non-masked area, so invert the mask first
#             maskind = np.ma.array(tmp, mask=np.logical_not(mask))
#             s[i, :] = np.ma.compressed(maskind)
#     return np.squeeze(s)


def writeNii(map, hdr, outDir, range=None, clip=False) -> None:
    """Write clipped NifTi images

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
    clip: bool
        Clip and apply values specified in range
        (Default: False)

    Returns
    -------
    None; writes out file

    Examples
    --------
    writeNii(matrix, header, output_directory, [0, 2], clip=True)

    See Also
    --------
    clipImage(img, range) : this function is wrapped around
    """
    if clip and not range:
        raise Exception("Range is required in order to clip an image.")
    if not clip:
        clipped_img = nib.Nifti1Image(map, hdr.affine, hdr.header)
    else:
        clipped_img = clipImage(map, range)
        clipped_img = nib.Nifti1Image(clipped_img, hdr.affine, hdr.header)
    nib.save(clipped_img, outDir)


def clipImage(img, range) -> np.ndarray[float]:
    """Clips input matrix within desired range. Min and max values are
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
    clippedImage = clipImage(image, [0, 3])
    Clips input matrix in the range 0 to 3
    """
    img[img > range[1]] = range[1]
    img[img < range[0]] = range[0]
    return img


def highprecisionexp(array, maxp=1e32) -> np.ndarray[float]:
    """Prevents overflow warning with numpy.exp by assigning overflows
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
    except:  # noqa: E722
        ans = np.full(array.shape, maxp)
    np.seterr(**defaultErrorState)
    return ans


def highprecisionpower(x1, x2, maxp=1e32) -> np.ndarray[float]:
    """Prevents overflow warning with numpy.powerr by assigning overflows
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
    except:  # noqa: E722
        ans = np.full(x1.shape, maxp)
    np.seterr(**defaultErrorState)
    return ans
