#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from matplotlib.collections import LineCollection

def plot(input, output, bval=None, mask=None):
    """
    Plots and saves the number of outliers in input 4D irlls output
    file as a PNG file

    Parameters
    ----------
    input:  str
        list of 4D DWI (nifti-format) paths to evaluate and plot
    output: str
        output path of plot in .ong format
    bval:   str
        path to relevant .bval file
    mask:   str
        path to brain mask in nifti format

    Returns
    -------
    none
        writes plot to path

    See Also
    --------
    motionplot: plots subject motion from eddy_qc output file
    snrplot:    plots DWI's SNR  
    """
    print('Plotting outliers...')
    if not op.exists(input):
        raise OSError('Input file {} does not exist'.format(input))
    if op.splitext(input)[-1] != '.nii':
        raise OSError('Input file {} is not nifti type'. format(input))
    if op.isdir(output):
        raise OSError('Output {} cannot be a directory. Please '
        'define the output to be an image file.'.format(output))
    if op.splitext(output)[-1] != '.png':
        raise OSError('Output path {} does not indicate a PNG file'
        ''. format(input))
    hdr = nib.load(input)
    img = np.array(hdr.dataobj)
    truncateIdx = np.isnan(img)
    img[truncateIdx] = 0
    dims = img.shape     # size per dimension
    nvox = dims[0] * dims[1] * dims[2]      # no. of voxels
    vols = dims[-1]     # number of volumes
    if np.ndim(img) != 4:
        raise Exception('Only 4D nifti files can be read. '
        'User-supplied file is not a 4D nifti.')
    if mask is not None:
        if op.exists(mask):
            if op.splitext(mask)[-1] != '.nii':
                raise OSError('Input maks {} is not nifti type '
                ''.format(mask))
            hdr_mask = nib.load(mask)
            bw = np.array(hdr_mask.dataobj)
        else:
            raise OSError('Mask path {} does not exist'.format(mask))
    else:
        bw = np.ones(dims[0:3], order='F')
    if bval is None:
        bvals = np.zeros(dims[-1], dtype=int)
    else:
        bvals = np.loadtxt(bval, dtype=int)
    # multiply mask by img
    for i in range(vols):
        img[:,:,:,i] = np.multiply(img[:,:,:,i], bw)
    # Create x-axis
    x = np.arange(start=1, stop=vols, step=1)
    # create y-axis
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = np.count_nonzero(img[:, :, :, i])
    # Normalize to percentage of voxels
    y = (y / np.count_nonzero(bw)) * 100
    # Plot
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    if bval:
        bvals = np.loadtxt(bval, dtype=int)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(bvals.min(), bvals.max())
        lc = LineCollection(segments, cmap='Set1', norm=norm, alpha=0.8)
        lc.set_array(bvals)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax)
        cbar.set_label('B-Value')
    else:
        plt.plot(x, y, "-", lw=0.5, color="black", alpha=0.8)
    plt.scatter(x, y, s=10, color="black", linewidths=0, alpha=0.90)
    plt.xlabel('Shell Number')
    plt.ylabel('Percentage of Outlier Voxels (%)')
    plt.xticks(rotation=45)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    if mask is None:
        plt.title('IRLLS Outlier Determination in DWI')
    else:
        plt.title('IRLLS Outlier Determination in DWI (Brain Masked)')
    plt.savefig(output, dpi=600)
