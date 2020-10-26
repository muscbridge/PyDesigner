#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Tools for exporting DSIStudio-compatible outputs. Adapted from
mattcieslak/dmri_convert/mrtrix_to_dsistudio.py to suit PyDesigner's
needs.

References:
    1. https://github.com/mattcieslak/dmri_convert/blob/master/mrtrix_to_dsistudio.py
"""


import os
import os.path as op
import subprocess
import numpy as np
import nibabel as nib
from dipy.core.geometry import cart2sphere
from dipy.core.sphere import HemiSphere
from dipy.direction import peak_directions
from scipy.io.matlab import loadmat, savemat
from tqdm import tqdm
ODF_COLS = 20000  # Number of columns in DSI Studio odf split
tqdmWidth = 70

def get_dsi_studio_ODF_geometry(odf_key):
    """
    Reads DSIStudio's ODF geometry in odfs.mat

    Parameters
    ----------
    odf_keys : str
        DSIStudio's direction set to load

    Returns
    -------
    odf_vertices : array_like(dtype=int)
        ODF vertices
    odf_faces : array_like(dtype=double)
        ODF faces
    """
    working_dir = os.path.abspath(os.path.dirname(__file__))
    m = loadmat(op.join(working_dir, 'odfs.mat'))
    odf_vertices = m[odf_key + '_vertices'].T
    odf_faces = m[odf_key + '_faces'].T
    return odf_vertices, odf_faces

def convertLPS(input, output):
    """
    Converts a nifti file to LPS for compatibility with DSIStudio

    Parameters
    ----------
    input : str
        Path to input nifti file; must possess .nii extension
    output : str
        Path to output nifti file; must possess .nii extension

    Returns
    -------
    None, writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if op.splitext(input)[-1] != '.nii':
        raise IOError('Input file needs to specified as a NifTI '
        '(.nii)')
    if op.splitext(output)[-1] != '.nii':
        raise IOError('Output file needs to specified as a NifTI '
        '(.nii)')
    arg = [
        'mrconvert',
        '-quiet',
        '-force',
        input,
        '-strides', '-1,-2,3',
        output
    ]
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('Conversion of NifTI file to LPS failed. '
        'Check above for errors.')

def makefib(input, output, mask=None, n_fibers=3):
    """
    Converts a NifTi ``.nii`` file containing sh coefficients to a DSI
    Studio fib file

    This function uses ``sh2amp`` to get amplitude values for each
    direction in DSI Studio's ``odf8`` direction set. These values are
    masked and loaded into the "odfN" matrices in the fib file.

    Parameters
    ----------
    input : str
        Path to input nifti file containing SH coefficients
    output : str
        Path to output fib file; must end with .fib
    mask : str, optional
        Path to nifti file containing brain mask
    n_fibers : int, optional
        The maximum number ODF maxima to extract per voxel
        (Default: 3)

    Returns
    -------
    None, writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if op.splitext(input)[-1] != '.nii':
        raise IOError('Input file needs to specified as a NifTI '
        '(.nii)')
    if op.splitext(output)[-1] != '.fib':
        raise IOError('Output file needs to specified as a .fib file')
    if not mask is None:
        if not op.exists(mask):
            raise OSError('Path to brain mask does not exist. Please '
            'ensure that the folder specified exists.')
    outdir = op.dirname(output)
    # Convert to LPS
    fname, ext = op.splitext(op.basename(input))
    fname = fname +'_lps'
    input_ = op.join(outdir, fname + ext)
    if not mask is None:
        (fname, ext) = op.splitext(op.basename(mask))
        fname = fname +'_lps'
        mask_ = op.join(outdir, fname + ext)
    convertLPS(input, input_)
    convertLPS(mask, mask_)
    # Get ODF geometry
    verts, faces = get_dsi_studio_ODF_geometry("odf8")
    num_dirs, _ = verts.shape
    hemisphere = num_dirs // 2
    x, y, z = verts[:hemisphere].T
    hs = HemiSphere(x=x, y=y, z=z)
    # Convert to DSI Studio LPS+ from MRTRIX3 RAS+
    _, theta, phi = cart2sphere(-x, -y, z)
    dirs_txt = op.join(outdir, "directions.txt")
    np.savetxt(dirs_txt, np.column_stack([phi, theta]))
    # Get SH amplitude
    odf_amplitudes_nii = op.join(outdir, 'amplitudes.nii')
    arg = [
        'sh2amp',
        '-quiet',
        '-force',
        '-nonnegative',
        input_, dirs_txt, odf_amplitudes_nii
    ]
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('Failed to determine amplitude of SH '
        'coefficients. Check above for errors.')
    # Load images
    amplitudes_img = nib.load(odf_amplitudes_nii)
    ampl_data = amplitudes_img.get_fdata()
    if not mask is None:
        mask_img = nib.load(mask_)
        if not np.allclose(mask_img.affine, amplitudes_img.affine):
            raise ValueError('Differing orientation between mask and '
            'amplitudes.')
        if not mask_img.shape == amplitudes_img.shape[:3]:
            raise ValueError('Differing grid between mask and '
            'amplitudes')
        mask_img = mask_img.get_fdata()
    else:
        mask_img = np.ones((ampl_data.shape[0], ampl_data.shape[1],
        ampl_data.shape[2]), order='F')
    # Make flat mask
    flat_mask = mask_img.flatten(order='F') > 0
    odf_array = ampl_data.reshape(-1, ampl_data.shape[3], order="F")
    masked_odfs = odf_array[flat_mask, :]
    n_odfs = masked_odfs.shape[0]
    peak_indices = np.zeros((n_odfs, n_fibers))
    peak_vals = np.zeros((n_odfs, n_fibers))
    dsi_mat = {}
    # Create matfile that can be read by dsi Studio
    dsi_mat['dimension'] = np.array(amplitudes_img.shape[:3])
    dsi_mat['voxel_size'] = np.array(amplitudes_img.header.get_zooms()[:3])
    n_voxels = int(np.prod(dsi_mat['dimension']))
    for odfnum in tqdm(range(masked_odfs.shape[0]),
        desc='ODF Peak Detection',
        bar_format='{desc}: [{percentage:0.0f}%]',
        unit='vox',
        ncols=tqdmWidth):
        dirs, vals, indices = peak_directions(masked_odfs[odfnum], hs)
        for dirnum, (val, idx) in enumerate(zip(vals, indices)):
            if dirnum == n_fibers:
                break
            peak_indices[odfnum, dirnum] = idx
            peak_vals[odfnum, dirnum] = val
    for nfib in range(n_fibers):
        # fill in the "fa" values
        fa_n = np.zeros(n_voxels)
        fa_n[flat_mask] = peak_vals[:, nfib]
        dsi_mat['fa%d' % nfib] = fa_n.astype(np.float32)
        # Fill in the index values
        index_n = np.zeros(n_voxels)
        index_n[flat_mask] = peak_indices[:, nfib]
        dsi_mat['index%d' % nfib] = index_n.astype(np.int16)
    # Add in the ODFs
    num_odf_matrices = n_odfs // ODF_COLS
    split_indices = (np.arange(num_odf_matrices) + 1) * ODF_COLS
    odf_splits = np.array_split(masked_odfs, split_indices, axis=0)
    for splitnum, odfs in enumerate(odf_splits):
        dsi_mat['odf%d' % splitnum] = odfs.T.astype(np.float32)
    dsi_mat['odf_vertices'] = verts.T
    dsi_mat['odf_faces'] = faces.T
    dsi_mat['z0'] = np.array([1.])
    savemat(output, dsi_mat, format='4', appendmat=False)
    # Remove unwanted files
    os.remove(input_)
    os.remove(mask_)
    os.remove(dirs_txt)
    os.remove(odf_amplitudes_nii)
