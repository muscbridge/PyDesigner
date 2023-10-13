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
from typing import Tuple

import nibabel as nib
import numpy as np
from dipy.core.geometry import cart2sphere
from dipy.core.sphere import HemiSphere
from dipy.direction import gfa, peak_directions
from scipy.io.matlab import loadmat, savemat
from tqdm import tqdm

from pydesigner.tractography import sphericalsampling

ODF_COLS = 20000  # Number of columns in DSI Studio odf split
tqdmWidth = 70


def get_dsi_studio_ODF_geometry(odf_key) -> Tuple[np.ndarray[int], np.ndarray[float]]:
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
    m = loadmat(op.join(working_dir, "odfs.mat"))
    odf_vertices = m[odf_key + "_vertices"].T
    odf_faces = m[odf_key + "_faces"].T
    return odf_vertices, odf_faces


def convertLPS(input, output) -> None:
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
        raise OSError("Input path does not exist. Please ensure that " "the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if op.splitext(input)[-1] != ".nii":
        raise IOError("Input file needs to specified as a NifTI " "(.nii)")
    if op.splitext(output)[-1] != ".nii":
        raise IOError("Output file needs to specified as a NifTI " "(.nii)")
    arg = ["mrconvert", "-quiet", "-force", input, "-strides", "-1,-2,3,4", output]
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception("Conversion of NifTI file to LPS failed. " "Check above for errors.")


def makefib(input, output, map=None, mask=None, n_fibers=5, scale=1, other_maps=None) -> None:
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
    map : str
        Path to dMRI map to control stopping criteria; usually an FA map. This
        map is stored under variable fa_n (n =  1, 2,..., n ) in .fib file.
        This maps is visualized by qa in DSI Studio. The peak amplitude response
        is used if no map is provided
    mask : str, optional
        Path to nifti file containing brain mask
    n_fibers : int, optional
        The maximum number ODF maxima to extract per voxel
        (Default: 3)
    scale: float; optional
        Affects overall size of odfs in .fib file. This only affects
        visualization.
        (Default: 1)
    other_maps : list of str; optional
        List containing paths to other metrics maps to include into the .fib
        file for computing metrics values along tracks. The maps could be AD,
        AK, MK, AWF, etc. as long as they are the same size as SH coefficient
        file.

    Returns
    -------
    None, writes out file
    """
    if not op.exists(input):
        raise OSError("Input path does not exist. Please ensure that " "the folder or file specified exists.")
    if not op.exists(op.dirname(output)):
        raise OSError(
            "Specifed directory for output file {} does not "
            "exist. Please ensure that this is a valid "
            "directory.".format(op.dirname(output))
        )
    if op.splitext(input)[-1] != ".nii":
        raise IOError("Input file needs to specified as a NifTI " "(.nii)")
    if op.splitext(output)[-1] != ".fib":
        raise IOError("Output file needs to specified as a .fib file")
    if map is not None:
        if not op.exists(map):
            raise OSError("Path to map image does not exist. Please " "ensure that the folder specified exists.")
    if mask is not None:
        if not op.exists(mask):
            raise OSError("Path to brain mask does not exist. Please " "ensure that the folder specified exists.")
    if isinstance(other_maps, list) or other_maps is None:
        if isinstance(other_maps, list):
            if any([not op.exists(x) for x in other_maps]):
                raise OSError("One of the paths defined in other maps does not " "exist Please ensure all files exist.")
    else:
        raise TypeError(
            "Path to other maps needs to be entered as a list of " "strings defining paths to other metric files."
        )
    outdir = op.dirname(output)
    # Convert to LPS
    fname, ext = op.splitext(op.basename(input))
    fname = fname + "_lps"
    input_ = op.join(outdir, fname + ext)
    convertLPS(input, input_)
    if map is not None:
        fname, ext = op.splitext(op.basename(map))
        fname = fname + "_lps"
        map_ = op.join(outdir, fname + ext)
        convertLPS(map, map_)
    if mask is not None:
        fname, ext = op.splitext(op.basename(mask))
        fname = fname + "_lps"
        mask_ = op.join(outdir, fname + ext)
        convertLPS(mask, mask_)
    # Get ODF geometry
    verts, faces = sphericalsampling.dsigrid("odf8")
    num_dirs, _ = verts.shape
    hemisphere = num_dirs // 2
    x, y, z = verts[:hemisphere].T
    hs = HemiSphere(x=x, y=y, z=z)
    # Convert to DSI Studio LPS+ from MRTRIX3 RAS+
    _, theta, phi = cart2sphere(-x, -y, z)
    dirs_txt = op.join(outdir, "directions.txt")
    np.savetxt(dirs_txt, np.column_stack([phi, theta]))
    # Get SH amplitude
    odf_amplitudes_nii = op.join(outdir, "amplitudes.nii")
    arg = [
        "sh2amp",
        "-quiet",
        "-force",
        "-nonnegative",
        input_,
        dirs_txt,
        odf_amplitudes_nii,
    ]
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception("Failed to determine amplitude of SH " "coefficients. Check above for errors.")
    # Load images
    amplitudes_img = nib.load(odf_amplitudes_nii)
    ampl_data = amplitudes_img.get_fdata()
    if mask is not None:
        mask_img = nib.load(mask_)
        if not np.allclose(mask_img.affine, amplitudes_img.affine):
            raise ValueError("Differing orientation between mask and " "amplitudes.")
        if not mask_img.shape == amplitudes_img.shape[:3]:
            raise ValueError("Differing grid between mask and " "amplitudes")
        mask_img = mask_img.get_fdata()
    else:
        mask_img = np.ones((ampl_data.shape[0], ampl_data.shape[1], ampl_data.shape[2]), order="F")
    # Make flat mask
    flat_mask = mask_img.flatten(order="F") > 0
    odf_array = ampl_data.reshape(-1, ampl_data.shape[3], order="F")
    masked_odfs = odf_array[flat_mask, :]
    masked_odfs = np.nan_to_num(masked_odfs)
    if map is not None:
        map_img = nib.load(map_)
        if not np.allclose(map_img.affine, amplitudes_img.affine):
            raise ValueError("Differing orientation between map image and " "amplitudes.")
        if not map_img.shape == amplitudes_img.shape[:3]:
            raise ValueError("Differing grid between map image and " "amplitudes")
        map_img = map_img.get_fdata().flatten(order="F")
        map_img[map_img < 0] = 0
        masked_map = map_img[flat_mask]

    # Scale ODF
    masked_odfs = scale * masked_odfs

    # Compute GFA
    masked_gfa = np.zeros(masked_odfs.shape[0])
    for vox in tqdm(
        range(masked_odfs.shape[0]),
        desc="Computing GFA",
        bar_format="{desc}: [{percentage:0.0f}%]",
        unit="vox",
        ncols=tqdmWidth,
    ):
        masked_gfa[vox] = gfa(masked_odfs[vox, :])

    # Remove voxels possibly containing errors in ODF calculations. This is
    # particularly important for volumes that were co-registered as it fixes
    # large amplitudes ODF amplitude at border voxels
    if map is not None:
        idx = np.logical_or(masked_map == 0, np.sum(masked_odfs, axis=1) == 0)
        masked_map[idx] = 0
    else:
        idx = np.sum(masked_odfs, axis=1) == 0
    masked_odfs[idx, :] = 0
    masked_gfa[idx] = 0
    if map is not None:
        masked_map[idx] = 0

    n_odfs = masked_odfs.shape[0]
    peak_indices = np.zeros((n_odfs, n_fibers), dtype=int)
    peak_vals = np.zeros((n_odfs, n_fibers))
    dsi_mat = {}
    # Create matfile that can be read by dsi Studio
    dsi_mat["dimension"] = np.array(amplitudes_img.shape[:3])
    dsi_mat["voxel_size"] = np.array(amplitudes_img.header.get_zooms()[:3])
    n_voxels = int(np.prod(dsi_mat["dimension"]))
    for odfnum in tqdm(
        range(masked_odfs.shape[0]),
        desc="ODF Peak Detection",
        bar_format="{desc}: [{percentage:0.0f}%]",
        unit="vox",
        ncols=tqdmWidth,
    ):
        dirs, vals, indices = peak_directions(masked_odfs[odfnum], hs)
        for dirnum, (val, idx) in enumerate(zip(vals, indices)):
            if dirnum == n_fibers:
                break
            peak_indices[odfnum, dirnum] = idx
            peak_vals[odfnum, dirnum] = val
    for nfib in range(n_fibers):
        # fill in the "fa" values
        fa_n = np.zeros(n_voxels)
        if map is not None:
            fa_fromidx = np.zeros(masked_map.shape)
            np.putmask(fa_fromidx, peak_vals[:, nfib] != 0, masked_map)
            fa_n[flat_mask] = fa_fromidx
        else:
            fa_n[flat_mask] = peak_vals[:, nfib]
        dsi_mat["fa%d" % nfib] = fa_n.astype(np.float32)
        # Fill in the index values
        index_n = np.zeros(n_voxels)
        index_n[flat_mask] = peak_indices[:, nfib]
        dsi_mat["index%d" % nfib] = index_n.astype(np.int16)
    # Add in the ODFs
    num_odf_matrices = n_odfs // ODF_COLS
    split_indices = (np.arange(num_odf_matrices) + 1) * ODF_COLS
    odf_splits = np.array_split(masked_odfs, split_indices, axis=0)
    for splitnum, odfs in enumerate(odf_splits):
        dsi_mat["odf%d" % splitnum] = odfs.T.astype(np.float32)
    dsi_mat["odf_vertices"] = verts.T
    dsi_mat["odf_faces"] = faces.T
    dsi_mat["z0"] = np.array([1.0])
    # Fill in GFA
    g_fa = np.zeros(n_voxels)
    g_fa[flat_mask] = masked_gfa
    dsi_mat["gfa"] = g_fa.astype(np.float32)
    # Fill in other metrics maps
    if other_maps is not None:
        for path_map in other_maps:
            map_name, ext = op.splitext(op.basename(path_map))
            map_dir = op.dirname(path_map)
            map_lps = op.join(map_dir, map_name + "_lps" + ext)
            convertLPS(path_map, map_lps)
            other_img = nib.load(map_lps)
            if not other_img.shape == amplitudes_img.shape[:3]:
                raise ValueError("Differing grid between other map image: {} " "and amplitudes".format(path_map))
            gmap = other_img.get_fdata().flatten(order="F")
            dsi_mat[map_name] = gmap.astype(np.float32)
            os.remove(map_lps)
    savemat(output, dsi_mat, format="4", appendmat=False)
    # Remove unwanted files
    os.remove(input_)
    if mask is not None:
        os.remove(mask_)
    if map is not None:
        os.remove(map_)
    os.remove(dirs_txt)
    os.remove(odf_amplitudes_nii)
