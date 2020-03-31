#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Utilities for running various MRtrix3's DWI preprocessing tools
"""

import os
import os.path as op
import subprocess
import numpy as np
from designer.preprocessing import preparation, util, smoothing, rician, mrinfoutil

def miftonii(input, output, strides='1,2,3,4', nthreads=None,
             force=True, verbose=False):
    """
    Converts input `.mif` images to output `.nii` images

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .nii file
    strides : str, optional
        Specify the strides of the output data in memory
        (Default: '1,2,3,4')
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file

    See Also
    --------
    niitomif
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if op.splitext(output)[-1] != '.nii':
        raise OSError('Output specified does not possess the .nii '
                      'extension.')
    if not isinstance(strides, str):
        raise Exception('Please specify strides as a string.')
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    arg = ['mrconvert']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    arg.extend(['-export_grad_fsl',
                op.splitext(output)[0] + '.bvec',
                op.splitext(output)[0] + '.bval'])
    arg.extend(['-json_export', op.splitext(output)[0] + '.json'])
    arg.extend(['-strides', strides])
    arg.extend([input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('Conversion from .mif to .nii failed; check '
                        'above for errors.')

def niitomif(input, output, strides='1,2,3,4', nthreads=None,
             force=True, verbose=False):
    """
    Converts input `.nii` images to output `.nif` images provided that
    all BVEC, BVAL and JSON files are provided and named same as input .nii

    Parameters
    ----------
    input : str
        Path to input .nii file
    output : str
        Path to output .mif file
    strides : str, optional
        Specify the strides of the output data in memory
        (Default: '1,2,3,4')
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file

    See Also
    --------
    miftonii
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if op.splitext(output)[-1] != '.mif':
        raise OSError('Output specified does not possess the .mif '
                      'extension.')
    if not op.exists(op.splitext(input)[0] + '.bvec'):
        raise OSError('Unable to locate BVEC file" {}'.format(op.splitext(
            output)[0] + '.bvec'))
    if not op.exists(op.splitext(input)[0] + '.bval'):
        raise OSError('Unable to locate BVAL file" {}'.format(op.splitext(
            output)[0] + '.bval'))
    if not op.exists(op.splitext(input)[0] + '.json'):
        raise OSError('Unable to locate JSON file" {}'.format(op.splitext(
            output)[0] + '.json'))
    if not isinstance(strides, str):
        raise Exception('Please specify strides as a string.')
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    arg = ['mrconvert']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    arg.extend(['-fslgrad',
                op.splitext(input)[0] + '.bvec',
                op.splitext(input)[0] + '.bval'])
    arg.extend(['-json_import', op.splitext(input)[0] + '.json'])
    arg.extend(['-strides', strides])
    arg.extend([input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('Conversion from .nii to .mif failed; check '
                        'above for errors.')

def denoise(input, output, noisemap=True, extent='5,5,5', nthreads=None,
            force=True, verbose=False):
    """
    Runs MRtrix3's `dwidenoise` command with optimal parameters for
    PyDesigner.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    noisemap : bool, optional
        Specify whether or not to save the noisemap as a 
        nifti file (Default: True)
    extent : str, optional
        Set the window size of the denoising filter.
        (Default: '5,5,5')
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if not isinstance(noisemap, bool):
        raise Exception('Please specify whether noisemap generation '
                        'is True or False.')
    if not isinstance(extent, str):
        raise Exception('Please specify extent as a string formatted as '
                        '"n,n,n".')
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    noisemap_path = op.join(op.dirname(input), 'noisemap.nii')
    arg = ['dwidenoise']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    if noisemap:
        arg.extend(['-noise', noisemap_path])
    if not (extent is None):
        arg.extend(['-extent', extent])
    arg.extend([input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('dwidenoise failed, please look above for error '
                        'sources.')

def degibbs(input, output, nthreads=None, force=False, verbose=False):
    """
    Runs MRtrix3's `mrdegibbs` command with optimal parameters for
    PyDesigner.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    arg = ['mrdegibbs']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    arg.extend([input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('mrdegibbs failed, please look above for error '
                        'sources.')

def undistort(input, output, rpe='rpe_header', epib0=1,
              qc=None, nthreads=None, force=False, verbose=False):
    """
    Runs MRtrix3's `dwipreproc` command with optimal parameters for
    PyDesigner.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    rpe : str, {'rpe_header', 'rpe-pair', 'rpe_all, 'rpe_all'}, optional
        Reverse phase encoding of the dataset. (Default: 'rpe_header')
    epib0 : int
        Number of reverse PE dir B0 pairs to use in TOPUP correction
        (Default: 1)
    qc : bool
        Specify whether to generate eddy QC metrics (Default: True)
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if not rpe in ['rpe_none', 'rpe_pair', 'rpe_all', 'rpe_header']:
        raise Exception('Entered RPE selection is not valid. Please '
                        'choose either "rpe_none", "rpe_pair", '
                        '"rpe_all", or "rpe_header".')
    if not isinstance(epib0, int):
        raise Exception('Number of TOPUP B0s need to be specified as '
                        'as an integer.')
    if not qc is None:
        if not isinstance(qc, str):
            raise Exception('Please specify QC directory as a string')
        if not op.exists(qc):
            raise OSError('Specified QC directory does not exist. '
                          'Please ensure that this is a valid '
                          'directory.')
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    rpe = '-' + rpe
    # Get output directory
    outdir = op.dirname(output)
    # Extract BVEC and BVALS for shell sampling deduction
    arg_extract = ['mrinfo']
    arg_extract.extend(['-export_grad_fsl',
                        op.join(outdir, 'dwiec.bvec'),
                        op.join(outdir, 'dwiec.bval')])
    arg_extract.append(input)
    completion = subprocess.run(arg_extract)
    if completion.returncode != 0:
        raise Exception('extracting FSL BVEC and BVEC gradients '
                        'failed during undistortion, please look '
                        'above for errors.')
    # Form main undistortion argument
    arg = ['dwipreproc']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    # Determine whether half or full sphere sampling
    repol_string = '--repol '
    if util.bvec_is_fullsphere(op.join(outdir, 'dwiec.bvec')):
        # is full, add appropriate dwipreproc option
        repol_string += '--data_is_shelled'
    else:
        # half
        repol_string += '--slm=linear'
    if epib0 > 0:
        try:
            epi_path = op.join(outdir, 'B0_EPI.mif')
            epiboost(input=input,
                    output=epi_path,
                    num=epib0,
                    nthreads=nthreads,
                    force=force,
                    verbose=verbose)
            arg.extend(['-se_epi', epi_path])
        except:
            print('[WARNING] Unable to apply EPI boost because DWI '
            'consists of single PE direction.')
    arg.extend(['-eddy_options', repol_string])
    arg.append(rpe)
    if not qc is None:
        arg.extend(['-eddyqc_all', qc])
    arg.extend([input, output])
    completion = subprocess.run(arg, cwd=outdir)
    if completion.returncode != 0:
        raise Exception('dwipreproc failed, please look above for '
                        'error sources.')
    # Remove temporarily generated files
    os.remove(op.join(outdir, 'dwiec.bvec'))
    os.remove(op.join(outdir, 'dwiec.bval'))
    if epib0 > 0:
        try:
            os.remove(epi_path)
        except:
            print('[Warning] unable to remove {} because it does not '
            'exist'.format(epi_path))
    
def brainmask(input, output, thresh=0.25, nthreads=None, force=False,
              verbose=False):
    """
    Creates a brainmask using FSL's Brain Extraction Tool (BET) and
    MRtrix3's file manipulation tools.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .nii brainmask file
    thresh : float
        BET threshold ranging from 0 to 1 (Default: 0.25)
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if (thresh < 0) or (thresh > 1):
        raise ValueError('BET Threshold needs to be within 0 to 1 range.')
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    # Read FSL NifTi output format and change it if not '.nii'
    fsl_suffix = os.getenv('FSLOUTPUTTYPE')
    if fsl_suffix is None:
        raise OSError('Unable to determine system environment variable '
                      'FSF_OUTPUT_FORMAT. Ensure that FSL is installed '
                      'correctly.')
    if fsl_suffix == 'NIFTI_GZ':
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI'
    outdir = op.dirname(output)
    B0_all = op.join(outdir, 'B0_all.mif')
    B0_mean = op.join(outdir, 'B0_mean.nii')
    B0_nan = op.join(outdir, 'B0.nii')
    mask = op.join(outdir, 'brain')
    tmp_brain = op.join(outdir, 'brain.nii')
    # Extract B0 from DWI
    arg_b0_all = ['dwiextract']
    if force:
        arg_b0_all.append('-force')
    if not verbose:
        arg_b0_all.append('-quiet')
    if not (nthreads is None):
        arg_b0_all.extend(['-nthreads', nthreads])
    arg_b0_all.append('-bzero')
    arg_b0_all.extend([input, B0_all])
    completion = subprocess.run(arg_b0_all)
    if completion.returncode != 0:
        raise Exception('Unable to extract B0s from DWI for computation '
                        'of brain mask. See above for errors.')
    # Now compute mean
    arg_b0_mean = (['mrmath', '-axis', '3', B0_all, 'mean', B0_mean])
    completion = subprocess.run(arg_b0_mean)
    if completion.returncode != 0:
        raise Exception('Unable to compute mean of B0s for computation '
                        'of brain mask. See above for errors.')

    # Now remove nan create mask using `fslmaths`
    arg_b0_nan = ['fslmaths', B0_mean, '-nan', B0_nan]
    completion = subprocess.run(arg_b0_nan)
    if completion.returncode != 0:
        raise Exception('Unable to remove NaN from B0 for the '
                        'computation of brain mask. See above for errors.')
    # Compute brain mask from
    arg_mask = ['bet', B0_nan, mask, '-m', '-f', str(thresh)]
    completion = subprocess.run(arg_mask)
    if completion.returncode != 0:
        raise Exception('Unable to compute brain mask from B0. See above '
                        'for errors')
    # Remove intermediary file
    os.remove(B0_all)
    os.remove(B0_mean)
    os.remove(tmp_brain)
    os.rename(op.join(outdir, mask + '_mask.nii'), output)

def smooth(input, output, fwhm=1.25):
    """
    Performs Gaussian smoothing on input .mif image

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    fwhm : float
        The full width half max in voxels to be smoothed
        (Default: 1.25)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if fwhm < 0:
        raise Exception('FWHM cannot be less than zero.')
    # Convert input .mif to .nii
    outdir = op.dirname(output)
    nii_path = op.join(outdir, 'dwism.nii')
    miftonii(input=input, output=nii_path, strides='1,2,3,4')
    # Perform smoothing
    smoothing.smooth_image(nii_path,
                           csfname=None,
                           outname=nii_path,
                           width=fwhm)
    # Convert .nii to .mif
    niitomif(input=nii_path, output=output, strides='1,2,3,4')
    # Remove converted files
    os.remove(nii_path)
    os.remove(op.splitext(nii_path)[0] + '.bvec')
    os.remove(op.splitext(nii_path)[0] + '.bval')
    os.remove(op.splitext(nii_path)[0] + '.json')

def riciancorrect(input, output, noise=None):
    """
    Performs Rician correction on input .mif

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    noise : str
        Path to noise map from dwidenoise in .nii format (Default: None)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if noise is not None:
        if not op.exists(noise):
            raise OSError('Input noisemap {} does not exist.'.format(
                noise))
        if op.splitext(noise)[-1] != '.nii':
            raise OSError('Noisemap needs to be in NifTi format.')
    else:
        raise Exception('Rician correction cannot be performed without a '
                        'noisemap.')
    # Convert input .mif to .nii
    outdir = op.dirname(output)
    nii_path = op.join(outdir, 'dwirc.nii')
    miftonii(input=input, output=nii_path, strides='1,2,3,4')
    # Perform Rician correction
    rician.rician_img_correct(nii_path,
                              noise,
                              outpath=nii_path)
    # Convert .nii to .mif
    niitomif(input=nii_path, output=output, strides='1,2,3,4')
    # Remove converted files
    os.remove(nii_path)
    os.remove(op.splitext(nii_path)[0] + '.bvec')
    os.remove(op.splitext(nii_path)[0] + '.bval')
    os.remove(op.splitext(nii_path)[0] + '.json')

def extractbzero(input, output, nthreads=None, force=False,
              verbose=False):
    """
    Extracts only bzero shells from an input mif file.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    arg = ['dwiextract']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    arg.extend(['-bzero', input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('Unable to extract B0s from DWI for computation '
                        'of brain mask. See above for errors.')

def extractnonbzero(input, output, nthreads=None, force=False,
              verbose=False):
    """
    Extracts only non-bzero shells from an input mif file.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    arg = ['dwiextract']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    arg.extend(['-no_bzero', input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('Unable to extract B0s from DWI for computation '
                        'of brain mask. See above for errors.')

def epiboost(input, output, num=1, nthreads=None, force=False,
              verbose=False):
    """
    Analyzes an input .mif's PE direction to split into two different
    phase encoding (PE) DWIs. B0s from opposing PE are then extracted
    and concatenated with the DWI. This reduces the number of B0s used
    in undistortion for a better and speedier estimation of the
    distortion field.

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    num : int
        Number of B0s pairs to use in EPI correction (Default: 1)
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    print('Applying EPIBOOST')
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if op.splitext(output)[-1] != '.mif':
        raise OSError('Output should be specified as a .mif file.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if not isinstance(num, int):
        raise Exception('Number of B0s to use needs to be specified '
                        'as an integer.')
    if not num > 0:
        raise Exception('Number of B0s to use needs to be a positive '
                        'integer greater than 0.')
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    outdir = op.dirname(output)
    fname_bzero = op.join(outdir, 'B0_ALL.mif')
    # Extract all B0s
    extractbzero(input, fname_bzero, nthreads=nthreads, force=force,
              verbose=verbose)
    # Start by figuring out whether DWI is composed of multiple PE dirs
    # or single PE dirs. If all PE dirs are the same, the dataset likely
    # comes with matching phase encoding and slice timing train,
    # indicating that it has a single PE direction.
    dw_scheme = np.array(mrinfoutil.dwscheme(fname_bzero), dtype=int)[:, -1]
    pe_scheme = np.array(mrinfoutil.pescheme(fname_bzero))
    if len(pe_scheme) != len(dw_scheme):
        raise Exception('It appears that the input volume possesses a '
                        'dw_scheme of length {}, and pe_scheme of length '
                         '{}. These number need to match. Please check '
                        'your dataset or contact us on GitHub'.format(
            len(dw_scheme), len(pe_scheme)))
    uPE, indPE, iPE = np.unique(pe_scheme, axis=0, return_index=True,
                                return_inverse=True)
    nPE = len(uPE)
    if nPE < 2:
        raise Exception('DWI consists of just one PE direction. '
                        'Unable to extract B0s.')
    # Index unique PE directions
    bval = []
    bind = []
    iteridx = np.unique(iPE)
    for i, val in enumerate(iteridx):
        bind.append(np.where(iPE == val)[0].tolist())
        bval.append(dw_scheme[np.where(iPE == val)].tolist())
    # Check whether number of B0s to extract exceed those in DWI
    if num > min([len(x) for x in bval]):
        raise Exception('Specified number of B0s pairs to extract '
                        '({}) exceed those physically present in DWI '
                        '({}), please ensure that variable `num` '
                        'suitably represents the number of B0s in DWI.'
                        .format(num, min([len(x) for x in bval])))
    # Extract the first `num` pairs from each PE direction
    num = np.arange(0, num, dtype=int).tolist()
    idx_extract = []
    for idx, val in enumerate(bind):
        idx_extract.extend([val[i] for i in num])
    # Extract EPI volume
    str_extract = [str(x) for x in idx_extract]
    arg_epi = ['mrconvert']
    if force:
        arg_epi.append('-force')
    if not verbose:
        arg_epi.append('-quiet')
    if not (nthreads is None):
        arg_epi.extend(['-nthreads', nthreads])
    arg_epi.extend(['-coord', '3', ','.join(str_extract)])
    arg_epi.extend([fname_bzero, output])
    completion = subprocess.run(arg_epi)
    if completion.returncode != 0:
        raise Exception('EPIBOOST: failed to extract specified '
                        'TOPUP B0 indices. See above for errors.')
    # Remove temp files
    os.remove(fname_bzero)

def reslice(input, output, voxel, interp='linear', nthreads=None,
            force=False, verbose=False):
    """
    Reslices input image to target voxel size

    Parameters
    ----------
    input : str
        Path to input .mif file
    output : str
        Path to output .mif file
    voxel : float or tuple of float
        x, y, z voxel size in mm
    interp : str, {'linear', 'nearest', 'cubic' , 'sinc'}, optional
        set the interpolation method to use when resizing (Default: 
        'linear')
    nthreads : int, optional
        Specify the number of threads to use in processing
        (Default: all available threads)
    force : bool, optional
        Force overwrite of output files if pre-existing
        (Default:False)
    verbose : bool, optional
        Specify whether to print console output (Default: False)

    Returns
    -------
    None; writes out file
    """
    if not op.exists(input):
        raise OSError('Input path does not exist. Please ensure that '
                      'the folder or file specified exists.')
    if op.splitext(output)[-1] != '.mif':
        raise OSError('Output should be specified as a .mif file.')
    if not op.exists(op.dirname(output)):
        raise OSError('Specifed directory for output file {} does not '
                      'exist. Please ensure that this is a valid '
                      'directory.'.format(op.dirname(output)))
    if not isinstance(voxel, str):
        raise Exception('Voxel size needs to be defined as a string '
                        ' of three values')
    if len(voxel.split(',')) != 3:
        raise Exception('Please specify voxel size for each axis '
                        'x, y, and z i.e. "3,3,3" for 3 mm isotropic')
    if not isinstance(interp, str):
        raise Exception('Interpolation method needs to be specified '
                        'as a string')
    if interp not in ('linear', 'nearest', 'cubic', 'sinc'):
        raise Exception('User specified interpoaltion method {} is '
                        'not a valid option'.format(interp))
    if not (nthreads is None):
        if not isinstance(nthreads, int):
            raise Exception('Please specify the number of threads as an '
                            'integer.')
    if not isinstance(force, bool):
        raise Exception('Please specify whether forced overwrite is True '
                        'or False.')
    if not isinstance(verbose, bool):
        raise Exception('Please specify whether verbose is True or False.')
    arg = ['mrresize']
    if force:
        arg.append('-force')
    if not verbose:
        arg.append('-quiet')
    if not (nthreads is None):
        arg.extend(['-nthreads', nthreads])
    arg.extend(['-voxel', voxel])
    arg.extend(['-interp', interp])
    arg.extend([input, output])
    completion = subprocess.run(arg)
    if completion.returncode != 0:
        raise Exception('EPIBOOST: failed to extract specified '
                        'TOPUP B0 indices. See above for errors.')
