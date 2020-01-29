#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Utilities for running various MRtrix3's DWI preprocessing tools
"""
import os
import os.path as op
import subprocess
from designer.preprocessing import preparation, util

def miftonii(input, output, strides='1,2,3,4', nthreads=None,
             force=True, verbose=False):
    """
    Converts input `.mif` images to output `.nii` images

    Parameters
    ----------
    input (str):    path to input .mif file
    output (str):   path to output .nii file
    strides (str):  specify the strides of the output data in memory
                    (default: '1,2,3,4')

    Returns
    -------
    system call:    (none)
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

def denoise(input, output, noisemap=True, extent='5,5,5', nthreads=None,
            force=True, verbose=False):
    """
    Runs MRtrix3's `dwidenoise` command with optimal parameters for
    PyDesigner.

    Parameters
    ----------
    input (str):      path to input .mif file

    output (str):     path to output .mif file
    noisemap (bool):  specify whether or not to save the noisemap as a
                      nifti file (default: True)
    extent (str):     set the window size of the denoising filter.
                      (default: 5,5,5)
    nthreads (int):   number of threads in multi-threaded applications
    force (bool):     force overwrite of output files (default: False)
    verbose (bool):   display information messages (default: False)

    Returns
    -------
    system call:    (none)
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
    input (str):      path to input .mif file

    output (str):     path to output .mif file
    nthreads (int):   number of threads in multi-threaded applications
    force (bool):     force overwrite of output files (default: False)
    verbose (bool):   display information messages (default: False)

    Returns
    -------
    system call:    (none)
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

def undistort(input, output, rpe='rpe_header', qc=None, 
              nthreads=None, force=False, verbose=False):
    """
    Runs MRtrix3's `dwipreproc` command with optimal parameters for
    PyDesigner.

    Parameters
    ----------
    input (str):      path to input .mif file

    output (str):     path to output .mif file
    nthreads (int):   number of threads in multi-threaded applications
    force (bool):     force overwrite of output files (default: False)
    verbose (bool):   display information messages (default: False)

    Returns
    -------
    system call:    (none)
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
    