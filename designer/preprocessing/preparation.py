#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Adds utilities for preparing the data for eddy and analysis
"""

import os #mkdir
import os.path as op # dirname, basename, join, splitext
import shutil #rmtree
import json # decode
from enum import Enum
import nibabel as nib # various utilities for reading Nifti images
import subprocess
import re # regex substitution
from designer.preprocessing import util # preprocessing
DWIFile = util.DWIFile

def fix_bval(bvalfile):
    """
    Converts all whitespace into newlines in the file

    Parameters
    ----------
    bvalfile : str
        The .bval to ensure is the correct format for mrtrix

    Returns
    -------
    None, overwrites bval
    """

    if not op.exists(bvalfile):
        raise Exception('File '+ bvalfile + ' does not exist.')

    with open(bvalfile, 'r') as f:
        data = f.read()

    # replace whitespace with lines
    data = re.sub(r'\s+', '\n', data)

    # write to file
    with open(bvalfile, 'w') as f:
        f.write(data)

def make_simple_mif(filetable):
    """
    Makes a single .mif from the HEAD dwi file

    Parameters
    ----------
    filetable : dict of str
        The filetable that pydesigner.py uses to track files

    Returns
    -------
    None, writes file
    """
    if not op.exists(filetable['dwi'].getJSON()):
        raise Exception('DWI does not have a .json file to use for eddy')

    #coerce bval into mrtrix3-friendly format
    fix_bval(op.join(filetable['dwi'].getPath(),
             filetable['dwi'].getName() + '.bval'))
    finalpath = filetable['outpath']
    finalmif = op.join(finalpath, 'HEADmif.mif')

    dwi_convert_args = ['mrconvert',
                        '-json_import',
                        filetable['dwi'].getJSON(),
                        '-fslgrad',
                        filetable['dwi'].getBVEC(),
                        filetable['dwi'].getBVAL(),
                        filetable['dwi'].getFull(),
                        '-quiet',
                        finalmif]

    completion = subprocess.run(dwi_convert_args)

    if completion.returncode != 0:
        raise Exception('topup conversion failed, please see above')

    filetable['dwimif'] = finalmif

def make_se_epi(filetable):
    """
    Makes a single spin-echo epi from the topup and the dwi

    Parameters
    ----------
    filetable : dict of str
        The filetable that pydesigner.py uses to track files

    Returns
    -------
    None, writes file
    """

    #----------------------------------------------------------------------
    # Check inputs and coerce .bval file
    #----------------------------------------------------------------------

    # check that we actually have a .json for the original files
    if not op.exists(filetable['dwi'].getJSON()):
        raise Exception('DWI does not have a .json file to use for eddy')

    if not op.exists(op.join(filetable['topup'].getJSON())):
        raise Exception('topup does not have a .json file to use for eddy')

    #NOTE: we assume that the DWI will have .bvec and .bval because
    # pydesigner will not allow the 'dwi' entry to not have them

    # coerce .bval into having the mrtrix3-friendly format
    fix_bval(op.join(filetable['dwi'].getPath(),
             filetable['dwi'].getName() + '.bval'))

    finalpath = filetable['outpath']

    # Make a temporary working directory
    outpath = op.join(finalpath, 'tmp_se_epi')
    if op.exists(outpath):
        shutil.rmtree(outpath)
    os.mkdir(outpath)

    #----------------------------------------------------------------------
    # Make dwipreproc-friendly files
    #----------------------------------------------------------------------

    # dwi
    tmp_dwi = op.join(outpath, 'tmp_dwi.mif')
    dwi_convert_args = ['mrconvert',
                        '-json_import',
                        filetable['dwi'].getJSON(),
                        '-fslgrad',
                        filetable['dwi'].getBVEC(),
                        filetable['dwi'].getBVAL(),
                        filetable['dwi'].getFull(),
                        '-quiet',
                        tmp_dwi]

    completion = subprocess.run(dwi_convert_args)

    # move dwi to non-temp directory as well
    final_dwi = op.join(finalpath, 'tmp_dwi.mif')
    shutil.copyfile(tmp_dwi, final_dwi)

    # add it to the filetable
    filetable['dwimif'] = final_dwi

    if completion.returncode != 0:
        raise Exception('DWI conversion failed, please see above.')

    # topup
    tmp_tp = op.join(outpath, 'tmp_tp.mif')
    topup_convert_args = ['mrconvert',
                          '-json_import',
                          filetable['topup'].getJSON(),
                          filetable['topup'].getFull(),
                          '-quiet',
                          tmp_tp]
    completion = subprocess.run(topup_convert_args)

    if completion.returncode != 0:
        raise Exception('topup conversion failed, please see above')

    # extract the b0 images from the dwi image
    b0extracted = op.join(outpath, 'b0extracted.mif')
    extract_b0_args = ['dwiextract',
                       '-bzero',
                       tmp_dwi,
                       '-quiet',
                       b0extracted]
    completion = subprocess.run(extract_b0_args)

    if completion.returncode != 0:
        raise Exception('b0 extraction failed, please see above')

    # separate b0 images into individual files
    # start by getting number of volumes in each image
    b0_info = op.join(outpath, 'b0x.txt')
    get_dwi_b0_info_args = ['mrinfo',
                            '-size',
                            b0extracted]
    completion = subprocess.run(get_dwi_b0_info_args, capture_output=True)
    if completion.returncode != 0:
        raise Exception('Extracted b0 information failed, please see above')

    # use an abomination unto good coding to extract 4th dim size
    ndb0xstr = completion.stdout.decode('utf-8').rstrip().split(' ')[-1]
    ndb0x = int(ndb0xstr.rstrip('"'))

    # repeat for rpe
    rpe_info = op.join(outpath, 'rpex.txt')
    get_rpe_info_args = ['mrinfo',
                            '-size',
                            tmp_tp]
    completion = subprocess.run(get_rpe_info_args, capture_output=True)
    if completion.returncode != 0:
        raise Exception('Extracted topup information failed, '
                        'please see above')

    ndrpexstr = completion.stdout.decode('utf-8').rstrip().split(' ')[-1]
    ndrpex = int(ndrpexstr.rstrip('"'))

    # iterate over the indices and align the data
    b0_basename = op.join(outpath, 'b0x_')
    regto = b0_basename+'0.mif'
    to_cat = [regto]
    for ii in range(ndb0x):
        # iterate over b0s and register to first b0
        # extract the volume
        i = str(ii)
        extracted_name = b0_basename+i+'.mif'
        reg_txt = b0_basename+i+'to0.txt'
        reg_mif = b0_basename+i+'to0.mif'
        xfm_b0 = b0_basename+i+'.mif'
        extract_args = ['mrconvert',
                        '-force',
                        '-coord',
                        '3',
                        i,
                        b0extracted,
                        '-quiet',
                        extracted_name]
        completion = subprocess.run(extract_args)
        if completion.returncode != 0:
            raise Exception('Failed splitting b0 volume '+i)
        # if first b0, no need to register
        if i == '0':
            continue
        # calculate the transform required
        reg_args = ['mrregister',
                    '-type','rigid',
                    '-noreorientation',
                    '-rigid', reg_txt,
                    '-quiet',
                    extracted_name,
                    regto]
        completion = subprocess.run(reg_args)
        if completion.returncode != 0:
            raise Exception('Failed registering volume '+i+' to b0')
        # apply transform
        transformed_basename = op.join(outpath, 'dwib0')
        transform_args = ['mrtransform',
                          '-linear',
                          reg_txt,
                          reg_mif,
                          '-quiet',
                          xfm_b0]
        to_cat.append(xfm_b0)

    # Repeat for RPE
    rpe_basename = op.join(outpath, 'rpex_')
    for ii in range(ndrpex):
        # iterate over rpes and register to first rpe
        # extract the volume
        i = str(ii)
        extracted_name = rpe_basename+i+'.mif'
        reg_txt = rpe_basename+i+'to0.txt'
        reg_mif = rpe_basename+i+'to0.mif'
        xfm_rpe = rpe_basename+i+'.mif'
        extract_args = ['mrconvert',
                        '-force',
                        '-coord',
                        '3',
                        i,
                        '-quiet',
                        tmp_tp,
                        extracted_name]
        completion = subprocess.run(extract_args)
        if completion.returncode != 0:
            raise Exception('Failed splitting rpe volume '+i)
        # calculate the transform required
        reg_args = ['mrregister',
                    '-type','rigid',
                    '-noreorientation',
                    '-rigid', reg_txt,
                    '-quiet',
                    extracted_name,
                    regto]
        completion = subprocess.run(reg_args)
        if completion.returncode != 0:
            raise Exception('Failed registering volume '+i+'  to rpe')
        # apply transform
        transformed_basename = op.join(outpath, 'dwirpe')
        transform_args = ['mrtransform',
                          '-linear',
                          reg_txt,
                          reg_mif,
                          '-quiet',
                          xfm_rpe]
        to_cat.append(xfm_rpe)

    # Concatenate all b0 into one se-epi
    # NOTE: uses finalpath instead of outpath
    se_epi = op.join(finalpath, 'se-epi.mif')
    mrcat_args = ['mrcat', '-force', '-quiet', '-axis', '3']
    for f in to_cat:
        mrcat_args.append(f)
    mrcat_args.append(se_epi)

    completion = subprocess.run(mrcat_args)
    if completion.returncode != 0:
        raise Exception('Concatenation of b0s into se-epi failed')

    filetable['se-epi'] = se_epi

    # Clean up temp dir
    # shutil.rmtree(outpath)
