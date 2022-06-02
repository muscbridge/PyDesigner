"""
Runs the PyDesigner pipeline
"""

#---------------------------------------------------------------------
# Package Management
#---------------------------------------------------------------------
import sys as sys
import subprocess #subprocess
import glob # recursive file search
import os # mkdir
import os.path as op # path
import shutil # which, rmtree
import gzip # handles fsl's .gz suffix
import argparse # ArgumentParser, add_argument
import textwrap # dedent
import json
import numpy as np # array, ndarray
from designer.info import __version__
from designer.preprocessing import util, preparation, mrinfoutil, mrpreproc
from designer.plotting import snrplot, outlierplot, motionplot
from designer.fitting import dwipy as dp
from designer.postprocessing import filters
from designer.tractography import dsistudio as ds
DWIFile = util.DWIFile
DWIParser = util.DWIParser

# Locate mrtrix3 via which-ing dwidenoise
dwidenoise_location = shutil.which('dwidenoise')
if dwidenoise_location == None:
    raise Exception('Cannot find mrtrix3, please see '
        'https://github.com/m-ama/PyDesigner/wiki'
        ' to troubleshoot.')

# Extract mrtrix3 path from dwidenoise_location
mrtrix3path = op.dirname(dwidenoise_location)

# Locate FSL via which-ing fsl
fsl_location = shutil.which('fsl')
if fsl_location == None:
    raise Exception('Cannot find FSL, please see '
        'https://github.com/m-ama/PyDesigner/wiki'
        ' to troubleshoot.')

# Extract FSL path from fsl_location
fslpath = op.dirname(fsl_location)

def main():
    #-----------------------------------------------------------------
    # Parse Arguments
    #-----------------------------------------------------------------
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(
            prog='pydesigner',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent('''\
    Appendix
    --------
    Filename note:
        Use the base name without the extension. This makes it easy to program
        in automatic .bvec/.bval detection for Niftis and makes your shell
        easier to read by others. The program will automatically search image
        filenames for .nii and .nii.gz extensions. If you use the --dicom
        option, then the program will assume that the entire directory
        consists of dicom files, and will warn you of any files which fail to
        be read in as dicoms.

    Example usage:
        In order to process in the standard way:
        python3 pydesigner.py \\
                --standard \\
                <dwi>

        In order to process in a custom pipeline with denoising, eddy, reverse
        phase encoding, and smoothing, but no diffusion metrics:
        python3 pydesigner.py \\
                --denoise \\
                --undistort \\
                --smooth \\
                --nofit \\
                <dwi>

        In order to just do denoising, eddy with reverse phase encode, and 
        diffusion metrics:
        python3 pydesigner.py \\
                --denoise \\
                --undistort \\
                <dwi>

    Standard pipeline steps:
        1. dwidenoise (thermal denoising)
        2. mrdegibbs (gibbs unringing)
        3. topup + eddy (undistortion)
        4. rician bias correction
        5. normalization to white matter in first b0 image
        6. IRWLLS, CWLLS DKI fit
        7. Outlier detection and removal

    See also:
        GitHub      https://github.com/m-ama/PyDesigner
        mrtrix3     https://www.mrtrix.org/
        fsl         https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

                                    '''))

    # Specify arguments below

    # Mandatory
    parser.add_argument('dwi',
                        nargs='+',
                        help='The diffusion dataset you would like '
                        'to process. ',
                        type=str)

    # Optional
    parser.add_argument('-o', '--output',
                        metavar='directory',
                        help='Output location. '
                        'Default: same path as dwi.',
                        type=str)
    parser.add_argument('-s', '--standard', action='store_true',
                        default=False,
                        help='Standard preprocessing, bypasses most other '
                        'options. See Appendix:Standard pipeline steps '
                        'for more information. ')
    parser.add_argument('-n', '--denoise', action='store_true', default=False,
                        help='Run thermal denoising with dwidenoise.')
    parser.add_argument('--extent', metavar='n,n,n', default='5,5,5',
                        help='Denoising extent formatted n,n,n (forces '
                        ' denoising. '
                        'Default: 5,5,5.')
    parser.add_argument('-g', '--degibbs', action='store_true', default=False,
                        help='Perform gibbs unringing. Only perform if you '
                        'have full Fourier encoding. The program will check '
                        'for you if you have a .json sidecar.')
    parser.add_argument('-u', '--undistort', action='store_true', default=False,
                        help='Run FSL eddy to perform image undistortion. '
                        'NOTE: needs a --topup to run.')
    parser.add_argument('--rpe_pairs', default=0, type=int,
                        metavar='n',
                        help='Number of reverse phase encoded B0 '
                        'pairs to use in TOPUP. Using less pairs '
                        'results in faster TOPUP correction. '
                        'Specfying 0 results in using all B0 pairs.'
                        'We recommend using just one pair. Default: 0')
    parser.add_argument('-z', '--smooth', action='store_true', default=False,
                        help='Perform smoothing on the DWI data.')
    parser.add_argument('--fwhm', type=float, default=1.25,
                        metavar='n',
                        help='The FWHM to use as a multiple of voxel size. '
                        'Default 1.25')
    parser.add_argument('-r', '--rician', action='store_true', default=False,
                        help='Perform Rician noise correction on the data '
                        '(requires --denoise to generate a noisemap).')
    parser.add_argument('--nofit', action='store_true', default=False,
                        help='Do not fit DTI or DKI tensors.')
    parser.add_argument('--noakc', action='store_true', default=False,
                        help='Do not brute force K tensor outlier rejection.')
    parser.add_argument('--nooutliers', action='store_true', default=False,
                        help='Do not perform outlier correction on kurtosis '
                        'fitting metrics.')
    parser.add_argument('-m', '--mask', action='store_true', default=False,
                        help='Compute a brain mask prior to tensor fitting '
                        'to strip skull and improve efficiency. Optionally, '
                        'use --maskthr to specify a threshold manually.')
    parser.add_argument('--maskthr', metavar='n',
                        default=0.25,
                        help='FSL bet threshold used for brain masking. '
                        'Default: 0.25')
    parser.add_argument('--user_mask', metavar='path',
                        help='Path to user-supplied brain mask.',
                        type=str)
    parser.add_argument('-cf', '--csf_fsl', action='store_true', default=False,
                        help='Compute a CSF mask for CSF-excluded '
                        'smoothing to minimize partial volume '
                        'effects using FSL FAST.')
    parser.add_argument('-cd', '--csf_adc', metavar='n', default=False,
                        help='Compute a CSF mask for CSF-excluded '
                        'smoothing to minimize partial volume '
                        'effects using thresholding a pseudo-ADC map '
                        'computed as ln(S0/S1000)/b1000. Default: 2')
    parser.add_argument('--reslice', metavar='x,y,z',
                        help='Relices DWI to voxel resolution '
                        'specified in millimeters (mm) or output '
                        'dimensions. Performing reslicing will skip '
                        'plotting of SNR curves. Providing dimensions '
                        'greater than 9 will switch from mm voxel '
                        'reslicing to output image reslicing.')
    parser.add_argument('--interp', action='store_true', default='linear',
                        help='Set the interpolation to use when '
                        'reslicing. Choices are linear (default), ' 
                        'nearest, cubic, and sinc.')
    parser.add_argument('-te', '--multite', action='store_true',
                        default=False,
                        help='Specify whether input DWI consists of '
                        'multiple TEs. PyDesigner will preprocess all '
                        'TEs together, then extract metric values of '
                        'each TE separately.')          
    parser.add_argument('--fit_constraints', default='0,1,0',
                        metavar='D>0,K>0,K < 3/(b*D)',
                        help='Constrain the WLLS fit. '
                        'Default: 0,1,0.')
    parser.add_argument('--l_max', default=6, type=int,
                        metavar='n',
                        help='Maximum spherical harmonic degree for '
                        'FBI spherical harmonic expansion')
    parser.add_argument('--no_rectify', action='store_true',
                        default=False,
                        help='Disable rectification of FBI fODF. Use '
                        'only when rectification of excellent '
                        'acquisitions results in degradation of FBI '
                        'or FBWM metric maps')
    parser.add_argument('--noqc', action='store_true', default=False,
                        help='Disable QC saving of QC metrics')
    parser.add_argument('--median', action='store_true', default=False,
                        help='Performs postprocessing median filtering of '
                        'final maps. WARNING: Use on a case-by-case '
                        'basis for bad data only. When applied, the '
                        'filter alters the values of most voxels, so '
                        'it should be used with caution and avoided '
                        'when data quality is otherwise adequate. '
                        'While maps appear visually soother with '
                        'this flag on, they may nonetheless be less '
                        'accurate.')
    parser.add_argument('--nthreads', type=int, default=None,
                        help='Number of threads to use for computation. '
                        'Note that using too many threads will cause a slow-'
                        'down.')
    parser.add_argument('--resume', action='store_true',
                        help='Continue from an aborted or partial previous '
                        'run of pydesigner.')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrites of existing files. Otherwise, '
                        'there will be an error at runtime.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print out all output. This is a very messy '
                        'option. We recommend piping output to a text file '
                        'if you use this option.')
    parser.add_argument('--adv', action='store_true',
                        help='Disables safety checks for advanced users who '
                        'want to force a preprocessing step. WARNING: '
                        'THIS FLAG IS FOR ADVANCED USERS ONLY WHO FULLY '
                        'UNDERSTAND THE MRI SYSTEM AND ITS OUTPUTS. '
                        'RUNNING WITH THIS FLAG COULD POTENTIALLY '
                        'RESULT IN IMPRECISE AND INACCURATE RESULTS.')
    parser.add_argument('-v', '--version', action='version',
                        version=__version__)

    # Use argument specification to actually get args
    args = parser.parse_args()

    #-----------------------------------------------------------------
    # Parse Input Image
    #-----------------------------------------------------------------
    image = DWIParser(args.dwi)
    # Variable fType indicates the extension to raw_dwi.X, where X take the
    # place of known dMRI file extensions (.mif, .nii, .nii.gz). This allows
    # easy switching based on any scenario for testing.
    fType = '.mif'
    multi_echo = False
    if not args.output:
        outpath = image.getPath()
    else:
        outpath = args.output
    image.cat(path=outpath,
            ext=fType,
            verbose=args.verbose,
            force=args.force,
            resume=args.resume)
    working_path = op.join(outpath, 'working' + fType)
    # Create index of DWI volumes with different TEs
    if np.unique(image.echotime).size > 1:
        multi_echo = True
        multi_echo_start = [0]
        multi_echo_end = [image.vols[0] - 1]
        for idx, vols in enumerate(image.vols[1:]):
            multi_echo_start.append(multi_echo_start[-1] + vols)
            multi_echo_end.append(multi_echo_end[-1] + vols)
        multi_echo_start = [int(x) for x in multi_echo_start]
        multi_echo_end = [int(x) for x in multi_echo_end]
            
    # Make an initial conversion to nifti
    init_nii = op.join(outpath, 'dwi_raw.nii')
    if not (args.resume and op.exists(init_nii)):
        mrpreproc.miftonii(input=working_path,
                        output=init_nii,
                        strides='1,2,3,4',
                        nthreads=args.nthreads,
                        force=args.force,
                        verbose=args.verbose)
                        
    #-----------------------------------------------------------------
    # Validate Arguments
    #-----------------------------------------------------------------
    errmsg = ''
    warningmsg = ''
    msgstart = 'Incompatible arguments: '
    override = '; overriding with '
    # Warn if --standard and cherry-picking
    if args.standard:
        stdmsg= '--standard but cherry-picking '
        override='; overriding with standard pipeline.\n'
        if args.denoise:
            warningmsg+=msgstart+stdmsg+'--denoise'+override
        if args.undistort:
            warningmsg+=msgstart+stdmsg+'--eddy'+override
        if args.smooth:
            warningmsg+=msgstart+stdmsg+'--smooth'+override
        # Coerce all of the above to be true
        args.denoise = True
        args.undistort = True
        args.smooth = True
        args.csf_adc = 2
        args.mask = True
        args.degibbs = True
        args.rician = True

    # Can't do WMTI if no fit
    if args.nofit:
        stdmsg='--nofit given but '
        if args.noakc:
            warningmsg+=msgstart+stdmsg+'--noakc'+override+'tensor fitting.\n'
            args.nofit = False
        if args.nooutliers:
            warningmsg+=msgstart+stdmsg+'--nooutliers'
            warningmsg+=override+'tensor fitting.\n'
            args.nofit = False

    # (Extent or Degibbs) and no Denoise
    if not args.denoise:
        stdmsg='No --denoise but '
        if args.extent != '5,5,5':
            warningmsg+=stdmsg+'--extent given; overriding with --denoise\n'
            args.denoise = True
        if args.rician:
            warningmsg+=stdmsg+'--rician given; overriding with --denoise\n'
            args.denoise = True

    # Cannot run --user_mask and --mask at the same time
    if args.user_mask and args.mask:
        errmsg+='Cannot run with both --mask and --user_mask; '
        errmsg+='--mask if you do not have a custom brain mask and ' \
                '--user_mask if you want to supply a mask.'
    
    # Cannot run --csf_fsl and --csf_adc at the same time
    if args.csf_fsl and args.csf_adc:
        errmsg+='Cannot run with both --csf_fsl and --csf_adc; '
        errmsg+='please supply only one option.'

    # Check to make sure brain mask exists if given
    if args.user_mask:
        if not op.exists(args.user_mask):
            errmsg+='--user_mask file '+args.user_mask+' not found\n'
        # Then check if it's a nifti file
        if not '.nii' in op.splitext(args.user_mask)[-1]:
            errmsg+='User supplied mask if not in NifTi (.nii) format.'

    # Check output directory exists if given
    if args.output:
        if not op.exists(args.output):
            try:
                os.makedirs(args.output, exist_ok=True)
            except:
                errmsg+='Cannot find or create output directory'

    # Check that --fit_constraints can be converted to int array
    fit_constraints = np.fromstring(args.fit_constraints,
                                        dtype=int, sep=',')
    for i in fit_constraints:
        if i < 0 or i > 1:
            errmsg+='Invalid --fit_constraints value, should be 0 or 1\n'
            break

    # Ensure l_max is an even integer
    if args.l_max % 2 != 0:
        errmsg+='User provided l_max = {} is not an even integer.'.format(args.l_max)

    # --force and --resume given
    if args.resume and args.force:
        errmsg+=msgstart+'--continue and --force\n'

    if args.output:
        if not op.isdir(args.output):
            try:
                os.makedirs(args.output, exist_ok=True)
            except:
                errmsg+=('Output directory does not exist and cannot '
                        'be made.')

    # Print warnings
    if warningmsg is not '':
        print(warningmsg)

    # If things are unsalvageable, point out all errors and quit
    if errmsg is not '':
        raise Exception(errmsg)

    # Begin keeping track of nifti files
    filetable = {'dwi' : DWIFile(init_nii)}
    if not filetable['dwi'].isAcquisition():
        raise Exception('Input dwi does not have .bval/.bvec pair')

    # Begin composing command history
    cmdtable = {'HEAD': 'none'}
    cmdtable['input'] = mrinfoutil.commandhistory(working_path)

    # Check to make sure no partial fourier if --degibbs given
    if args.degibbs and args.adv:
        args.degibbs = True
    else:
        if args.degibbs and filetable['dwi'].isPartialFourier():
            print('[WARNING] Given DWI is partial fourier, overriding '
                '--degibbs; no unringing correction will be done to '
                'avoid artifacts.Use the "--adv" flag to run forced '
                'corrections.')
            args.degibbs = False
    
    # Handle FBI rectification
    fbi_rectify = True
    if args.no_rectify:
        fbi_rectify = False

    #-----------------------------------------------------------------
    # Path Handling
    #-----------------------------------------------------------------
    qcpath = op.join(outpath, 'metrics_qc')
    eddyqcpath = op.join(qcpath, 'eddy')
    fitqcpath = op.join(qcpath, 'fitting')
    metricpath = op.join(outpath, 'metrics')
    intermediatepath = op.join(outpath, 'intermediate_nifti')
    if not args.nofit:
        if op.exists(metricpath):
            if args.force:
                shutil.rmtree(metricpath)
            elif not args.resume:
                raise Exception(
                    'Running fitting would cause an overwrite. '
                    'In order to run this please delete the '
                    'files, use --force, use --resume, or '
                    'change output destination.')
        else:
            os.makedirs(metricpath, exist_ok=True)
    if not args.noqc:
        if op.exists(qcpath):
            if args.force:
                shutil.rmtree(qcpath)
            elif not args.resume:
                raise Exception('Running QCing would cause an overwrite. '
                                'In order to run this please delete the '
                                'files, use --force, use --resume, or '
                                'change output destination.')
        else:
            os.makedirs(qcpath, exist_ok=True)
        if op.exists(eddyqcpath) and args.undistort:
            if args.force:
                shutil.rmtree(eddyqcpath)
            elif not args.resume:
                raise Exception('Running dwidenoise would cause an '
                                'overwrite. '
                                'In order to run this please delete the '
                                'files, use --force, or change output '
                                'destination.')
        if op.exists(fitqcpath) and not args.nofit:
            if args.force:
                shutil.rmtree(fitqcpath)
            elif not args.resume:
                raise Exception('Running fitting would cause an '
                                'overwrite. '
                                'In order to run this please delete the '
                                'files, use --force, or change output '
                                'destination.')
        if args.undistort:
            os.makedirs(eddyqcpath, exist_ok=True)
        if not args.nofit:
            os.makedirs(fitqcpath, exist_ok=True)
            
    if not op.exists(intermediatepath):
        os.makedirs(intermediatepath, exist_ok=True)

    # TODO: add non-json RPE support, additional RPE type support

    # Get naming and location information
    dwiname = filetable['dwi'].getName()
    if not args.output:
        outpath = filetable['dwi'].getPath()
    else:
        outpath = args.output
    filetable['outpath'] = outpath

    # Make the pipeline point to dwi as the last file since it's the only one
    # so far
    filetable['HEAD'] = filetable['dwi']

    if args.nthreads and args.verbose:
        print('Using ' + str(args.nthreads) + ' threads.')

    # Create processing step variable to count preprocessing stage
    step_count = 0

    #-----------------------------------------------------------------
    # Run Denoising
    #-----------------------------------------------------------------
    if args.denoise:
        step_count += 1
        denoised_name = 'dwi_denoised'
        # hardcoding this to be the initial file per dwidenoise
        # recommmendation
        # file names
        denoised_name_full = str(step_count)+ '_' + denoised_name 
        nii_denoised = op.join(intermediatepath, denoised_name_full + '.nii')
        mif_denoised = op.join(outpath, denoised_name_full + '.mif')
        # output the noise map even without user permission, space is cheap
        noisemap_name = 'noisemap.nii'
        nii_noisemap = op.join(outpath, noisemap_name)
        # check to see if this already exists
        if not (args.resume and op.exists(nii_denoised) and \
            op.exists(nii_noisemap)):
            # run denoise function
            mrpreproc.denoise(input=working_path,
                              output=mif_denoised,
                              noisemap=True,
                              extent=args.extent,
                              nthreads=args.nthreads,
                              force=args.force,
                              verbose=args.verbose)
            mrpreproc.miftonii(input=mif_denoised,
                               output=nii_denoised,
                               strides='1,2,3,4',
                               nthreads=args.nthreads,
                               force=args.force,
                               verbose=False)
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_denoised, working_path)
            # update command history
            cmdtable['denoise'] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable['HEAD'] = cmdtable['denoise']
        # update nifti file tracking
        filetable['denoised'] = DWIFile(nii_denoised)
        filetable['HEAD'] = filetable['denoised']
        filetable['noisemap'] = DWIFile(nii_noisemap)

    #-----------------------------------------------------------------
    # Run Reslicing
    #-----------------------------------------------------------------
    if args.reslice:
        step_count += 1
        reslice_name = 'dwi_reslice'
        noise_name = 'noisemap_resliced'
        # file names
        reslice_name_full = str(step_count)+ '_' + reslice_name
        nii_reslice = op.join(intermediatepath, reslice_name_full + '.nii')
        mif_reslice = op.join(outpath, reslice_name_full + '.mif')
        nii_noise = op.join(outpath, noise_name + '.nii')
        # check to see if this already exists
        if not (args.resume and op.exists(nii_reslice)):
            # run reslice function on both DWI and noisemap
            mrpreproc.reslice(input=working_path,
                              output=mif_reslice,
                              size=args.reslice,
                              interp=args.interp,
                              nthreads=args.nthreads,
                              force=args.force,
                              verbose=args.verbose)
            mrpreproc.reslice(input=nii_noisemap,
                              output=nii_noise,
                              size=args.reslice,
                              interp=args.interp,
                              nthreads=args.nthreads,
                              force=True,
                              verbose=args.verbose)
            mrpreproc.miftonii(input=mif_reslice,
                               output=nii_reslice,
                               strides='1,2,3,4',
                               nthreads=args.nthreads,
                               force=args.force,
                               verbose=False)
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_reslice, working_path)
            os.rename(nii_noise, nii_noisemap)
            # update command history
            cmdtable['reslice'] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable['HEAD'] = cmdtable['reslice']
        # update nifti file tracking
        filetable['reslice'] = DWIFile(nii_reslice)
        filetable['HEAD'] = filetable['reslice']

    #-----------------------------------------------------------------
    # Run Gibbs Unringing
    #-----------------------------------------------------------------
    if args.degibbs:
        step_count += 1
        degibbs_name = 'dwi_degibbs'
        # file names
        degibbs_name_full = str(step_count)+ '_' + degibbs_name
        nii_degibbs = op.join(intermediatepath, degibbs_name_full + '.nii')
        mif_degibbs = op.join(outpath, degibbs_name_full + '.mif')
        # check to see if this already exists
        if not (args.resume and op.exists(nii_degibbs)):
            # run degibbs function
            mrpreproc.degibbs(input=working_path,
                              output=mif_degibbs,
                              nthreads=args.nthreads,
                              force=args.force,
                              verbose=args.verbose)
            mrpreproc.miftonii(input=mif_degibbs,
                               output=nii_degibbs,
                               strides='1,2,3,4',
                               nthreads=args.nthreads,
                               force=args.force,
                               verbose=False)
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_degibbs, working_path)
            # update command history
            cmdtable['degibbs'] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable['HEAD'] = cmdtable['degibbs']
        # update nifti file tracking
        filetable['unrung'] = DWIFile(nii_degibbs)
        filetable['HEAD'] = filetable['unrung']

    #-----------------------------------------------------------------
    # Undistort
    #-----------------------------------------------------------------
    if args.undistort:
        step_count += 1
        undistorted_name = 'dwi_undistorted'
        # file names
        undistorted_name_full = str(step_count)+ '_' + undistorted_name
        nii_undistorted = op.join(intermediatepath, undistorted_name_full + '.nii')
        mif_undistorted = op.join(outpath, undistorted_name_full + '.mif')
        if args.noqc:
            eddyqcpath = None
        # check to see if this already exists
        if not (args.resume and op.exists(nii_undistorted)):
            # run undistort function
            mrpreproc.undistort(input=working_path,
                                output=mif_undistorted,
                                rpe='rpe_header',
                                qc=eddyqcpath,
                                epib0=args.rpe_pairs,
                                nthreads=args.nthreads,
                                force=args.force,
                                verbose=args.verbose)
            mrpreproc.miftonii(input=mif_undistorted,
                            output=nii_undistorted,
                            strides='1,2,3,4',
                            nthreads=args.nthreads,
                            force=args.force,
                            verbose=False)
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_undistorted, working_path)
            # update command history
            cmdtable['undistort'] = mrinfoutil.commandhistory(working_path)
            cmdtable['HEAD'] = cmdtable['undistort']
        # update nifti file tracking
        filetable['undistorted'] = DWIFile(nii_undistorted)
        filetable['HEAD'] = filetable['undistorted']

        # Plot head motion
        if not args.noqc:
            plot_path_full = op.join(qcpath, 'head_motion.png')
            motionplot.plot(op.join(eddyqcpath, 'eddy_restricted_movement_rms'),
                            plot_path_full,
                            voxel=mrinfoutil.spacing(working_path))

    #-----------------------------------------------------------------
    # Create CSF Mask
    #-----------------------------------------------------------------
    csfmask_name = 'csf_mask.nii'
    csfmask_out = op.join(outpath, csfmask_name)
    # FSL Method
    if args.csf_fsl:
        mrpreproc.csfmask(input=working_path,
                            output=csfmask_out,
                            method='fsl',
                            thresh=args.maskthr,
                            nthreads=args.nthreads,
                            force=args.force,
                            verbose=args.verbose)
        filetable['csfmask'] = DWIFile(csfmask_out)
    # ADC Method
    if args.csf_adc:
        mrpreproc.csfmask(input=working_path,
                            output=csfmask_out,
                            method='adc',
                            coeff=args.csf_adc,
                            nthreads=args.nthreads,
                            force=args.force,
                            verbose=args.verbose)
        filetable['csfmask'] = DWIFile(csfmask_out)

    #-----------------------------------------------------------------
    # Create Brain Mask
    #-----------------------------------------------------------------
    if args.mask:
        brainmask_name = 'brain_mask.nii'
        brainmask_out = op.join(outpath, brainmask_name)
        mrpreproc.brainmask(input=working_path,
                            output=brainmask_out,
                            thresh=args.maskthr,
                            nthreads=args.nthreads,
                            force=args.force,
                            verbose=args.verbose)
        filetable['mask'] = DWIFile(brainmask_out)
    
    if args.user_mask:
        # Rotates user mask to same orientation as PyDesigner's working
        # file to prevent incorrect masking
        brainmask_name = 'brain_mask.nii'
        brainmask_out = op.join(outpath, 'brain_mask.nii')
        mrpreproc.stride_match(
            target=working_path,
            moving=args.user_mask,
            output=brainmask_out,
            nthreads=args.nthreads,
            force=args.force,
            verbose=args.verbose
        )
        filetable['mask'] = DWIFile(brainmask_out)

    #-----------------------------------------------------------------
    # Multiply Brain Mask with CSF Mask if both present
    #-----------------------------------------------------------------
    if (args.mask or args.user_mask) and (args.csf_fsl or args.csf_adc):
        cmd = [
            'mrcalc',
            '-force',
            brainmask_out,
            csfmask_out,
            '-mult',
            csfmask_out
        ]
        completion = subprocess.run(cmd)
        if completion.returncode != 0:
            raise Exception('Unable to multiply CSF mask with brain '
                            'mask. See above for errors.')
    #-----------------------------------------------------------------
    # Smooth
    #-----------------------------------------------------------------
    if args.smooth:
        csfname = None
        if 'csfmask' in filetable:
            csfname = filetable['csfmask'].getFull()
        step_count += 1
        smoothing_name = 'dwi_smoothed'
        # file names
        smoothing_name_full = str(step_count)+ '_' + smoothing_name
        nii_smoothing = op.join(intermediatepath, smoothing_name_full + '.nii')
        mif_smoothing = op.join(outpath, smoothing_name_full + '.mif')
        # check to see if this already exists
        if not (args.resume and op.exists(nii_smoothing)):
            mrpreproc.smooth(input=working_path,
                            csfname=csfname,
                            output=mif_smoothing,
                            fwhm=args.fwhm)

            mrpreproc.miftonii(input=mif_smoothing,
                                output=nii_smoothing,
                                strides='1,2,3,4',
                                nthreads=args.nthreads,
                                force=args.force,
                                verbose=False)
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_smoothing, working_path)
            # update command history
            cmdtable['smooth'] = ['designer.preprocessing.mrpreproc.smooth(input={}, '
                                  'output={}, '
                                  'fwhm={}'.format(working_path,
                                                   mif_smoothing,
                                                   args.fwhm)]
            cmdtable['smooth'].append(mrinfoutil.commandhistory(working_path)[-1])
            cmdtable['HEAD'] = cmdtable['smooth']
        # update nifti file tracking
        filetable['smoothed'] = DWIFile(nii_smoothing)
        filetable['HEAD'] = filetable['smoothed']

    #-----------------------------------------------------------------
    # Rician Noise Correction
    #-----------------------------------------------------------------
    if args.rician:
        step_count += 1
        rician_name = 'dwi_rician'
        # file names
        rician_name = str(step_count)+ '_' + rician_name
        nii_rician = op.join(intermediatepath, rician_name + '.nii')
        mif_rician = op.join(outpath, rician_name + '.mif')
        # check to see if this already exists
        if not (args.resume and op.exists(nii_rician)):
            mrpreproc.riciancorrect(input=working_path,
                                    output=mif_rician,
                                    noise=filetable['noisemap'].getFull())
            nii_rician_name = 'r' + filetable['HEAD'].getName() + '.nii'
            nii_rician_full = op.join(outpath, nii_rician_name)
            mrpreproc.miftonii(input=mif_rician,
                                output=nii_rician,
                                strides='1,2,3,4',
                                nthreads=args.nthreads,
                                force=args.force,
                                verbose=False)
            # remove old working.mif and replace with new corrected .mif
            os.remove(working_path)
            os.rename(mif_rician, working_path)
            # update command history
            cmdtable['rician'] = ['designer.preprocessing.mrpreproc.'
                                  'riciancorrect(input={}, '
                                  'output={}, '
                                  'noise={})'.format(working_path,
                                                      mif_rician,
                                                      filetable['noisemap'].getFull())]
            cmdtable['rician'].append(mrinfoutil.commandhistory(working_path)[-1])
            cmdtable['HEAD'] = cmdtable['rician']
        # update nifti file tracking
        filetable['rician_corrected'] = DWIFile(nii_rician)
        filetable['HEAD'] = filetable['rician_corrected']

    #-----------------------------------------------------------------
    # Extract averaged B0
    #-----------------------------------------------------------------
    # file names
    b0_name = 'B0'
    nii_b0 = op.join(outpath, b0_name + '.nii')
    # check to see if this already exists
    if not (args.resume and op.exists(nii_b0)):
        # extract mean B0
        mrpreproc.extractmeanbzero(input=working_path,
                                    output=nii_b0,
                                    nthreads=args.nthreads,
                                    force=args.force,
                                    verbose=args.verbose)
        # update command history
        cmdtable['B0'] = mrinfoutil.commandhistory(working_path)[-1]
        cmdtable['HEAD'] = cmdtable['B0']
    # update nifti file tracking
    filetable['B0'] = DWIFile(nii_b0)
    filetable['HEAD'] = filetable['B0']

    #-----------------------------------------------------------------
    # Extract averaged non-B0 shells
    #-----------------------------------------------------------------
    # get non B0 shells
    b_shells = [x for x in mrinfoutil.shells(working_path) if x != 0]
    # remove 
    # file names
    b_names = ['B' + str(x) for x in b_shells]
    b_paths = [op.join(outpath, x + '.nii') for x in b_names]
    # check to see if this already exists
    for b_value, b_nii in zip(b_shells, b_paths):
        if not (args.resume and op.exists(b_nii)):
            # extract mean shells
            mrpreproc.extractmeanshell(
                input=working_path,
                output=b_nii,
                shell=b_value,
                nthreads=args.nthreads,
                force=args.force,
                verbose=args.verbose
            )
            # update command history
            cmdtable['B' + str(b_value)] = mrinfoutil.commandhistory(working_path)[-1]
            cmdtable['HEAD'] = cmdtable['B' + str(b_value)]
            # update nifti file tracking
            filetable['B' + str(b_value)] = DWIFile(b_nii)
            filetable['HEAD'] = filetable['B' + str(b_value)]

    #-----------------------------------------------------------------
    # Make preprocessed file
    #-----------------------------------------------------------------
    preprocessed = op.join(outpath, 'dwi_preprocessed.nii')
    if not (args.resume and op.exists(preprocessed)):
        mrpreproc.miftonii(input=working_path,
                            output=preprocessed,
                            strides='1,2,3,4',
                            nthreads=args.nthreads,
                            force=args.force,
                            verbose=False)
    filetable['preprocessed'] = DWIFile(preprocessed)
    filetable['HEAD'] = filetable['preprocessed']

    #-----------------------------------------------------------------
    # Compute SNR
    #-----------------------------------------------------------------
    if (args.denoise and not args.reslice) and not args.noqc:
        files = []
        files.append(init_nii)
        files.append(filetable['HEAD'].getFull())
        try:
            if 'mask' in filetable:
                snr = snrplot.makesnr(dwilist=files,
                                    noisepath=nii_noisemap,
                                    maskpath=filetable['mask'].getFull())
            else:
                snr = snrplot.makesnr(dwilist=files,
                                    noisepath=filetable['noisemap'].getFull(),
                                    maskpath=None)
            snr.makeplot(path=qcpath, smooth=True, smoothfactor=3)
        except:
            print('[WARNING] SNR plotting failed, see above. '
            'Proceeding with processing.')
    
    #-----------------------------------------------------------------
    # Write logs
    #-----------------------------------------------------------------
    with open(op.join(outpath, 'log_command.json'), 'w') as fp:
        json.dump(cmdtable, fp, indent=2)

    #-----------------------------------------------------------------
    # Handle multi-echo data
    #-----------------------------------------------------------------
    imPath = filetable['HEAD'].getFull()
    if multi_echo and args.multite:
        imPath = []
        for i in range(len(image.echotime)):
            echo_out = op.join(outpath, 'TE' + str(image.echotime[i]) + '_dwi_preprocessed.nii')
            mrpreproc.dwiextract(working_path,
                                echo_out,
                                start=multi_echo_start[i],
                                end=multi_echo_end[i],
                                nthreads=args.nthreads,
                                force=args.force,
                                verbose=False)
            imPath.append(echo_out)

    # Remove working.mif
    os.remove(working_path)

    #-----------------------------------------------------------------
    # Tensor Fitting
    #-----------------------------------------------------------------
    ext = '.nii'
    fit_mask = None
    if 'mask' in filetable:
        fit_mask = filetable['mask'].getFull()
    if not args.nofit:
        # create dwi fitting object
        if multi_echo and args.multite:
            for path, echo in zip(imPath, image.echotime):
                # qcpath = op.join(op.dirname(path), 'qc_fitting')
                os.makedirs(qcpath, exist_ok=True)
                dp.fit_regime(
                    input=path,
                    output=metricpath,
                    prefix='TE' + str(echo) + '_',
                    suffix=None,
                    ext=ext,
                    irlls=not args.nooutliers,
                    akc=not args.noakc,
                    l_max=args.l_max,
                    rectify = fbi_rectify,
                    qcpath=fitqcpath,
                    fit_constraints=fit_constraints,
                    mask=fit_mask,
                    nthreads=args.nthreads
                )
        else:
            dp.fit_regime(
                input=imPath,
                output=metricpath,
                prefix=None,
                suffix=None,
                ext=ext,
                irlls=not args.nooutliers,
                akc=not args.noakc,
                l_max=args.l_max,
                rectify = fbi_rectify,
                qcpath=fitqcpath,
                fit_constraints=fit_constraints,
                mask=fit_mask,
                nthreads=args.nthreads
            )

    #-----------------------------------------------------------------
    # Post Processing
    #-----------------------------------------------------------------
    if args.median:
        f_metrics = glob.glob(op.join(metricpath, '*' + ext))
        f_metrics = [x for x in f_metrics if not x.endswith('DT.nii')]
        f_metrics = [x for x in f_metrics if not x.endswith('KT.nii')]
        f_metrics = [x for x in f_metrics if not x.endswith('fodf.nii')]
        for f in f_metrics:
            filters.median(f, f)

    #-----------------------------------------------------------------
    # Fiber Tracking
    #-----------------------------------------------------------------
    f_odf = glob.glob(op.join(metricpath, '*fodf*'))
    for f in f_odf:
        if 'mask' in filetable:
            path, ext = op.splitext(f)
            f_fib = path + '.fib'
            ds.makefib(
                input=f,
                output=f_fib,
                mask=filetable['mask'].getFull()
            )
        else:
            ds.makefib(
                input=f,
                output=f_fib,
                mask=None
            )
if __name__ == '__main__':
    main()
