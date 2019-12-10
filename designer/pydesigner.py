"""
Runs the PyDesigner pipeline
"""

#---------------------------------------------------------------------- 
# Package Management
#----------------------------------------------------------------------
import subprocess #subprocess
import os # mkdir
import os.path as op # path
import shutil # which, rmtree
import gzip # handles fsl's .gz suffix
import argparse # ArgumentParser, add_argument
import textwrap # dedent
import numpy as np # array, ndarray
from preprocessing import util, smoothing, rician, preparation, snrplot
from fitting import dwipy as dp
from system import systemtools as sys
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

# Configure system for Intel MKL
if sys.isAMD():
    sys.setenv([('MKL_DEBUG_CPU_TYPE','5')])

#----------------------------------------------------------------------
# Parse Arguments
#---------------------------------------------------------------------- 
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
            --eddy \\
            --rpe_pair <rpe> \\
            --pe_dir <dir> \\
            --smooth \\
            <dwi>

    In order to just do denoising, eddy with reverse phase encode, and 
    diffusion metrics:
    python3 pydesigner.py \\
            --denoise \\
            --eddy \\
            --rpe_pair <rpe> \\
            --pe_dir <dir> \\
            --DKI \\
            <dwi>

Standard pipeline steps:
    1. dwidenoise (thermal denoising)
    2. mrdegibbs (gibbs unringing)
    3. topup + eddy (undistortion)
    4. b1 bias correction
    4. CSF-excluded smoothing
    5. rician bias correction
    6. normalization to white matter in first b0 image
    7. IRWLLS, CWLLS DKI fit
    8. Outlier detection and removal

See also:
    GitHub      https://github.com/m-ama/PyDesigner
    mrtrix3     https://www.mrtrix.org/
    fsl         https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

                                '''))

# Specify arguments below

# Mandatory
parser.add_argument('dwi', help='the diffusion dataset you would like to '
                    'process',
                    type=str)

# Optional
parser.add_argument('-o', '--output',
                    help='Output location. '
                    'Default: same path as dwi.',
                    type=str)
parser.add_argument('-s', '--standard', action='store_true',
                    default=False,
                    help='Standard preprocessing, bypasses most other '
                    'options. See Appendix:Standard pipeline steps '
                    'for more information. ')
parser.add_argument('--denoise', action='store_true', default=False,
                    help='Run thermal denoising with dwidenoise.')
parser.add_argument('--extent', metavar='n,n,n', default='5,5,5',
                    help='Denoising extent formatted n,n,n (forces '
                    ' denoising. '
                    'Default: 5,5,5.')
parser.add_argument('--degibbs', action='store_true', default=False,
                    help='Perform gibbs unringing. Only perform if you '
                    'have full Fourier encoding. The program will check '
                    'for you if you have a .json sidecar.')
parser.add_argument('--undistort', action='store_true', default=False,
                    help='Run FSL eddy to perform image undistortion. '
                    'NOTE: needs a --topup to run.')
parser.add_argument('--topup', default=None,
                    help='The topup b0 series with a reverse phase encode '
                    'direction opposite the dwi. REQUIRED for '
                    '--undistort')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='Perform smoothing on the DWI data. '
                    'Recommended to also supply --csfmask in order to '
                    'avoid contaminating the voxels which border CSF.')
parser.add_argument('--fwhm', type=float,
                    help='The FWHM to use as a multiple of voxel size. '
                    'Default 1.25')
parser.add_argument('--csfmask', default=None,
                    help='CSF mask for exclusion during smoothing. '
                    'Must be in the DWI space and resolution. ')
parser.add_argument('--rician', action='store_true', default=False,
                    help='Perform Rician noise correction on the data '
                    '(requires --denoise to generate a noisemap).')
parser.add_argument('--nofit', action='store_true', default=False,
                    help='Do not fit DTI or DKI tensors.')
parser.add_argument('--noakc', action='store_true', default=False,
                    help='Do not brute force K tensor outlier rejection.')
parser.add_argument('--nooutliers', action='store_true', default=False,
                    help='Do not perform outlier correction on kurtosis '
                    'fitting metrics.')
parser.add_argument('-w', '--wmti', action='store_true', default=False,
                    help='Include DKI WMTI parameters (forces DKI): '
                    'AWF, IAS_params, EAS_params. ')
parser.add_argument('--kcumulants', action='store_true', default=False,
                    help='output the kurtosis tensor with W cumulant '
                    'rather than K. ')
parser.add_argument('--mask', action='store_true', default=False,
                    help='Compute a brain mask prior to tensor fitting '
                    'to strip skull and improve efficiency. Use '
                     '--maskthr to specify a threshold manually.')
parser.add_argument('--maskthr', metavar='<fractional intensity '
                                             ' threshold>',
                    help='FSL bet threshold used for brain masking. '
                    'Default: 0.25')
parser.add_argument('--fit_constraints', default='0,1,0',
                    help='Constrain the WLLS fit. '
                    'Default: 0,1,0.')
parser.add_argument('--rpe_none', action='store_true', default=False,
                    help='No reverse phase encode is available; FSL eddy '
                    'will perform eddy current and motion correction '
                    ' only. ')
parser.add_argument('--rpe_pair', metavar='<reverse PE b=0 image>',
                    help='Specify reverse phase encoding image.')
parser.add_argument('--rpe_all', metavar='<reverse PE dwi>',
                    help='All DWIs have been acquired with an opposite '
                    'phase encoding direction. This information will be '
                    'used to perform a recombination of image volumes '
                    '(each pair of volumes with the same b-vector but '
                    'different phase encoding directions will be '
                    'combined into a single volume). The argument to '
                    'this option is the set of volumes with '
                    'reverse phase encoding but the same b-vectors the '
                    'same as the input image.')
parser.add_argument('--pe_dir', metavar='<phase encoding direction>',
                    help='Specify the phase encoding direction of the '
                    'input series. NOTE: REQUIRED for eddy due to a bug '
                    'in dwipreproc. Can be signed axis number, (-0,1,+2) '
                    'axis designator (RL, PA, IS), or '
                    'NIfTI axis codes (i-,j,k)')
parser.add_argument('--noqc', action='store_true', default=False,
                    help='Disable QC saving of QC metrics')
parser.add_argument('--nthreads', type=int,
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

# Use argument specification to actually get args
args = parser.parse_args()

#---------------------------------------------------------------------
# Parse Input Image
#----------------------------------------------------------------------
image = DWIParser(args.dwi)
# Variable fType indicates the extension to raw_dwi.X, where X take the
# place of known dMRI file extensions (.mif, .nii, .nii.gz). This allows
# easy switching based on any scenario for testing.
fType = '.nii'
if not args.output:
    outpath = image.getPath()
else:
    outpath = args.output
image.cat(path=outpath,
          ext=fType,
          verbose=args.verbose,
          force=args.force)
args.dwi = op.join(outpath, 'raw_dwi' + fType)

#---------------------------------------------------------------------
# Validate Arguments
#----------------------------------------------------------------------

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
    args.b1correct = True
    args.smooth = True

# Can't do WMTI if no fit
if args.nofit:
    stdmsg='--nofit given but '
    if args.wmti:
        warningmsg+=msgstart+stdmsg+'--wmti'+override+'tensor fitting.\n'
        args.nofit = False
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

# Incompatible eddy args
if not args.topup and not args.rpe_none and args.undistort:
    errmsg+='Cannot undistort without rpe selection'
elif args.rpe_pair:
    errmsg+='We are sorry but this feature is unsupported for now.'

# FWHM given but not smoothing
if not args.smooth and args.fwhm:
    warningmsg+='No --smooth given but --fwhm given; '
    warningmsg+=' overriding with --smooth\n'
    args.smooth = True

# Check to make sure CSF mask exists if given
if args.csfmask:
    if not op.exists(args.csfmask):
        errmsg+='--csfmask file '+args.csfmask+' not found\n'

# Check output directory exists if given
if args.output:
    if not op.exists(args.output):
        try:
            os.mkdir(args.output)
        except:
            errmsg+='Cannot find or create output directory'

# Check that --fit_constraints can be converted to int array
fit_constraints = np.fromstring(args.fit_constraints,
                                    dtype=int, sep=',')
for i in fit_constraints:
    if i < 0 or i > 1:
        errmsg+='Invalid --fit_constraints value, should be 0 or 1\n'
        break

# --force and --resume given
if args.resume and args.force:
    errmsg+=msgstart+'--continue and --force\n'

if args.output:
    if not op.isdir(args.output):
        try:
            os.mkdir(args.output)
        except:
            errmsg+=('Output directory does not exist and cannot '
                     'be made.')

# Print warnings
if warningmsg is not '':
    print(warningmsg)

# If things are unsalvageable, point out all errors and quit
if errmsg is not '':
    raise Exception(errmsg)

# Begin importing important data files
filetable = {'dwi' : DWIFile(args.dwi)}
if not filetable['dwi'].isAcquisition():
    raise Exception('Input dwi does not have .bval/.bvec pair')

# Check to make sure no partial fourier if --degibbs given
if args.degibbs and args.adv:
    args.degibbs = True
else:
    if args.degibbs and filetable['dwi'].isPartialFourier():
        print('Given DWI is partial fourier, overriding --degibbs; '
              'no unringing correction will be done to avoid artifacts.')
        args.degibbs = False

if args.rpe_pair:
    filetable['rpe_pair'] = DWIFile(args.rpe_pair)
if args.rpe_all:
    filetable['rpe_all'] = DWIFile(args.rpe_all)

if args.topup:
    filetable['topup'] = DWIFile(args.topup)

#----------------------------------------------------------------------
# Path Handling
#----------------------------------------------------------------------
qcpath = op.join(outpath, 'metrics_qc')
eddyqcpath = op.join(qcpath, 'eddy')
fitqcpath = op.join(qcpath, 'fitting')
metricpath = op.join(outpath, 'metrics')
if not args.nofit:
    if op.exists(metricpath):
        if args.force:
            shutil.rmtree(metricpath)
        else:
            raise Exception(
                'Running fitting would cause an overwrite. '
                'In order to run this please delete the '
                'files, use --force, use --resume, or '
                'change output destination.')
    else:
        os.mkdir(metricpath)
if not args.noqc:
    if op.exists(qcpath):
        if args.force:
            shutil.rmtree(qcpath)
        else:
            raise Exception('Running QCing would cause an overwrite. '
                            'In order to run this please delete the '
                            'files, use --force, use --resume, or '
                            'change output destination.')
    else:
        os.mkdir(qcpath)
    if op.exists(eddyqcpath) and args.undistort:
        if args.force:
            shutil.rmtree(eddyqcpath)
        else:
            raise Exception('Running dwidenoise would cause an '
                            'overwrite. '
                            'In order to run this please delete the '
                            'files, use --force, or change output '
                            'destination.')
    if op.exists(fitqcpath) and not args.nofit:
        if args.force:
            shutil.rmtree(fitqcpath)
        else:
            raise Exception('Running fitting would cause an '
                            'overwrite. '
                            'In order to run this please delete the '
                            'files, use --force, or change output '
                            'destination.')
    if args.undistort:
        os.mkdir(eddyqcpath)
    if not args.nofit:
        os.mkdir(fitqcpath)

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

#----------------------------------------------------------------------
# Run Denoising
#----------------------------------------------------------------------
if args.denoise:
    # hardcoding this to be the initial file per dwidenoise
    # recommmendation
    denoised_name = 'd' + filetable['dwi'].getName() + '.nii'
    denoised = op.join(outpath, denoised_name)
    # output the noise map even without user permission, space is cheap
    noisemap_name = 'n' + filetable['dwi'].getName() + '.nii'
    noisemap = op.join(outpath, noisemap_name)
    # check to see if this already exists
    if not (args.resume and op.exists(denoised) and op.exists(noisemap)):
        # system call
        denoise_args = ['dwidenoise', '-noise', noisemap]
        if args.force:
            denoise_args.append('-force')
        else:
            if (op.exists(denoised) or op.exists(noisemap) and
                not args.resume):
                raise Exception('Running dwidenoise would cause an '
                                'overwrite. '
                                'In order to run this please delete the '
                                'files, use --force, or change output '
                                'destination.')
        if not args.verbose:
            denoise_args.append('-quiet')

        if args.extent:
            denoise_args.append('-extent')
            denoise_args.append(args.extent)
        if args.nthreads:
            denoise_args.append('-nthreads')
            denoise_args.append(str(args.nthreads))
        denoise_args.append(filetable['dwi'].getFull())
        denoise_args.append(denoised)
        completion = subprocess.run(denoise_args)
        if completion.returncode != 0:
            raise Exception('dwidenoise failed, please look above for '
                            ' error sources')
    filetable['denoised'] = DWIFile(denoised)
    filetable['noisemap'] = DWIFile(noisemap)
    filetable['HEAD'] = filetable['denoised']

#----------------------------------------------------------------------
# Run Gibbs Unringing
#----------------------------------------------------------------------
if args.degibbs:
    # add to HEAD name
    degibbs_name = 'g' + filetable['HEAD'].getName() + '.nii'
    degibbs = op.join(outpath, degibbs_name)
    # check to see if this already exists
    if not (args.resume and op.exists(degibbs)):
        # system call
        degibbs_args = ['mrdegibbs']
        if args.force:
            degibbs_args.append('-force')
        else:
            if op.exists(degibbs) and not args.resume:
                raise Exception('Running mrdegibbs would cause an '
                                'overwrite. '
                                'In order to run this please delete the '
                                'files, use --force, use --resume, or '
                                'change output destination.')
        if not args.verbose:
            degibbs_args.append('-quiet')
        if args.nthreads:
            degibbs_args.append('-nthreads')
            degibbs_args.append(str(args.nthreads))
        degibbs_args.append(filetable['HEAD'].getFull())
        degibbs_args.append(degibbs)
        completion = subprocess.run(degibbs_args)
        if completion.returncode != 0:
            raise Exception('mrdegibbs failed, please look above for '
                            'error sources')
    filetable['unrung'] = DWIFile(degibbs)
    filetable['HEAD'] = filetable['unrung']

#----------------------------------------------------------------------
# Undistort
#----------------------------------------------------------------------
if args.undistort:
    # Add to HEAD name
    undistorted_name = 'u' + filetable['HEAD'].getName() + '.nii'
    undistorted_full = op.join(outpath, undistorted_name)

    # check to see if this already exists
    if not (args.resume and op.exists(undistorted_full)):
        # prepare; makes se-epi.mif
        if args.topup:
            preparation.make_se_epi(filetable)
        else:
            preparation.make_simple_mif(filetable)

        # system call
        dwipreproc_args = ['dwipreproc']
        if args.force:
            dwipreproc_args.append('-force')
        else:
            if op.exists(undistorted_full) and not args.resume:
                raise Exception('Running undistortion would cause an '
                                'overwrite. '
                                'In order to run this please delete the '
                                'files, use --force, use --resume, or '
                                'change output destination')
        if not args.verbose:
            dwipreproc_args.append('-quiet')
        if not args.noqc:
            dwipreproc_args.append('-eddyqc_all')
            dwipreproc_args.append(eddyqcpath)
        dwipreproc_args.append('-rpe_header')
        # full vs half sphere
        dwipreproc_args.append('-eddy_options')
        repol_string = '--repol '
        if util.bvec_is_fullsphere(filetable['dwi'].getBVEC()):
            # is full, add appropriate dwipreproc option
            repol_string += '--data_is_shelled'
        else:
            # half
            repol_string += '--slm=linear'
        dwipreproc_args.append(repol_string)
        if args.topup:
            dwipreproc_args.append('-se_epi')
            dwipreproc_args.append(filetable['se-epi'])
        # Note: we skip align_seepi because it's handled in make_se_epi
        dwipreproc_args.append(filetable['dwimif'])
        dwipreproc_args.append(undistorted_full)
        if args.verbose:
            print(*dwipreproc_args)
        completion = subprocess.run(dwipreproc_args)
        if completion.returncode != 0:
            raise Exception('dwipreproc failed, please look above for '
                            'error sources.')
        filetable['undistorted'] = DWIFile(undistorted_full)
        filetable['HEAD'] = filetable['undistorted']

#---------------------------------------------------------------------- 
# Create Brain Mask
#----------------------------------------------------------------------
if args.mask:
    fsl_suffix = '.gz'
    brainmask_fsl_name = 'brain'
    brainmask_fsl_full = op.join(outpath, brainmask_fsl_name)
    brainmask_fsl_out = op.join(outpath, brainmask_fsl_name + '_mask' +
    '.nii' + fsl_suffix)
    brainmask_out = op.join(outpath, brainmask_fsl_name + '_mask' + '.nii')
    B0_name = 'B0.nii'
    B0_mean = 'B0_mean.nii'
    B0_full = op.join(outpath, B0_name)
    B0_mean_full = op.join(outpath, B0_mean)
    # check to see if this already exists
    if (op.exists(brainmask_out) and not args.resume) and not args.force:
            raise Exception('Running mask would cause an overwrite. '
                            'In order to run please delete the files, use '
                            '--force, use --resume, or change output '
                            'destination.')
    if not (args.resume and op.exists(brainmask_out)):
        if args.maskthr is None:
            maskthr = 0.25
        else:
            maskthr = args.maskthr
        # Extract B0s
        mask_arg = ['dwiextract', '-force', '-fslgrad',
                       filetable['dwi'].getBVEC(), filetable['dwi'].getBVAL(),
                       '-bzero']
        if args.nthreads:
            mask_arg.append('-nthreads')
            mask_arg.append(str(args.nthreads))
        if not args.verbose:
            mask_arg.append('-quiet')
        mask_arg.extend([filetable['HEAD'].getFull(), B0_full])
        completion = subprocess.run(mask_arg)
        # Compute mean B0s
        mask_arg = ['mrmath', '-force']
        if args.nthreads:
            mask_arg.append('-nthreads')
            mask_arg.append(str(args.nthreads))
        if not args.verbose:
            mask_arg.append('-quiet')
        mask_arg.extend(['-axis', '3', B0_full, 'mean', B0_mean_full])
        completion = subprocess.run(mask_arg)
        if completion.returncode != 0:
            raise Exception('B0 extraction failed: check your .bval file')
        # Remove NaNs
        mask_arg = ['fslmaths', B0_mean_full, '-nan', B0_full + fsl_suffix]
        completion = subprocess.run(mask_arg)
        if completion.returncode != 0:
            raise Exception('Unable to remove NaNs from B0.nii. '
                            'Try manually extracting a brain mask '
                            'and saving it in working directory '
                            'as brain_mask.nii. Then use the --resume '
                            'option to continue from here.')
        with gzip.open(B0_full + fsl_suffix, 'r') as f_in, \
                open(B0_full,'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        if os.path.exists(B0_full + fsl_suffix):
            os.remove(B0_full + fsl_suffix)
        mask_arg = ['bet', B0_full, brainmask_fsl_full, '-m', '-f',
                       np.str(maskthr)]
        completion = subprocess.run(mask_arg)
        if completion.returncode != 0:
            raise Exception('Brain extraction failed. Check your B0.nii file '
                            'to verify correct extraction, then run with '
                            '--resume flag to continue preprocessing from '
                            'here.')
        # Decompress fsl's gunzip format
        with gzip.open(brainmask_fsl_out, 'r') as f_in, \
                open(brainmask_out,'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Remove .gz file
        if op.exists(brainmask_fsl_out):
            os.remove(brainmask_fsl_out)
        if op.exists(brainmask_fsl_full + '.nii.gz'):
            os.remove(brainmask_fsl_full + '.nii.gz')
        # Remove all other files
        if op.exists(B0_mean_full):
            os.remove(B0_mean_full)
        # Update filetable
        filetable['mask'] = DWIFile(brainmask_out)
        filetable['b0'] = DWIFile(B0_full)

#----------------------------------------------------------------------
# Smooth
#---------------------------------------------------------------------- 
if args.smooth:
    # add to HEAD name
    smoothing_name = 's' + filetable['HEAD'].getName() + '.nii'
    smoothing_full = op.join(outpath, smoothing_name)
    # check to see if this already exists
    if op.exists(smoothing_full):
        if not (args.resume or args.force):
            raise Exception('Running smoothing would cause an overwrite. '
                            'In order to run please delete the files, use '
                            '--force, use --resume, or change output '
                            'destination.')
    if not args.resume:
        if args.fwhm:
            fwhm = args.fwhm
        else:
            fwhm = 1.25
        smoothing.smooth_image(filetable['HEAD'].getFull(),
                               csfname=args.csfmask,
                               outname=smoothing_full,
                               width=fwhm)
    filetable['smoothed'] = DWIFile(smoothing_full)
    filetable['HEAD'] = filetable['smoothed']

#----------------------------------------------------------------------
# Rician Noise Correction
#----------------------------------------------------------------------
if args.rician:
    # add to HEAD name
    rician_name = 'r' + filetable['HEAD'].getName() + '.nii'
    rician_full = op.join(outpath, rician_name)
    # check to see if this already exists
    if op.exists(rician_full):
        # system call
        if not (args.resume or args.force):
            raise Exception('Running rician correction would cause an '
                            'overwrite. '
                            'In order to run this please delete the '
                            'files, use --force, use --resume, or '
                            'change output destination.')

    if not args.resume:
        rician.rician_img_correct(filetable['HEAD'].getFull(),
                      filetable['noisemap'].getFull(),
                      outpath=rician_full)

    filetable['rician_corrected'] = DWIFile(rician_full)
    filetable['HEAD'] = filetable['rician_corrected']

#---------------------------------------------------------------------- 
# Make preprocessed file
#----------------------------------------------------------------------
preprocessed = op.join(outpath, 'preprocessed_dwi')
shutil.copyfile(filetable['HEAD'].getFull(), preprocessed + '.nii')
shutil.copyfile(filetable['dwi'].getBVAL(), preprocessed + '.bval')
shutil.copyfile(filetable['dwi'].getBVEC(), preprocessed + '.bvec')
filetable['preprocessed'] = DWIFile(preprocessed)
filetable['HEAD'] = filetable['preprocessed']

#----------------------------------------------------------------------
# Compute SNR
#----------------------------------------------------------------------
if args.denoise and not args.noqc:
    files = []
    files.append(op.join(outpath, 'raw_dwi.nii'))
    files.append(filetable['HEAD'].getFull())
    snrplot = snrplot.makesnr(dwilist=files,
                              noisepath=filetable['noisemap'].getFull(),
                              maskpath=filetable['mask'].getFull())
    snrplot.makeplot(path=qcpath, smooth=True, smoothfactor=3)

#----------------------------------------------------------------------
# Tensor Fitting
#----------------------------------------------------------------------
if not args.nofit:
    # create dwi fitting object
    if not args.nthreads:
        img = dp.DWI(filetable['HEAD'].getFull())
    else:
        img = dp.DWI(filetable['HEAD'].getFull(), args.nthreads)
    # detect outliers
    if not args.nooutliers:
        if not img.isdki():
            outliers, dt_est = img.irlls(mode='DTI')
        else:
            outliers, dt_est = img.irlls(mode='DKI')
        # write outliers to qc folder
        if not args.noqc:
            outlier_full = op.join(fitqcpath, 'outliers_irlls.nii')
            dp.writeNii(outliers, img.hdr, outlier_full)
        # fit while rejecting outliers
        img.fit(fit_constraints, reject=outliers)
    else:
        # fit without rejecting outliers
        img.fit(fit_constraints)

    md, rd, ad, fa, fe, trace = img.extractDTI()
    dp.writeNii(md, img.hdr, op.join(metricpath, 'md'))
    dp.writeNii(rd, img.hdr, op.join(metricpath, 'rd'))
    dp.writeNii(ad, img.hdr, op.join(metricpath, 'ad'))
    dp.writeNii(fa, img.hdr, op.join(metricpath, 'fa'))
    dp.writeNii(fe, img.hdr, op.join(metricpath, 'fe'))
    if not img.isdki():
        dp.writeNii(trace, img.hdr, op.join(metricpath, 'trace'))
    else:
        # do akc, DKI fitting
        if not args.noakc:
            # do akc
            akc_out = img.akcoutliers()
            img.akccorrect(akc_out)
            if not args.noqc:
                dp.writeNii(akc_out, img.hdr,
                            op.join(fitqcpath, 'outliers_akc'))
        mk, rk, ak, kfa, mkt, trace = img.extractDKI()
        # naive implementation of writing these variables
        dp.writeNii(mk, img.hdr, op.join(metricpath, 'mk'))
        dp.writeNii(rk, img.hdr, op.join(metricpath, 'rk'))
        dp.writeNii(ak, img.hdr, op.join(metricpath, 'ak'))
        dp.writeNii(kfa, img.hdr, op.join(metricpath, 'kfa'))
        dp.writeNii(mkt, img.hdr, op.join(metricpath, 'mkt'))
        dp.writeNii(trace, img.hdr, op.join(metricpath, 'trace'))
        if args.wmti:
            awf, eas_ad, eas_rd, eas_tort, ias_ad, ias_rd, ias_tort = \
                img.extractWMTI()
            dp.writeNii(awf, img.hdr,
                        op.join(metricpath, 'wmti_awf'))
            dp.writeNii(eas_ad, img.hdr,
                        op.join(metricpath, 'wmti_eas_ad'))
            dp.writeNii(eas_rd, img.hdr,
                        op.join(metricpath, 'wmti_eas_rd'))
            dp.writeNii(eas_tort, img.hdr,
                        op.join(metricpath, 'wmti_eas_tort'))
            dp.writeNii(ias_ad, img.hdr,
                        op.join(metricpath, 'wmti_ias_ad'))
            dp.writeNii(ias_rd, img.hdr,
                        op.join(metricpath, 'wmti_ias_rd'))
            dp.writeNii(ias_tort, img.hdr,
                        op.join(metricpath, 'wmti_ias_tort'))
        # reorder tensor for mrtrix3
        DT, KT = img.tensorReorder(img.tensorType())
        dp.writeNii(DT, img.hdr, op.join(metricpath, 'DT'))
        dp.writeNii(KT, img.hdr, op.join(metricpath, 'KT'))
