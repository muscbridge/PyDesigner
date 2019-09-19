"""
Runs the PyDesigner pipeline
"""

#---------------------------------------------------------------------- 
# Package Management
#----------------------------------------------------------------------
import subprocess #subprocess
import os.path as op # path
import shutil # which
import argparse # ArgumentParser, add_argument
import textwrap # dedent
import numpy # array, ndarray
from preprocessing import util, smoothing, rician
DWIFile = util.DWIFile

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
                    'NOTE: needs a phase encoding '
                    'specification to run.')
parser.add_argument('--smooth', action='store_true', default=False,
                    help='Perform smoothing on the DWI data. '
                    'Recommended to also supply --csfmask in order to '
                    'avoid contaminating the voxels which border CSF.')
parser.add_argument('--csfmask', default=None,
                    help='CSF mask for exclusion during smoothing. '
                    'Must be in the DWI space and resolution. ')
parser.add_argument('--rician', action='store_true', default=False,
                    help='Perform Rician noise correction on the data '
                    '(requires --denoise to generate a noisemap).')
parser.add_argument('--nofit', action='store_true', default=False,
                    help='Do not fit DTI or DKI tensors.')
parser.add_argument('-w', '--WMTI', action='store_true', default=False,
                    help='Include DKI WMTI parameters (forces DKI): '
                    'AWF, IAS_params, EAS_params. ')
parser.add_argument('--kcumulants', action='store_true', default=False,
                    help='output the kurtosis tensor with W cumulant '
                    'rather than K. ')
parser.add_argument('--mask', action='store_true', default=False,
                    help='Compute a brain mask prior to tensor fitting '
                    'to strip skull and improve efficiency. ')
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

# Use argument specification to actually get args
args = parser.parse_args()

#---------------------------------------------------------------------
# Validate Arguments
#----------------------------------------------------------------------

errmsg = ''
warningmsg = ''
msgstart = 'Incompatible arguments: '
override = '; overriding with '
# --rpe*
if args.rpe_pair and args.rpe_all:
    errmsg+=msgstart+'--rpe_pair and --rpe_all\n'

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
    if args.WMTI:
        warningmsg+=msgstart+stdmsg+'--WMTI'+override+'tensor fitting.'
        args.nofit = False

# (Extent or Degibbs) and no Denoise
if not args.denoise:
    stdmsg='No --denoise but '
    if args.extent != '5,5,5':
        print(args)
        warningmsg+=stdmsg+'--extent given; overriding with --denoise\n'
        args.denoise = True
    if args.rician:
        warningmsg+=stdmsg+'--rician given; overriding with --denoise\n'
        args.denoise = True

# Check to make sure CSF mask exists if given
if args.csfmask:
    if not op.exists(args.csfmask):
        errmsg+='--csfmask file '+args.csfmask+' not found\n'

# --force and --resume given
if args.resume and args.force:
    errmsg+=msgstart+'--continue and --force\n'

if args.output:
    if not op.isdir(args.output):
        try:
            os.mkdir(args.output)
        except:
            raise Exception('Output directory does not exist and cannot '
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

if args.rpe_pair:
    filetable['rpe_pair'] = DWIFile(args.rpe_pair)
if args.rpe_all:
    filetable['rpe_all'] = DWIFile(args.rpe_all)

# TODO: add check for the rpe specifiers so we fail BEFORE running things

# Get naming and location information
dwiname = filetable['dwi'].getName()
if not args.output:
    outpath = filetable['dwi'].getPath()
else:
    outpath = args.output

# Make the pipeline point to dwi as the last file since it's the only one
# so far
filetable['HEAD'] = filetable['dwi']

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
    # TODO: construct
    print('UNDER CONSTRUCTION, SORRY, SKIPPING...');

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
    smoothing.smooth_image(filetable['HEAD'].getFull(),
                           csfname=args.csfmask,
                           outname=smoothing_full,
                           width=1.2)
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
    if not (args.resume and op.exists(rician_full)):
        # system call
        if op.exists(rician_full) and not args.resume:
            raise Exception('Running rician correction would cause an '
                            'overwrite. '
                            'In order to run this please delete the '
                            'files, use --force, use --resume, or '
                            'change output destination.')
        else:
            rician.rician_img_correct(filetable['HEAD'].getFull(),
                          filetable['noisemap'].getFull(),
                          outpath=rician_full)
    filetable['rician_corrected'] = DWIFile(rician_full)
    filetable['HEAD'] = filetable['rician_corrected']
