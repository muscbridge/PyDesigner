"""
Adds utilities for the command-line interface
"""

import os.path as op # dirname, basename, join, splitext
import os
import sys # exit
import subprocess
import json # decode
import pprint #pprint
import numpy as np
import math
import warnings
from designer.preprocessing import mrinfoutil

def bvec_is_fullsphere(bvec):
    """Determines if .bvec file is full or half-sphere

    Parameters
    ----------
    bvec : string
        The filename of the bvec

    Returns
    -------
    True if full-sphere, False if half
    """

    # Check for existence
    if not op.exists(bvec):
        raise Exception('.bvec file '+bvec+' does not exist.')
    # Attempt to load file
    try:
        data = np.loadtxt(bvec)
    except:
        raise Exception('.bvec file '+bvec+' cannot be read.')

    # Transpose data so 2nd dimension is the axis i,j,k (x,y,z)
    data = np.transpose(data)
    # Check that data has correct dimensions
    size_dim0 = np.size(data, 0)
    if np.size(data, 1) != 3:
        raise Exception('.bvec file '+bvec+' is not 3D; is '+str(size_dim0))

    return vecs_are_fullsphere(data)

def vecs_are_fullsphere(bvecs):
    """Determines if input vectors are full or half-sphere

    Parameters
    ----------
    bvecs : ndarray
        Matrix of size [n_vectors x 3]

    Returns
    -------
    True if full-sphere, False if half

    Notes
    -----
    Adapted from Chris Rorden's nii_preprocess as seen here:
    https://github.com/neurolabusc/nii_preprocess/blob/dd1c84f23f8828923dd5fc493a22156b7006a3d4/nii_preprocess.m#L1786-L1824
    and reproduced from the original designer pipeline here:
    https://github.com/m-ama/PyDesigner/blob/7a39ec4cb9679f1c3e9ead68baa8f8c111b0618a/designer/designer.py#L347-L368
    """
    # TODO: figure out why this math works
    # Check dimensions
    if np.size(bvecs, 1) != 3:
        raise Exception('bvecs are not 3-dimensional.')
    # Remove NaNs
    bvecs[~np.isnan(bvecs.any(axis=1))]
    # Remove any 0-vectors
    bvecs[~np.all(bvecs == 0, axis=1)]
    # Assume half-sphere if no vectors remaining
    if not bvecs.any():
        raise Exception('bvecs do not point anywhere.')
    # Get mean length
    mean = np.mean(bvecs, 0)
    # Get x-component of direction
    x_component = np.sqrt(np.sum(np.power(mean, 2)))
    # Create a matrix to divide by this length
    Dx = np.repeat(x_component, 3)
    # Get mean unit length
    mean_ulength = np.divide(mean, Dx)
    # Scale by mean unit length
    mean = np.ones([np.size(bvecs, 0), 3])* mean_ulength
    # UNKNOWN
    minV = min(np.sum(bvecs.conj() * mean, axis=1))
    # Get angle in degrees
    theta = math.degrees(math.acos(minV))
    return (theta >= 110)

def find_valid_ext(pathname):
    """Finds valid extensions for dwifile, helper function

    Parameters
    ----------
    pathname : string
        The name to try and find extensions for

    Returns
    -------
    array
        Array of valid file extensions for the basename
    """
    # Go ahead and return blank if the pathname is blank
    if pathname is '':
        return []

    # Coerce into a basename only in case somebody is lazy before calling
    pathname = op.splitext(pathname)[0]

    exts = []

    # Figure out which extensions are valid
    valid_extensions = ('.nii.gz', '.nii', '.json', '.bval',
                        '.bvec')
    for ext in valid_extensions:
        if op.exists(pathname + ext):
            exts.append(ext)
    
    return exts

class DWIFile:
    """
    Diffusion data file object, used for handling paths and extensions.

    Helps interface different extensions, group .bval/.bvec seamlessly to
    the programmer. Offers interactive tools to try and locate a file if
    it can't be found. Likely a bit overkill, but reduces uberscript
    coding.

    Attributes
    ----------
    path : string
        The path to the file
    name : string
        The base name of the file
    ext : array
        The valid extensions for this grouping of dwi data
    acquisition : bool
        Indicates if a DWI acquisition or not, relevant for .bval/.bvec
    json : struct
        The contents of the .json metadata if available
    """

    def __init__(self, name):
        """Constructor for dwifile

        Attempts to find the file and launches interactive file-finder if
        it doesn't exist or can't be found.

        Parameters
        ----------
        name : string
            The name that the user has supplied for the file. Could be a
            whole path, could be a filename, could be basename.
        """
        full = name
        [pathname, ext] = op.splitext(full)
        self.path = op.dirname(pathname)
        self.name = op.basename(pathname)

        # Check for existence of the name
        self.ext = find_valid_ext(pathname)
        if not self.ext:
            raise Exception('File '+name+' is not a valid file.')

        # Figure out if dwi acquisition
        if (('.bval' in self.ext) and
            ('.bvec' in self.ext) and
            (('.nii.gz' in self.ext) or ('.nii' in self.ext))):
            self.acquisition = True
        else:
            self.acquisition = False

        # If JSON available, load it
        if ('.json' in self.ext):
            with open(op.join(self.path, self.name + '.json')) as f:
                self.json = json.load(f)
        else:
            self.json = None

    def getName(self):
        """Get the name without the path for this dwifile

        Returns
        -------
        string
            Name of the file without extensions
        """
        return self.name

    def getPath(self):
        """Get the path without the name for this dwifile

        Returns
        -------
        string
            The path to this file
        """
        return self.path

    def getFull(self):
        """Get the path and name combined for thsi dwifile

        Returns
        -------
        string
            The full path and filename with extension
        """

        if '.nii' in self.ext:
            return op.join(self.path, self.name + '.nii')
        else:
            return op.join(self.path, self.name + '.nii.gz')

    def isAcquisition(self):
        """Check if this object is an acquisition

        Returns
        -------
        bool
            True if acquisition, False if not
        """

        return self.acquisition

    def hasJSON(self):
        """Checks if this object has a .json file

        Returns
        -------
            True if has .json, False if not
        """
        if self.json:
            return True
        else:
            return False

    def getJSON(self):
        """Returns the .json filename for this DWIFile

        Returns
        -------
        string
            The full path to the .json file
        """
        if self.hasJSON():
            if self.path:
                return op.join(self.path, self.name + '.json')
            else:
                return self.name + '.json'
        else:
            return None

    def getBVAL(self):
        """Returns the .bval filename for this DWIFile

        Returns
        -------
        string
            The full path to the .bval
        """
        if self.isAcquisition():
            if self.path:
                return op.join(self.path, self.name + '.bval')
            else:
                return self.name + '.bval'
        else:
            return None

    def getBVEC(self):
        """Returns the .bvec filename for this DWIFile

        Returns
        -------
        string
            The full path to the .bvec
        """

        if self.isAcquisition():
            if self.path:
                return op.join(self.path, self.name + '.bvec')
            else:
                return self.name + '.bvec'
        else:
            return None
    def isPartialFourier(self):
        """Returns whether the volume is partial fourier encoded

        Returns
        -------
        boolean
            Whether the encoding is partial fourier or not
        """

        if not self.isAcquisition():
            raise Exception('Volume is not an acquisition volume.')
        else:
            if not self.getJSON():
                raise Exception('No access to Partial Fourier information.')
            else:
                if 'PartialFourier' in self.json:
                    encoding = self.json['PartialFourier']
                    encodingnumber = float(encoding)
                    if encodingnumber != 1:
                        return True
                    else:
                        return False
                elif ('PhaseEncodingSteps' in self.json) and \
                        ('AcquisitionMatrixPE' in self.json):
                    steps = int(self.json['PhaseEncodingSteps'])
                    acqmat = int(self.json['AcquisitionMatrixPE'])
                    if steps != acqmat:
                        return False
                    else:
                        return True

    def print(self, json=False):
        print('Path: ' + self.path)
        print('Name: ' + self.name)
        print('Extensions: ' + str(self.ext))
        print('Acquisition: ' + str(self.acquisition))
        if json:
            pprint.pprint(self.json)

class DWIParser:
    """
    Parses a list of DWIs and concatenates them into a single 4D NifTi
    with appropriate BVEC, BVALS.

    Attributes
    ----------
    DWIlist :   list of strings
        Contains paths to all input series
    DWInlist:   list of strings
        Contains path to file names without extension
    DWIext:     list of strings
        Contains extension of input files
    BVALlist:   list of strings
        Contains paths to all BVAL files
    BVEClist:   list of strings
        Contains paths to all BVEC files
    JSONlist:   list of strings
        Contains paths to all JSON files
    nDWI:       int
        Number of DWIs entered   
    """
    def __init__(self, path):
        UserCpath = path.rsplit(',')
        self.DWIlist = [op.realpath(i) for i in UserCpath]
        # This loop determines the types of inputs parsed into the function
        ftype = []
        i = 0
        while i < len(self.DWIlist):
            ftype.append(mrinfoutil.format(self.DWIlist[i]).lower())
            i += 1
        self.InputType = np.unique(ftype)
        if self.InputType.size > 1:
            raise IOError('It appears that multiple types of files '
                            'have been parsed as input. Please input '
                            'a specific type of input exclusively. '
                            'Detected inputs were: {}'.format(
                self.InputType))
        self.InputType = self.InputType[0]
        DWIflist = [op.splitext(i) for i in self.DWIlist]
        # Compressed nifti (.nii.gz) have double extensions, and so
        # require double ext-splitting. The following loop takes care of
        # that.
        if '.gz' in np.unique(np.array(DWIflist)[:,-1])[0]:
            for idx, i in enumerate(DWIflist):
                DWIflist[idx] = (op.splitext(i[0])[0], '.nii.gz')

        self.DWInlist = [i[0] for i in DWIflist]
        if 'nifti' in self.InputType:
            self.BVALlist = [i + '.bval' for i in self.DWInlist]
            self.BVEClist = [i + '.bvec' for i in self.DWInlist]
            self.JSONlist = [i + '.json' for i in self.DWInlist]
        self.DWIext = [i[1] for i in DWIflist]
        self.nDWI = len(self.DWIlist)

    # def makeext(self, dwipath):
    #     """
    #     Makes FSL grad and JSON extensions for files that don't posses
    #     header information
    #
    #     Parameters
    #     ----------
    #     dwipath:    string
    #                 path to dwi without extension
    #
    #     Returns
    #     -------
    #     BVALpath:   string
    #                 path to BVAL
    #     BVECpath:   string
    #                 path to BVEC
    #     JSONpath:   string
    #                 path to JSON
    #     """
    #     # First, check whether an extension to input path exists
    #     splitpath = op.splitext(dwipath)
    #     # If an extension does not exist, entered path is valid
    #     if not splitpath[1]:
    #         BVALpath = dwipath + '.bval'
    #         BVECpath = dwipath + '.bvec'
    #         JSONpath = dwipath + '.json'
    #     else:
    #         raise Exception('Ensure that the path to DWI is entered '
    #                         'without an extension')
    #     return BVALpath, BVECpath, JSONpath

    def cat(self, path, ext='.nii', verbose=False, force=False,
            resume=False):
        """Concatenates all input series when nDWI > 1 into a 4D NifTi
        along with a appropriate BVAL, BVEC and JSON files.
        Concatenation of series via MRTRIX3 requires every NifTi file to
        come with BVAL/BVEC to produce a .json with `dw_scheme`.

        Parameters
        ----------
        path:       string
            Directory where to store concatenated series
        ext:        string
            Extenstion to save concatenated file in. Refer to MRTRIX3's
            `mrconvert` function for a list of possible extensions
        verbose:    bool
            Displays MRTRIX3's console output
        force:      bool
            Forces file overwrite if they already exist
        """
        # Check whether working.(ext) exists
        if op.exists(op.join(path, 'working' + ext)):
            if force:
                os.remove(op.join(path, 'working' + ext))
                for i in range(self.nDWI):
                    if op.exists(op.join(path, ('dwi' + str(i) + '.mif'))):
                        os.remove(op.join(path, ('dwi' + str(i) + '.mif')))
            elif not resume:
                raise IOError(
                    'Concatenated series already exists. '
                    'In order to run this please delete the '
                    'file working, use --force, use --resume, or '
                    'change output destination.')

        if not (resume and op.exists(op.join(path, 'working' + ext))):
            miflist = []
            # The following loop converts input file into .mif
            for (idx, i) in enumerate(self.DWIlist):
                if 'nifti' in self.InputType and \
                        not op.exists(self.JSONlist[idx]):
                    try:
                        self.json2fslgrad(i)
                    except:
                        raise IOError('Please supply a valid JSON file '
                                      'accompanying {}'.format(i))
                convert_args = ['mrconvert -stride 1,2,3,4']
                if verbose is False:
                    convert_args.append('-quiet')
                if force is True:
                    convert_args.append('-force')
                if hasattr(self, 'BVEClist') or hasattr(self, 'BVALlist'):
                    if op.exists(self.BVEClist[idx]) or \
                            op.exists(self.BVALlist[idx]):
                        convert_args.append('-fslgrad')
                        convert_args.append(self.BVEClist[idx])
                        convert_args.append(self.BVALlist[idx])
                    else:
                        raise FileNotFoundError('BVEC and BVAL pairs for the '
                                                'input paths do not exist. '
                                                'Ensure that they exist or '
                                                'have the same name as DWI.')
                if hasattr(self, 'JSONlist'):
                    if op.exists(self.JSONlist[idx]):
                        convert_args.append('-json_import')
                        convert_args.append(self.JSONlist[idx])
                    else:
                        warnings.warn('JSON file(s) {} not found. '
                                      'Attempting to process without. '
                                      'If processing fails, please use the '
                                      '"--adv" flag'.format(JSONlist))
                convert_args.append(i)
                convert_args.append(
                    op.join(path, ('dwi' + str(idx) + '.mif')))
                miflist.append(op.join(path, ('dwi' + str(idx) + '.mif')))
                cmd = ' '.join(str(e) for e in convert_args)
                completion = subprocess.run(cmd, shell=True)
                if completion.returncode != 0:
                    raise Exception('Please use the "--force" flag to '
                                    'overwrite existing outputs, or clear '
                                    'the output directory')
            # The following command concatenates all DWI(i) into a single
            # .mif file if nDWI > 1
            if self.nDWI > 1:
                cat_arg = ['mrcat -axis 3']
                if verbose is False:
                    cat_arg.append('-quiet')
                if force is True:
                    cat_arg.append('-force')
                for i,fname in enumerate(miflist):
                    cat_arg.append(fname)
                cat_arg.append(
                    op.join(path, ('working' + '.mif')))
                cmd = ' '.join(str(e) for e in cat_arg)
                completion = subprocess.run(cmd, shell=True)
                if completion.returncode != 0:
                    raise Exception('Failed to concatenate multiple '
                'series.')
            else:
                cat_arg = ['mrconvert']
                if verbose is False:
                    cat_arg.append('-quiet')
                if force is True:
                    cat_arg.append('-force')
                cat_arg.append(op.join(path, 'dwi0.mif'))
                cat_arg.append(op.join(path, 'working.mif'))
                cmd = ' '.join(str(e) for e in cat_arg)
                print(cmd)
                completion = subprocess.run(cmd, shell=True)
                if completion.returncode != 0:
                    raise Exception('Failed to convert single series')
            if '.mif' not in ext:
                miflist.append(op.join(path, 'working' + '.mif'))

            # Output concatenated .mif into other formats
            if '.mif' not in ext:
                convert_args = ['mrconvert -stride 1,2,3,4']
                if verbose is False:
                    convert_args.append('-quiet')
                if force is True:
                    convert_args.append('-force')
                convert_args.append('-export_grad_fsl')
                convert_args.append(op.join(path, 'working.bvec'))
                convert_args.append(op.join(path, 'working.bval'))
                convert_args.append('-json_export')
                convert_args.append(op.join(path, 'working.json'))
                convert_args.append(op.join(path, 'working.mif'))
                convert_args.append(op.join(path, 'working' + ext))
                cmd = ' '.join(str(e) for e in convert_args)
                completion = subprocess.run(cmd, shell=True)
                if completion.returncode != 0:
                    for i, fname in enumerate(miflist):
                        os.remove(fname)
                    os.remove(op.join(path, 'working' + ext))
                    raise Exception('Concatenation to ' + str(ext) + ' '
                                    'failed. Please ensure that your input '
                                    'NifTi files have the same phase '
                                    'encoding directions, and are '
                                    'accompanied by valid .bval, .bvec, '
                                    'and .json. If this is not possible, '
                                    'please provide manually concatenated '
                                    'DWIs or run with single series input.')
            for i, fname in enumerate(miflist):
                os.remove(fname)

    def getPath(self):
        """Returns directory where first file in DWI list is stored
        """
        path = os.path.dirname(os.path.abspath(self.DWIlist[0]))
        return path

    def json2fslgrad(self, path):
        """Creates FSL .bvec and .bval for series missing that information.
        Some datasets have their B0s separately that do not produce fsl
        gradients upon conversion to NifTi. This function creates those
        missing features for complete concatenation from .json file. Use
        with caution if and only if you know your input series is a DWI.

        Parameters
        ----------
        path:   string
            Path to NifTi file
        """
        image = DWIFile(path)
        if not image.hasJSON():
            raise Exception('It is not advisable to run multi-series '
                            'processing without `.json` files. Please '
                            'ensure your NifTi files come with .json '
                            'files.')
        args_info = ['mrinfo', path]
        cmd = ' '.join(str(e) for e in args_info)
        # Reads the "Dimension line of `mrinfo` and extracts the size
        # of NifTi
        pipe = subprocess.Popen(cmd, shell=True,
                                stdout=subprocess.PIPE)
        strlist = pipe.stdout.readlines()[3].split()
        dims = [int(i) for i in strlist if i.isdigit()]
        nDWI = dims[-1]
        # Check whether for inexistence of gradient table in JSON and
        # some mention of B0 in EPI
        if ('b0' in image.json['SeriesDescription'] or \
                'B0' in image.json['SeriesDescription'] or \
                'b0' in image.json['ProtocolName'] or \
                'B0' in image.json['ProtocolName']):
            bval = np.zeros(nDWI, dtype=int)
            bvec = np.zeros((3, nDWI), dtype=int)
            fPath = op.splitext(path)[0]
            np.savetxt((fPath + '.bvec'), bvec, delimiter=' ', fmt='%d')
            np.savetxt((fPath + '.bval'), np.c_[bval], delimiter=' ',
                       fmt='%d')