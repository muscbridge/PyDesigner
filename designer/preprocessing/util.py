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

def find_valid_ext(pathname):
    """
    Finds valid extensions for dwifile, helper function

    Parameters
    ----------
    pathname : str
        The name to try and find extensions for

    Returns
    -------
    list of str
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

def json2fslgrad(path):
    """
    Creates FSL .bvec and .bval for series missing that information.
    Some datasets have their B0s separately that do not produce fsl
    gradients upon conversion to NifTi. This function creates those
    missing features for complete concatenation from .json file. Use
    with caution if and only if you know your input series is a DWI.

    Parameters
    ----------
    path : str
        Path to NifTi file

    Returns
    -------
    None; writes out file
    """
    image = DWIFile(path)
    if (image.getBVAL() is None) or (image.getBVEC() is None):
        if not image.hasJSON():
            raise Exception('It is not advisable to run multi-series '
                            'processing without `.json` files. Please '
                            'ensure your NifTi files come with .json '
                            'files.')
        # Get number of 3D volumes in image
        ndim = mrinfoutil.ndim(image.getFull())
        if ndim == 4:
            nDWI = mrinfoutil.size(image.getFull())[-1]
        else:
            nDWI = 1
        if nDWI <= 15:
            bval = np.zeros((1, nDWI), dtype=int)
            bvec = np.zeros((3, nDWI), dtype=int)
            fPath = op.splitext(path)[0]
            np.savetxt(op.join(image.getPath(), image.getName() + '.bvec'), 
            bvec, delimiter=' ', fmt='%d')
            np.savetxt(op.join(image.getPath(), image.getName() + '.bval'),
            np.c_[bval], delimiter=' ', fmt='%d')
        else:
            raise Exception('PyDesigner currently only supports '
                            'B0s without BVAL or BVEC pairs if '
                            'the number of volumes is less than '
                            '15. Please ensure all your input '
                            'volumes come with valid BVEC/BVAL '
                            'pairs and JSON.')

class DWIFile:
    """
    Diffusion data file object, used for handling paths and extensions.

    Helps interface different extensions, group .bval/.bvec seamlessly to
    the programmer. Offers interactive tools to try and locate a file if
    it can't be found. Likely a bit overkill, but reduces uberscript
    coding.

    Attributes
    ----------
    path : str
        The path to the file
    name : str
        The base name of the file
    ext : list of str
        The valid extensions for this grouping of dwi data
    acquisition : bool
        Indicates if a DWI acquisition or not, relevant for .bval/.bvec
    json : struct
        The contents of the .json metadata if available
    """

    def __init__(self, name):
        """
        Constructor for dwifile

        Attempts to find the file and launches interactive file-finder if
        it doesn't exist or can't be found.

        Parameters
        ----------
        name : str
            The name that the user has supplied for the file. Could be
            a whole path, could be a filename, could be basename.
        """
        full = name
        [pathname, ext] = op.splitext(full)
        if ext == '.gz':
            # split again
            pathname = op.splitext(pathname)[0]
            ext = '.nii.gz'
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
        """
        Get the name without the path for this dwifile

        Returns
        -------
        str
            Name of the file without extensions
        """
        return self.name

    def getPath(self):
        """
        Get the path without the name for this dwifile

        Returns
        -------
        str
            The path to this file
        """
        return self.path

    def getFull(self):
        """
        Get the path and name combined for this dwifile

        Returns
        -------
        str
            The full path and filename with extension
        """

        if '.nii' in self.ext:
            return op.join(self.path, self.name + '.nii')
        else:
            return op.join(self.path, self.name + '.nii.gz')

    def isAcquisition(self):
        """
        Check if this object is an acquisition

        Returns
        -------
        bool
            True if acquisition, False if not
        """

        return self.acquisition

    def hasJSON(self):
        """
        Checks if this object has a .json file

        Returns
        -------
        bool
            True if has .json, False if not
        """
        if self.json:
            return True
        else:
            return False

    def getJSON(self):
        """
        Returns the .json filename for this DWIFile

        Returns
        -------
        str
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
        """
        Returns the .bval filename for this DWIFile

        Returns
        -------
        str
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
        """
        Returns the .bvec filename for this DWIFile

        Returns
        -------
        str
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
        """
        Returns whether the volume is partial fourier encoded

        Returns
        -------
        bool
            True if encoding is partial fourier; False otherwise
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
    DWIlist : list of str
        Contains paths to all input series
    DWInlist : list of str
        Contains path to file names without extension
    DWIext : list of str
        Contains extension of input files
    BVALlist : list of str
        Contains paths to all BVAL files
    BVEClist : list of str
        Contains paths to all BVEC files
    JSONlist : list of str
        Contains paths to all JSON files
    nDWI : int
        Number of DWIs entered   
    """
    def __init__(self, path):
        """
        DWIParser class initiator

        Parameters
        ----------
        path : str
            path to input DWI

        Returns
        -------
        self : class
            DWIParser class object
        """
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

    def cat(self, path, ext='.nii', verbose=False, force=False,
            resume=False):
        """
        Concatenates all input series when nDWI > 1 into a 4D NifTi
        along with a appropriate BVAL, BVEC and JSON files.
        Concatenation of series via MRTRIX3 requires every NifTi file to
        come with BVAL/BVEC to produce a .json with `dw_scheme`.

        Parameters
        ----------
        path : str
            Directory where to store concatenated series
        ext : str
            Extenstion to save concatenated file in. Refer to MRTRIX3's
            `mrconvert` function for a list of possible extensions
        force : bool, optional
            Force overwrite of output files if pre-existing
            (Default:False)
        verbose : bool, optional
            Specify whether to print console output (Default: False)
        resume : bool, optional
            Continue from an aborted or partial previous run of
            pydesigner (Default: False)
        
        Returns
        -------
        None; writes out file
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
                        (not (op.exists(self.BVEClist[idx])) or \
                                (op.exists(self.BVALlist[idx]))):
                    try:
                        json2fslgrad(i)
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
        """
        Returns directory where first file in DWI list is stored

        Returns
        -------
        str
            directory of first DWI
        """
        path = os.path.dirname(os.path.abspath(self.DWIlist[0]))
        return path
