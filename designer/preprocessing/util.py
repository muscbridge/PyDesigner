"""
Adds utilities for the command-line interface
"""

import os.path as op # dirname, basename, join, splitext
import sys # exit
import json # decode
import pprint #pprint
import numpy as np
import math

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
                encoding = self.json['PartialFourier']
                encodingnumber = float(encoding)
                if encodingnumber != 1:
                    return True
                else:
                    return False

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
        self.DWIlist = [os.path.realpath(i) for i in UserCpath]
        acq = np.zeros((len(DWIlist)),dtype=bool)
        for i,fname in enumerate(DWIlist):
            acq[i] = DWIFile(fname).isAcquisition
        if not np.any(acq):
            raise Exception('One of the input sequences in not a '
            'valid DWI acquisition. Ensure that the NifTi file is '
            'present with its BVEC/BVAL pair.')
        DWIflist = [os.path.splitext(i) for i in DWIlist]
        self.DWInlist = [i[0] for i in DWIflist]
        self.BVALlist = [i + '.bval' for i in DWInlist]
        self.BVEClist = [i + '.bvec' for i in DWInlist]
        self.JSONlist = [i + '.json' for i in DWInlist]
        self.DWIext = [i[1] for i in DWIflist]
        self.nDWI = len(DWIlist)

    def cat(self, path):
        """Concatenates all input series when nDWI > 1 into a 4D NifTi
        along with a appropriate BVAL, BVEC and JSON files.

        Parameters
        ----------
        path:   string
            Directory where to store concatenated series
        """
        if self.nDWI <= 1:
            raise Exception('Nothing to concatenate when there is '
        'only one input series.')
        else:
            miflist = []
            for (idx, i) in enumerate(self.DWInlist):
                convert_arg = 'mrconvert -stride -1,2,3,4 -fslgrad ' + \
                    self.BVEClist[idx] + \
                    ' ' + \
                    self.BVALlist[idx] + \
                    ' ' + \
                    i + \
                    self.DWIext[idx] + \
                    ' ' + \
                    os.path.join(path, ('dwi' + str(idx) + '.mif'))
                miflist.append(os.path.join(path,
                ('dwi' + str(idx) + '.mif'))
                completion = subprocess.run(convert_arg)
                if completion.returncode != 0:
                    raise Exception('Conversion to .mif failed.')
            cat_arg = 'mrcat -axis 3 ' + \
                    DWImif + ' ' + \
                    os.path.join(path, ('dwi_designer' + '.mif'))
                    completion = subprocess.run(convert_arg)
                if completion.returncode != 0:
                    raise Exception('Failed to concatenate multiple '
                'series.')
