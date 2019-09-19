"""
Adds utilities for the command-line interface
"""

import os.path as op # dirname, basename, join, splitext
import sys # exit
import json # decode
import pprint #pprint

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

    def print(self, json=False):
        print('Path: ' + self.path)
        print('Name: ' + self.name)
        print('Extensions: ' + str(self.ext))
        print('Acquisition: ' + str(self.acquisition))
        if json:
            pprint.pprint(self.json)
