import numpy as np
import scipy as scp
import nibabel as nib
import os
import scipy.optimize as opt

class DWI(object):
    def __init__(self, imPath):
        if os.path.exists(imPath):
            self.hdr = nib.load(imPath)
            self.dwi = hdr.get_fdata
            (path, file) = os.path.split(niiPath)               # Get just NIFTI filename + extension
            fName = os.path.splitext(file)[0]                   # Remove extension from NIFTI filename
            bvalPath = os.path.join(path, fName + '.bval')      # Add .bval to NIFTI filename
            bvecPath = os.path.join(path, fName + '.bvec')      # Add .bvec to NIFTI filename
            if os.path.exists(bvalPath) and os.path.exists(bvecPath)
                bvecs = np.loadtxt(bvecPath)                    # Load bvecs
                bvals = np.rint(np.loadtxt(bvalPath))           # Load bvals
                self.grad = np.c_[np.transpose(bvecs), bvals]   # Combine bvecs and bvals into [n x 4] array where n is
                                                                #   number of DWI volumes. [Gx Gy Gz Bval]
            else:
                assert('Unable to locate BVAL or BVEC files')
        else:
            assert('File in path not found. Please locate file and try again')

    def
