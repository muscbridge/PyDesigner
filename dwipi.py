import numpy as np
import scipy as scp
import nibabel as nib
import os
import numba
from numba import jit
import scipy.optimize as opt

class DWI(object):
    def __init__(self, imPath):
        if os.path.exists(imPath):
            assert isinstance(imPath, object)
            self.hdr = nib.load(imPath)
            self.img = np.array(self.hdr.dataobj)
            (path, file) = os.path.split(imPath)               # Get just NIFTI filename + extension
            fName = os.path.splitext(file)[0]                   # Remove extension from NIFTI filename
            bvalPath = os.path.join(path, fName + '.bval')      # Add .bval to NIFTI filename
            bvecPath = os.path.join(path, fName + '.bvec')      # Add .bvec to NIFTI filename
            if os.path.exists(bvalPath) and os.path.exists(bvecPath):
                bvecs = np.loadtxt(bvecPath)                    # Load bvecs
                bvals = np.rint(np.loadtxt(bvalPath))           # Load bvals
                self.grad = np.c_[np.transpose(bvecs), bvals]   # Combine bvecs and bvals into [n x 4] array where n is
                                                                #   number of DWI volumes. [Gx Gy Gz Bval]
            else:
                assert('Unable to locate BVAL or BVEC files')
            maskPath = os.path.join(path,'brain_mask.nii')
            if os.path.exists(maskPath):
                tmp = nib.load(maskPath)
                self.mask = np.array(tmp.dataobj)
                self.maskStatus = True
                print('Found brain mask')
            else:
                self.mask = np.ones((self.img.shape[0], self.img.shape[1], self.img.shape[2]), order='F')
                self.maskStatus = False
                print('No brain mask found')
        else:
            assert('File in path not found. Please locate file and try again')
        print('Image ' + fName + '.nii loaded successfully')

    def getBvals(self):
        # Loads a vector of bvals from .bval
        return self.grad[:,3]

    def getBvecs(self):
        # Loads a [N x 3] array of gradient directions from .bvec
        return self.grad[:,0:3]

    def maxBval(self):
        # Finds the maximum bval in a dataset to determine between DTI and DKI
        maxval = max(np.unique(self.grad[:,3]))
        return maxval

    def tensorType(self):
        # Determines whether the function is DTI
        if self.maxbval() <= 1500:
            type = 'dti'
            print('Maximum BVAL < 1500, image is DTI')
        elif self.maxbval() > 1500:
            type = 'dki'
            print('Maximum BVAL > 1500, image is DKI')
        else:
            raise ValueError('tensortype: Error in determining maximum BVAL')
        return type

    def createTensorOrder(self):
        # Creates the appropriate tensor order for ADC or AKC calculations
        if self.tensortype() == 'dti':
            cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
            ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3])) - 1
        if self.tensortype() == 'dki':
            cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1], dtype=int)
            ind = np.array(([1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,2,2],[1,1,2,3],[1,1,3,3],\
                [1,2,2,2],[1,2,2,3],[1,2,3,3],[1,3,3,3],[2,2,2,2],[2,2,2,3],[2,2,3,3],[2,3,3,3],[3,3,3,3])) - 1
        return cnt, ind

    def vectorize(self):
        # if the input is 1D or 2D, unpatch it to 3D or 4D using a mask
        # if the input is 3D or 4D, vectorize it using a mask
        if self.img.ndim == 1:
            self.img = np.expand_dims(self.img, axis=0)
        if self.img.ndim == 2:
            n = self.img.shape[0]
            s = np.zeros((self.mask.shape[0], self.mask.shape[1], self.mask.shape[2], n), order='F')
            for i in range(0, n):
                s[:,:,:,i] = np.reshape(self.img[i,:], (self.mask.shape), order='F')
        if self.img.ndim == 3:
            dwi = np.expand_dims(self.img, axis=-1)
        if self.img.ndim == 4:
            s = np.zeros((self.img.shape[-1], np.prod(self.mask.shape).astype(int)), order='F')
            for i in range(0, self.img.shape[-1]):
                tmp = self.img[:,:,:,i]
                maskind = np.ma.array(tmp, mask=self.mask)
                s[i,:] = np.ma.ravel(maskind, order='F').data
        return np.squeeze(s)
