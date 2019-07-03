import numpy as np
import scipy as scp
import nibabel as nib
import os
import numba
from numba import jit
import scipy.optimize as opt
import warnings
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

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
        if self.maxBval() <= 1500:
            type = 'dti'
            print('Maximum BVAL < 1500, image is DTI')
        elif self.maxBval() > 1500:
            type = 'dki'
            print('Maximum BVAL > 1500, image is DKI')
        else:
            raise ValueError('tensortype: Error in determining maximum BVAL')
        return type

    def createTensorOrder(self, order=None):
        # Creates the appropriate tensor order for ADC or AKC calculations
        imType = self.tensorType()
        if order is None:
            if imType == 'dti':
                cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
                ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3])) - 1
            elif imType == 'dki':
                cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1], dtype=int)
                ind = np.array(([1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 3], [1, 1, 2, 2], [1, 1, 2, 3], [1, 1, 3, 3], \
                                [1, 2, 2, 2], [1, 2, 2, 3], [1, 2, 3, 3], [1, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 3],
                                [2, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3])) - 1
        elif order == 2:
            print('User enforced tensor order 2 for DTI')
            cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
            ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3])) - 1
        elif order == 4:
            print('User enforced tensor order 4 for DKI')
            cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1], dtype=int)
            ind = np.array(([1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,2,2],[1,1,2,3],[1,1,3,3],\
                [1,2,2,2],[1,2,2,3],[1,2,3,3],[1,3,3,3],[2,2,2,2],[2,2,2,3],[2,2,3,3],[2,3,3,3],[3,3,3,3])) - 1
        else:
            raise ValueError('createTensorOrder: Please enter valid order values (2 or 4).')
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

    def fit(self):
        # Run fitting
        order = np.floor(np.log(np.abs(np.max(self.grad[:,-1])+1))/np.log(10))
        if order >= 2:
            self.grad[:, -1] = self.grad[:, -1]/1000

        self.img.astype(np.double)
        self.img[self.img <= 0] = np.finfo(np.double).eps

        self.grad.astype(np.double)
        normgrad = np.sqrt(np.sum(self.grad[:,:3]**2, 1))
        normgrad[normgrad == 0] = 1

        self.grad[:,:3] = self.grad[:,:3]/np.tile(normgrad, (3,1)).T
        self.grad[np.isnan(self.grad)] = 0

        dcnt, dind = self.createTensorOrder(2)
        wcnt, wind = self.createTensorOrder(4)

        ndwis = self.img.shape[-1]
        bs = np.ones((ndwis, 1))
        bD = np.tile(dcnt,(ndwis, 1))*self.grad[:,dind[:, 0]]*self.grad[:,dind[:, 1]]
        bW = np.tile(wcnt,(ndwis, 1))*self.grad[:,wind[:, 0]]*self.grad[:,wind[:, 1]]*self.grad[:,wind[:, 2]]*self.grad[:,wind[:, 3]]
        b = np.concatenate((bs, (np.tile(-self.grad[:,-1], (6,1)).T*bD), np.squeeze(1/6*np.tile(self.grad[:,-1], (15,1)).T**2)*bW), 1)

        dwi_ = self.vectorize()
        init = np.matmul(np.linalg.pinv(b), np.log(dwi_))
        shat = np.exp(np.matmul(b, init))

        print('...fitting with wlls')
        inputs = tqdm(range(0, dwi_.shape[1]))
        num_cores = multiprocessing.cpu_count()
        dt = Parallel(n_jobs=num_cores,prefer='processes')\
            (delayed(self.img)(shat[:,i], dwi_[:,i], b) for i in inputs)
        dt = np.reshape(dt, (dwi_.shape[1], b.shape[1])).T

        s0 = np.exp(dt[0,:])
        dt = dt[1:,:]
        D_apprSq = 1/(np.sum(dt[(0,3,5),:], axis=0)/3)**2
        dt[6:,:] = dt[6:,:]*np.tile(D_apprSq, (15,1))
        return dt, s0, b
