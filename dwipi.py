import numpy as np
import scipy as scp
import nibabel as nib
import os
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
        """
        Returns a vector of b-values, requires no input arguments
        Classification: Method

        Usage
        -----
        bvals = dwi.getBvals(), where dwi is the DWI class object
        """
        return self.grad[:,3]

    def getBvecs(self):
        """
        Returns an array of gradient vectors, requires no input parameters
        Classification: Method

        Usage
        -----
        bvecs = dwi.getBvecs(), where dwi is the DWI class object
        """
        return self.grad[:,0:3]

    def maxBval(self):
        """
        Returns the maximum b-value in a dataset to determine between DTI and DKI, requires no input parameters
        Classification: Method

        Usage
        -----
        a = dwi.maxBval(), where dwi is the DWI class object

        """
        return max(np.unique(self.grad[:,3]))

    def tensorType(self):
        """
        Returns whether input image is DTI or DKI compatible, requires no input parameters
        Classification: Method

        Usage
        -----
        a = dwi.tensorType(), where dwi is the DWI class object

        Returns
        -------
        a: 'dti' or 'dki' (string)
        """
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
        """
        Creates tensor order array and indices
        Classification: Method

        Usage
        -----
        (cnt ind) = dwi.createTensorOrder(order)

        Parameters
        ----------
        order: 2 or 4 (int or None)
               Tensor order number, 2 for diffusion and 4 for kurtosis. Default: None; auto-detect

        Returns
        -------
        cnt: vector (int)
        ind: array (int)

        Additional Information
        ----------------------
        The tensors for this pipeline are based on NYU's designer layout as depicted in the table below. This will soon
        be depreciated and updated with MRTRIX3's layout.
        =============================
        ------D------
        1  |    D11
        2  |    D12
        3  |    D13
        4  |    D22
        5  |    D23
        6  |    D33
        ------K------
       1  |   W1111
       2  |   W1112
       3  |   W1113
       4  |   W1122
       5  |   W1123
       6  |   W1133
       7  |   W1222
       8  |   W1223
       9  |   W1233
       10 |   W1333
       11 |   W2222
       12 |   W2223
       13 |   W2233
       14 |   W2333
       15 |   W3000
        """
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
        """
        Returns vectorized image based on brain mask, requires no input parameters
        If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
        If the input is 3D or 4D, vectorize it using a mask
        Classification: Method

        Usage
        -----
        vec = dwi.vectorize()

        Returns
        -------
        vec: N X number_of_voxels vector or array, where N is the number of DWI volumes
        """
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

    def wlls(self, shat, dwi, b):
        # compute a wlls fit using weights from inital fit shat
        w = np.diag(shat)
        dt = np.matmul(np.linalg.pinv(np.matmul(w, b)), np.matmul(w, np.log(dwi)))
        # for constrained fitting I'll need to modify this line. It is much slower than pinv so lets ignore for now.
        #dt = opt.lsq_linear(np.matmul(w, b), np.matmul(w, np.log(dwi)), \
        #     method='bvls', tol=1e-12, max_iter=22000, lsq_solver='exact')
        return dt

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
        self.b = np.concatenate((bs, (np.tile(-self.grad[:,-1], (6,1)).T*bD), np.squeeze(1/6*np.tile(self.grad[:,-1], (15,1)).T**2)*bW), 1)

        dwi_ = self.vectorize()
        init = np.matmul(np.linalg.pinv(self.b), np.log(dwi_))
        shat = np.exp(np.matmul(self.b, init))

        print('...fitting with wlls')
        inputs = tqdm(range(0, dwi_.shape[1]))
        num_cores = multiprocessing.cpu_count()
        self.dt = Parallel(n_jobs=num_cores,prefer='processes')\
            (delayed(self.wlls)(shat[:,i], dwi_[:,i], self.b) for i in inputs)
        self.dt = np.reshape(self.dt, (dwi_.shape[1], self.b.shape[1])).T
        self.s0 = np.exp(self.dt[0,:])
        self.dt = self.dt[1:,:]
        D_apprSq = 1/(np.sum(self.dt[(0,3,5),:], axis=0)/3)**2
        self.dt[6:,:] = self.dt[6:,:]*np.tile(D_apprSq, (15,1))