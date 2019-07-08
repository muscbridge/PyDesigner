import numpy as np
import scipy as scp
import nibabel as nib
import os
import scipy.optimize as opt
import warnings
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# Define the lowest number possible before it is considred a zero
minZero = 1e-8

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

    def vectorize(self, dwi, mask):
        """
        Returns vectorized image based on brain mask, requires no input parameters
        If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
        If the input is 3D or 4D, vectorize it using a mask
        Classification: Method

        Usage
        -----
        vec = dwi.vectorize(img) if there's no mask
        vec = dwi.vectorize(img, mask) if there's a mask

        Returns
        -------
        vec: N X number_of_voxels vector or array, where N is the number of DWI volumes
        """
        if mask is None:
            mask = np.ones((dwi.shape[0], dwi.shape[1], dwi.shape[2]), order='F')
        if dwi.ndim == 1:
            dwi = np.expand_dims(dwi, axis=0)
        if dwi.ndim == 2:
            n = dwi.shape[0]
            s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], n), order='F')
            for i in range(0, n):
                s[:,:,:,i] = np.reshape(dwi[i,:], (mask.shape), order='F')
        if dwi.ndim == 3:
            dwi = np.expand_dims(dwi, axis=-1)
        if dwi.ndim == 4:
            s = np.zeros((dwi.shape[-1], np.sum(mask).astype(int)), order='F')
            for i in range(0, dwi.shape[-1]):
                tmp = dwi[:,:,:,i]
                maskind = np.ma.array(tmp, mask=mask)
                s[i,:] = np.ma.ravel(maskind, order='F').data
        return np.squeeze(s)

    def fibonacciSphere(self, samples=1, randomize=True):
        """
        Returns "samples" evenly spaced points on a sphere
        :param samples:
        :param randomize:
        :return:
        """
        import random
        rnd = 1
        if randomize:
            rnd = random.random() * samples
        points = []
        offset = 2/samples
        increment = np.pi * (3. - np.sqrt(5.))
        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - pow(y,2))
            phi = ((i + rnd) % samples) * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            points.append([x,y,z])
        return points

    def radialSampling(self, dir, n):
        # get the radial component of a set of directions
        dt = 2*np.pi/n
        theta = np.arange(0,2*np.pi-dt,dt)
        dirs = np.vstack((np.cos(theta), np.sin(theta), 0*theta))
        v = np.hstack((-dir[1], dir[0], 0))
        s = np.sqrt(np.sum(v**2))
        c = dir[2]
        V = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        R = np.eye(3) + V + np.matmul(V,V) * (1-c)/(s**2)
        dirs = np.matmul(R, dirs)
        return dirs

    def diffusionCoeff(self, dt, dir):
        # compute ADC
        dcnt, dind = self.createTensorOrder(2)
        ndir = dir.shape[0]
        bD = np.tile(dcnt,(ndir, 1)) * dir[:,dind[:, 0]] * dir[:,dind[:, 1]]
        adc = np.matmul(bD, dt)
        return adc

    def kurtosisCoeff(self, dt, dir):
        # compute AKC
        wcnt, wind = self.createTensorOrder(4)
        ndir = dir.shape[0]
        adc = self.diffusionCoeff(dt[:6], dir)
        md = np.sum(dt[np.array([0,3,5])], 0)/3
        bW = np.tile(wcnt,(ndir, 1)) * dir[:,wind[:, 0]] * dir[:,wind[:, 1]] * dir[:,wind[:, 2]] * dir[:,wind[:, 3]]
        akc = np.matmul(bW, dt[6:])
        akc = (akc * np.tile(md**2, (adc.shape[0], 1)))/(adc**2)
        return akc

    def dtiTensorParams(self, nn):
        # compute dti tensor eigenvalues and eigenvectors and sort them
        values, vectors = np.linalg.eig(nn)
        idx = np.argsort(-values)
        values = -np.sort(-values)
        vectors = vectors[:, idx]
        return values, vectors

    def dkiTensorParams(self, v1, dt):
        # kurtosis tensor parameters use average directional
        # statistics to approximate ak and rk
        dirs = np.vstack((v1, -v1))
        akc = self.kurtosisCoeff(dt, dirs)
        ak = np.mean(akc)
        dirs = self.radialSampling(v1, 256).T
        akc = self.kurtosisCoeff(dt, dirs)
        rk = np.mean(akc)
        return ak, rk

    def wlls(self, shat, dwi, b):
        # compute a wlls fit using weights from inital fit shat
        w = np.diag(shat)
        dt = np.matmul(np.linalg.pinv(np.matmul(w, b)), np.matmul(w, np.log(dwi)))
        # for constrained fitting I'll need to modify this line. It is much slower than pinv so lets ignore for now.
        #dt = opt.lsq_linear(np.matmul(w, b), np.matmul(w, np.log(dwi)), \
        #     method='bvls', tol=1e-12, max_iter=22000, lsq_solver='exact')
        return dt

    def fit(self):
        """
        Returns fitted diffusion or kurtosis tensor
        :return:
        """
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

        dwi_ = self.vectorize(self.img, None)
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
        return self.dt

    def extract(self):
        # extract all tensor parameters from dt
        num_cores = multiprocessing.cpu_count()

        print('...extracting dti parameters')
        DT = np.reshape(
            np.concatenate((self.dt[0, :], self.dt[1, :], self.dt[2, :], self.dt[1, :], self.dt[3, :], self.dt[4, :], self.dt[2, :], self.dt[4, :], self.dt[5, :])),
            (3, 3, self.dt.shape[1]))

        # get the trace
        rdwi = scp.special.expit(np.matmul(self.b[:, 1:], self.dt))
        B = np.round(-(self.b[:, 0] + self.b[:, 3] + self.b[:, 5]) * 1000)
        uB = np.unique(B)
        trace = np.zeros((self.dt.shape[1], uB.shape[0]))
        for ib in range(0, uB.shape[0]):
            t = np.where(B == uB[ib])
            trace[:, ib] = np.mean(rdwi[t[0], :], axis=0)

        nvox = self.dt.shape[1]
        inputs = range(0, nvox)
        values, vectors = zip(*Parallel(n_jobs=num_cores, prefer='processes') \
            (delayed(self.dtiTensorParams)(DT[:, :, i]) for i in inputs))
        values = np.reshape(np.abs(values), (nvox, 3))
        vectors = np.reshape(vectors, (nvox, 3, 3))

        print('...extracting dki parameters')
        dirs = np.array(self.fibonacciSphere(256, True))
        akc = self.kurtosisCoeff(self.dt, dirs)
        mk = np.mean(akc, 0)
        ak, rk = zip(*Parallel(n_jobs=num_cores, prefer='processes') \
            (delayed(self.dkiTensorParams)(vectors[i, :, 0], self.dt[:, i]) for i in inputs))
        ak = np.reshape(ak, (nvox))
        rk = np.reshape(rk, (nvox))

        l1 = self.vectorize(values[:, 0], self.mask)
        l2 = self.vectorize(values[:, 1], self.mask)
        l3 = self.vectorize(values[:, 2], self.mask)
        v1 = self.vectorize(vectors[:, :, 0].T, self.mask)

        md = (l1 + l2 + l3) / 3
        rd = (l2 + l3) / 2
        ad = l1
        fa = np.sqrt(1 / 2) * np.sqrt((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / np.sqrt(
            l1 ** 2 + l2 ** 2 + l3 ** 2)
        trace = self.vectorize(trace.T, self.mask)
        fe = np.abs(np.stack((fa * v1[:, :, :, 0], fa * v1[:, :, :, 1], fa * v1[:, :, :, 2]), axis=3))
        ak = self.vectorize(ak, self.mask)
        rk = self.vectorize(rk, self.mask)
        mk = self.vectorize(mk, self.mask)
        return md, rd, ad, fa, fe, trace, mk, ak, rk

    def findViols(self, img, c=[0, 1, 0]):
        """
        Returns a 3D violation map of voxels that violate constraints\
        Classification: Method

        Usage
        -----
        map = findViols(img, [0 1 0]

        Parameters
        ----------
        img: 3D metric array such as mk or fa
        c:   [3 x 1] vector that toggles which constraints to check
             c[0]: Check D < 0 constraint
             c[1]: Check K < 0 constraint
             c[2]: Check K > 3/(b*D) constraint

        Returns
        -------
        map: 3D array containing locations of voxels that incur directional violations

        """
        if c == None:
            c = [0, 0, 0]

        nVoxels = np.prod(img.shape)
        sumViols = np.zeros(nVoxels)
        maxB = self.maxBval()
        adc = self.diffusionCoeff(self.dt[:6], dirs)
        akc = self.kurtosisCoeff(self.dt, dirs)
        nDirs = dirs.shape[0]
        tmp = np.zeros(3)
        for i in range(nVoxels):
            # C[0]: D < 0
            tmp[0] = np.size(np.nonzero(adc[:, i] < minZero))
            # C[1]: K < 0
            tmp[1] = np.size(np.nonzero(akc[:, i] < minZero))
            #c[2]:
            tmp[2] = np.size(np.nonzero(akc[:, i] > (3/adc[:, i])))
            sumViols[i] = tmp[0] + tmp[1] + tmp[2]



