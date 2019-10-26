#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
import os
import random as rnd
import warnings
import cvxpy as cvx
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit as sigmoid
from tqdm import tqdm
from . import dwidirs
warnings.filterwarnings("ignore")

# Define the lowest number possible before it is considered a zero
minZero = 1e-8

# Define number of directions to resample after computing all tensors
dirSample = 256

# Progress bar Properties
tqdmWidth = 70  # Number of columns of progress bar

class DWI(object):
    def __init__(self, imPath, nthreads=-1):
        if os.path.exists(imPath):
            assert isinstance(imPath, object)
            self.hdr = nib.load(imPath)
            self.img = np.array(self.hdr.dataobj)
            nanidx = np.isnan(self.img)
            self.img[nanidx] = minZero
            # Get just NIFTI filename + extensio
            (path, file) = os.path.split(imPath)
            # Remove extension from NIFTI filename
            fName = os.path.splitext(file)[0]
            # Add .bval to NIFTI filename
            bvalPath = os.path.join(path, fName + '.bval')
            # Add .bvec to NIFTI filename
            bvecPath = os.path.join(path, fName + '.bvec')
            bvecPath = os.path.join(path, fName + '.bvec')
            if os.path.exists(bvalPath) and os.path.exists(bvecPath):
                # Load bvecs
                bvecs = np.loadtxt(bvecPath)
                # Load bvals
                bvals = np.rint(np.loadtxt(bvalPath)) / 1000
                # Combine bvecs and bvals into [n x 4] array where n is
                # number of DWI volumes. [Gx Gy Gz Bval]
                self.grad = np.c_[np.transpose(bvecs), bvals]
            else:
                raise NameError('Unable to locate BVAL or BVEC files')
            maskPath = os.path.join(path,'brain_mask.nii')
            if os.path.exists(maskPath):
                tmp = nib.load(maskPath)
                self.mask = np.array(tmp.dataobj).astype(bool)
                self.maskStatus = True
            else:
                self.mask = np.ones((self.img.shape[0], self.img.shape[
                    1], self.img.shape[2]), order='F')
                self.maskStatus = False
                print('No brain mask supplied')
        else:
            assert('File in path not found. Please locate file and try '
                   'again')
        tqdm.write('Image ' + fName + '.nii loaded successfully')
        if not isinstance(nthreads, int):
            raise Exception('Variable nthreads need to be an integer')
        if nthreads < -1 or nthreads == 0:
            raise Exception('Variable nthreads is a positive integer or '
                            '-1')
        if nthreads is None:
            self.workers = -1
        else:
            self.workers = nthreads
        if self.workers == -1:
            tqdm.write('Processing with ' +
                       np.str(multiprocessing.cpu_count()) +
                       ' workers...')
        else:
            tqdm.write('Processing with ' +
                       np.str(self.workers) +
                       ' workers...')

    def getBvals(self):
        """Returns a vector of b-values, requires no input arguments
        Classification: Method

        Usage
        -----
        bvals = dwi.getBvals(), where dwi is the DWI class object
        """
        return self.grad[:,3]

    def getBvecs(self):
        """Returns an array of gradient vectors, requires no input
        parameters
        Classification: Method

        Usage
        -----
        bvecs = dwi.getBvecs(), where dwi is the DWI class object
        """
        return self.grad[:,0:3]

    def maxBval(self):
        """ Returns the maximum b-value in a dataset to determine between
        DTI and DKI, requires no input parameters
        Classification: Method

        Usage
        -----
        a = dwi.maxBval(), where dwi is the DWI class object

        """
        return max(np.unique(self.grad[:,3])).astype(int)

    def getndirs(self):
        """Returns the number of gradient directions acquired from scanner

        Usage
        -----
        n = dwi.getndirs(), where dwi is the DWI class object

        Retun
        n: integer quantifying number of directions
        """
        return np.sum(self.grad[:, 3] == self.maxBval())

    def tensorType(self):
        """Returns whether input image is DTI or DKI compatible, requires
        no input parameters
        Classification: Method

        Usage
        -----
        a = dwi.tensorType(), where dwi is the DWI class object

        Returns
        -------
        a: 'dti' or 'dki' (string)
        """
        if self.maxBval() <= 1.5:
            type = 'dti'
        elif self.maxBval() > 1.5:
            type = 'dki'
        else:
            raise ValueError('tensortype: Error in determining maximum '
                             'BVAL')
        return type

    def isdki(self):
        """Returns logical value to answer the mystical question whether
        the input image is DKI

        Usage
        -----
        ans = dwi.isdki(), where dwi is the DWI class object

        Returns
        -------
        ans:    True or False (bool)
        """
        if self.tensorType() == 'dki':
            ans = True
        else:
            ans = False
        return ans

    def createTensorOrder(self, order=None):
        """Creates tensor order array and indices
        Classification: Method

        Usage
        -----
        (cnt, ind) = dwi.createTensorOrder(order)

        Parameters
        ----------
        order:  2 or 4 (int or None)
                Tensor order number, 2 for diffusion and 4 for kurtosis.
                Default: None; auto-detect

        Returns
        -------
        cnt:    vector (int)
        ind:    array (int)

        Additional Information
        ----------------------
        The tensors for this pipeline are based on NYU's designer layout as
        depicted in the table below. This will soon be depreciated and
        updated with MRTRIX3's layout.
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
       15 |   W3333
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
            cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
            ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3])) - 1
        elif order == 4:
            cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1], dtype=int)
            ind = np.array(([1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,2,2],[1,1,2,3],[1,1,3,3],\
                [1,2,2,2],[1,2,2,3],[1,2,3,3],[1,3,3,3],[2,2,2,2],[2,2,2,3],[2,2,3,3],[2,3,3,3],[3,3,3,3])) - 1
        else:
            raise ValueError('createTensorOrder: Please enter valid order values (2 or 4).')
        return cnt, ind

    def fibonacciSphere(self, samples=1, randomize=True):
        """Returns evenly spaced points on a sphere
        Classification: Method

        Usage
        ----
        dirs = dwi.fibonacciSphere(256, True)

        Parameters
        ----------
        samples:    Positive real integer
                    number of points to compute from sphere

        randomize:  True or False (bool)
                    choose whether sampling is random or not

        Returns
        ------
        points:     [3 x samples] array containing evenly spaced points
        from a sphere
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
        """Get the radial component of a metric from a set of directions
        Classification: Method

        Usage
        -----
        grad = dwi.radiansampling(dir, number_of_dirs)

        Parameters
        ----------
        dir:    [n x 3] input matrix
        n:      number of rows, n

        Returns
        -------
        dirs:   Matrix containing radial components

        """
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
        """Computes apparent diffusion coefficient (ADC)
        Classification: Method

        Usage
        -----
        adc = dwi.diffusionCoeff(dt, dir)

        Parameters
        ----------
        dt:     [21 x nvoxel] array containing diffusion tensor
        dir:    [n x 3] array containing gradient directions

        Returns
        -------
        adc:    matrix containing apparent diffusion coefficient
        """
        dcnt, dind = self.createTensorOrder(2)
        ndir = dir.shape[0]
        bD = np.tile(dcnt,(ndir, 1)) * dir[:,dind[:, 0]] * \
             dir[:, dind[:, 1]]
        adc = np.matmul(bD, dt)
        return adc

    def kurtosisCoeff(self, dt, dir):
        """Computes apparent kurtosis coefficient (AKC)
        Classification: Method

        Usage
        -----
        adc = dwi.kurtosisCoeff(dt, dir)

        Parameters
        ----------
        dt:     [21 x nvoxel] array containing diffusion tensor
        dir:    [n x 3] array containing gradient directions

        Returns
        ------
        adc:    matrix containing apparent kurtosis coefficient
        """
        wcnt, wind = self.createTensorOrder(4)
        ndir = dir.shape[0]
        adc = self.diffusionCoeff(dt[:6], dir)
        adc[adc == 0] = minZero
        md = np.sum(dt[np.array([0,3,5])], 0)/3
        bW = np.tile(wcnt,(ndir, 1)) * dir[:,wind[:, 0]] * \
             dir[:,wind[:,1]] * dir[:,wind[:, 2]] * dir[:,wind[:, 3]]
        akc = np.matmul(bW, dt[6:])
        akc = (akc * np.tile(md**2, (adc.shape[0], 1)))/(adc**2)
        return akc

    def dtiTensorParams(self, nn):
        """Computes sorted DTI tensor eigenvalues and eigenvectors
        Classification: Method

        Usage
        -----
        (values, vectors) = dwi.dtiTensorParams(DT)

        Parameters
        ----------
        DT:         diffusion tensor

        Returns
        -------
        values:     sorted eigenvalues
        vectors:    sorted eigenvectors
        """
        values, vectors = np.linalg.eig(nn)
        idx = np.argsort(-values)
        values = -np.sort(-values)
        vectors = vectors[:, idx]
        return values, vectors

    def dkiTensorParams(self, v1, dt):
        """Uses average directional statistics to approximate axial
        kurtosis(AK) and radial kurtosis (RK)
        Classification: Method

        Usage
        -----
        (rk, ak) = dwi.dkiTensorParams(v1, dt)

        Parameters
        ----------
        v1:     first eigenvector
        dt:     diffusion tensor

        Returns
        -------
        rk:     radial kurtosis
        ak:     axial kurtosis
        kfa:    kurtosis fractional anisotropy
        mkt:    mean kurtosis tensor
        """
        dirs = np.vstack((v1, -v1))
        akc = self.kurtosisCoeff(dt, dirs)
        ak = np.mean(akc)
        dirs = self.radialSampling(v1, dirSample).T
        akc = self.kurtosisCoeff(dt, dirs)
        rk = np.mean(akc)
        W_F = np.sqrt(dt[6]**2 + \
                      dt[16]**2 + \
                      dt[20]**2 + \
                      6 * dt[9]**2+ \
                      6 * dt[11]**2 + \
                      6 * dt[18]**2 + \
                      4 * dt[7]**2 + \
                      4 * dt[8]**2 + \
                      4 * dt[12]**2 + \
                      4 * dt[17]**2 + \
                      4 * dt[15]**2 + \
                      4 * dt[19]**2 + \
                      12 * dt[10]**2 + \
                      12 * dt[13]**2 + \
                      12 * dt[14]**2)
        Wbar = 1/5 * (dt[6] + dt[16] + dt[20] + 2 *
                      (dt[9] + dt[11] + dt[18]))
        if W_F < minZero:
            kfa = 0
        else:
            W_diff = np.sqrt(
                (dt[6] - Wbar)**2 + \
                (dt[16] - Wbar) ** 2 + \
                (dt[20] - Wbar)**2 + \
                6 * (dt[9] - Wbar / 3)**2 + \
                6 * (dt[11] - Wbar / 3)**2 + \
                6 * (dt[18] - Wbar / 3)**2 + \
                4 * dt[7]**2 + \
                4 * dt[8]**2 + \
                4 * dt[12]**2 + \
                4 * dt[17]**2 + \
                4 * dt[15]**2 + \
                4 * dt[19]**2 + \
                12 * dt[10]**2 + \
                12 * dt[13]**2 + \
                12 * dt[14]**2)
            kfa = W_diff / W_F
        mkt = Wbar
        return ak, rk, kfa, mkt

    def wlls(self, shat, dwi, b, cons=None, warmup=None):
        """Estimates diffusion and kurtosis tenor at voxel with
        unconstrained Moore-Penrose pseudoinverse or constrained
        quadratic convex optimization. This is a helper function for
        dwi.fit() so a multiprocessing parallel loop can be iterated over
        voxels
        Classification: Method

        For Unconstrained Fitting:
        In the absence of constraints, an exact formulation in the form
        Cx = b is produced. This is further simplified to x_hat = C^+ *
        b. One can use the Moore-Penrose method to compute the
        pseudoinverse to approximate diffusion tensors.

        For Constrained Fitting:
        The equation |Cx -b|^2 expands to 0.5*x.T(C.T*A)*x -(C.T*b).T
                                                 -------     ------
                                                    P           q
        where A is denoted by multiplier matrix (w * b)
        Multiplying by a positive constant (0.5) does not change the value
        of optimum x*. Similarly, the constant offset b.T*b does not
        affect x*, therefore we can leave these out.

        Minimize: || C*x -b ||_2^2
            subject to A*x <= b
            No lower or upper bounds

        Usage
        -----
        dt = dwi.wlls(shat, dwi, b, constraints)

        Parameters
        ----------
        shat:   [ndir x 1] vector (float)
                S_hat, approximated signal intensity at voxel
        dwi:    [ndir x 1] vector (float)
                diffusion weighted intensity values at voxel, for all
                b-values
        b:      [ndir x 1] vector (float)
                b-values vector
        cons:   [(n * dir) x 22) vector (float)
                matrix containing inequality constraints for fitting
        warmup: estimate dt vector (22, 0) at each voxel

        Returns
        -------
        dt:     diffusion tensor
        """
        w = np.diag(shat)
        # Unconstrained Fitting
        if cons is None:
            dt = np.matmul(np.linalg.pinv(np.matmul(w, b)),
                           np.matmul(w, np.log(dwi)))

        # Constrained fitting
        else:
            C = np.matmul(w, b).astype('double')
            d = np.matmul(w, np.log(dwi)).astype('double').reshape(-1)
            m, n = C.shape
            x = cvx.Variable(n)
            if warmup is not None:
                x.value = warmup
            objective = cvx.Minimize(0.5 * cvx.sum_squares(C * x - d))
            constraints = [cons * x >= np.zeros((len(cons)))]
            prob = cvx.Problem(objective, constraints)
            try:
                prob.solve(warm_start=True,
                           max_iter=20000)
                dt = x.value
            except:
                dt = np.full_like(x.value, minZero)
        return dt

    def fit(self, constraints=None, reject=None, dt_hat=None):
        """
        Returns fitted diffusion or kurtosis tensor

        Usage
        -----
        dwi.fit()
        dwi.fit(constraints=[0,1,0], reject=irlls_output)

        Parameters
        ----------
        constraits: [1 x 3] vector (int)
                    Specifies which constraints to use
        reject:     4D matrix containing information on voxels to exclude
                    from DT estimation
        dt_hat:     [22 x nvox] matrix with estimated dt to speedup
                    minimization (optional)

        Returns
        -------
        self.dt:    return diffusion tensor within DWI class
        """
        # Create constraints
        if reject is None:
            reject = np.zeros(self.img.shape)
        grad = self.grad
        order = np.floor(np.log(np.abs(np.max(grad[:,-1])+1))/np.log(10))
        if order >= 2:
            grad[:, -1] = grad[:, -1]
        self.img.astype(np.double)
        self.img[self.img <= 0] = np.finfo(np.double).eps
        grad.astype(np.double)
        normgrad = np.sqrt(np.sum(grad[:,:3]**2, 1))
        normgrad[normgrad == 0] = 1
        grad[:,:3] = grad[:,:3]/np.tile(normgrad, (3,1)).T
        grad[np.isnan(grad)] = 0
        dcnt, dind = self.createTensorOrder(2)
        wcnt, wind = self.createTensorOrder(4)
        ndwis = self.img.shape[-1]
        bs = np.ones((ndwis, 1))
        bD = np.tile(dcnt,(ndwis, 1))*grad[:,dind[:, 0]]*grad[:,dind[:, 1]]
        bW = np.tile(wcnt, (ndwis, 1)) * self.grad[:,wind[:, 0]] * \
             self.grad[:, wind[:, 1]] * self.grad[:, wind[:,2]] *  \
             self.grad[:,wind[:,3]]
        self.b = np.concatenate((bs, (
                    np.tile(-self.grad[:, -1], (6, 1)).T * bD), np.squeeze(
            1 / 6 * np.tile(self.grad[:, -1], (15, 1)).T ** 2) * bW), 1)
        dwi_ = vectorize(self.img, self.mask)
        reject_ = vectorize(reject, self.mask).astype(bool)
        init = np.matmul(np.linalg.pinv(self.b), np.log(dwi_))
        shat = np.exp(np.matmul(self.b, init))
        self.dt = np.zeros(init.shape)
        if constraints is None or (constraints[0] == 0 and
                                   constraints[1] == 0 and
                                   constraints[2] == 0):
            inputs = tqdm(range(0, dwi_.shape[1]),
                          desc='Unconstrained Tensor Fit',
                          unit='vox',
                          ncols=tqdmWidth)
            self.dt = Parallel(n_jobs=self.workers, prefer='processes') \
                (delayed(self.wlls)(shat[~reject_[:, i], i], \
                                    dwi_[~reject_[:, i], i], \
                                    self.b[~reject_[:, i]]) \
                 for i in inputs)
        else:
            C = self.createConstraints(constraints)  # Linear inequality constraint matrix A_ub
            inputs = tqdm(range(0, dwi_.shape[1]),
                          desc='Constrained Tensor Fit  ',
                          unit='vox',
                          ncols=tqdmWidth)
            if dt_hat is None:
                self.dt = Parallel(n_jobs=self.workers, prefer='processes') \
                    (delayed(self.wlls)(shat[~reject_[:, i], i],
                                             dwi_[~reject_[:, i], i],
                                             self.b[~reject_[:, i]],
                                             cons=C) for i in inputs)
            else:
                self.dt = Parallel(n_jobs=self.workers, prefer='processes') \
                    (delayed(self.wlls)(shat[~reject_[:, i], i],
                                        dwi_[~reject_[:, i], i],
                                        self.b[~reject_[:, i]],
                                        cons=C,
                                        warmup=dt_hat[:, i]) \
                     for i in inputs)
        self.dt = np.reshape(self.dt, (dwi_.shape[1], self.b.shape[1])).T
        self.s0 = np.exp(self.dt[0,:])
        self.dt = self.dt[1:,:]
        D_apprSq = 1/(np.sum(self.dt[(0,3,5),:], axis=0)/3)**2
        self.dt[6:,:] = self.dt[6:,:]*np.tile(D_apprSq, (15,1))

    def createConstraints(self, constraints=[0, 1, 0]):
        """
        Generates constraint array for constrained minimization quadratic
        programming

        Usage
        -----
        C = dwi.createConstraints([0, 1, 0])

        Parameter(s)
        -----------
        constraints:    [1 X 3] logical vector indicating which constraints
        out of three to enable
                        C1 is Dapp > 0
                        C1 is Kapp > 0
                        C3 is Kapp < 3/(b*Dapp)

        Return(s)
        ---------
        C:              array containing constraints to consider during
                        minimization
                        C is shaped [number of constraints enforced *
                        number of directions, 22]
        """
        if sum(constraints) >= 0 and sum(constraints) <= 3:
            dcnt, dind = self.createTensorOrder(2)
            wcnt, wind = self.createTensorOrder(4)
            cDirs = dwidirs.dirs30
            ndirs = cDirs.shape[0]
            C = np.empty((0, 22))
            if constraints[0] > 0:  # Dapp > 0
                C = np.append(C, np.hstack((np.zeros((ndirs, 1)),np.tile(dcnt, [ndirs, 1]) * cDirs[:, dind[:, 0]] * cDirs[:, dind[:, 1]],np.zeros((ndirs, 15)))), axis=0)
            if constraints[1] > 0:  # Kapp > 0
                C = np.append(C, np.hstack((np.zeros((ndirs, 7)), np.tile(wcnt, [ndirs, 1]) * cDirs[:, wind[:, 0]] * cDirs[:, wind[:, 1]] * cDirs[:,wind[:,2]] * cDirs[:,wind[:,3]])),axis=0)
            if constraints[2] > 0:  # K < 3/(b*Dapp)
                C = np.append(C, np.hstack((np.zeros((ndirs, 1)), 3 / self.maxBval() * np.tile(dcnt, [ndirs, 1]) * cDirs[:, dind[:, 0]],np.tile(-wcnt, [ndirs, 1]) * cDirs[:, wind[:, 1]] * cDirs[:,wind[:, 2]] * cDirs[:,wind[:, 3]])),axis=0)
        else:
            print('Invalid constraints. Please use format "[0, 0, 0]"')
        return C

    def extractDTI(self):
        """Extract all DTI parameters from DT tensor. Warning, this can
        only be run after tensor fitting dwi.fit()

        Usage
        -----
        (md, rd, ad, fa) = dwi.extractDTI(), where dwi is the DWI class
        object

        Parameters
        ----------
        (none)

        Returns
        -------
        md:     mean diffusivity
        rd:     radial diffusivity
        ad:     axial diffusivity
        fa:     fractional anisotropy
        fe:     first eigenvectors
        trace:  sum of first eigenvalues
        """
        # extract all tensor parameters from dt

        DT = np.reshape(
            np.concatenate((self.dt[0, :], self.dt[1, :], self.dt[2, :],
                            self.dt[1, :], self.dt[3, :], self.dt[4, :],
                            self.dt[2, :], self.dt[4, :], self.dt[5, :])),
            (3, 3, self.dt.shape[1]))

        # get the trace
        rdwi = sigmoid(np.matmul(self.b[:, 1:], self.dt))
        B = np.round(-(self.b[:, 0] + self.b[:, 3] + self.b[:, 5]) * 1000)
        uB = np.unique(B)
        trace = np.zeros((self.dt.shape[1], uB.shape[0]))
        for ib in range(0, uB.shape[0]):
            t = np.where(B == uB[ib])
            trace[:, ib] = np.mean(rdwi[t[0], :], axis=0)
        nvox = self.dt.shape[1]
        inputs = tqdm(range(0, nvox),
                      desc='DTI params              ',
                      unit='vox',
                      ncols=tqdmWidth)
        values, vectors = zip(
            *Parallel(n_jobs=self.workers, prefer='processes') \
                (delayed(self.dtiTensorParams)(DT[:, :, i]) for i in
                 inputs))
        values = np.reshape(np.abs(values), (nvox, 3))
        vectors = np.reshape(vectors, (nvox, 3, 3))
        self.DTIvectors = vectors
        l1 = vectorize(values[:, 0], self.mask)
        l2 = vectorize(values[:, 1], self.mask)
        l3 = vectorize(values[:, 2], self.mask)
        v1 = vectorize(vectors[:, :, 0].T, self.mask)
        md = (l1 + l2 + l3) / 3
        rd = (l2 + l3) / 2
        ad = l1
        fa = np.sqrt(1 / 2) * np.sqrt(
            (l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / np.sqrt(
            l1 ** 2 + l2 ** 2 + l3 ** 2)
        fe = np.abs(np.stack((fa * v1[:, :, :, 0], fa * v1[:, :, :, 1],
                              fa * v1[:, :, :, 2]), axis=3))
        trace = vectorize(trace.T, self.mask)
        return md, rd, ad, fa, fe, trace

    def extractDKI(self):
        """Extract all DKI parameters from DT tensor. Warning, this can
        only be run after tensor fitting dwi.fit()

        Usage
        -----
        (mk, rk, ak, fe, trace) = dwi.extractDTI(), where dwi is the DWI
        class object

        Parameters
        ----------
        (none)

        Returns
        -------
        mk:     mean diffusivity
        rk:     radial diffusivity
        ak:     axial diffusivity
        kfa:    kurtosis fractional anisotropy
        mkt:    mean kurtosis tensor
        trace:  sum of first eigenvalues
        """
        # get the trace
        rdwi = sigmoid(np.matmul(self.b[:, 1:], self.dt))
        B = np.round(-(self.b[:, 0] + self.b[:, 3] + self.b[:, 5]) * 1000)
        uB = np.unique(B)
        trace = np.zeros((self.dt.shape[1], uB.shape[0]))
        for ib in range(0, uB.shape[0]):
            t = np.where(B == uB[ib])
            trace[:, ib] = np.mean(rdwi[t[0], :], axis=0)
        dirs = dwidirs.dirs256
        akc = self.kurtosisCoeff(self.dt, dirs)
        mk = np.mean(akc, 0)
        nvox = self.dt.shape[1]
        inputs = tqdm(range(0, nvox),
                      desc='DKI params              ',
                      unit='vox',
                      ncols=tqdmWidth)
        ak, rk, kfa, mkt = zip(*Parallel(n_jobs=self.workers,
                                  prefer='processes') \
            (delayed(self.dkiTensorParams)(self.DTIvectors[i, :, 0],
                                           self.dt[:, i]) for i in inputs))
        ak = np.reshape(ak, (nvox))
        rk = np.reshape(rk, (nvox))
        kfa = np.reshape(kfa, (nvox))
        mkt = np.reshape(mkt, (nvox))
        trace = vectorize(trace.T, self.mask)
        ak = vectorize(ak, self.mask)
        rk = vectorize(rk, self.mask)
        mk = vectorize(mk, self.mask)
        kfa = vectorize(kfa, self.mask)
        mkt = vectorize(mkt, self.mask)
        return mk, rk, ak, kfa, mkt, trace

    def findViols(self, c=[0, 1, 0]):
        """
        Returns a 3D violation map of voxels that violate constraints.
        Classification: Method

        Usage
        -----
        map = findViols(img, [0 1 0]

        Parameters
        ----------
        img: 3D metric array such as mk or fa
        c:   [3 x 1] vector that toggles which constraints to check
             c[0]: Check D < 0 constraint
             c[1]: Check K < 0 constraint (default)
             c[2]: Check K > 3/(b*D) constraint

        Returns
        -------
        map: 3D array containing locations of voxels that incur directional
             violations. Voxels with values contain violaions and voxel
             values represent proportion of directional violations.

        """
        if c == None:
            c = [0, 0, 0]

        nvox = self.dt.shape[1]
        sumViols = np.zeros(nvox)
        maxB = self.maxBval()
        adc = self.diffusionCoeff(self.dt[:6], self.dirs)
        akc = self.kurtosisCoeff(self.dt, self.dirs)
        tmp = np.zeros(3)
        print('...computing directional violations')
        for i in range(nvox):
            # C[0]: D < 0
            tmp[0] = np.size(np.nonzero(adc[:, i] < minZero))
            # C[1]: K < 0
            tmp[1] = np.size(np.nonzero(akc[:, i] < minZero))
            #c[2]:
            tmp[2] = np.size(np.nonzero(akc[:, i] > (3/(maxB * adc[:, i]))))
            sumViols[i] = np.sum(tmp)

        map = np.zeros((sumViols.shape))
        if c[0] == 0 and c[1] == 0 and c[2] == 0:
            # [0 0 0]
            print('0 0 0')
            map = pViols

        elif c[0] == 1 and c[1] == 0 and c[2] == 0:
            # [1 0 0]
            print('1 0 0')
            map = sumViols/dirSample

        elif c[0] == 0 and c[1] == 1 and c[2] == 0:
            # [0 1 0]
            print('0 1 0')
            map = sumViols/dirSample

        elif c[0] == 0 and c[1] == 0 and c[2] == 1:
            # [0 0 1]
            print('0 0 1')
            map = sumViols/dirSample

        elif c[0] == 1 and c[1] == 1 and c[2] == 0:
            # [1 1 0]
            print('1 1 0')
            map = sumVioms/(2 * dirSample)

        elif c[0] == 1 and c[1] == 0 and c[2] == 1:
            # [1 0 1]
            print('1 0 1')
            map = sumViols/(2 * dirSample)

        elif c[0] == 0 and c[1] == 1 and c[2] == 1:
            # [0 1 1]
            print('0 1 1')
            map = sumViols / (2 * dirSample)

        elif c[0] == 1 and c[1] == 1 and c[2] == 1:
            # [1 1 1]
            print('1 1 1')
            map = sumViols / (3 * dirSample)

        map = np.reshape(map, nvox)
        map = vectorize(map, self.mask)
        return map

    def goodDirections(self, outliers):
        """Creates a 3D maps of good directions from IRLLS outlier map
        For any given voxel, a violation is computed using logical or
        operant for all b-values. Whether
        an outlier occurs at b1000 or b1000 and b2000, that voxel is still
        a violation unless none of the b-values have outliers.

        Usage
        -----
        map = dwi.goodDirections(outliers)

        Parameters
        ----------
        outliers:   4D maps of outliers from IRLLS

        Returns
        -------
        map:        3D maps of number of good directions

        """
        # Compute number of good directions
        maxB = self.maxBval()
        outliers_ = vectorize(outliers, self.mask)
        nvox = outliers_.shape[1]
        nonB0 = ~(self.grad[:, -1] < 0.01)
        bvals = np.unique(self.grad[nonB0, -1])

        # bidx is an index locator where rows are indexes of b-value and
        # columns are unique b-values
        bidx = np.zeros((self.getndirs(), bvals.size), dtype='int')
        for i in range(bvals.size):
            bidx[:, i] = np.array(np.where(self.grad[:, -1] == bvals[i]))

        tmpVals = np.zeros(bidx.shape, dtype=bool)
        sumViols = np.zeros(nvox, dtype=int)
        for i in range(nvox):
            for j in range(bvals.size):
                tmpVals[:,j] = outliers_[bidx[:,j], i]
                sumViols[i] = np.sum(np.any(tmpVals, axis=1))

        # Number of good directions
        map = np.full(sumViols.shape, self.getndirs()) - sumViols
        map = vectorize(map, self.mask)
        return map

    def findVoxelViol(self, adcVox, akcVox, maxB, c):
        """
        Returns the proportions of vioaltions occuring at a voxel.
        Classification: Method

        Usage
        -----
        map = findViols(voxel, [0 1 0]

        Parameters
        ----------
        img: 3D metric array such as mk or fa
        c:   [3 x 1] vector that toggles which constraints to check
             c[0]: Check D < 0 constraint
             c[1]: Check K < 0 constraint (default)
             c[2]: Check K > 3/(b*D) constraint

        Returns
        -------
        n: Number ranging from 0 to 1 that indicates proportion of
        violations occuring at voxel.

        """
        tmp = np.zeros(3)
        # C[0]: D < 0
        tmp[0] = np.size(np.nonzero(adcVox < minZero))
        # C[1]: K < 0
        tmp[1] = np.size(np.nonzero(akcVox < minZero))
        #c[2]:K > 3/(b * D)
        tmp[2] = np.size(np.nonzero(akcVox > (3/(maxB * adcVox))))
        sumViols = np.sum(tmp)

        if c[0] == 0 and c[1] == 0 and c[2] == 0:
            # [0 0 0]
            n = 0

        elif c[0] == 1 and c[1] == 0 and c[2] == 0:
            # [1 0 0]
            n = sumViols/dirSample

        elif c[0] == 0 and c[1] == 1 and c[2] == 0:
            # [0 1 0]
            n = sumViols/dirSample

        elif c[0] == 0 and c[1] == 0 and c[2] == 1:
            # [0 0 1]
            n = sumViols/dirSample

        elif c[0] == 1 and c[1] == 1 and c[2] == 0:
            # [1 1 0]
            n = sumVioms/(2 * dirSample)

        elif c[0] == 1 and c[1] == 0 and c[2] == 1:
            # [1 0 1]
            n = sumViols/(2 * dirSample)

        elif c[0] == 0 and c[1] == 1 and c[2] == 1:
            # [0 1 1]
            n = sumViols/(2 * dirSample)

        elif c[0] == 1 and c[1] == 1 and c[2] == 1:
            # [1 1 1]
            n = sumViols/(3 * dirSample)
        return n

    def parfindViols(self, c=[0, 0, 0]):
        if c == None:
            c = [0, 0, 0]
        print('...computing directional violations (parallel)')
        nvox = self.dt.shape[1]
        map = np.zeros(nvox)
        maxB = self.maxBval()
        adc = self.diffusionCoeff(self.dt[:6], self.dirs)
        akc = self.kurtosisCoeff(self.dt, self.dirs)
        inputs = tqdm(range(0, nvox))
        map = Parallel(n_jobs=self.workers, prefer='processes') \
            (delayed(self.findVoxelViol)(adc[:,i], akc[:,i], maxB, [0, 1, 0]) for i in inputs)
        map = np.reshape(pViols2, nvox)
        map = self.multiplyMask(vectorize(map,self.mask))
        return map

    def multiplyMask(self, img):
        # Returns an image multiplied by the brain mask to remove all values outside the brain
        return np.multiply(self.mask.astype(bool), img)

    def akcoutliers(self, iter=10):
        """
        Uses 100,000 direction in chunks of 10 to iteratively find
        outliers. Returns a mask of locations where said violations
        occur. Multiprocessing is disabled because this is a
        memory-intensive task.
        To be run only after tensor fitting.
        Classification: Method

        Usage
        -----
        akc_out = dwi.akoutliers(), where dwi is the DWI class object

        Parameters
        ----------
        iter:       number of iterations to perform out of 10. Default: 10
                    reduce this number if your computer does not have
                    sufficient RAM

        Returns
        -------
        akc_out:    3D map containing outliers where AKC falls fails the
                    inequality test -2 < AKC < 10
        """
        dir = dwidirs.dirs10000
        nvox = self.dt.shape[1]
        akc_out = np.zeros(nvox, dtype=bool)
        N = dir.shape[0]
        nblocks = 10
        if iter > nblocks:
            print('Entered iteration value exceeds 10...resetting to 10')
            iter = 10
        inputs = tqdm(range(iter),
                      desc='AKC Outlier Detection   ',
                      unit='blk',
                      ncols=tqdmWidth)
        for i in inputs:
            akc = self.kurtosisCoeff(self.dt, dir[int(N/nblocks*i):int(N/nblocks*(i+1))])
            akc_out[np.where(np.any(np.logical_or(akc < -2, akc > 10), axis=0))] = True
            akc_out.astype('bool')
        return vectorize(akc_out, self.mask)

    def akccorrect(self, akc_out, window=5, connectivity='all'):
        """Applies AKC outlier map to DT to replace outliers with a
        moving median.
        Run this only after tensor fitting and akc outlier detection
        Classification: Method

        Usage
        -----
        dwi.akccorrect(akc_out), where dwi is the DWI class object

        Parameters
        ----------
        akc_out:        3D map containing outliers from DWI.akcoutliers
        window:         width of square matrix filter.
                        default: 5
                        type: int
        connectivity:   specifies what kind of connected-component
                        connectivity to use for median determination
                        choices: 'all' (default) or 'face'
                        type: string
        """
        # Get box filter properties
        centralIdx = np.median(range(window))
        d2move = np.int(np.abs(window - (centralIdx + 1)))  # Add 1 to
        # central idx because first index starts with zero

        # Vectorize and Pad
        dt = np.pad(vectorize(self.dt, self.mask),
                     ((d2move, d2move), (d2move, d2move),
                     (d2move, d2move), (0, 0)),
                     'constant', constant_values=np.nan)
        akc_out = np.pad(akc_out, d2move, 'constant',
                         constant_values=False)

        violIdx = np.array(
            np.where(akc_out))  # Locate coordinates of violations
        nvox = violIdx.shape[1]

        for i in tqdm(range(dt.shape[-1]),
                      desc='AKC Correction          ',
                      unit='tensor',
                      ncols=tqdmWidth):
            for j in range(nvox):
                # Index beginning and ending of patch
                Ib = violIdx[0, j] - d2move
                Ie = violIdx[0, j] + d2move + 1
                Jb = violIdx[1, j] - d2move
                Je = violIdx[1, j] + d2move + 1
                Kb = violIdx[2, j] - d2move
                Ke = violIdx[2, j] + d2move + 1

                if connectivity == 'all':
                    patchViol = np.delete(
                        np.ravel(akc_out[Ib:Ie, Jb:Je, Kb:Ke]),
                        np.median(range(np.power(window,3))))  # Remove
                    # centroid element
                    patchImg = np.delete(
                        np.ravel(dt[Ib:Ie, Jb:Je, Kb:Ke, i]),
                        np.median(range(np.power(window,3))))  # Remove
                    # centroid element
                    connLimit = np.power(window,3) -1
                elif connectivity == 'face':
                    patchViol = akc_out[
                        [Ib, Ie], violIdx[1, j], violIdx[2, j]]
                    patchViol = np.hstack((patchViol, akc_out[
                        violIdx[0, j], [Jb, Je], violIdx[2, j]]))
                    patchViol = np.hstack((patchViol, akc_out[
                        violIdx[0, j], violIdx[1, j], [Kb, Ke]]))
                    patchImg = dt[
                        [Ib, Ie], violIdx[1, j], violIdx[2, j], i]
                    patchImg = np.hstack((patchImg, dt[
                        violIdx[0, j], [Jb, Je], violIdx[2, j], i]))
                    patchImg = np.hstack((patchImg, dt[
                        violIdx[0, j], violIdx[1, j], [Kb, Ke], i]))
                    if window == 3:
                        connLimit = 6
                    elif window == 5:
                        connLimit = 12
                    elif window == 7:
                        connLimit = 18
                    elif window == 9:
                        connLimit = 24
                else:
                    raise Exception(
                        'Connectivity choice "{}" is invalid. Please '
                        'enter either "all" or "face".'.format(
                            connectivity))

                nVoil = np.sum(patchViol)

                # Here a check is performed to compute the number of
                # violations in a patch. If all voxels are violations,
                # do nothing. Otherwise, exclude violation voxels from
                # the median calculation
                if nVoil == connLimit:
                    continue
                else:
                   dt[violIdx[0, j], violIdx[1, j], violIdx[2, j],
                      i] = np.nanmedian(patchImg)

        # Remove padding
        dt = dt[d2move:-d2move, d2move:-d2move,
             d2move:-d2move, :]
        self.dt = vectorize(dt, self.mask)

    def irlls(self, excludeb0=True, maxiter=25, convcrit=1e-3, mode='DKI', leverage=0.85, bounds=3):
        """This functions performs outlier detection and robust parameter
        estimation for diffusion MRI using the iterative reweigthed
        linear least squares (IRLLS) approach.
        Classification: Method

        Usage
        -----
        outliers = dwi.irlls()

        Parameters
        ----------
        exludeb0:   True (default) or False (bool)
                    Exlude the b0 images when removing outliers?
        maxiter:    Integer; default: 25
                    Maximum number of iterations in the iterative
                    reweighting loop
        convcrit:   Real positive double; default: 1e-3
                    Fraction of L2-norm of estimated diffusion parameter
                    vector that the L2-norm of different vector should
                    get under un order to reach convergence in the iterative
                    reweighted loop
        mode:       'DTI' or 'DKI' (string); default: 'DKI'
                    Specifies whether to use DTi or DKI model
        leverage:   Double ranging from 0 to 1; default: 0.85
                    Measurements with a leverage above this threshold will
                    not be excluded in estimation of DT after
                    outlier detection
        Bounds:     Integer; default: 3
                    Set the threshold of the number of standard deviation
                    that are needed to exclude a measurement

        Returns
        -------
        outliers:   4D image same size as input dwi marking voxels
                    that are outliers
        dt:         IRLLS method of DT estimation
        """

        # if not excludeb0.dtype:
        #     assert('option: Excludeb0 should be set to True or False')

        if maxiter < 1 or  maxiter > 200:
            assert('option: Maxiter should be set to a value between 1 '
                   'and 200')

        if convcrit < 0 or convcrit > 1:
            assert('option: Maxiter should be set to a value between 1 '
                   'and 200')

        if not (mode == 'DKI' or mode == 'DTI'):
            assert('Mode should be set to DKI or DTI')

        if leverage < 0 or leverage > 1:
            assert('option: Leverage should be set to a value between 0 '
                   'and 1')

        if bounds < 1:
            assert('option: Bounds should be set to a value >= 1')

        # Vectorize DWI
        dwi = vectorize(self.img, self.mask)
        (ndwi, nvox) = dwi.shape
        b = np.array(self.grad[:, 3])
        b = np.reshape(b, (len(b), 1))
        g = self.grad[:, 0:3]

        # Apply Scaling
        scaling = False
        if np.sum(dwi < 1)/np.size(dwi) < 0.001:
            dwi[dwi < 1] = 1
        else:
            scaling = True
            if self.maxBval() < 10:
                tmp = dwi[dwi < 0.05]
            else:
                tmp = dwi[dwi < 50]
            sc = np.median(tmp)
            dwi[dwi < sc/1000] = sc/1000
            dwi = dwi * 1000 / sc

        # Create B-matrix
        (dcnt, dind) = self.createTensorOrder(2)
        if mode == 'DTI':
            bmat = np.hstack((np.ones((ndwi, 1)), np.matmul((-np.tile(b, (1, 6)) * g[:,dind[:,0]] * g[:,dind[:,1]]), np.diag(dcnt))))
        else:
            (wcnt, wind) = self.createTensorOrder(4)
            bmat = np.hstack((np.ones((ndwi,1)),
                              np.matmul((-np.tile(b, (1, 6)) * g[:,dind[:,0]] * g[:,dind[:,1]]), np.diag(dcnt)),
                              (1/6)*np.matmul((np.square(np.tile(b, (1, 15))) * g[:,wind[:,0]] * g[:,wind[:,1]] * g[:,wind[:,2]] * g[:,wind[:,3]]),
                                              np.diag(wcnt))))
        nparam = bmat.shape[1]
        ndof = ndwi - nparam

        # Initialization
        b0_pos = np.zeros(b.shape,dtype=bool, order='F')
        if excludeb0:
            if self.maxBval() < 10:
                b0_pos = b < 0.01
            else:
                b0_pos = b < 10

        reject = np.zeros(dwi.shape, dtype=bool, order='F')
        conv = np.zeros((nvox, 1))
        dt = np.zeros((nparam, nvox))
        # fa = np.zeros((nvox, 1))
        # md = np.zeros((nvox, 1))

        # Attempt basic noise estimation
        try:
            sigma
        except NameError:
            def estSigma(dwi, bmat):
                dwi = np.reshape(dwi, (len(dwi), 1))
                # dt_ = np.linalg.lstsq(bmat, np.log(dwi), rcond=None)[0]
                try:
                    dt_ = np.linalg.solve(np.dot(bmat.T, bmat), np.dot(
                        bmat.T, np.log(dwi)))
                except:
                    dt_ = minZero
                w = np.exp(np.matmul(bmat, dt_)).reshape((ndwi, 1))
                # dt_ = np.linalg.lstsq((bmat * np.tile(w, (1, nparam))), (np.log(dwi) * w), rcond=None)[0]
                try:
                    dt_ = np.linalg.solve(
                        np.dot((bmat * np.tile(w, (1, nparam))).T,
                            (bmat * np.tile(w, (1, nparam)))), \
                        np.dot((bmat * np.tile(w, (1, nparam))).T, (np.log(
                            dwi) * w)))
                except:
                    dt_ = minZero
                e = np.log(dwi) - np.matmul(bmat, dt_)
                m = np.median(np.abs((e * w) - np.median(e * w)))
                try:
                    sigma_ = np.sqrt(ndwi / ndof) * 1.4826 * m
                except:
                    sigma_ = minZero
                return sigma_
            sigma_ = np.zeros((nvox,1))
            inputs = tqdm(range(nvox),
                          desc='IRLLS: Noise Estimation ',
                          unit='vox',
                          ncols=tqdmWidth)
            sigma_ = Parallel(n_jobs=self.workers, prefer='processes') \
                (delayed(estSigma)(dwi[:, i], bmat) for i in inputs)
            sigma = np.median(sigma_)
            sigma = np.tile(sigma,(nvox,1))
        if scaling:
            sigma = sigma*1000/sc

        def outlierHelper(dwi, bmat, sigma, b, b0_pos, maxiter=25, convcrit=1e-3, leverage=3, bounds=3):
            # Preliminary rough outlier check
            dwi_i = dwi.reshape((len(dwi), 1))
            dwi0 = np.median(dwi_i[b.reshape(-1) < 0.01])
            out = dwi_i > (dwi0 + 3 * sigma)
            if np.sum(~out[b.reshape(-1) > 0.01]) < (bmat.shape[1] - 1):
                out = np.zeros((out.shape),dtype=bool)
            out[b0_pos.reshape(-1)] = False
            bmat_i = bmat[~out.reshape(-1)]
            dwi_i = dwi_i[~out.reshape(-1)]
            n_i = dwi_i.size
            ndof_i = n_i - bmat_i.shape[1]

            # WLLS estimation
            # dt_i = np.linalg.lstsq(bmat_i, np.log(dwi_i), rcond=None)[0]
            try:
                dt_i = np.linalg.solve(np.dot(bmat_i.T, bmat_i), np.dot(bmat_i.T, np.log(dwi_i)))
            except:
                dt_i = minZero
            w = np.exp(np.matmul(bmat_i, dt_i))
            # dt_i = np.linalg.lstsq((bmat_i * np.tile(w, (1, nparam))), (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w),
            #                        rcond=None)[0]
            try:
                dt_i = np.linalg.solve(np.dot((bmat_i * np.tile(w, (1, nparam))).T, (bmat_i * np.tile(w, (1, nparam)))), \
                                    np.dot((bmat_i * np.tile(w, (1, nparam))).T, (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w)))
            except:
                dwi_hat = minZero
            dwi_hat = np.exp(np.matmul(bmat_i, dt_i))

            # Goodness-of-fit
            residu = np.log(dwi_i.reshape((dwi_i.shape[0],1))) - np.log(dwi_hat)
            residu_ = dwi_i.reshape((dwi_i.shape[0],1)) - dwi_hat
            try:
                chi2 = np.sum((residu_ * residu_) / np.square(sigma)) / (ndof_i) -1
            except:
                chi2 = minZero
            try:
                gof = np.abs(chi2) < 3 * np.sqrt(2/ndof_i)
            except:
                gof = True  # If ndof_i = 0, right inequality becomes inf and makes the logic True
            gof2 = gof

            # Iterative reweighning procedure
            iter = 0
            while (not gof) and (iter < maxiter):
                try:
                    C = np.sqrt(n_i/(n_i-nparam)) * 1.4826 * np.median(np.abs(residu_ - np.median(residu_))) / dwi_hat
                except:
                    C = minZero
                try:
                    GMM = np.square(C) / np.square(np.square(residu) + np.square(C))
                except:
                    GMM = minZero
                w = np.sqrt(GMM) * dwi_hat
                dt_imin1 = dt_i
                # dt_i = np.linalg.lstsq((bmat_i * np.tile(w, (1, nparam))), (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w),
                #                        rcond=None)[0]
                try:
                    dt_i = np.linalg.solve(np.dot((bmat_i * np.tile(w, (1, nparam))).T, (bmat_i * np.tile(w, (1, nparam)))), \
                                        np.dot((bmat_i * np.tile(w, (1, nparam))).T,  (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w)))
                except:
                    dt_i = minZero
                dwi_hat = np.exp(np.matmul(bmat_i, dt_i))
                dwi_hat[dwi_hat < 1] = 1
                residu = np.log(dwi_i.reshape((dwi_i.shape[0],1))) - np.log(dwi_hat)
                residu_ = dwi_i.reshape((dwi_i.shape[0], 1)) - dwi_hat

                # Convergence check
                iter = iter + 1
                gof = np.linalg.norm(dt_i - dt_imin1) < np.linalg.norm(dt_i) * convcrit
                conv = iter

            # Outlier detection
            if ~gof2:
                # lev = np.diag(np.matmul(bmat_i, np.linalg.lstsq(np.matmul(np.transpose(bmat_i),
                #                                         np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i)),
                #                       np.matmul(np.transpose(bmat_i), np.diag(np.square(w.reshape(-1)))), rcond=None)[0]))
                try:
                    lev = np.diag(\
                        np.matmul(bmat_i, \
                                np.linalg.solve(\
                                    np.dot((np.matmul(np.transpose(bmat_i), np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i))).T, \
                                            (np.matmul(np.transpose(bmat_i), np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i)))), \
                                    np.dot((np.matmul(np.transpose(bmat_i), np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i))).T, \
                                            np.matmul(np.transpose(bmat_i), np.diag(np.square(w.reshape(-1))))))))
                except:
                    lev = minZero
                lev = lev.reshape((lev.shape[0], 1))
                try:
                    lowerbound_linear = -bounds * np.lib.scimath.sqrt(1 -lev) * sigma / dwi_hat
                except:
                    lowerbound_linear = minZero
                upperbound_nonlinear = bounds * np.lib.scimath.sqrt(1 - lev) * sigma

                tmp = np.zeros(residu.shape, dtype=bool, order='F')
                tmp[residu < lowerbound_linear] = True
                tmp[residu > upperbound_nonlinear] = True
                tmp[lev > leverage] = False
                tmp2 = np.ones(b.shape, dtype=bool, order='F')
                tmp2[~out.reshape(-1)] = tmp
                tmp2[b0_pos] = False
                reject = tmp2
            else:
                tmp2 = np.zeros(b.shape, dtype=bool, order='F')
                tmp2[out.reshape(-1)] = True
                reject = tmp2

            # Robust parameter estimation
            keep = ~reject.reshape(-1)
            bmat_i = bmat[keep,:]
            dwi_i = dwi[keep]
            # dt_ = np.linalg.lstsq(bmat_i, np.log(dwi_i), rcond=None)[0]
            try:
                dt_ = np.linalg.solve(np.dot(bmat_i.T, bmat_i), \
                                    np.dot(bmat_i.T, np.log(dwi_i)))
            except:
                dt_ = minZero
            w = np.exp(np.matmul(bmat_i, dt_))
            # dt = np.linalg.lstsq((bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam))), (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w.reshape((len(w),1))),
            #                            rcond=None)[0]
            try:
                dt = np.linalg.solve(\
                    np.dot((bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam))).T, (bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam)))), \
                    np.dot((bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam))).T, \
                        (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w.reshape((len(w),1)))))
            except:
                dt = minZero
            # dt_tmp = dt.reshape(-1)
            # dt2 = np.array([[dt_tmp[1], dt_tmp[2]/2, dt_tmp[3]],
            #        [dt_tmp[2]/2, dt_tmp[4], dt_tmp[5]/2],
            #        [dt_tmp[3]/2, dt_tmp[5]/2, dt_tmp[6]]])
            # eigv, tmp = np.linalg.eig(dt2)
            # fa = np.sqrt(1/2) * \
            #      (np.sqrt(np.square(eigv[0] - eigv[1]) + np.square(eigv[0] - eigv[2]) + np.square(eigv[1] - eigv[2])) / \
            #      np.sqrt(np.square(eigv[0]) + np.square(eigv[1]) + np.square(eigv[2])))
            # md = np.sum(eigv)/3
            return reject.reshape(-1), dt.reshape(-1)#, fa, md

        inputs = tqdm(range(nvox),
                          desc='IRLLS: Outlier Detection',
                          unit='vox',
                          ncols=tqdmWidth)
        (reject, dt) = zip(*Parallel(n_jobs=self.workers, prefer='processes') \
            (delayed(outlierHelper)(dwi[:, i], bmat, sigma[i,0], b, b0_pos) for i in inputs))
        # for i in inputs:
        #     reject[:,i], dt[:,i] = outlierHelper(dwi[:, i], bmat, sigma[i,0], b, b0_pos)
        dt = np.array(dt)
        # self.dt = dt
        #Unscaling
        if scaling:
            dt[1, :] = dt[1, :] + np.log(sc/1000)
        #Unvectorizing
        reject = vectorize(np.array(reject).T, self.mask)
        # dt = np.array(dt)
        # fa = vectorize(np.array(fa), self.mask)
        # md = vectorize(np.array(md), self.mask)

        return reject, dt.T #, fa, md

    def tensorReorder(self, dwiType):
        """Reorders tensors in DT to those of MRTRIX in accordance to
        the table below

        MRTRIX3 Tensors                     DESIGNER Tensors
        ---------------                     ----------------

        0   D0      1   1                       1   1
        1   D1      2   2                       1   2
        2   D2      3   3                       1   3
        3   D3      1   2                       2   2
        4   D4      1   3                       2   3
        5   D5      2   3                       3   3

        6   K0      1   1   1   1               1   1   1   1
        7   K1      2   2   2   2               1   1   1   2
        8   K2      3   3   3   3               1   1   1   3
        9   K3      1   1   1   2               1   1   2   2
        10  K4      1   1   1   3               1   1   2   3
        11  K5      1   2   2   2               1   1   3   3
        12  K6      1   3   3   3               1   2   2   2
        13  K7      2   2   2   3               1   2   2   3
        14  K8      2   3   3   3               1   2   3   3
        15  K9      1   1   2   2               1   3   3   3
        16  K10     1   1   3   3               2   2   2   2
        17  K11     2   2   3   3               2   2   2   3
        18  K12     1   1   2   3               2   2   3   3
        19  K13     1   2   2   3               2   3   3   3
        20  K14     1   2   3   3               3   3   3   3

        Value Assignment
        ----------------

        MRTRIX3         DESIGNER
        -------         --------
            0               0
            1               3
            2               5
            3               1
            4               2
            5               4

            6               6
            7               16
            8               20
            9               7
            10              8
            11              12
            12              15
            13              17
            14              19
            15              9
            16              11
            17              18
            18              10
            19              13
            20              14

        Usage
        -----
        dt = dwi.tensorReorder()

        Parameters
        ----------
        dwiType:   'dti' or 'dki' (string)
                    Indicates whether image is DTI or DKI

        Returns
        -------
        DT:         4D image containing DT tensor
        KT:         4D image containing KT tensor
        """
        if self.dt is None:
            raise Exception('Please run dwi.fit() to generate a tensor '
                            'prior to reordering tensors.')

        if dwiType == 'dti':
            dt = np.zeros((6, self.dt.shape[1]))
            dt[0,:] =   self.dt[0, :]       # D0
            dt[1, :] =  self.dt[3, :]       # D1
            dt[2, :] =  self.dt[5, :]       # D2
            dt[3, :] =  self.dt[1, :]       # D3
            dt[4, :] =  self.dt[2, :]       # D4
            dt[5, :] =  self.dt[4, :]       # D5
            DT = vectorize(dt[0:6, :], self.mask)
            return DT

        if dwiType == 'dki':
            dt = np.zeros(self.dt.shape)
            dt[0, :] =  self.dt[0, :]       # D0
            dt[1, :] =  self.dt[3, :]       # D1
            dt[2, :] =  self.dt[5, :]       # D2
            dt[3, :] =  self.dt[1, :]       # D3
            dt[4, :] =  self.dt[2, :]       # D4
            dt[5, :] =  self.dt[4, :]       # D5
            dt[6, :] =  self.dt[6, :]       # K0
            dt[7, :] =  self.dt[16, :]      # K1
            dt[8, :] =  self.dt[20, :]      # K2
            dt[9, :] =  self.dt[7, :]       # K3
            dt[10, :] = self.dt[8, :]       # K4
            dt[11, :] = self.dt[12, :]      # K5
            dt[12, :] = self.dt[15, :]      # K6
            dt[13, :] = self.dt[17, :]      # K7
            dt[14, :] = self.dt[19, :]      # K8
            dt[15, :] = self.dt[9, :]       # K9
            dt[16, :] = self.dt[11, :]      # K10
            dt[17, :] = self.dt[18, :]      # K11
            dt[18, :] = self.dt[10, :]      # K12
            dt[19, :] = self.dt[13, :]      # K13
            dt[20, :] = self.dt[14, :]      # K14
            DT = vectorize(dt[0:6, :], self.mask)
            KT = vectorize(dt[6:21, :], self.mask)
            return (DT, KT)

    def irllsviolmask(self, reject):
        """Computes 3D violation mask of outliers detected from IRLLS
        method
        Classification: Method

        Usage
        -----
        mask = dwi.irllsviolmask(outliers)

        Parameters
        ----------
        reject:     4D input outlier map from IRLLS

        Returns
        propviol:   3D mask where voxel value is the percentage of
        directional violations
        """
        img = vectorize(reject, self.mask)
        (ndwi, nvox) = img.shape
        b = np.array(self.grad[:, 3])
        b = np.reshape(b, (len(b), 1))
        b_pos = ~(b < 0.01).reshape(-1)
        img = img[b_pos, :]
        propViol = np.sum(img,axis=0).astype(int) / np.sum(b_pos)
        propViol = vectorize(propViol, self.mask)
        return propViol

    def optimPipeline(self, savePath=os.getcwd()):
        """Runs the recommended tensor fitting pipeline

        Usage
        -----
        DWI.optimPipeline()

        Parameters
        ----------
        savePath:   directory where to save maps

        Returns
        -------
        MD, AD, RD, FA, trace, fe, MK, AK, RK
        """
        if savePath == os.getcwd():
            savePath = os.path.join(savePath, 'Output')
            qcPath = os.path.join(savePath, 'QC')
            os.mkdir(savePath)
            os.mkdir(qcPath)
        else:
            qcPath = os.path.join(savePath, 'QC')
            os.mkdir(qcPath)
        outliers, dt_hat = self.irlls()
        outlierPath = os.path.join(qcPath, 'Outliers_IRLLS.nii')
        writeNii(outliers, self.hdr, outlierPath)
        self.fit(constraints=[0, 1, 0], reject=outliers)
        if not self.isdki():
            tqdm.write('Detected DTI')
            md, rd, ad, fa, fe, trace = self.extractDTI()
            mdPath = os.path.join(savePath, 'MD.nii')
            rdPath = os.path.join(savePath, 'RD.nii')
            adPath = os.path.join(savePath, 'AD.nii')
            faPath = os.path.join(savePath, 'FA.nii')
            fePath = os.path.join(savePath, 'FE.nii')
            tracePath = os.path.join(savePath, 'Trace.nii')
            writeNii(md, self.hdr, mdPath)
            writeNii(rd, self.hdr, rdPath)
            writeNii(ad, self.hdr, adPath)
            writeNii(fa, self.hdr, faPath)
            writeNii(fe, self.hdr, fePath)
            writeNii(trace, self.hdr, tracePath)
        else:
            tqdm.write('Detected DKI')
            md, rd, ad, fa, fe, trace = self.extractDTI()
            akc_out = self.akcoutliers()
            self.akccorrect(akc_out=akc_out)
            mk, rk, ak, trace = self.extractDKI()
            mdPath = os.path.join(savePath, 'MD.nii')
            rdPath = os.path.join(savePath, 'RD.nii')
            adPath = os.path.join(savePath, 'AD.nii')
            faPath = os.path.join(savePath, 'FA.nii')
            fePath = os.path.join(savePath, 'FE.nii')
            tracePath = os.path.join(savePath, 'Trace.nii')
            mkPath = os.path.join(savePath, 'MK.nii')
            rkPath = os.path.join(savePath, 'RK.nii')
            akPath = os.path.join(savePath, 'AK.nii')
            akcPath = os.path.join(qcPath, 'Outliers_AKC.nii')
            writeNii(md, self.hdr, mdPath)
            writeNii(rd, self.hdr, rdPath)
            writeNii(ad, self.hdr, adPath)
            writeNii(fa, self.hdr, faPath)
            writeNii(fe, self.hdr, fePath)
            writeNii(trace, self.hdr, tracePath)
            writeNii(mk, self.hdr, mkPath)
            writeNii(rk, self.hdr, rkPath)
            writeNii(ak, self.hdr, akPath)
            writeNii(akc_out, self.hdr, akcPath)
        DT, KT = self.tensorReorder(self.tensorType())
        dtPath = os.path.join(savePath, 'DT.nii')
        ktPath = os.path.join(savePath, 'KT.nii')
        writeNii(DT, self.hdr, dtPath)
        writeNii(KT, self.hdr, ktPath)

# class medianFilter(object):
#     def __init__(self, img, violmask, th=15, sz=3, conn='face'):
#         assert th > 0, 'Threshold cannot be less than zero, disable median filtering instead'
#         assert violmask.shape == img.shape, 'Image dimensions not the same as violation mask dimensions'
#         self.Threshold = th
#         self.Size = sz
#         self.Connectivity = conn
#         self.Mask = np.logical_and(violmask < th, violmask > 0)
#         self.Img = img
#
#         # Get box filter properties
#         centralIdx = np.median(range(sz))
#         d2move = np.int(np.abs(sz - (centralIdx + 1))) # Add 1 to central idx because first index starts with zero
#
#         # Apply a nan padding to all 3 dimensions of the input image and a nan padding to mask. Padding widths is same
#         # distance between centroid of patch to edge. This enables median filtering of edges.
#         self.Img = np.pad(self.Img, d2move, 'constant', constant_values=np.nan)
#         self.Mask = np.pad(self.Mask, d2move, 'constant', constant_values=False)
#
#         (Ix, Iy, Iz) = img.shape
#         (Mx, My, Mz) = self.Mask.shape
#
#     def findReplacement(self, bias='rand'):
#         """
#         Returns information on replacements for violating voxels
#
#         Usage
#         -----
#         m = med.findReplacement(bias='rand')
#
#         Parameters
#         ----------
#         bias: 'left', 'right', or 'rand'
#                In the even the number of voxels in patch is even (for
#                median calculation), 'left' will pick a median to the left
#                of mean and 'right' will pick a median to the right of
#                mean. 'rand' will randomny pick a bias.
#
#         Returns
#         -------
#         m: Vector containing index of replacement voxel in patch. In
#         conn = 'face' max is 5 and conn =
#         """
#         # Get box filter properties
#         centralIdx = np.median(range(self.Size))
#         d2move = np.int(np.abs(self.Size - (centralIdx + 1)))  # Add 1 to central idx because first index starts with zero
#
#         violIdx = np.array(np.where(self.Mask))   # Locate coordinates of violations
#         nvox = violIdx.shape[1]
#         self.PatchIdx = np.zeros(violIdx.shape[1])
#
#         inputs = tqdm(range(nvox))
#         cntSkip = 0
#         for i in inputs:
#             # Index beginning and ending of patch
#             Ib = violIdx[0, i] - d2move
#             Ie = violIdx[0, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
#             Jb = violIdx[1, i] - d2move
#             Je = violIdx[1, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
#             Kb = violIdx[2, i] - d2move
#             Ke = violIdx[2, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
#
#             if self.Connectivity == 'all':
#                 patchViol = np.delete(np.ravel(self.Mask[Ib:Ie, Jb:Je, Kb:Ke]), 13)     # Remove 14th (centroid) element
#                 patchImg = np.delete(np.ravel(self.Img[Ib:Ie, Jb:Je, Kb:Ke]), 13)            # Remove 14th (centroid) element
#                 connLimit = 26
#             elif self.Connectivity == 'face':
#                 patchViol = self.Mask[[Ib, Ie], violIdx[1, i], violIdx[2, i]]
#                 patchViol = np.hstack((patchViol, self.Mask[violIdx[0,i], [Jb, Je], violIdx[2, i]]))
#                 patchViol = np.hstack((patchViol, self.Mask[violIdx[0, i], violIdx[1, i], [Kb, Ke]]))
#                 patchImg = self.Img[[Ib, Ie], violIdx[1, i], violIdx[2, i]]
#                 patchImg = np.hstack((patchImg, self.Img[violIdx[0, i], [Jb, Je], violIdx[2, i]]))
#                 patchImg = np.hstack((patchImg, self.Img[violIdx[0, i], violIdx[1, i], [Kb, Ke]]))
#                 connLimit = 6
#             else:
#                 raise Exception('Connectivity choice "{}" is invalid. Please enter either "all" or "face".'.format(self.Connectivity))
#
#             nVoil = np.sum(patchViol)
#
#             # Here a check is performed to compute the number of violations in a patch. If all voxels are violations,
#             # do nothing. Otherwise, exclude violation voxels from the median calculation
#             if nVoil == connLimit:
#                 self.PatchIdx[i] = np.nan
#                 cntSkip = cntSkip + 1
#                 continue
#             else:
#                 # Sort all patch values in ascending order and remove NaNs
#                 patchVals = np.array(np.sort(patchImg[~np.isnan(patchImg)],kind='quicksort'))
#                 nVals = patchVals.size
#
#                 # Median algorithm dependent on whether number of valid voxels (nVals) is even or odd
#                 if np.mod(nVals, 2) == 0:                                       # If even
#                     medianIdxTmp = np.array([nVals/2 - 1, nVals/2],dtype=int)   # Convert to Py index (-1)
#                     if bias == 'left':
#                         medianIdx = medianIdxTmp[0]
#                     elif bias == 'right':
#                         medianIdx = medianIdxTmp[1]
#                     elif bias == 'rand':
#                         medianIdx = medianIdxTmp[rnd.randint(0,1)]
#                 else:                                                           # If odd
#                     medianIdx = (nVals-1)/2                                     # Convert to Py index (-1)
#
#                 # Now that a the index of a median voxel is located, determine it's value in sorted list and find the
#                 # location of that voxel in patch. The median index needs to be relative to patch, not sorted list. In
#                 # the event that there are more than one indexes of the same value, use the first one.
#                 medianIdxP = np.where(patchImg == patchVals[np.int(medianIdx)])[0][0]
#                 self.PatchIdx[i] = medianIdxP
#         self.PatchIdx = np.array(self.PatchIdx, dtype='int')  # Convert to integer
#         print('%d voxels out of %d were completely surrounded by violations and were ignored' %(cntSkip, violIdx.shape[1]))
#         return self.PatchIdx
#
#     def applyReplacement(self, img):
#         """
#         Applies median filter onto input images.
#
#         Usage
#         -----
#         filteredImage = med.applyReplacement(image)
#
#         Parameters
#         ----------
#         image:          input image to apply the median voxel replacement on
#
#         Returns
#         -------
#         filteredImage:  median filtered image
#         """
#         # Get box filter properties
#         centralIdx = np.median(range(self.Size))
#         d2move = np.int(
#             np.abs(self.Size - (centralIdx + 1)))  # Add 1 to central idx because first index starts with zero
#
#         # Pad image with zeros
#         img = np.pad(img, d2move, 'constant', constant_values=0)
#
#         violIdx = np.array(np.where(self.Mask))  # Locate coordinates of violations
#         # self.PatchIdx = np.array(np.zeros(violIdx.shape[1]),dtype='int')
#
#         inputs = tqdm(range(self.PatchIdx.size))
#         for i in inputs:
#             # Index beginning and ending of patch
#             Ib = violIdx[0, i] - d2move
#             Ie = violIdx[0, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
#             Jb = violIdx[1, i] - d2move
#             Je = violIdx[1, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
#             Kb = violIdx[2, i] - d2move
#             Ke = violIdx[2, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
#
#             if self.Connectivity == 'all':
#                 patchViol = np.delete(np.ravel(self.Mask[Ib:Ie, Jb:Je, Kb:Ke]), 13)     # Remove 14th (centroid) element
#                 patchImg = np.delete(np.ravel(self.Img[Ib:Ie, Jb:Je, Kb:Ke]), 13)            # Remove 14th (centroid) element
#             elif self.Connectivity == 'face':
#                 patchViol = self.Mask[[Ib, Ie], violIdx[1, i], violIdx[2, i]]
#                 patchViol = np.hstack((patchViol, self.Mask[violIdx[0,i], [Jb, Je], violIdx[2, i]]))
#                 patchViol = np.hstack((patchViol, self.Mask[violIdx[0, i], violIdx[1, i], [Kb, Ke]]))
#                 patchImg = img[[Ib, Ie], violIdx[1, i], violIdx[2, i]]
#                 patchImg = np.hstack((patchImg, img[violIdx[0, i], [Jb,
#                                                                   Je], violIdx[2, i]]))
#                 patchImg = np.hstack((patchImg, img[violIdx[0, i],
#                                                   violIdx[1, i], [Kb, Ke]]))
#
#             if np.isnan(self.PatchIdx[i]) == True:
#                 continue
#             else:
#                 img[violIdx[0, i], violIdx[1, i], violIdx[2, i]] = patchImg[self.PatchIdx[i]]
#
#         # Unpad image by removing first and last slices along each axis
#         img = np.delete(img, [0, img.shape[0] - 1], axis=0)
#         img = np.delete(img, [0, img.shape[1] - 1], axis=1)
#         img = np.delete(img, [0, img.shape[2] - 1], axis=2)
#         return img

class medianFilter(object):
    """
    Create a median filter class object that holds vital information on
    filtering properties. Class initializes by computing a violation map
    based in input image

    Parameters
    ----------
    img:        3D numpy array
                input reference image used to compute an outlier mask
    brainmask:  3D boolean numpy array
                brain mask to speed up calculation
    multiplier: double with minimum at 0 | default: 1.5
                sensitivity to detecting outliers; lower value increases
                sensitivity.
    sz:         integer | default: 3
                size of 3D searching matrix; sz=3 corresponds to a 3x3x3
                searching matrix
    conn:       string| default: 'face'
                connectivity to use in computing median
    """
    def __init__(self, img, brainmask=None, th=0.5, sz=3,
                 conn='face', bias='rand'):
        self.Threshold = th
        self.Size = sz
        self.Connectivity = conn
        self.Img = np.array(img)
        if brainmask is None:
            self.BrainMask = np.ones((self.Img.shape[0], self.Img.shape[
                1], self.Img.shape[2]), dtype=bool, order='F')
        else:
            self.BrainMask = np.array(brainmask).astype(bool)
        if self.Threshold > 1 or self.Threshold < 0:
            raise Exception('Threshold cannot be less than 0 or greater '
                            'than 1. Please specifty a between the range '
                            '[0 1].')
        # Get box filter properties
        centralIdx = np.median(range(self.Size)).astype(int)
        # Add 1 to central idx because first index starts with zero
        d2move = np.int(np.abs(self.Size - (centralIdx + 1)))
        # Apply a nan padding to all 3 dimensions of the input image and a
        # nan padding to mask. Padding widths is same distance between
        # centroid of patch to edge. This enables median filtering of edges
        self.Img = np.pad(self.Img, d2move, 'constant',
                          constant_values=np.nan)
        self.BrainMask = np.pad(self.BrainMask, d2move, 'constant',
                                constant_values=np.nan)
        # Create (3 x nvox) cartesian coordinate representation of image
        # within a brainmask
        cartIdx = np.array(np.where(
            np.multiply(self.Img, self.BrainMask) > 0))
        nvox = cartIdx.shape[1]
        inputs = tqdm(range(nvox),
                      desc='Filter: Outlier Mask    ',
                      unit='vox',
                      ncols=tqdmWidth)
        self.OutlierMask = np.zeros(self.Img.shape, dtype=bool)
        for i in inputs:
            # Index beginning and ending of patch
            Ib = cartIdx[0, i] - d2move
            Ie = cartIdx[0, i] + d2move
            Jb = cartIdx[1, i] - d2move
            Je = cartIdx[1, i] + d2move
            Kb = cartIdx[2, i] - d2move
            Ke = cartIdx[2, i] + d2move
            if self.Connectivity == 'all':
                # Remove 14th (centroid) element
                patchImg = np.delete(np.ravel(self.Img[
                                              Ib:Ie+1, Jb:Je+1, Kb:Ke+1]),
                                     13)
                connLimit = 26
            elif self.Connectivity == 'face':
                patchImg = self.Img[[Ib, Ie], cartIdx[1, i], cartIdx[2, i]]
                patchImg = np.hstack((patchImg,
                                      self.Img[cartIdx[0, i], [Jb, Je],
                                               cartIdx[2, i]]))
                patchImg = np.hstack((patchImg, self.Img[cartIdx[0, i],
                                                         cartIdx[1, i],
                                                         [Kb, Ke]]))
                connLimit = 6
            # Now compare voxel value to IQR. If more than 1.5 * IQR,
            # voxel is an outlier
            # IQR = np.subtract(*np.percentile(patchImg, [75, 25]))
            # if self.Img[cartIdx[0, i], cartIdx[1, i], cartIdx[2, i]] > \
            #         multiplier * IQR:
            #     self.OutlierMask[cartIdx[0, i],
            #                      cartIdx[1, i],
            #                      cartIdx[2, i]] = True
            pdiff = np.absolute((self.Img[cartIdx[0, i],
                                          cartIdx[1, i],
                                          cartIdx[2, i]]) - \
                                np.mean(patchImg)) / self.Img[cartIdx[0,i],
                                                              cartIdx[1,i],
                                                              cartIdx[2,i]]
            if pdiff > self.Threshold:
                # Mark outlier mask voxel as true
                self.OutlierMask[cartIdx[0, i],
                                 cartIdx[1, i],
                                 cartIdx[2, i]] = True

        # Create (3 x nvox) cartesian coordinate representation of image
        # within a brainmask
        cartIdx = np.array(np.where(self.OutlierMask > 0))
        nvox = cartIdx.shape[1]
        self.PatchIdx = np.zeros(nvox)
        inputs = tqdm(range(nvox),
                      desc='Filter: Find Replacement',
                      unit='vox',
                      ncols=tqdmWidth)
        cntSkip = 0
        for i in inputs:
            # Index beginning and ending of patch
            Ib = cartIdx[0, i] - d2move
            Ie = cartIdx[0, i] + d2move
            Jb = cartIdx[1, i] - d2move
            Je = cartIdx[1, i] + d2move
            Kb = cartIdx[2, i] - d2move
            Ke = cartIdx[2, i] + d2move
            if self.Connectivity == 'all':
                # Remove 14th (centroid) element
                patchViol = np.delete(np.ravel(
                    self.OutlierMask[Ib:Ie+1, Jb:Je+1, Kb:Ke+1]), 13)
                patchImg = np.delete(np.ravel(
                    self.Img[Ib:Ie + 1, Jb:Je + 1,Kb:Ke + 1]),
                                     13)
                connLimit = 26
            elif self.Connectivity == 'face':
                patchViol = self.OutlierMask[[Ib, Ie],
                                             cartIdx[1, i],
                                             cartIdx[2, i]]
                patchViol = np.hstack((patchViol,
                                       self.OutlierMask[cartIdx[0, i],
                                                        [Jb, Je],
                                                        cartIdx[2, i]]))
                patchViol = np.hstack((patchViol,
                                       self.OutlierMask[cartIdx[0, i],
                                                        cartIdx[1, i],
                                                        [Kb, Ke]]))
                patchImg = self.Img[[Ib, Ie], cartIdx[1, i], cartIdx[2, i]]
                patchImg = np.hstack((patchImg,
                                      self.Img[cartIdx[0, i], [Jb, Je],
                                               cartIdx[2, i]]))
                patchImg = np.hstack((patchImg, self.Img[cartIdx[0, i],
                                                         cartIdx[1, i],
                                                         [Kb, Ke]]))
                nVoil = np.sum(patchViol)
                # Here a check is performed to compute the number of
                # violations in a patch. If all voxels are violations,do
                # nothing. Otherwise, exclude violation voxels from the
                # median calculation
                # Assign a value of -1 to voxel that need to be skipped
                # because nan cannot be assigned to a vector of integers
                if nVoil == connLimit:
                    self.PatchIdx[i] = -1
                    cntSkip = cntSkip + 1
                    continue
                else:
                    # Sort all patch values in ascending order and remove NaNs
                    patchVals = np.array(np.sort(
                        patchImg[~np.isnan(patchImg)],
                        kind='quicksort'))
                    nVals = patchVals.size
                    # Median algorithm dependent on whether number of valid
                    # voxels (nVals) is even or odd
                    # If even:
                    if np.mod(nVals, 2) == 0:
                        medianIdxTmp = np.array([nVals/2 - 1, nVals/2],
                                                dtype=int)
                        if bias == 'left':
                            medianIdx = medianIdxTmp[0]
                        elif bias == 'right':
                            medianIdx = medianIdxTmp[1]
                        elif bias == 'rand':
                            medianIdx = medianIdxTmp[rnd.randint(0,1)]
                    # if odd:
                    else:
                        medianIdx = (nVals-1)/2
                    # Now that a the index of a median voxel is located,
                    # determine it's value in sorted list and find the
                    # location of that voxel in patch. The median index
                    # needs to be relative to patch, not sorted list. In
                    # the event that there are more than one indices of the
                    # same value, use the first one.
                    medianIdxP = np.where(patchImg == patchVals[np.int(medianIdx)])[0][0]
                    self.PatchIdx[i] = medianIdxP
            self.PatchIdx = np.array(self.PatchIdx, dtype='int')
        print('%d voxels out of %d were completely surrounded by '
              'violations and were ignored' \
              % (cntSkip, self.PatchIdx.shape[0]))
        # Unpad mask for saving
        self.OutlierMask = np.delete(self.OutlierMask,
                                     [0, self.OutlierMask.shape[0] - 1],
                                     axis=0)
        self.OutlierMask = np.delete(self.OutlierMask,
                                     [0, self.OutlierMask.shape[1] - 1],
                                     axis=1)
        self.OutlierMask = np.delete(self.OutlierMask,
                                     [0, self.OutlierMask.shape[2] - 1],
                                     axis=2)

    def apply(self, img, weight=1):
        """
        Applies the median filter object to an input image

        Parameters
        ----------
        img:    3D numpy array
                image to apply the filter on
        weight: integer | default = 1
                specify the weightage in logic
                ```
                    if weightage * img_threshold > ref_threshold:
                        substitute voxel
                ```
                this is to provide a lower weightage to diffusion maps
                as they are less likely to require median filtering
        """
        # Get box filter properties
        if weight < 0 or weight > 1:
             raise Exception('Threshold cannot be less than 0 or greater '
                            'than 1. Please specifty a between the range '
                            '[0 1].')
        centralIdx = np.median(range(self.Size)).astype(int)
        # Add 1 to central idx because first index starts with zero
        d2move = np.int(np.abs(self.Size - (centralIdx + 1)))
        # Pad image and Outlier again
        img = np.pad(img, d2move, 'constant', constant_values=np.nan)
        self.OutlierMask = np.pad(self.OutlierMask, d2move, 'constant',
                                  constant_values=np.nan)
        # Create (3 x nvox) cartesian coordinate representation of image
        # the outlier mask
        cartIdx = np.array(np.where(self.OutlierMask))
        nvox = cartIdx.shape[1]
        inputs = tqdm(range(nvox),
                      desc='Filter: Substitution    ',
                      unit='vox',
                      ncols=tqdmWidth)
        for i in inputs:
            # Index beginning and ending of patch
            Ib = cartIdx[0, i] - d2move
            Ie = cartIdx[0, i] + d2move
            Jb = cartIdx[1, i] - d2move
            Je = cartIdx[1, i] + d2move
            Kb = cartIdx[2, i] - d2move
            Ke = cartIdx[2, i] + d2move
            if self.Connectivity == 'all':
                # Remove 14th (centroid) element
                patchImg = np.delete(np.ravel(img[Ib:Ie+1, Jb:Je+1,
                                              Kb:Ke+1]), 13)
            elif self.Connectivity == 'face':
                print(cartIdx[:,i])
                patchImg = img[[Ib, Ie], cartIdx[1, i], cartIdx[2, i]]
                patchImg = np.hstack((patchImg, img[cartIdx[0, i],
                                                    [Jb, Je],
                                                    cartIdx[2, i]]))
                patchImg = np.hstack((patchImg, img[cartIdx[0, i],
                                                    cartIdx[1, i],
                                                    [Kb, Ke]]))
            pdiff = np.absolute((self.Img[cartIdx[0, i],
                                          cartIdx[1, i],
                                          cartIdx[2, i]]) - \
                                np.mean(patchImg)) / self.Img[
                        cartIdx[0, i],
                        cartIdx[1, i],
                        cartIdx[2, i]]
            if pdiff > self.Threshold:
                # Mark outlier mask voxel as true
                self.OutlierMask[cartIdx[0, i],
                                 cartIdx[1, i],
                                 cartIdx[2, i]] = True
            if self.PatchIdx[i] > 0:
                img[cartIdx[0, i],
                    cartIdx[1, i],
                    cartIdx[2, i]] = patchImg[self.PatchIdx[i]]
            # Unpad image
            img = np.delete(img, [0, img.shape[0] - 1], axis = 0)
            img = np.delete(img, [0, img.shape[1] - 1], axis = 1)
            img = np.delete(img, [0, img.shape[2] - 1], axis = 2)
        return img





    # def (self, bias='rand'):
    #     """
    #     Returns information on replacements for violating voxels
    #
    #     Usage
    #     -----
    #     m = med.findReplacement(bias='rand')
    #
    #     Parameters
    #     ----------
    #     bias: 'left', 'right', or 'rand'
    #            In the even the number of voxels in patch is even (for
    #            median calculation), 'left' will pick a median to the left
    #            of mean and 'right' will pick a median to the right of
    #            mean. 'rand' will randomny pick a bias.
    #
    #     Returns
    #     -------
    #     m: Vector containing index of replacement voxel in patch. In
    #     conn = 'face' max is 5 and conn =
    #     """
    #     # Get box filter properties
    #     centralIdx = np.median(range(self.Size)).astype(int)
    #     d2move = np.int(np.abs(self.Size - (centralIdx + 1)))  # Add 1 to central idx because first index starts with zero
    #
    #     violIdx = np.array(np.where(self.Mask))   # Locate coordinates of violations
    #     nvox = violIdx.shape[1]
    #     self.PatchIdx = np.zeros(violIdx.shape[1])
    #
    #
    #     inputs = tqdm(range(nvox),
    #                   desc='Median: Finding Outliers',
    #                   unit='vox',
    #                   ncols=tqdmWidth)
    #     cntSkip = 0
    #     for i in inputs:
    #         # Index beginning and ending of patch
    #         Ib = violIdx[0, i] - d2move
    #         Ie = violIdx[0, i] + d2move
    #         Jb = violIdx[1, i] - d2move
    #         Je = violIdx[1, i] + d2move
    #         Kb = violIdx[2, i] - d2move
    #         Ke = violIdx[2, i] + d2move
    #
    #         if self.Connectivity == 'all':
    #             # Remove 14th (centroid) element
    #             patchImg = np.delete(np.ravel(self.Img[
    #                                           Ib:Ie, Jb:Je, Kb:Ke]),13)
    #             connLimit = 26
    #         elif self.Connectivity == 'face':
    #             patchImg = self.Img[[Ib, Ie], violIdx[1, i], violIdx[2, i]]
    #             patchImg = np.hstack((patchImg,
    #                                   self.img[violIdx[0, i], [Jb, Je],
    #                                       violIdx[2, i]]))
    #             patchImg = np.hstack((patchImg, self.Img[violIdx[0, i],
    #                                                      violIdx[1, i],
    #                                                      [Kb, Ke]]))
    #             connLimit = 6
    #         else:
    #             raise Exception('Connectivity choice "{}" is invalid. "'
    #                             'Please enter either "all" or "face".'
    #                             .format(self.Connectivity))
    #         # Now compare voxel value to IQR. If more than 1.5 * IQR,
    #         # voxel is an outlier
    #         IQR = np.subtract(*np.percentile(patchImg, [75, 25]))
    #         if img[violIdx[0, i], violIdx[1, i], violIdx[2, i]] > \
    #                 1.5 * IQR:
    #             self.Mask[i] = True
    #         nVoil = np.sum(patchViol)
    #
    #         # Here a check is performed to compute the number of violations
    #         # in a patch. If all voxels are violations, do nothing.
    #         # Otherwise, exclude violation voxels from the median
    #         # calculation
    #         if nVoil == connLimit:
    #             self.PatchIdx[i] = np.nan
    #             cntSkip = cntSkip + 1
    #             continue
    #         else:
    #             # Sort all patch values in ascending order and remove NaNs
    #             patchVals = np.array(np.sort(patchImg[~np.isnan(patchImg)],kind='quicksort'))
    #             nVals = patchVals.size
    #
    #             # Median algorithm dependent on whether number of valid voxels (nVals) is even or odd
    #             if np.mod(nVals, 2) == 0:                                       # If even
    #                 medianIdxTmp = np.array([nVals/2 - 1, nVals/2],dtype=int)   # Convert to Py index (-1)
    #                 if bias == 'left':
    #                     medianIdx = medianIdxTmp[0]
    #                 elif bias == 'right':
    #                     medianIdx = medianIdxTmp[1]
    #                 elif bias == 'rand':
    #                     medianIdx = medianIdxTmp[rnd.randint(0,1)]
    #             else:                                                           # If odd
    #                 medianIdx = (nVals-1)/2                                     # Convert to Py index (-1)
    #
    #             # Now that a the index of a median voxel is located, determine it's value in sorted list and find the
    #             # location of that voxel in patch. The median index needs to be relative to patch, not sorted list. In
    #             # the event that there are more than one indexes of the same value, use the first one.
    #             medianIdxP = np.where(patchImg == patchVals[np.int(medianIdx)])[0][0]
    #             self.PatchIdx[i] = medianIdxP
    #     self.PatchIdx = np.array(self.PatchIdx, dtype='int')  # Convert to integer
    #     print('%d voxels out of %d were completely surrounded by violations and were ignored' %(cntSkip, violIdx.shape[1]))
    #     return self.PatchIdx
    #
    # def applyReplacement(self, img):
    #     """
    #     Applies median filter onto input images.
    #
    #     Usage
    #     -----
    #     filteredImage = med.applyReplacement(image)
    #
    #     Parameters
    #     ----------
    #     image:          input image to apply the median voxel replacement on
    #
    #     Returns
    #     -------
    #     filteredImage:  median filtered image
    #     """
    #     # Get box filter properties
    #     centralIdx = np.median(range(self.Size))
    #     d2move = np.int(
    #         np.abs(self.Size - (centralIdx + 1)))  # Add 1 to central idx because first index starts with zero
    #
    #     # Pad image with zeros
    #     img = np.pad(img, d2move, 'constant', constant_values=0)
    #
    #     violIdx = np.array(np.where(self.Mask))  # Locate coordinates of violations
    #     # self.PatchIdx = np.array(np.zeros(violIdx.shape[1]),dtype='int')
    #
    #     inputs = tqdm(range(self.PatchIdx.size))
    #     for i in inputs:
    #         # Index beginning and ending of patch
    #         Ib = violIdx[0, i] - d2move
    #         Ie = violIdx[0, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
    #         Jb = violIdx[1, i] - d2move
    #         Je = violIdx[1, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
    #         Kb = violIdx[2, i] - d2move
    #         Ke = violIdx[2, i] + d2move + 1     # Mitigate Python's [X,Y) indexing
    #
    #         if self.Connectivity == 'all':
    #             patchViol = np.delete(np.ravel(self.Mask[Ib:Ie, Jb:Je, Kb:Ke]), 13)     # Remove 14th (centroid) element
    #             patchImg = np.delete(np.ravel(self.Img[Ib:Ie, Jb:Je, Kb:Ke]), 13)            # Remove 14th (centroid) element
    #         elif self.Connectivity == 'face':
    #             patchViol = self.Mask[[Ib, Ie], violIdx[1, i], violIdx[2, i]]
    #             patchViol = np.hstack((patchViol, self.Mask[violIdx[0,i], [Jb, Je], violIdx[2, i]]))
    #             patchViol = np.hstack((patchViol, self.Mask[violIdx[0, i], violIdx[1, i], [Kb, Ke]]))
    #             patchImg = img[[Ib, Ie], violIdx[1, i], violIdx[2, i]]
    #             patchImg = np.hstack((patchImg, img[violIdx[0, i], [Jb,
    #                                                               Je], violIdx[2, i]]))
    #             patchImg = np.hstack((patchImg, img[violIdx[0, i],
    #                                               violIdx[1, i], [Kb, Ke]]))
    #
    #         if np.isnan(self.PatchIdx[i]) == True:
    #             continue
    #         else:
    #             img[violIdx[0, i], violIdx[1, i], violIdx[2, i]] = patchImg[self.PatchIdx[i]]
    #
    #     # Unpad image by removing first and last slices along each axis
    #     img = np.delete(img, [0, img.shape[0] - 1], axis=0)
    #     img = np.delete(img, [0, img.shape[1] - 1], axis=1)
    #     img = np.delete(img, [0, img.shape[2] - 1], axis=2)
    #     return img

def vectorize(img, mask):
    """ Returns vectorized image based on brain mask, requires no input
    parameters
    If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
    If the input is 3D or 4D, vectorize it using a mask
    Classification: Method

    Usage
    -----
    vec = dwi.vectorize(img) if there's no mask
    vec = dwi.vectorize(img, mask) if there's a mask

    Returns
    -------
    vec: N X number_of_voxels vector or array, where N is the number of DWI
    volumes
    """
    if mask is None:
        mask = np.ones((img.shape[0], img.shape[1], img.shape[2]), order='F')
    mask = mask.astype(bool)
    if img.ndim == 1:
        n = img.shape[0]
        s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), order='F')
        s[mask] = img
    if img.ndim == 2:
        n = img.shape[0]
        s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], n), order='F')
        for i in range(0, n):
            s[mask, i] = img[i,:]
    if img.ndim == 3:
        maskind = np.ma.array(img, mask=np.logical_not(mask))
        s = np.ma.compressed(maskind)
    if img.ndim == 4:
        s = np.zeros((img.shape[-1], np.sum(mask).astype(int)), order='F')
        for i in range(0, img.shape[-1]):
            tmp = img[:,:,:,i]
            # Compressed returns non-masked area, so invert the mask first
            maskind = np.ma.array(tmp, mask=np.logical_not(mask))
            s[i,:] = np.ma.compressed(maskind)
    return np.squeeze(s)

def writeNii(map, hdr, outDir, range=None):
    """Write clipped NifTi images
    Classification: Function

    Usage
    -----
    writeNii(matrix, header, output_directory, [0, 2])

    Parameters
    ----------
    map:    3D matrix to write
    header: NifTi header file containing NifTi properties
    outDir: string
            Output file name
    range:  Range to clip images, inclusive of values in range

    Returns
    -------
    None
    """
    if range == None:
        clipped_img = nib.Nifti1Image(map, hdr.affine, hdr.header)
    else:
        clipped_img = clipImage(map, range)
        clipped_img = nib.Nifti1Image(clipped_img, hdr.affine, hdr.header)
    nib.save(clipped_img, outDir)

def clipImage(img, range):
    """ Clips input matrix within desired range. Min and max values are
    inclusive of range

    Usage
    -----
    clippedImage = clipImage(image, [0 3])
    Clips input matrix in the range 0 to 3

    Parameters
    ----------
    img:            input image matrix
    range:          [1 x 2] vector specifying range to clip

    Returns
    -------
    clippedImage:   clipped image; same size as img
    """
    img[img > range[1]] = range[1]
    img[img < range[0]] = range[0]
    return img
