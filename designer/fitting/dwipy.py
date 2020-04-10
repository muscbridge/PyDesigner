#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing
import os
import random
import cvxpy as cvx
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
import scipy.linalg as sla
from tqdm import tqdm
from . import dwidirs

# Define the lowest number possible before it is considered a zero
minZero = 1e-8
# Define number of directions to resample after computing all tensors
dirSample = 256
# Progress bar Properties
tqdmWidth = 70  # Number of columns of progress bar
# Set default numpy errorstates
np.seterr(all = 'ignore')
defaultErrorState = np.geterr()

class DWI(object):
    """
    The DWI object handles tensor estimation and parameter extraction
    of dwiffusion weighted images

    Attributes
    ----------
    hdr : class
        Nibabel class object of input nifti file
    img : ndarray
        3D or 4D input image array
    grad : ndarray
        [N x 4] gradient and bvalue scheme, where the first three
        columns are the X, Y, and Z gradient vectors respectively, and
        the fourth column is B-values
    mask : ndarray(dtype=bool)
        3D array of brain mask
    MaskStatus : bool
        True if brain_mask.nii is present, False otherwise
    workers : int
        Number of CPU workers to use in processing
    """
    def __init__(self, imPath, nthreads=-1):
        """
        DWI class initializer

        Parameters
        ----------
        imPath : str
            Path to input nifti file

        nthreads : int
            Number of CPU workers to use in processing (Defaults to
            all physically present workers)
        """
        if not os.path.exists(imPath):
            raise OSError('Input image {} not found'.format(imPath))
        self.hdr = nib.load(imPath)
        self.img = np.array(self.hdr.dataobj)
        truncateIdx = np.logical_or(np.isnan(self.img),
                                (self.img < minZero))
        self.img[truncateIdx] = minZero
        # Get just NIFTI filename + extensio
        (path, file) = os.path.split(imPath)
        # Remove extension from NIFTI filename
        fName = os.path.splitext(file)[0]
        # Add .bval to NIFTI filename
        bvalPath = os.path.join(path, fName + '.bval')
        # Add .bvec to NIFTI filename
        bvecPath = os.path.join(path, fName + '.bvec')
        if os.path.exists(bvalPath) and os.path.exists(bvecPath):
            # Load bvecs
            bvecs = np.loadtxt(bvecPath)
            # Load bvals
            bvals = np.rint(np.loadtxt(bvalPath) / 1000)
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
        """
        Returns a vector of b-values, requires no input arguments

        Returns
        -------
        ndarray(dtype=float)
            Vector array of b-values

        Examples
        --------
        bvals = dwi.getBvals(), where dwi is the DWI class object
        """
        return self.grad[:,3]

    def getBvecs(self):
        """
        Returns an array of gradient vectors, requires no input
        parameters
        
        Returns
        -------
        ndarray(dtype=float)
            [N x 3] array of gradient vectors

        Examples
        --------
        bvecs = dwi.getBvecs(), where dwi is the DWI class object
        """
        return self.grad[:,0:3]

    def maxBval(self):
        """
        Returns the maximum b-value in a dataset to determine between
        DTI and DKI, requires no input parameters

        Returns
        -------
        float
            maximum B-value in DWI

        Examples
        --------
        a = dwi.maxBval(), where dwi is the DWI class object

        """
        return max(np.unique(self.grad[:,3])).astype(int)

    def getndirs(self):
        """
        Returns the number of gradient directions acquired from the
        scanner

        Returns
        -------
        ndarray
            number of gradient directions

        Examples
        --------
        n = dwi.getndirs(), where dwi is the DWI class object
        """
        return np.sum(self.grad[:, 3] == self.maxBval())

    def tensorType(self):
        """
        Returns whether input image is DTI or DKI compatible, requires
        no input parameters

        Returns
        -------
        str
            'dti' or 'dki'

        Examples
        --------
        a = dwi.tensorType(), where dwi is the DWI class object
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
        """
        Returns logical value to answer the mystical question whether
        the input image is DKI

        Returns
        -------
        ans : bool
            True if DKI; false otherwise

        Examples
        --------
        ans = dwi.isdki(), where dwi is the DWI class object
        """
        if self.tensorType() == 'dki':
            ans = True
        else:
            ans = False
        return ans

    def createTensorOrder(self, order=None):
        """
        Creates tensor order array and indices

        Parameters
        ----------
        order :  2 or 4 (int or None)
            Tensor order number, 2 for diffusion and 4 for kurtosis.
            Default: None; auto-detect

        Returns
        -------
        cnt : ndarray(dtype=int)
            Count of number of times a tensor appears
        ind : ndarray(dtype=int)
            Indices of count

        Examples
        --------
        (cnt, ind) = dwi.createTensorOrder(order)

        Notes
        -----
        The tensors for this pipeline are based on NYU's designer layout as
        depicted in the table below. This will soon be depreciated and
        updated with MRTRIX3's layout.
        
         .. code-block:: none

            ~~~~~~D~~~~~~
            1  |    D11
            2  |    D12
            3  |    D13
            4  |    D22
            5  |    D23
            6  |    D33
            ~~~~~~K~~~~~~
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
                ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3],
                                [3, 3])) - 1
            elif imType == 'dki':
                cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6,
                                4, 1],
                               dtype=int)
                ind = np.array(([1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 1, 3],
                                [1, 1, 2, 2], [1, 1, 2, 3], [1, 1, 3, 3],
                                [1, 2, 2, 2], [1, 2, 2, 3], [1, 2, 3, 3],
                                [1, 3, 3, 3], [2, 2, 2, 2], [2, 2, 2, 3],
                                [2, 2, 3, 3], [2, 3, 3, 3], [3, 3, 3, 3]))\
                      - 1
        elif order == 2:
            cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
            ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3],
                            [3, 3])) - 1
        elif order == 4:
            cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4,
                            1],
                           dtype=int)
            ind = np.array(([1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,2,2],
                            [1,1,2,3],[1,1,3,3],[1,2,2,2],[1,2,2,3],
                            [1,2,3,3],[1,3,3,3],[2,2,2,2],[2,2,2,3],
                            [2,2,3,3],[2,3,3,3],[3,3,3,3])) - 1
        else:
            raise ValueError('createTensorOrder: Please enter valid '
                             'order values (2 or 4).')
        return cnt, ind

    def fibonacciSphere(self, samples=1, randomize=True):
        """
        Returns evenly spaced points on a sphere

        Parameters
        ----------
        samples : int
            Number of points to compute from sphere, must be a
            positive and real integer (Default: 1)

        randomize : bool
                    True if sampling is randomized; False otherwise
                    (Default: True)

        Returns
        ------
        points : ndarray(dtype=float)
            [3 x samples] array containing evenly spaced points
            from a sphere

        Examples
        --------
        dirs = dwi.fibonacciSphere(256, True)
        """
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
        """
        Get the radial component of a metric from a set of directions

        Parameters
        ----------
        dir : ndarray(dtype=float)
            [n x 3] input array of directions
        n : int
            number of rows, n

        Returns
        -------
        dirs :   Matrix containing radial components

        Examples
        --------
        grad = dwi.radiansampling(dir, number_of_dirs)

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
        """
        Computes apparent diffusion coefficient (ADC)

        Parameters
        ----------
        dt : ndarray(dtype=float)
            [21 x nvoxel] array containing diffusion tensor
        dir :  ndarray(dtype=float)
            [n x 3] array containing gradient directions

        Returns
        -------
        adc : ndarray(dtype=float)
            Array containing apparent diffusion coefficient

        Examples
        --------
        adc = dwi.diffusionCoeff(dt, dir)
        """
        dcnt, dind = self.createTensorOrder(2)
        ndir = dir.shape[0]
        bD = np.tile(dcnt,(ndir, 1)) * dir[:,dind[:, 0]] * \
             dir[:, dind[:, 1]]
        adc = np.matmul(bD, dt)
        return adc

    def kurtosisCoeff(self, dt, dir):
        """
        Computes apparent kurtosis coefficient (AKC)

        Parameters
        ----------
        dt : ndarray(dtype=float)
            [21 x nvoxel] array containing diffusion tensor
        dir : ndarray(dtype=float)
            [n x 3] array containing gradient directions

        Returns
        ------
        adc : ndarray(dtype=float)
            Array containing apparent kurtosis coefficient

        Examples
        --------
        adc = dwi.kurtosisCoeff(dt, dir)
        """
        wcnt, wind = self.createTensorOrder(4)
        ndir = dir.shape[0]
        adc = self.diffusionCoeff(dt[:6], dir)
        adc[adc < minZero] = minZero
        md = np.sum(dt[np.array([0,3,5])], 0)/3
        bW = np.tile(wcnt,(ndir, 1)) * dir[:,wind[:, 0]] * \
             dir[:,wind[:,1]] * dir[:,wind[:, 2]] * dir[:,wind[:, 3]]
        akc = np.matmul(bW, dt[6:])
        akc = (akc * np.tile(md**2, (adc.shape[0], 1)))/(adc**2)
        return akc

    def dtiTensorParams(self, nn):
        """
        Computes sorted DTI tensor eigenvalues and eigenvectors

        Parameters
        ----------
        DT : ndarray(dtype=float)
            Diffusion tensor array

        Returns
        -------
        values : ndarray(dtype=float)
            Array of sorted eigenvalues
        vectors : ndarray(dtype=float)
            Array pf sorted eigenvectors

        Examples
        --------
        (values, vectors) = dwi.dtiTensorParams(DT)
        """
        values, vectors = np.linalg.eig(nn)
        idx = np.argsort(-values)
        values = -np.sort(-values)
        vectors = vectors[:, idx]
        return values, vectors

    def dkiTensorParams(self, v1, dt):
        """
        Uses average directional statistics to approximate axial
        kurtosis(AK) and radial kurtosis (RK)

        Parameters
        ----------
        v1 : ndarray(dtype=float)
            Array of first eigenvectors from DWI.dtiTensorParams()
        dt : ndarray(dtype=float)
            Array of diffusion tensor

        Returns
        -------
        rk : ndarray(dtype=float)
            Radial Kurtosis
        ak : ndarray(dtype=float)
            Axial Kurtosis
        kfa : ndarray(dtype=float)
            Kurtosis Fractional Anisotropy
        mkt : ndarray(dtype=float)
            Mean Kurtosis Tensor

        Examples
        --------
        (rk, ak) = dwi.dkiTensorParams(v1, dt)
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
        """
        Estimates diffusion and kurtosis tenor at voxel with
        unconstrained Moore-Penrose pseudoinverse or constrained
        quadratic convex optimization. This is a helper function for
        dwi.fit() so a multiprocessing parallel loop can be iterated over
        voxels

        Parameters
        ----------
        shat : ndarray(dtype=float)
            [ndir x 1] array of S_hat, approximated signal intensity
            at voxel
        dwi : ndarray(dtype=float)
            [ndir x 1] array of diffusion weighted intensity values at
            voxel, for all b-values
        b : ndarray(dtype=float)
            [ndir x 1] array of b-values vector
        cons : ndarray(dtype=float)
            [(n * dir) x 22) array containing inequality constraints
            for fitting (Default: None)
        warmup : ndarray(dtype=float)
            Estimated dt vector (22, 0) at each voxel for warm
            starting constrianed tensor fitting (Default: None)

        Returns
        -------
        dt : ndarray(dtype=float)
            Diffusion tensor

        Examples
        --------
        dt = dwi.wlls(shat, dwi, b, constraints)

        Notes
        -----
        For Unconstrained Fitting:
        In the absence of constraints, an exact formulation in the form
        Cx = b is produced. This is further simplified to x_hat = C^+ *
        b. One can use the Moore-Penrose method to compute the
        pseudoinverse to approximate diffusion tensors.

        For Constrained Fitting:
        .. code-block:: none

            The equation |Cx -b|^2 expands to 0.5*x.T(C.T*A)*x -(C.T*b).T
                                                      ~~~~~      ~~~~~
                                                        P          q
        
        where A is denoted by multiplier matrix (w * b)
        Multiplying by a positive constant (0.5) does not change the value
        of optimum x*. Similarly, the constant offset b.T*b does not
        affect x*, therefore we can leave these out.

        Minimize: || C*x -b ||_2^2
            subject to A*x <= b
            No lower or upper bounds
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
                prob.solve(solver=cvx.OSQP,
                           warm_start=True,
                           max_iter=20000,
                           polish=True,
                           linsys_solver='qdldl')
                dt = x.value
                if prob.status != 'optimal':
                    dt = np.full(n, minZero)
            except:
                dt = np.full(n, minZero)
        return dt

    def fit(self, constraints=None, reject=None):
        """
        Returns fitted diffusion or kurtosis tensor

        Parameters
        ----------
        constraints : array_like(dtype=int)
            [1 x 3] vector that specifies which constraints to use
            (Default: None)
        reject : ndarray(dtype=bool)
            4D array containing information on voxels to exclude
            from DT estimation (Default: None)

        Examples
        --------
        dwi.fit()
        dwi.fit(constraints=[0,1,0], reject=irlls_output)
        """
        # Handle rejected voxels from IRLLS
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
        shat = highprecisionexp(np.matmul(self.b, init))
        if constraints is None or (constraints[0] == 0 and
                                   constraints[1] == 0 and
                                   constraints[2] == 0):
            inputs = tqdm(range(0, dwi_.shape[1]),
                          desc='Unconstrained Tensor Fit',
                          bar_format='{desc}: [{percentage:0.0f}%]',
                          unit='vox',
                          ncols=tqdmWidth)
            self.dt = Parallel(n_jobs=self.workers, prefer='processes') \
                (delayed(self.wlls)(shat[~reject_[:, i], i], \
                                    dwi_[~reject_[:, i], i], \
                                    self.b[~reject_[:, i]]) \
                 for i in inputs)
        else:
            # C is linear inequality constraint matrix A_ub
            C = self.createConstraints(constraints)
            inputs = tqdm(range(0, dwi_.shape[1]),
                          desc='Constrained Tensor Fit',
                          bar_format='{desc}: [{percentage:0.0f}%]',
                          unit='vox',
                          ncols=tqdmWidth)
            self.dt = Parallel(n_jobs=self.workers,
                               prefer='processes') \
                (delayed(self.wlls)(shat[~reject_[:, i], i],
                                         dwi_[~reject_[:, i], i],
                                         self.b[~reject_[:, i]],
                                         cons=C) for i in inputs)
        self.dt = np.reshape(self.dt, (dwi_.shape[1], self.b.shape[1])).T
        self.s0 = highprecisionexp(self.dt[0,:])
        self.dt = self.dt[1:,:]
        D_apprSq = 1/(np.sum(self.dt[(0,3,5),:], axis=0)/3)**2
        self.dt[6:,:] = self.dt[6:,:]*np.tile(D_apprSq, (15,1))

    def createConstraints(self, constraints=[0, 1, 0]):
        """
        Generates constraint array for constrained minimization quadratic
        programming

        Parameters
        ----------
        constraints :   array_like(dtype=int)
            [1 X 3] logical vector indicating which constraints
            out of three to enable (Default: [0, 1, 0])
            C1 is Dapp > 0
            C1 is Kapp > 0
            C3 is Kapp < 3/(b*Dapp)

        Returns
        -------
        C : ndarray(dtype=float)
            Array containing constraints to consider during
            minimization, C is shaped [number of constraints enforced *
            number of directions, 22]

        Examples
        --------
        C = dwi.createConstraints([0, 1, 0])
        """
        if sum(constraints) >= 0 and sum(constraints) <= 3:
            dcnt, dind = self.createTensorOrder(2)
            wcnt, wind = self.createTensorOrder(4)
            cDirs = dwidirs.dirs30
            ndirs = cDirs.shape[0]
            C = np.empty((0, 22))
            if constraints[0] > 0:  # Dapp > 0
                C = np.append(C, np.hstack(
                    (np.zeros((ndirs, 1)),
                     np.tile(dcnt, [ndirs, 1]) * cDirs[:, dind[:, 0]] * \
                     cDirs[:, dind[:, 1]],np.zeros((ndirs, 15)))), axis=0)
            if constraints[1] > 0:  # Kapp > 0
                C = np.append(C, np.hstack(
                    (np.zeros((ndirs, 7)),
                     np.tile(wcnt, [ndirs, 1]) * cDirs[:, wind[:, 0]] * \
                     cDirs[:, wind[:, 1]] * cDirs[:,wind[:,2]] * \
                     cDirs[:,wind[:,3]])),axis=0)
            if constraints[2] > 0:  # K < 3/(b*Dapp)
                C = np.append(C, np.hstack(
                    (np.zeros((ndirs, 1)),
                     3 / self.maxBval() * \
                     np.tile(dcnt, [ndirs, 1]) * cDirs[:, dind[:, 0]],
                     np.tile(-wcnt, [ndirs, 1]) * cDirs[:, wind[:, 1]] * \
                     cDirs[:,wind[:, 2]] * cDirs[:,wind[:, 3]])),axis=0)
        else:
            print('Invalid constraints. Please use format "[0, 0, 0]"')
        return C

    def extractDTI(self):
        """
        Extract all DTI parameters from DT tensor. Warning, this can
        only be run after tensor fitting dwi.fit()

        Returns
        -------
        md : ndarray(dtype=float)
            Mean Diffusivity
        rd : ndarray(dtype=float)
            Radial Diffusivity
        ad : ndarray(dtype=float)
            Axial Diffusivity
        fa : ndarray(dtype=float) 
            Fractional Anisotropy
        fe : ndarray(dtype=float)
            First Eigenvectors
        trace : ndarray(dtype=float)
            Sum of first eigenvalues

        Examples
        --------
        (md, rd, ad, fa) = dwi.extractDTI(), where dwi is the DWI class
        object
        """
        # extract all tensor parameters from dt
        DT = np.reshape(
            np.concatenate((self.dt[0, :], self.dt[1, :], self.dt[2, :],
                            self.dt[1, :], self.dt[3, :], self.dt[4, :],
                            self.dt[2, :], self.dt[4, :], self.dt[5, :])),
            (3, 3, self.dt.shape[1]))
        # get the trace
        rdwi = highprecisionexp(np.matmul(self.b[:, 1:], self.dt))
        B = np.round(-(self.b[:, 0] + self.b[:, 3] + self.b[:, 5]) * 1000)
        uB = np.unique(B)
        trace = np.zeros((self.dt.shape[1], uB.shape[0]))
        for ib in range(0, uB.shape[0]):
            t = np.where(B == uB[ib])
            trace[:, ib] = np.mean(rdwi[t[0], :], axis=0)
        nvox = self.dt.shape[1]
        inputs = tqdm(range(0, nvox),
                      desc='DTI parameters',
                      bar_format='{desc}: [{percentage:0.0f}%]',
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
        fa = np.sqrt(1 / 2) * \
             np.sqrt((l1 - l2) ** 2 + \
                     (l2 - l3) ** 2 + \
                     (l3 - l1) ** 2) / \
             np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
        fe = np.abs(np.stack((fa * v1[:, :, :, 0], fa * v1[:, :, :, 1],
                              fa * v1[:, :, :, 2]), axis=3))
        trace = vectorize(trace.T, self.mask)
        return md, rd, ad, fa, fe, trace

    def extractDKI(self):
        """
        Extract all DKI parameters from DT tensor. Warning, this can
        only be run after tensor fitting dwi.fit()

        Returns
        -------
        mk : ndarray(dtype=float)
            Mean Diffusivity
        rk : ndarray(dtype=float)
            Radial Diffusivity
        ak : ndarray(dtype=float)
            Axial Diffusivity
        kfa : ndarray(dtype=float)
            Kurtosis Fractional Anisotropy
        mkt : ndarray(dtype=float)
            Mean Kurtosis Tensor
        trace : ndarray(dtype=float)
            Sum of first eigenvalues

        Examples
        --------
        (mk, rk, ak, fe, trace) = dwi.extractDTI(), where dwi is the DWI
        class object
        """
        # get the trace
        rdwi = highprecisionexp(np.matmul(self.b[:, 1:], self.dt))
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
                      desc='DKI parameters',
                      bar_format='{desc}: [{percentage:0.0f}%]',
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

    def extractWMTI(self):
        """
        Returns white matter tract integrity (WMTI) parameters. Warning:
        this can only be run after fitting and DWI.extractDTI().

        Returns
        -------
        awf : ndarray(dtype=float)
            Axonal Water Fraction
        eas_ad : ndarray(dtype=float)
            Extra-axonal space Axial Diffusivity
        eas_rd : ndarray(dtype=float)
            Extra-axonal Space Radial Diffusivity
        eas_md : ndarray(dtype=float)
            Extra-axonal Space Mean Diffusivity
        eas_tort : ndarray(dtype=float)
            Extra-axonal Space Tortuosity
        ias_ad : ndarray(dtype=float)
            Intra-axonal Space Axial Diffusivity
        ias_rd : ndarray(dtype=float)
            Intra-axonal Space Radial Diffusivity
        ias_da : ndarray(dtype=float)
            Intra-axonal Space Intrinsic Diffusivity
        ias_tort : ndarray(dtype=float)
            Intra-axonal Space Tortuosity
        """
        def wmtihelper(dt, dir, adc, akc, awf, adc2dt):
            # Avoid complex output. However,
            # negative AKC might be taken care of by applying constraints
            with np.errstate(invalid='ignore'):
                akc[akc < minZero] = minZero 
            try:
                # Eigenvalue decomposition of De
                De = np.multiply(
                    adc,
                    1 + np.sqrt(
                        (np.multiply(akc, awf) / (3 * (1 - awf)))))
                dt_e = np.matmul(adc2dt, De)
                DTe = dt_e[[0, 1, 2, 1, 3, 4, 2, 4, 5]]
                DTe = np.reshape(DTe, (3, 3), order='F')
                eigval = sla.eigh(DTe, eigvals_only=True)
                eigval = np.sort(eigval)[::-1]
                eas_ad = eigval[0]
                eas_rd = 0.5 * (eigval[1] + eigval[2])
                eas_md = np.add(eas_ad, (2 * eas_rd)) / 3
                try:
                    eas_tort = eas_ad / eas_rd
                except:
                    eas_tort = minZero
            except:
                eas_ad = minZero
                eas_rd = minZero
                eas_md = minZero
                eas_tort = minZero
            try:
                # Eigenvalue decomposition of Da
                Di = np.multiply(
                    adc,
                    1 - np.sqrt(
                        (np.multiply(akc, (1 - awf)) / (3 * awf))))
                dt_i = np.matmul(adc2dt, Di)
                DTi = dt_i[[0, 1, 2, 1, 3, 4, 2, 4, 5]]
                DTi = np.reshape(DTi, (3, 3), order='F')
                eigval = sla.eigh(DTi, eigvals_only=True)
                eigval = np.sort(eigval)[::-1]
                ias_ad = eigval[0]
                ias_rd = 0.5 * (eigval[1] + eigval[2])
                ias_da = np.add(ias_ad, (2 * ias_rd))
                np.seterr(invalid='raise')
                try:
                    ias_tort = ias_ad / ias_rd
                except:
                    ias_tort = minZero
            except:
                ias_ad = minZero
                ias_rd = minZero
                ias_da = minZero
                ias_tort = minZero
            return eas_ad, eas_rd, eas_md, eas_tort, ias_ad, ias_rd, ias_da, ias_tort
        dir = dwidirs.dirs10000
        nvox = self.dt.shape[1]
        N = dir.shape[0]
        nblocks = 10
        maxk = np.zeros((nvox, nblocks)).astype(float)
        inputs = tqdm(range(nblocks),
                      desc='Extracting AWF',
                      bar_format='{desc}: [{percentage:0.0f}%]',
                      unit='iter',
                      ncols=tqdmWidth)
        for i in inputs:
            maxk = np.stack(self.kurtosisCoeff(
                self.dt,dir[int(N/nblocks*i):int(N/nblocks*(i+1))])).astype(float)
            maxk = np.nanmax(maxk, axis=0)
        awf = np.divide(maxk, (maxk + 3)).astype(float)
        # Changes voxels less than minZero, nans and infs to minZero
        truncateIdx = np.logical_or(
            np.logical_or(np.isnan(awf), np.isinf(awf)),
            (awf < minZero))
        awf[truncateIdx] = minZero
        dirs = dwidirs.dirs30
        adc = self.diffusionCoeff(self.dt[:6], dirs)
        akc = self.kurtosisCoeff(self.dt, dirs)
        (dcnt, dind) = self.createTensorOrder(2)
        adc2dt = np.linalg.pinv(np.matmul(
                                (dirs[:, dind[:, 0]] * \
                                 dirs[:, dind[:, 1]]),
                                 np.diag(dcnt)))
        eas_ad = np.zeros(nvox)
        eas_rd = np.zeros(nvox)
        eas_md = np.zeros(nvox)
        eas_tort = np.zeros(nvox)
        ias_ad = np.zeros(nvox)
        ias_rd = np.zeros(nvox)
        ias_da = np.zeros(nvox)
        ias_tort = np.zeros(nvox)
        inputs = tqdm(range(nvox),
                      desc='Extracting EAS and IAS',
                      bar_format='{desc}: [{percentage:0.0f}%]',
                      unit='vox',
                      ncols=tqdmWidth)
        eas_ad, eas_rd, eas_md, eas_tort, ias_ad, ias_rd, ias_da, ias_tort = zip(*Parallel(
            n_jobs=self.workers, prefer='processes')(
            delayed(wmtihelper)(self.dt[:, i],
                                dirs,
                                adc[:, i],
                                akc[:,i],
                                awf[i],
                                adc2dt) for i in inputs))
        awf = vectorize(awf, self.mask)
        eas_ad = vectorize(np.array(eas_ad), self.mask)
        eas_rd = vectorize(np.array(eas_rd), self.mask)
        eas_md = vectorize(np.array(eas_md), self.mask)
        eas_tort = vectorize(np.array(eas_tort), self.mask)
        ias_ad = vectorize(np.array(ias_ad), self.mask)
        ias_rd = vectorize(np.array(ias_rd), self.mask)
        ias_da = vectorize(np.array(ias_da), self.mask)
        ias_tort = vectorize(np.array(ias_tort), self.mask)
        return awf, eas_ad, eas_rd, eas_md, eas_tort, ias_ad, ias_rd, ias_da, ias_tort

    def findViols(self, c=[0, 1, 0]):
        """
        Returns a 3D violation map of voxels that violate constraints.

        Parameters
        ----------
        img : ndarray(dtype=float)
            3D metric array such as mk or fa
        c : array_like(dtype=int)
            [3 x 1] vector that toggles which constraints to check
            c[0]: Check D < 0 constraint
            c[1]: Check K < 0 constraint (Default)
            c[2]: Check K > 3/(b*D) constraint

        Returns
        -------
        map : ndarray(dtype=bool)
            3D array containing locations of voxels that incur directional
            violations. Voxels with values contain violaions and voxel
            values represent proportion of directional violations.

        Examples
        --------
        map = findViols(img, [0 1 0]

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
            tmp[2] = np.size(np.nonzero(akc[:, i] > \
                                        (3/(maxB * adc[:, i]))))
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
        """
        Creates a 3D maps of good directions from IRLLS outlier map
        For any given voxel, a violation is computed using logical `or`
        operant for all b-values. Whether an outlier occurs at b1000
        or b1000 and b2000, that voxel is still a violation unless
        none of the b-values have outliers.

        Parameters
        ----------
        outliers : ndarray(dtype=bool)
            4D maps of outliers from IRLLS

        Returns
        -------
        map : ndarray(dtype=int)
            3D map of number of good directions

        Examples
        --------
        map = dwi.goodDirections(outliers)
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
        Returns the proportions of violations occurring at a voxel.

        Parameters
        ----------
        img : ndarray(dtype=float)
            3D metric array such as mk or fa
        c : array_like(dtype=int)
            [3 x 1] vector that toggles which constraints to check
            c[0]: Check D < 0 constraint
            c[1]: Check K < 0 constraint (Default)
            c[2]: Check K > 3/(b*D) constraint

        Returns
        -------
        n : ndarray(dtype=float)
            percentaghhe ranging from 0 to 1 that indicates proportion
            of violations occuring at voxel.

        Examples
        --------
        map = findViols(voxel, [0 1 0]
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
            (delayed(self.findVoxelViol)(adc[:,i],
                                         akc[:,i], maxB, [0, 1, 0]) for\
                i in inputs)
        map = np.reshape(pViols2, nvox)
        map = self.multiplyMask(vectorize(map,self.mask))
        return map

    def multiplyMask(self, img):
        """
        Multiplies a 3D image by the brain mask

        Parameters
        ----------
        img : ndarray(dtype=float)
            3D image to be multiplied

        Returns
        -------
        ndarray(dtype=float)
            multiplied image
        """
        # Returns an image multiplied by the brain mask to remove all
        # values outside the brain
        return np.multiply(self.mask.astype(bool), img)

    def akcoutliers(self, iter=10):
        """
        Uses 100,000 direction in chunks of 10 to iteratively find
        outliers. Returns a mask of locations where said violations
        occur. Multiprocessing is disabled because this is a
        memory-intensive task.
        To be run only after tensor fitting.

        Parameters
        ----------
        iter : int, optional
            number of iterations to perform out of 10. Reduce this
            number if your computer does not have sufficient RAM.
            (Default: 10)

        Returns
        -------
        akc_out : ndarray(dtype=bool)
            3D map containing outliers where AKC falls fails the
            inequality test -2 < AKC < 10

        Examples
        --------
        akc_out = dwi.akoutliers(), where dwi is the DWI class object
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
                      desc='AKC Outlier Detection',
                      bar_format='{desc}: [{percentage:0.0f}%]',
                      unit='blk',
                      ncols=tqdmWidth)
        for i in inputs:
            akc = self.kurtosisCoeff(
                self.dt, dir[int(N/nblocks*i):int(N/nblocks*(i+1))])
            akc_out[np.where(np.any(np.logical_or(akc < -2, akc > 10),
                                    axis=0))] = True
            akc_out.astype('bool')
        return vectorize(akc_out, self.mask)

    def akccorrect(self, akc_out, window=3, connectivity='face'):
        """
        Applies AKC outlier map to DT to replace outliers with a
        moving median. Run this only after tensor fitting and akc
        outlier detection.

        Parameters
        ----------
        akc_out : ndarray(dtype=bool)
            3D map containing outliers from DWI.akcoutliers
        window : int, optional
            Width of square matrix filter (Default: 5)
        connectivity : str, {'face', 'all'}, optional
            Specifies what kind of connected-component connectivity to
            use for median determination

        Examples
        --------
        dwi.akccorrect(akc_out), where dwi is the DWI class object
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
                      desc='AKC Correction',
                      bar_format='{desc}: [{percentage:0.0f}%]',
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
                    patchViol = np.delete(akc_out[
                        Ib:Ie, violIdx[1, j], violIdx[2, j]], d2move)
                    patchViol = np.hstack((patchViol, np.delete(akc_out[
                        violIdx[0, j], Jb:Je, violIdx[2, j]], d2move)))
                    patchViol = np.hstack((patchViol, np.delete(akc_out[
                        violIdx[0, j], violIdx[1, j], Kb:Ke], d2move)))
                    patchImg = np.delete(dt[
                        Ib:Ie, violIdx[1, j], violIdx[2, j], i], d2move)
                    patchImg = np.hstack((patchImg, np.delete(dt[
                        violIdx[0, j], Jb:Je, violIdx[2, j], i],
                                                              d2move)))
                    patchImg = np.hstack((patchImg, np.delete(dt[
                        violIdx[0, j], violIdx[1, j], Kb:Ke, i],
                                                              d2move)))
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

    def irlls(self, excludeb0=True, maxiter=25, convcrit=1e-3, mode='DKI',
              leverage=0.85, bounds=3):
        """
        This functions performs outlier detection and robust parameter
        estimation for diffusion MRI using the iterative reweigthed
        linear least squares (IRLLS) approach.

        Parameters
        ----------
        exludeb0 : bool, optional
            Exlude the b0 images when removing outliers (Default: True)
        maxiter : int, optional   Integer; default: 25
            Maximum number of iterations in the iterative reweighting
            loop (Default: 25)
        convcrit : float, optional
            Fraction of L2-norm of estimated diffusion parameter
            vector that the L2-norm of different vector should get in
            order to reach convergence in the iterative reweighted
            loop (Default: 1e-3)
        mode : str, {'DKI', 'DTI'}, optional
            Specifies whether to use DTI or DKI model (Default: 'DKI')
        leverage : float, optional
            Measurement ranging from 0 to 1 where a leverage above
            this threshold will not be excluded in estimation of DT
            after outlier (Default: 0.85)
        Bounds : int, optional
            Set the threshold of the number of standard deviation
            that are needed to exclude a measurement (Default: 3)

        Returns
        -------
        outliers : ndarray(dtype=bool)
            4D image same size as input DWI marking voxels
            that are outliers
        dt : ndarray(dtype=float)
            IRLLS method of DT estimation

        Examples
        --------
        outliers = dwi.irlls()
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
            bmat = np.hstack(
                (np.ones((ndwi, 1)),
                 np.matmul((-np.tile(b, (1, 6)) * g[:,dind[:,0]] * \
                            g[:,dind[:,1]]), np.diag(dcnt))))
        else:
            (wcnt, wind) = self.createTensorOrder(4)
            bmat = np.hstack(
                (np.ones((ndwi,1)),
                 np.matmul((-np.tile(b, (1, 6)) * g[:,dind[:,0]] * \
                            g[:,dind[:,1]]), np.diag(dcnt)),
                 (1/6)*np.matmul((np.square(np.tile(b, (1, 15))) * \
                                  g[:,wind[:,0]] * g[:,wind[:,1]] * \
                                  g[:,wind[:,2]] * g[:,wind[:,3]]),
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
        # Attempt basic noise estimation
        try:
            sigma
        except NameError:
            def estSigma(dwi, bmat):
                dwi = np.reshape(dwi, (len(dwi), 1))
                try:
                    dt_ = np.linalg.lstsq(bmat, np.log(dwi), rcond=None)[0]
                    # dt_ = np.linalg.solve(np.dot(bmat.T, bmat), np.dot(
                    #     bmat.T, np.log(dwi)))
                except:
                    dt_ = np.full((bmat.shape[1], 1), minZero)
                w = highprecisionexp(np.matmul(bmat, dt_)).reshape((ndwi, 1))
                try:
                    dt_ = np.linalg.lstsq((bmat * np.tile(w, (1, nparam))),
                                          (np.log(dwi) * w), rcond=None)[0]
                    # dt_ = np.linalg.solve(
                    #     np.dot((bmat * np.tile(w, (1, nparam))).T,
                    #         (bmat * np.tile(w, (1, nparam)))), \
                    #     np.dot((bmat * np.tile(w, (1, nparam))).T, (np.log(
                    #         dwi) * w)))
                except:
                    dt_ = np.full((bmat.shape[1], 1), minZero)
                e = np.log(dwi) - np.matmul(bmat, dt_)
                m = np.median(np.abs((e * w) - np.median(e * w)))
                try:
                    sigma_ = np.sqrt(ndwi / ndof) * 1.4826 * m
                except:
                    sigma_ = minZero
                return sigma_
            sigma_ = np.zeros((nvox,1))
            inputs = tqdm(range(nvox),
                          desc='IRLLS Noise Estimation',
                          bar_format='{desc}: [{percentage:0.0f}%]',
                          unit='vox',
                          ncols=tqdmWidth)
            sigma_ = Parallel(n_jobs=self.workers, prefer='processes') \
                (delayed(estSigma)(dwi[:, i], bmat) for i in inputs)
            sigma = np.median(sigma_)
            sigma = np.tile(sigma,(nvox,1))
        if scaling:
            sigma = sigma*1000/sc
        def outlierHelper(dwi, bmat, sigma, b, b0_pos, maxiter=25,
                          convcrit=1e-3, leverage=3, bounds=3):
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
            try:
                dt_i = np.linalg.lstsq(bmat_i, np.log(dwi_i), rcond=None)[0]
                # dt_i = np.linalg.solve(np.dot(bmat_i.T, bmat_i),
                #                        np.dot(bmat_i.T, np.log(dwi_i)))
            except:
                dt_i = np.full((bmat_i.shape[1], 1), minZero)
            w = highprecisionexp(np.matmul(bmat_i, dt_i))
            try:
                dt_i = np.linalg.lstsq((bmat_i * np.tile(w, (1, nparam))),
                                       (np.log(dwi_i).reshape(
                                           (dwi_i.shape[0], 1)) * w),
                                       rcond=None)[0]
                # dt_i = np.linalg.solve(
                #     np.dot((bmat_i * np.tile(w, (1, nparam))).T,
                #            (bmat_i * np.tile(w, (1, nparam)))),
                #     np.dot((bmat_i * np.tile(w, (1, nparam))).T,
                #            (np.log(dwi_i).reshape(
                #                (dwi_i.shape[0], 1)) * w)))
            except:
                dt_i = np.full((bmat_i.shape[1], 1), minZero)
            dwi_hat = highprecisionexp(np.matmul(bmat_i, dt_i))
            # Goodness-of-fit
            residu = np.log(dwi_i.reshape((dwi_i.shape[0],1))) - \
                     np.log(dwi_hat)
            residu_ = dwi_i.reshape((dwi_i.shape[0],1)) - dwi_hat
            
            try:
                chi2 = np.sum((residu_ * residu_) /\
                              np.square(sigma)) / (ndof_i) -1
            except:
                chi2 = minZero
            try:
                gof = np.abs(chi2) < 3 * np.sqrt(2/ndof_i)
            except:
                gof = True  # If ndof_i = 0, right inequality becomes inf
                # and makes the logic True
            gof2 = gof
            # Iterative reweighning procedure
            iter = 0
            np.seterr(divide='raise', invalid='raise')
            while (not gof) and (iter < maxiter):
                try:
                    C = np.sqrt(n_i/(n_i-nparam)) * \
                        1.4826 * \
                        np.median(np.abs(residu_ - \
                                         np.median(residu_))) / dwi_hat
                except:
                    C = np.full(dwi_hat.shape, minZero)
                try:
                    GMM = np.square(C) / np.square(np.square(residu) + \
                                                   np.square(C))
                except:
                    # The following line produces a lot of Intel MKL
                    # warnings that should be ignored. This is a known
                    # Intel and Numpy bug that has not yet been resolved.
                    GMM = np.full(C.shape, minZero)
                w = np.sqrt(GMM) * dwi_hat
                dt_imin1 = dt_i
                try:
                    dt_i = np.linalg.lstsq(
                        (bmat_i * np.tile(w, (1, nparam))),
                        (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w),
                                           rcond=None)[0]
                    # dt_i = np.linalg.solve(
                    #     np.dot((bmat_i * np.tile(w, (1, nparam))).T,
                    #            (bmat_i * np.tile(w, (1, nparam)))),
                    #     np.dot((bmat_i * np.tile(w, (1, nparam))).T,
                    #            (np.log(dwi_i).reshape(
                    #                (dwi_i.shape[0], 1)) * w)))
                except:
                    dt_i = np.full((bmat_i.shape[1], 1), minZero)
                dwi_hat = highprecisionexp(np.matmul(bmat_i, dt_i))
                dwi_hat[dwi_hat < 1] = 1
                residu = np.log(
                    dwi_i.reshape((dwi_i.shape[0],1))) - np.log(dwi_hat)
                residu_ = dwi_i.reshape((dwi_i.shape[0], 1)) - dwi_hat
                # Convergence check
                iter = iter + 1
                gof = np.linalg.norm(
                    dt_i - dt_imin1) < np.linalg.norm(dt_i) * convcrit
                conv = iter
            np.seterr(**defaultErrorState)
            # Outlier detection
            if ~gof2:
                try:
                    lev = np.diag(np.matmul(bmat_i, np.linalg.lstsq(np.matmul(np.transpose(bmat_i),
                                                            np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i)),
                                          np.matmul(np.transpose(bmat_i), np.diag(np.square(w.reshape(-1)))), rcond=None)[0]))
                    # lev_helper = np.linalg.solve(\
                    #                 np.dot((np.matmul(np.transpose(bmat_i), np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i))).T, \
                    #                         (np.matmul(np.transpose(bmat_i), np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i)))), \
                    #                 np.dot((np.matmul(np.transpose(bmat_i), np.matmul(np.diag(np.square(w).reshape(-1)), bmat_i))).T, \
                    #                         np.matmul(np.transpose(bmat_i), np.diag(np.square(w.reshape(-1))))))
                    # lev = np.diag(np.matmul(bmat_i, lev_helper))
                except:
                    lev = np.full(residu.shape, minZero)
                lev = lev.reshape((lev.shape[0], 1))
                try:
                    lowerbound_linear = -bounds * \
                                        np.lib.scimath.sqrt(1-lev) * \
                                        sigma / dwi_hat
                except:
                    lowerbound_linear = np.full(lev.shape, minZero)
                try:
                    upperbound_nonlinear = bounds * \
                                           np.lib.scimath.sqrt(1 -lev) * sigma
                except:
                    upperbound_nonlinear = np.full(lev.shape, minZero)
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
            try:
                dt_ = np.linalg.lstsq(bmat_i, np.log(dwi_i), rcond=None)[0]
                # dt_ = np.linalg.solve(np.dot(bmat_i.T, bmat_i), \
                #                     np.dot(bmat_i.T, np.log(dwi_i)))
            except:
                dt_ = np.full((bmat_i.shape[1], 1), minZero)
            w = highprecisionexp(np.matmul(bmat_i, dt_))
            try:
                dt = np.linalg.lstsq((bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam))), (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w.reshape((len(w),1))),
                                           rcond=None)[0]
                # dt = np.linalg.solve(\
                #     np.dot((bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam))).T, (bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam)))), \
                #     np.dot((bmat_i * np.tile(w.reshape((len(w),1)), (1, nparam))).T, \
                #         (np.log(dwi_i).reshape((dwi_i.shape[0], 1)) * w.reshape((len(w),1)))))
            except:
                dt = np.full((bmat_i.shape[1], 1), minZero)
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
                          desc='IRLLS Outlier Detection',
                          bar_format='{desc}: [{percentage:0.0f}%]',
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
        return reject, dt.T

    def tensorReorder(self, dwiType):
        """
        Reorders tensors in DT to those of MRTRIX in accordance to
        the table below

        Parameters
        ----------
        dwiType : str, {'dti', 'dki'}
            Indicates whether image is DTI or DKI

        Returns
        -------
        DT : ndarray(dtype=float)
            4D image containing DT tensor
        KT : ndarray(dtype=float)
            4D image containing KT tensor

        Examples
        --------
        dt = dwi.tensorReorder()

        Notes
        -----
        MRTRIX3 and Designer tensors are described below.

        .. code-block:: none
        
            MRTRIX3 Tensors                     DESIGNER Tensors
            ~~~~~~~~~~~~~~~                     ~~~~~~~~~~~~~~~~

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
            ~~~~~~~~~~~~~~~~

            MRTRIX3         DESIGNER
            ~~~~~~~         ~~~~~~~~
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
        """
        Computes 3D violation mask of outliers detected from IRLLS
        method

        Parameters
        ----------
        reject : ndarray(dtype=bool)
            4D input outlier map from IRLLS

        Returns
        -------
        propviol : ndarray(dtype=float)
            3D mask where voxel value is the percentage of directional
            violations

        Examples
        --------
        mask = dwi.irllsviolmask(outliers)
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

def vectorize(img, mask):
    """
    Returns vectorized image based on brain mask, requires no input
    parameters
    If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
    If the input is 3D or 4D, vectorize it using a mask
    Classification: Function

    Parameters
    ----------
    img : ndarray
        1D, 2D, 3D or 4D image array to vectorize
    mask : ndarray
        3D image array for masking

    Returns
    -------
    vec : N X number_of_voxels vector or array, where N is the number
        of DWI volumes

    Examples
    --------
    vec = vectorize(img) if there's no mask
    vec = vectorize(img, mask) if there's a mask
    """
    if mask is None:
        mask = np.ones((img.shape[0],
                        img.shape[1],
                        img.shape[2]),
                       order='F')
    mask = mask.astype(bool)
    if img.ndim == 1:
        n = img.shape[0]
        s = np.zeros((mask.shape[0],
                      mask.shape[1],
                      mask.shape[2]),
                     order='F')
        s[mask] = img
    if img.ndim == 2:
        n = img.shape[0]
        s = np.zeros((mask.shape[0],
                      mask.shape[1],
                      mask.shape[2], n),
                     order='F')
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
    """
    Write clipped NifTi images

    Parameters
    ----------
    map : ndarray(dtype=float)
        3D array to write
    header : class
        Nibabel class header file containing NifTi properties
    outDir : str
        Output file name
    range : array_like
        [1 x 2] vector specifying range to clip, inclusive of value
        in range, e.g. range = [0, 1] for FA map

    Returns
    -------
    None; writes out file

    Examples
    --------
    writeNii(matrix, header, output_directory, [0, 2])

    See Also
    clipImage(img, range) : this function is wrapped around
    """
    if range == None:
        clipped_img = nib.Nifti1Image(map, hdr.affine, hdr.header)
    else:
        clipped_img = clipImage(map, range)
        clipped_img = nib.Nifti1Image(clipped_img, hdr.affine, hdr.header)
    nib.save(clipped_img, outDir)

def clipImage(img, range):
    """
    Clips input matrix within desired range. Min and max values are
    inclusive of range
    Classification: Function

    Parameters
    ----------
    img : ndarray(dtype=float)
        Input 3D image array
    range : array_like
        [1 x 2] vector specifying range to clip

    Returns
    -------
    clippedImage:   clipped image; same size as img

    Examples
    --------
    clippedImage = clipImage(image, [0 3])
    Clips input matrix in the range 0 to 3
    """
    img[img > range[1]] = range[1]
    img[img < range[0]] = range[0]
    return img

def highprecisionexp(array, maxp=1e32):
    """
    Prevents overflow warning with numpy.exp by assigning overflows
    to a maxumum precision value
    Classification: Function

    Parameters
    ----------
    array : ndarray
        Array or scalar of number to run np.exp on
    maxp : float, optional
        Maximum preicison to assign if overflow (Default: 1e32)

    Returns
    -------
    exponent or max-precision

    Examples
    --------
    a = highprecisionexp(array)
    """
    np.seterr(all='ignore')
    defaultErrorState = np.geterr()
    np.seterr(over='raise', invalid='raise')
    try:
        ans = np.exp(array)
    except:
        ans = np.full(array.shape, maxp)
    np.seterr(**defaultErrorState)
    return ans
