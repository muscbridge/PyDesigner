#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Function for computing DTI and DKI spherical harmonics from diffusion and
kurtosis tensors
"""
import warnings
import multiprocessing
import os.path as op
from joblib import Parallel, delayed
import numpy as np
import nibabel as nib
from scipy.special import sph_harm
from dipy.core.geometry import sphere2cart
from designer.fitting.dwipy import vectorize, writeNii
from designer.fitting import thresholds as th
from designer.tractography import sphericalsampling
from tqdm import tqdm

class odfmodel():
    """
    DTI/DKI tractograpy class for computing ODFs and preparing spherical
    harmonics for DTI or DKI fiber tracking.
    """
    def __init__(self, dt, kt=None, mask=None, scale=None, res='med', l_max=6,
        radial_weight=4, nthreads=None):
        """
        Parameters
        ----------
        dt : str
            Path to diffusion tensor, which is a 4D NifTI volume composed of six (6)
            components    
        kt : str; optional
            Path to kurtosis tensor, which is a 4D NifTI compose of fifteen (15)
            components
            (Default: None)
        mask :  str; optional
            Path to brain mask in NifTI format
        scale : str; optional
            Path to dMRI metric map to use for ODF scaling, where metric value
            at a voxel is multiplied by the ODF.
        res : str; optional, {'low', 'med', 'high'}
            Resolution of directions for ODF calculation. Higher resolution
            implies slower computation.
            (Default: 'med')
        l_max : int
            Maximum spherical harmonic degree to use for spherical harmonic
            expansion of ODF
            (Default: 6)
        radial_weight : float
            Radial weighting power for detecting directional differences
            (Default: 4)
        """
        if not op.exists(dt):
            raise OSError('Input DT path does not exist. Please ensure that '
                        'the folder or file specified exists.')
        if not kt is None:
            if not op.exists(kt):
                raise OSError('Input KT path does not exist. Please ensure that '
                            'the folder or file specified exists.')
        if not mask is None:
            if not op.exists(mask):
                raise OSError('Path to brain mask does not exist. Please '
                'ensure that the file specified exists.')
        if not scale is None:
            if not op.exists(scale):
                raise OSError('Path to scale image does not exist. Please '
                'ensure that the file specified exists.')
        if not isinstance(res, str):
            raise Exception('Please specify resolution as a string. Possible '
            'choices are "low", "med", or "high"')
        # Load images
        self.hdr = nib.load(dt)
        self.DT = self.hdr.get_fdata()
        if not kt is None:
            self.KT = nib.load(kt).get_fdata()
        else:
            self.KT = None
        if not mask is None:
            self.mask_img = nib.load(mask).get_fdata()
        else:
            self.mask_img = None
        if not scale is None:
            self.scale_img = nib.load(scale).get_fdata()
        else:
            self.scale_img = np.ones(self.DT.shape[0:3])
        if l_max % 2 != 0:
            raise Exception('Please provide l_max as a postive '
        'and even integer')
        self.l_max = l_max
        if radial_weight is None:
            warnings.warn('Radial weight for dODF computation not specified. '
            'Using default value of 4.')
            self.radial_weight = 4
        else:
            self.radial_weight=radial_weight
        self.vertices, self.idx, self.idx8, self.area, self.faces, \
            self.separation_angle = sphericalsampling.odfgrid(res)
        if not nthreads is None:
            if nthreads > multiprocessing.cpu_count():
                warnings.warn('Number of workers/threads specified exceed more '
                'than available. Using the maximum workers/threads available.')
                self.workers = -1
        if nthreads is None:
            self.workers = -1
        else:
            self.workers = nthreads

    def dkiodfhelper(self, dt, kt, radial_weight=4, fa_t=None, form='spherical'):
        """
        Computes DKI fODF coefficient at a voxel. This function is intended to
        parallelize computations across the brain.

        Parameters
        ----------
        dt : array_like(dtype=float)
            Diffusion tensor containing 6 elements
        kt: array_like(dtype=float)
            Kurtosis tensor containing 15 elements
        radial_weighing : float; optional
            Radial weighting power for detecting directional differences (Default: 4)
        fa_t : float64
            In rare cases the diffusion tensor may be extremely isotropic with
            very small eigenvalues, causing the kurtosis dODF to have erratic
            behavior with very large values, as the kurtosis dODF evaluates the
            inverse of D. Setting a threshold removes negative eigenvalues while
            preserving principal orientation in voxels where FA >= threshold 
        form : str; optional; {'spherical', 'cartesian', 'coefficient'}
            Form of ODF to return in
            (Default: 'spherical')

        Returns
        -------
        odf : array_like(dtype=float)
            DKI ODFs in either coefficient, spherical, or cartesian form
        """
        D = np.array(
            [
                [dt[0], dt[3], dt[4]],
                [dt[3], dt[1], dt[5]],
                [dt[4], dt[5], dt[2]]
            ]
        )

        W = np.zeros((3,3,3,3))
        W[0,0,0,0] = kt[0]
        W[1,1,1,1] = kt[1]
        W[2,2,2,2] = kt[2]
        W[0,0,0,1] = kt[3];  W[0,0,1,0] = W[0,0,0,1]; W[0,1,0,0] = W[0,0,0,1]; W[1,0,0,0] = W[0,0,0,1]
        W[0,0,0,2] = kt[4];  W[0,0,2,0] = W[0,0,0,2]; W[0,2,0,0] = W[0,0,0,2]; W[2,0,0,0] = W[0,0,0,2]
        W[0,1,1,1] = kt[5];  W[1,0,1,1] = W[0,1,1,1]; W[1,1,0,1] = W[0,1,1,1]; W[1,1,1,0] = W[0,1,1,1]
        W[0,2,2,2] = kt[6];  W[2,0,2,2] = W[0,2,2,2]; W[2,2,0,2] = W[0,2,2,2]; W[2,2,2,0] = W[0,2,2,2]
        W[1,1,1,2] = kt[7];  W[1,1,2,1] = W[1,1,1,2]; W[1,2,1,1] = W[1,1,1,2]; W[2,1,1,1] = W[1,1,1,2]
        W[1,2,2,2] = kt[8];  W[2,1,2,2] = W[1,2,2,2]; W[2,2,1,2] = W[1,2,2,2]; W[2,2,2,1] = W[1,2,2,2]
        W[0,0,1,1] = kt[9];  W[0,1,0,1] = W[0,0,1,1]; W[0,1,1,0] = W[0,0,1,1]; W[1,0,0,1] = W[0,0,1,1]; W[1,0,1,0] = W[0,0,1,1]; W[1,1,0,0] = W[0,0,1,1]
        W[0,0,2,2] = kt[10]; W[0,2,0,2] = W[0,0,2,2]; W[0,2,2,0] = W[0,0,2,2]; W[2,0,0,2] = W[0,0,2,2]; W[2,0,2,0] = W[0,0,2,2]; W[2,2,0,0] = W[0,0,2,2]
        W[1,1,2,2] = kt[11]; W[1,2,1,2] = W[1,1,2,2]; W[1,2,2,1] = W[1,1,2,2]; W[2,1,1,2] = W[1,1,2,2]; W[2,1,2,1] = W[1,1,2,2]; W[2,2,1,1] = W[1,1,2,2]
        W[0,0,1,2] = kt[12]; W[0,0,2,1] = W[0,0,1,2]; W[0,1,0,2] = W[0,0,1,2]; W[0,1,2,0] = W[0,0,1,2]; W[0,2,0,1] = W[0,0,1,2]; W[0,2,1,0] = W[0,0,1,2]; W[1,0,0,2] = W[0,0,1,2]; W[1,0,2,0] = W[0,0,1,2]; W[1,2,0,0] = W[0,0,1,2]; W[2,0,0,1] = W[0,0,1,2]; W[2,0,1,0] = W[0,0,1,2]; W[2,1,0,0] = W[0,0,1,2]
        W[0,1,1,2] = kt[13]; W[0,1,2,1] = W[0,1,1,2]; W[0,2,1,1] = W[0,1,1,2]; W[1,0,1,2] = W[0,1,1,2]; W[1,0,2,1] = W[0,1,1,2]; W[1,1,0,2] = W[0,1,1,2]; W[1,1,2,0] = W[0,1,1,2]; W[1,2,0,1] = W[0,1,1,2]; W[1,2,1,0] = W[0,1,1,2]; W[2,0,1,1] = W[0,1,1,2]; W[2,1,0,1] = W[0,1,1,2]; W[2,1,1,0] = W[0,1,1,2]
        W[0,1,2,2] = kt[14]; W[0,2,1,2] = W[0,1,2,2]; W[0,2,2,1] = W[0,1,2,2]; W[1,0,2,2] = W[0,1,2,2]; W[1,2,0,2] = W[0,1,2,2]; W[1,2,2,0] = W[0,1,2,2]; W[2,0,1,2] = W[0,1,2,2]; W[2,0,2,1] = W[0,1,2,2]; W[2,1,0,2] = W[0,1,2,2]; W[2,1,2,0] = W[0,1,2,2]; W[2,2,0,1] = W[0,1,2,2]; W[2,2,1,0] = W[0,1,2,2]

        # Reglarize tensor if fa is more than threshold specified (fa_t)
        if not fa_t is None:
            L, V = np.linalg.eig(D)
            L[L < th.__minZero__] = th.__minZero__
            idx = np.argsort(L)[::-1]
            L = L[idx]
            V = V[:, idx]
            fa = np.sqrt(
                (
                    (L[0] - L[1])**2 + \
                    (L[0] - L[2])**2 + \
                    (L[1] - L[2])**2 \
                ) / (2 * (np.sum(L**2)))
            )
            if fa > fa_t:
                x = np.roots([2*(1-2*fa_t**2)/3, -4*L[0]/3, 2*(1-fa_t**2)/3*L[0]**2])
                if x[np.logical_and(x > 0, x < L[0])].size != 0:
                    L[1:3] = x[np.logical_and(x > 0, x < L[0])]
                else:
                    Davg = np.trace(D)/3
                    L[L < 0.1 * Davg] = 0.1 * Davg
                D = np.matmul(np.matmul(V, np.diag(L)), np.linalg.inv(V))
                W = np.zeros((3,3,3,3))

        Davg = np.trace(D)/3
        try:
            U = Davg * np.linalg.inv(D)
        except:
            U = Davg * np.linalg.pinv(D)
        A1=0
        B11=0
        B12=0
        B13=0
        B22=0
        B23=0
        B33=0
        C1111=0
        C1112=0
        C1113=0
        C1122=0
        C1123=0
        C1133=0
        C1222=0
        C1223=0
        C1233=0
        C1333=0
        C2222=0
        C2223=0
        C2233=0
        C2333=0
        C3333=0

        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                for k in [0, 1, 2]:
                    for l in [0, 1, 2]:
                        # Coefficients for: 3UijWijklUkl
                        A1 = A1 + 3 * U[i,j] * W[i,j,k,l] * U[k,l]
                        # Coefficients for: -6(a+1)UijWijklVkl
                        B0 = -6*(radial_weight+1) * U[i,j] * W[i,j,k,l]
                        B11 = B11 + B0 * (U[k,0] * U[l,0])              
                        B12 = B12 + B0 * (U[k,0] * U[l,1] + U[k,1] * U[l,0])
                        B13 = B13 + B0 * (U[k,0] * U[l,2] + U[k,2] * U[l,0])
                        B22 = B22 + B0 * (U[k,1] * U[l,1])              
                        B23 = B23 + B0 * (U[k,1] * U[l,2] + U[k,2] * U[l,1])
                        B33 = B33 + B0 * (U[k,2] * U[l,2])
                        # Coefficients for: (alpha+1)(alpha+3)W(i,j,k,l)VijVkl
                        C0 = (radial_weight+1) * (radial_weight+3) * W[i,j,k,l]
                        C1111 = C1111 + C0 * (U[i,0] * U[j,0] * U[k,0] * U[l,0])                                                                                                                                                                                                                                                                                                                    
                        C1112 = C1112 + C0 * (U[i,0] * U[j,0] * U[k,0] * U[l,1] + U[i,0] * U[j,0] * U[k,1] * U[l,0] + U[i,0] * U[j,1] * U[k,0] * U[l,0] + U[i,1] * U[j,0] * U[k,0] * U[l,0])                                                                                                                                                                                                                                
                        C1113 = C1113 + C0 * (U[i,0] * U[j,0] * U[k,0] * U[l,2] + U[i,0] * U[j,0] * U[k,2] * U[l,0] + U[i,0] * U[j,2] * U[k,0] * U[l,0] + U[i,2] * U[j,0] * U[k,0] * U[l,0])                                                                                                                                                                                                                                
                        C1122 = C1122 + C0 * (U[i,0] * U[j,0] * U[k,1] * U[l,1] + U[i,0] * U[j,1] * U[k,0] * U[l,1] + U[i,0] * U[j,1] * U[k,1] * U[l,0] + U[i,1] * U[j,0] * U[k,0] * U[l,1] + U[i,1] * U[j,0] * U[k,1] * U[l,0] + U[i,1] * U[j,1] * U[k,0] * U[l,0])                                                                                                                                                                        
                        C1123 = C1123 + C0 * (U[i,0] * U[j,0] * U[k,1] * U[l,2] + U[i,0] * U[j,0] * U[k,2] * U[l,1] + U[i,0] * U[j,1] * U[k,0] * U[l,2] + U[i,0] * U[j,1] * U[k,2] * U[l,0] + U[i,0] * U[j,2] * U[k,0] * U[l,1] + U[i,0] * U[j,2] * U[k,1] * U[l,0] + U[i,1] * U[j,0] * U[k,0] * U[l,2] + U[i,1] * U[j,0] * U[k,2] * U[l,0] + U[i,1] * U[j,2] * U[k,0] * U[l,0] + U[i,2] * U[j,0] * U[k,0] * U[l,1] + U[i,2] * U[j,0] * U[k,1] * U[l,0] + U[i,2] * U[j,1] * U[k,0] * U[l,0])
                        C1133 = C1133 + C0 * (U[i,0] * U[j,0] * U[k,2] * U[l,2] + U[i,0] * U[j,2] * U[k,0] * U[l,2] + U[i,0] * U[j,2] * U[k,2] * U[l,0] + U[i,2] * U[j,0] * U[k,0] * U[l,2] + U[i,2] * U[j,0] * U[k,2] * U[l,0] + U[i,2] * U[j,2] * U[k,0] * U[l,0])                                                                                                                                                                        
                        C1222 = C1222 + C0 * (U[i,0] * U[j,1] * U[k,1] * U[l,1] + U[i,1] * U[j,0] * U[k,1] * U[l,1] + U[i,1] * U[j,1] * U[k,0] * U[l,1] + U[i,1] * U[j,1] * U[k,1] * U[l,0])                                                                                                                                                                                                                                
                        C1223 = C1223 + C0 * (U[i,0] * U[j,1] * U[k,1] * U[l,2] + U[i,0] * U[j,1] * U[k,2] * U[l,1] + U[i,0] * U[j,2] * U[k,1] * U[l,1] + U[i,1] * U[j,0] * U[k,1] * U[l,2] + U[i,1] * U[j,0] * U[k,2] * U[l,1] + U[i,1] * U[j,1] * U[k,0] * U[l,2] + U[i,1] * U[j,1] * U[k,2] * U[l,0] + U[i,1] * U[j,2] * U[k,0] * U[l,1] + U[i,1] * U[j,2] * U[k,1] * U[l,0] + U[i,2] * U[j,0] * U[k,1] * U[l,1] + U[i,2] * U[j,1] * U[k,0] * U[l,1] + U[i,2] * U[j,1] * U[k,1] * U[l,0])
                        C1233 = C1233 + C0 * (U[i,0] * U[j,1] * U[k,2] * U[l,2] + U[i,0] * U[j,2] * U[k,1] * U[l,2] + U[i,0] * U[j,2] * U[k,2] * U[l,1] + U[i,1] * U[j,0] * U[k,2] * U[l,2] + U[i,1] * U[j,2] * U[k,0] * U[l,2] + U[i,1] * U[j,2] * U[k,2] * U[l,0] + U[i,2] * U[j,0] * U[k,1] * U[l,2] + U[i,2] * U[j,0] * U[k,2] * U[l,1] + U[i,2] * U[j,1] * U[k,0] * U[l,2] + U[i,2] * U[j,1] * U[k,2] * U[l,0] + U[i,2] * U[j,2] * U[k,0] * U[l,1] + U[i,2] * U[j,2] * U[k,1] * U[l,0])
                        C1333 = C1333 + C0 * (U[i,0] * U[j,2] * U[k,2] * U[l,2] + U[i,2] * U[j,0] * U[k,2] * U[l,2] + U[i,2] * U[j,2] * U[k,0] * U[l,2] + U[i,2] * U[j,2] * U[k,2] * U[l,0])                                                                                                                                                                                                                                
                        C2222 = C2222 + C0 * (U[i,1] * U[j,1] * U[k,1] * U[l,1])                                                                                                                                                                                                                                                                                                                    
                        C2223 = C2223 + C0 * (U[i,1] * U[j,1] * U[k,1] * U[l,2] + U[i,1] * U[j,1] * U[k,2] * U[l,1] + U[i,1] * U[j,2] * U[k,1] * U[l,1] + U[i,2] * U[j,1] * U[k,1] * U[l,1])                                                                                                                                                                                                                                
                        C2233 = C2233 + C0 * (U[i,1] * U[j,1] * U[k,2] * U[l,2] + U[i,1] * U[j,2] * U[k,1] * U[l,2] + U[i,1] * U[j,2] * U[k,2] * U[l,1] + U[i,2] * U[j,1] * U[k,1] * U[l,2] + U[i,2] * U[j,1] * U[k,2] * U[l,1] + U[i,2] * U[j,2] * U[k,1] * U[l,1])                                                                                                                                                                        
                        C2333 = C2333 + C0 * (U[i,1] * U[j,2] * U[k,2] * U[l,2] + U[i,2] * U[j,1] * U[k,2] * U[l,2] + U[i,2] * U[j,2] * U[k,1] * U[l,2] + U[i,2] * U[j,2] * U[k,2] * U[l,1])                                                                                                                                                                                                                                
                        C3333 = C3333 + C0 * (U[i,2] * U[j,2] * U[k,2] * U[l,2])
        coeff = np.array(
            [
                A1, B11, B12, B13, B22, B23, B33, C1111, C1112, C1113, C1122, C1123, C1133, C1222, C1223, C1233, C1333, C2222, C2223, C2233, C2333, C3333, U[0,0], U[1,1], U[2,2], U[0,1], U[0,2], U[1,2], radial_weight
            ]
        )
        if form == 'coefficient':
            odf = coeff
        if form == 'spherical':
            odf = dkiodfspherical(coeff, self.vertices[:, 0], self.vertices[:, 1])
        if form == 'cartesian':
            x, y, z = sphere2cart(1, self.vertices[:, 1], self.vertices[:, 0])
            odf = dkiodfcartesian(coeff, x, y, z)
        return odf
    
    def dkiodf(self, form='spherical', fa_t=0.90):
        """
        Computes DKI ODFs for the whole brain.

        Parameters
        ----------
        form : str; optional; {'spherical', 'cartesial', 'coefficient'}
            Form of ODF to return in
            (Default: 'spherical')
        fa_t : float64; optional
            In rare cases the diffusion tensor may be extremely isotropic with
            very small eigenvalues, causing the kurtosis dODF to have erratic
            behavior with very large values, as the kurtosis dODF evaluates the
            inverse of D. Setting a threshold removes negative eigenvalues while
            preserving principal orientation in voxels where FA >= threshold
            (Default: 0.95)
        Returns
        -------
        DKI ODF in defined form
        """
        if not form in ['spherical', 'cartesian', 'coefficient']:
            raise Exception('Please select a valid form of ODF to receive')
        if self.KT is None:
                raise AttributeError('WOAH! Cannot compute DKI ODFs without '
                'kurtosis tensor (KT). Try using dtiodf(), Jumbo.')
        # Vectorize images
        DT = vectorize(self.DT, self.mask_img)
        KT = vectorize(self.KT, self.mask_img)
        nvox = DT.shape[-1]
        inputs = tqdm(range(nvox),
                        desc='DKI fODFs',
                        bar_format='{desc}: [{percentage:0.0f}%]',
                        unit='vox',
                        ncols=70)
        odf = Parallel(n_jobs=self.workers, prefer='processes') (delayed(self.dkiodfhelper)\
            (DT[:, i], KT[:, i], self.radial_weight, fa_t, form) for i in inputs)
        odf = np.array(odf).T
        odf = vectorize(odf, self.mask_img)
        return(odf)

    def dtiodfhelper(self, dt, form='spherical'):
        """
        Computes DTI fODF coefficient at a voxel. This function is intended to
        parallelize computations across the brain. Use only for diffusion
        ellipsoids.

        Parameters
        ----------
        dt : array_like(dtype=float)
            Diffusion tensor containing 6 elements
        radial_weighing : float; optional
            Radial weighting power for detecting directional differences (Default: 4)
        form : str; optional; {'spherical', 'coefficient'}
            Form of ODF to return in
            (Default: 'spherical')

        Returns
        -------
        odf : array_like(dtype=float)
            DKI ODFs in either coefficient, spherical, or cartesian form
        """
        D = np.array(
            [
                [dt[0], dt[3], dt[4]],
                [dt[3], dt[1], dt[5]],
                [dt[4], dt[5], dt[2]]
            ]
        )
        Davg = np.trace(D)/3
        try:
            U = Davg * np.linalg.inv(D)
        except:
            U = Davg * np.linalg.pinv(D)
        U11 = U[0,0]
        U22 = U[1,1]
        U33 = U[2,2]
        U12 = U[0,1]
        U13 = U[0,2]
        U23 = U[1,2]
        coeff = np.array(
            [U11, U12, U13, U22, U23, U33]
        )
        if form == 'coefficient':
            odf = coeff
        if form == 'spherical':
            odf = dtiodfspherical(coeff, self.vertices[:, 0], self.vertices[:, 1], self.radial_weight)
        return odf

    def dtiodf(self, form='spherical'):
        """
        Computed DTI ODFs for the whole brain (ellipsoids)

        Parameters
        ----------
        form : str; optional; {'spherical', 'cartesian', 'coefficient'}
            Form of ODF to return in
            (Default: 'spherical')
        Returns
        -------
        DTI ODF in defined form
        """
        if self.DT is None:
                raise AttributeError('WOAH! Cannot compute DTI ODFs without '
                'diffusion tensor (DT), Jumbo.')
        # Vectorize images
        DT = vectorize(self.DT, self.mask_img)
        nvox = DT.shape[-1]
        inputs = tqdm(range(nvox),
                        desc='DTI ODF',
                        bar_format='{desc}: [{percentage:0.0f}%]',
                        unit='vox',
                        ncols=70)
        odf = Parallel(n_jobs=self.workers, prefer='processes') (delayed(self.dtiodfhelper)\
            (DT[:, i], form) for i in inputs)
        odf = np.array(odf).T
        odf = vectorize(odf, self.mask_img)
        return(odf)

    def odfmaxhelper(self, odf):
        """
        Find local maxima of ODF over spherical grid at voxel

        Parameters
        ----------
        odf : array_like(dtype=float64)
            Spherical ODF values at a voxel

        Returns
        -------
        odfmax : array_like(dtype=float64)
            Local maxima of ODF over spherical grid in descending order
        dirmax : array_like(dtype=float64)
            Corresponding direction vector where local ODF maxima occur
        """
        maxidx = self.idx[odf[self.idx[:, 1]] == np.amax(odf[self.idx], axis=1), 0]
        odf_max = odf[maxidx]
        dir_max = self.vertices[maxidx, :]
        # Sort by magnitude in descending order
        idx = np.argsort(odf_max)[::-1]
        odfmax = odf_max[idx]
        dirmax = dir_max[idx]
        if odfmax.size == 0:
            odfmax = np.array([1])
            dirmax = np.array([0, 0])
        return odfmax, dirmax 

    def odf2shhelper(self, odf, B, scale):
        """
        Helper function to parallelize computation spherical harmonic expansion
        at a voxel.

        Parameters
        ----------
        odf : array_like(dtype=float64)
            Spherical ODF values at a voxel
        B : array_like(dtype=complex)
            Spherical harmonic basis set to compute expansion
        scale : float64
            Value of dMRI metric to multiply ODF with to control stopping
            criteria in tractography
        
        Returns
        -------
        sh : Shpherical harmonic expansion of ODF at voxel
        """
        odfmax, dirmax = self.odfmaxhelper(odf)
        odfmax = odfmax[0]
        sh = np.dot(np.linalg.pinv(B), odf / odfmax) * scale
        sh[np.isnan(sh)] = 0
        sh[np.isinf(sh)] = 0
        return sh

    def odf2sh(self, odf):
        """
        Converts whole-brain ODFs to spherical harmonics sampled at direction
        set specified by resolution. Only the real portion is returned.

        Parameters
        ----------
        odf : 4D ODF file containing spherical ODFs

        Returns
        -------
        sh : array_like(dtype=float64)
            Shperical harmonic expansion of ODF
        """
        odf = vectorize(odf, self.mask_img)
        scale = vectorize(self.scale_img, self.mask_img)
        # Create shperical harmonic (SH) base set
        degs = np.arange(self.l_max + 1, dtype=int)
        l_num = 2 * degs[::2] + 1 # how many per degree (evens only)
        harmonics = []
        sh_end = 1 # initialize the SH set for indexing
        for h in range(0,len(degs[::2])):
            sh_start = sh_end + l_num[h] - 1
            sh_end = sh_start + l_num[h] - 1
            harmonics.extend(np.arange(sh_start - 1, sh_end))
        # MRtrix does not have (-1)^m in formulas so we index where this would
        # occur and multiply those volumes by -1
        sh_idx = []
        sh_start = 2
        for h in range(1,len(degs[::2])):
            sh_end = sh_start + l_num[h] - 2
            sh_idx.extend(np.arange(sh_start, sh_end, 2))
            sh_start = sh_end + 2
        B = shbasis(degs, self.vertices[:, 0], self.vertices[:, 1])
        B = B[:, harmonics]
        nvox = odf.shape[-1]
        inputs = tqdm(range(nvox),
                        desc='ODF SH Expansion',
                        bar_format='{desc}: [{percentage:0.0f}%]',
                        unit='vox',
                        ncols=70)
        sh = Parallel(n_jobs=self.workers, prefer='processes') (delayed(self.odf2shhelper)\
            (odf[:, i], B, scale[i]) for i in inputs)
        sh = np.array(sh).T.real
        sh[sh_idx,:] = -sh[sh_idx,:]
        sh = vectorize(sh, self.mask_img)
        return sh

    def savenii(self, var, path):
        """
        Write out NifTI output of associated spherical harmonic file

        Parameters
        ----------
        var : array_like
            variable to write out
        path : str
            Path to output file
        
        Returns
        -------
        None; writes out file
        """
        writeNii(var, self.hdr, path)

def dkiodfspherical(odf, phi, theta):
    """
    Convert DKI ODFs coefficients at voxel to spherical form.

    Parameters
    ----------
    odf : array_like(dtype=float64)
        ODF coefficients at a voxel. There are 29 coefficients for DKI ODFs
    phi : array_like(dtype=float64)
        Polar phi angles
    theta : array_like(dtype=float64)
        Polar theta angles
    
    Returns
    -------
    spherical : array_like(dtype=float64)
        ODF in spherical form
    """
    if len(theta) != len(phi):
        raise Exception('Inputs theta and phi are not the same size')
    spherical = (1 / ((np.sin(phi) * np.cos(theta))**2 *  odf[22] + (np.sin(phi) * np.sin(theta))**2 *  odf[23] + np.cos(phi)**2 *  odf[24] + 2 * (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)) *  odf[25] + \
    2 * (np.sin(phi) * np.cos(theta)) * np.cos(phi) *  odf[26] + 2 * (np.sin(phi) * np.sin(theta)) * np.cos(phi) *  odf[27]))**(( odf[28] + 1) / 2) * (1 + ( odf[0] + \
    ( odf[1] * (np.sin(phi) * np.cos(theta))**2 +  odf[2] * (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)) +  odf[3] * (np.sin(phi) * np.cos(theta)) * np.cos(phi) +  odf[4] * (np.sin(phi) * np.sin(theta))**2 + \
    odf[5] * (np.sin(phi) * np.sin(theta)) * np.cos(phi) +  odf[6] * np.cos(phi)**2) / ((np.sin(phi) * np.cos(theta))**2 *  odf[22] + (np.sin(phi) * np.sin(theta))**2 *  odf[23] + np.cos(phi)**2 *  odf[24] + \
    2 * (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)) *  odf[25] + 2 * (np.sin(phi) * np.cos(theta)) * np.cos(phi) *  odf[26] + 2 * (np.sin(phi) * np.sin(theta)) * np.cos(phi) *  odf[27]) + \
    ( odf[7] * (np.sin(phi) * np.cos(theta))**4 +  odf[8] * (np.sin(phi) * np.cos(theta))**3 * (np.sin(phi) * np.sin(theta)) +  odf[9] * (np.sin(phi) * np.cos(theta))**3 * np.cos(phi) +  odf[10] * (np.sin(phi) * np.cos(theta))**2 * (np.sin(phi) * np.sin(theta))**2 + \
    odf[11] * (np.sin(phi) * np.cos(theta))**2 * (np.sin(phi) * np.sin(theta)) * np.cos(phi) +  odf[12] * (np.sin(phi) * np.cos(theta))**2 * np.cos(phi)**2 +  odf[13] * (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta))**3 + \
    odf[14] * (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta))**2 * np.cos(phi) +  odf[15] * (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)) * np.cos(phi)**2 +  odf[16] * (np.sin(phi) * np.cos(theta)) * np.cos(phi)**3 +  odf[17] * (np.sin(phi) * np.sin(theta))**4 + \
    odf[18] * (np.sin(phi) * np.sin(theta))**3 * np.cos(phi) +  odf[19] * (np.sin(phi) * np.sin(theta))**2 * np.cos(phi)**2 +  odf[20] * (np.sin(phi) * np.sin(theta)) * np.cos(phi)**3 +  odf[21] * np.cos(phi)**4) / ((np.sin(phi) * np.cos(theta))**2 *  odf[22] + (np.sin(phi) * np.sin(theta))**2 *  odf[23] + \
    np.cos(phi)**2 *  odf[24] + 2 * (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)) *  odf[25] + 2 * (np.sin(phi) * np.cos(theta)) * np.cos(phi) *  odf[26] + 2 * (np.sin(phi) * np.sin(theta)) * np.cos(phi) *  odf[27])**2) / 24)
    return spherical

def dkiodfcartesian(odf, x, y, z):
    """
    Convert DKI ODF coefficients at voxel to Cartesian form.

    Parameters
    ----------
    odf : array_like(dtype=float64)
        ODF coefficients at a voxel. There are 29 coefficients for DKI ODFs
    x : array_like(dtype=float64)
        Cartesian x coordinates
    y : array_like(dtype=float64)
        Cartesian y coordinates
    z : array_like(dtype=float64)
        Cartesian z coordinates
    
    Returns
    -------
    cart : array_like(dtype=float64)
        ODF in cartesian form
    """
    if len(x) != len(y):
        raise Exception('Input x, y and z coordinates are not the same size')
    if len(x) != len(z):
        raise Exception('Input x, y and z coordinates are not the same size')
    cart = (1 / ((x)**2 * odf[22] + (y)**2 * odf[23] + z**2 * odf[24] + 2 * (x) * (y) * odf[25] + \
    2 * (x) * z * odf[26] + 2 * (y) * z * odf[27]))**((odf[28] + 1) / 2) * (1 + (odf[0] + \
    (odf[1] * (x)**2 + odf[2] * (x) * (y) + odf[3] * (x) * z + odf[4] * (y)**2 + \
    odf[5] * (y) * z + odf[6] * z**2) / ((x)**2 * odf[22] + (y)**2 * odf[23] + z**2 * odf[24] + \
    2 * (x) * (y) * odf[25] + 2 * (x) * z * odf[26] + 2 * (y) * z * odf[27]) + \
    (odf[7] * (x)**4 + odf[8] * (x)**3 * (y) + odf[9] * (x)**3 * z + odf[10] * (x)**2 * (y)**2 + \
    odf[11] * (x)**2 * (y) * z + odf[12] * (x)**2 * z**2 + odf[13] * (x) * (y)**3 + \
    odf[14] * (x) * (y)**2 * z + odf[15] * (x) * (y) * z**2 + odf[16] * (x) * z**3 + odf[17] * (y)**4 + \
    odf[18] * (y)**3 * z + odf[19] * (y)**2 * z**2 + odf[20] * (y) * z**3 + odf[21] * z**4) / ((x)**2 * odf[22] + (y)**2 * odf[23] + \
    z**2 * odf[24] + 2 * (x) * (y) * odf[25] + 2 * (x) * z * odf[26] + 2 * (y) * z * odf[27])**2) / 24)
    return cart

def dtiodfspherical(odf, phi, theta, radial_weight=4):
    """
    Convert DTI ODFs coefficients at voxel to spherical form.

    Parameters
    ----------
    odf : array_like(dtype=float64)
        ODF coefficients at a voxel. There are 29 coefficients for DKI ODFs
    phi : array_like(dtype=float64)
        Polar phi angles
    theta : array_like(dtype=float64)
        Polar theta angles
    radial_weight : float
        Radial weighting power for detecting directional differences
        (Default: 4)
    
    Returns
    -------
    spherical : array_like(dtype=float64)
        ODF in spherical form
    """
    if len(theta) != len(phi):
        raise Exception('Inputs theta and phi are not the same size')
    spherical = (1/((np.sin(phi) * np.cos(theta))**2 * odf[0] + (np.sin(phi) *\
         np.sin(theta))**2 * odf[3] + np.cos(phi)**2 * odf[5] + 2 * \
             (np.sin(phi) * np.cos(theta)) * (np.sin(phi) * np.sin(theta)) * \
                 odf[1] + 2 * (np.sin(phi) * np.cos(theta)) * np.cos(phi) * \
                     odf[2] + 2 * (np.sin(phi) * np.sin(theta)) * np.cos(phi) \
                         * odf[4]))**((radial_weight + 1)/2)
    return spherical

def shbasis(deg, theta, phi):
    """
    Computes shperical harmonic basis set for even degrees of
    harmonics

    Parameters
    ----------
    deg : list of ints
        Degrees of harmonic
    theta : array_like
        (n, ) vector denoting azimuthal coordinates
    phi : array_like
        (n, ) vector denoting polar coordinates
    
    Returns
    -------
    complex array_like
        Harmonic samples at theta and phi at specified order
    """
    if not any([isinstance(x, int) for x in deg]):
        try:
            deg = [int(x) for x in deg]
        except:
            raise TypeError('Please supply degree of '
            'shperical harmonic as an integer')
    SH = []
    for n in deg:
        for m in range(-n, n + 1):
            SH.append(sph_harm(m, n, phi, theta))
    return np.array(SH, dtype=np.complex, order='F').T
