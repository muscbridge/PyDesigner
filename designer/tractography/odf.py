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
from designer.fitting.dwipy import vectorize, writeNii
from tqdm import tqdm

class odfmodel():
    """
    DTI/DKI tractograpy class for computing ODFs and preparing spherical
    harmonics for DTI or DKI fiber tracking.
    """
    def __init__(self, dt, kt=None, mask=None, l_max=6, radial_weight=4,
        nthreads=None):
        """
        Parameters
        ----------
        dt : str
            Path to diffusion tensor, which is a 4D NifTI volume composed of six (6)
            components    
        kt : str
            Path to kurtosis tensor, which is a 4D NifTI compose of fifteen (15)
            components
            (Default: None)
        mask :  str
            Path to brain mask in NifTI format
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
                'ensure that the folder specified exists.')
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
        if radial_weight is None:
            warnings.warn('Radial weight for dODF computation not specified. '
            'Using default value of 4.')
            self.radial_weight=4
        if nthreads > multiprocessing.cpu_count():
            warnings.warn('Number of workers/threads specified exceed more '
            'than available. Using the maximum workers/threads available.')
            self.workers = -1
        if nthreads is None:
            self.workers = -1
        else:
            self.workers = nthreads

    def dkiodf(self, output):
        """
        Computes DKI fODFs for the whole brain.

        Parameters
        ----------
        output : str
            Path to output NifTi file containing DKI fODFs
        Returns
        -------
        None, writes out file
        """
        if self.kt is None:
                raise AttributeError('WOAH! Cannot compute DKI ODFs without '
                'kurtosis tensor (KT). Try using dtiodf(), Jumbo.')
        if not op.exists(op.dirname(output)):
            raise OSError('Specifed directory for output file {} does not '
                        'exist. Please ensure that this is a valid '
                        'directory.'.format(op.dirname(output)))
        # Vectorize images
        DT = vectorize(self.DT, self.mask_img)
        KT = vectorize(self.KT, self.mask_img)
        nvox = DT.shape[-1]
        # coeff = np.zeros((29, nvox))
        # for i in tqdm(range(nvox),
        #     desc='Computing fODFs',
        #     bar_format='{desc}: [{percentage:0.0f}%]',
        #     unit='vox',
        #     ncols=70):
        #     coeff[:, i] = dkiodfhelper(DT[:, i], KT[:, i], radial_weight)
        inputs = tqdm(range(nvox),
                        desc='DKI fODFs',
                        bar_format='{desc}: [{percentage:0.0f}%]',
                        unit='vox',
                        ncols=70)
        coeff = Parallel(n_jobs=8, prefer='processes') (delayed(dkiodfhelper)\
            (DT[:, i], KT[:, i], self.radial_weight) for i in inputs)
        coeff = np.array(coeff).T
        coeff = vectorize(coeff, self.mask_img)
        coeff = nib.Nifti1Image(coeff, self.hdr.affine, self.hdr.header)
        return(coeff)

    def odf2sh(odf, deg, theta, phi, l_max=6):
        # Create shperical harmonic (SH) base set
        degs = np.arange(l_max + 1, dtype=int)
        l_tot = 2*degs + 1 # total harmonics in the degree
        l_num = 2 * degs[::2] + 1 # how many per degree (evens only)
        harmonics = []
        sh_end = 1 # initialize the SH set for indexing
        for h in range(0,len(degs[::2])):
            sh_start = sh_end + l_num[h] - 1
            sh_end = sh_start + l_num[h] - 1
            harmonics.extend(np.arange(sh_start,sh_end+1))
        B = shbasis(degs, theta, phi)

def dkiodfhelper(dt, kt, radial_weight=4):
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

    Returns
    -------
    coeff : array_like(dtype=float)
        DKI ODF coefficients containing 29 elements
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

    Davg = np.trace(D)/3
    U = Davg* np.linalg.inv(D)
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
                    B12 = B12 + B0 * (U[k,0] * U[l,1] + U[k,2] * U[l,0])
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
    return coeff

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
            if (n % 2) == 0:
                SH.append(sph_harm(m, n, phi, theta))
    return np.array(SH, dtype=np.complex, order='F').T
