#!/usr/bin/env python
# -*- coding : utf-8 -*-

"""
Function for computing DTI and DKI spherical harmonics from diffusion and
kurtosis tensors
"""
import numpy as np
import nibabel as nib
from designer.fitting import dwipy as dp

def dkiodfhelper(dt, kt, radial_weight=4):
    """
    Computes DKI fODFs at a voxel. This function is intended to parallelize
    computations across the brain.

    Parameters
    ----------
    dt : array_like(dtype=float)
        Diffusion tensor containing 6 elements
    kt: array_like(dtype=float)
        Kurtosis tensor containing 15 elements
    radial_weighing : float; optional
        Radial weighting power for detecting directional differences (Default: 4)
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
