#!/usr/bin/python
import numpy as np
import scipy as scp
import nibabel as nib
import os

def fetchbvecbval(niiPath):
    (path, file)  = os.path.split(niiPath)
    fName = os.path.splitext(file)[0]
    bvalPath = os.path.join(path, fName + '.bval')
    bvecPath = os.path.join(path, fName + '.bvec')
    return bvecPath,bvalPath

def loadbval(bvalPath):
    bvals = np.rint(np.loadtxt(bvalPath))
    return bvals

def loadbvec(bvecPath):
    bvecs = np.loadtxt(bvecPath)
    return bvecs

def loadgrad(niiPath):
    (bvecPath, bvalPath) = fetchbvecbval(niiPath)
    bvals = np.rint(np.loadtxt(bvalPath))
    bvecs = np.loadtxt(bvecPath)
    grad = np.c_[np.transpose(bvecs), bvals]
    return grad

def loadnii(niiPath):
    hdr = nib.load(niiPath)
    dwi = hdr.get_fdata
    print("Loaded into memory: %s" % (niiPath))
    return dwi

