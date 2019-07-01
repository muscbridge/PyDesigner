#!/usr/bin/python
import numpy as np
import scipy as scp
import nibabel as nib
import os

print('Import successful!')

# Generate paths to files
niiPath = '/Users/sid/Downloads/nii_test/DWI/DKI_BIPOLAR_2mm_64dir_62slices_20190206093716_18.nii'
(path, file)  = os.path.split(inFile)
fName = os.path.splitext(file)[0]
bvalPath = os.path.join(path,fName + '.bval')
bvecPath = os.path.join(path,fName + '.bvec')

# Read Files
hdr = nib.load(inFile)
dwi = hdr.get_fdata()

bvals = np.rint(np.loadtxt(bvalPath))
bvecs = np.loadtxt(bvecPath)
grad = np.concatenate([np.transpose(bvecs),bvals],1)

# Tolerance Level: Smallest acceptable diffusion and kurtosis value. Values above and below this are non-zeros and zeros
# respectively
minparam = 1e-8

