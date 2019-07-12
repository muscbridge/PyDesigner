#!/usr/bin/python
import numpy as np
import nibabel as nib
import dwipi as dp
import matplotlib.pyplot as plt
import time
import matplotlib.image as mpimg


niiPath = "D:\SystemFiles\siddh\Box Sync\Home-Work\PARAMAPS\dwi_designer.nii"
dwi = dp.DWI(niiPath)
dwi.fit()
viols = dwi.detectOutliers(1)

dwi.fit()
md, rd, ad, fa, fe, trace, mk, rk, ak = dwi.extract()
pViols = dwi.findViols([0, 1, 0])

# Plot
imShow = mk
for i in range(0,imShow.shape[2]):
    plt.imshow(imShow[:,:,i])
    time.sleep(1/20)
    plt.draw()
    plt.pause(1/30)

# Write Test
clipped_img = nib.Nifti1Image(imShow, dwi.hdr.affine, dwi.hdr.header)
nib.save(clipped_img,'/Users/sid/Downloads/nii_test/DWI/PARAMAPS/test_map.nii')

