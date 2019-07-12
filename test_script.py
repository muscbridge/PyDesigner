#!/usr/bin/python
import numpy as np
import nibabel as nib
import dwipi as dp
import matplotlib.pyplot as plt
import time
import matplotlib.image as mpimg


niiPath = "/Users/sid/Downloads/nii_test/DWI/PARAMAPS/dwi_designer.nii"
dwi = dp.DWI(niiPath)
dwi.fit()
viols = dwi.detectOutliers(1)
md, rd, ad, fa, fe, trace, mk, rk, ak = dwi.extract()

md = dwi.multiplyMask(md)
rd = dwi.multiplyMask(rd)
ad = dwi.multiplyMask(ad)
fa = dwi.multiplyMask(fa)
mk = dwi.multiplyMask(mk)
rk = dwi.multiplyMask(rk)
ak = dwi.multiplyMask(rk)
viols = dwi.multiplyMask(viols)


dp.writeNii(md, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/md.nii')
dp.writeNii(rd, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rd.nii')
dp.writeNii(ad, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ad.nii')
dp.writeNii(fa, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fa.nii')
dp.writeNii(fe, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fe.nii')
dp.writeNii(trace, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/trace.nii')
dp.writeNii(mk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/mk.nii')
dp.writeNii(rk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rk.nii')
dp.writeNii(ak, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ak.nii')
dp.writeNii(viols, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/outliers.nii')

med = dp.medianFilter(mk, viols, th=1, sz=3, conn='face')
reps = med.findReplacement(bias='rand')
md = med.applyReplacement(md)
rd = med.applyReplacement(rd)
ad = med.applyReplacement(ad)
fa = med.applyReplacement(fa)
fe = med.applyReplacement(fe)
trace = med.applyReplacement(trace)
rk = med.applyReplacement(rk)
ak = med.applyReplacement(ak)

dp.writeNii(md, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/md_m.nii')
dp.writeNii(rd, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rd_m.nii')
dp.writeNii(ad, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ad_m.nii')
dp.writeNii(fa, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fa_m.nii')
dp.writeNii(fe, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fe_m.nii')
dp.writeNii(trace, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/trace_m.nii')
dp.writeNii(mk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/mk_m.nii')
dp.writeNii(rk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rk_m.nii')
dp.writeNii(ak, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ak_m.nii')


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
