#!/usr/bin/python
import numpy as np
import nibabel as nib
import dwipi as dp
import matplotlib.pyplot as plt
import time
import os
import matplotlib.image as mpimg


niiPath = '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/dwi_designer.nii'
savePath = '/Users/sid/Downloads/PyDesigner_Test'

dwi = dp.DWI(niiPath)
# excludeb0=True
# maxiter=25
# convcrit=1e-3
# mode='DKI'
# leverage=3
# bounds=3
reject = dwi.irlls()
propviol = dwi.irllsviolmask(reject)
viols = dwi.detectOutliers(1)
md, rd, ad, fa, fe, trace, mk, rk, ak = dwi.extract()

i=50000
shat=shat[:,i]
dwi = dwi_[:,i]
b=self.b
constraints=[0,1,0]
G=-C


md = dwi.multiplyMask(md)
rd = dwi.multiplyMask(rd)
ad = dwi.multiplyMask(ad)
fa = dwi.multiplyMask(fa)
mk = dwi.multiplyMask(mk)
rk = dwi.multiplyMask(rk)
ak = dwi.multiplyMask(rk)
viols = dwi.multiplyMask(viols)

mdPath = os.path.join(savePath, 'md.nii')
rdPath = os.path.join(savePath, 'rd.nii')
adPath = os.path.join(savePath, 'ad.nii')
faPath = os.path.join(savePath, 'fa.nii')
fePath = os.path.join(savePath, 'fe.nii')
tracePath = os.path.join(savePath, 'trace.nii')
mkPath = os.path.join(savePath, 'mk.nii')
rkPath = os.path.join(savePath, 'rk.nii')
akPath = os.path.join(savePath, 'ak.nii')
violPath = os.path.join(savePath, 'outliers.nii')

dp.writeNii(md, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/md.nii', [0, 3])
dp.writeNii(rd, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rd.nii',[0, 3])
dp.writeNii(ad, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ad.nii',[0, 3])
dp.writeNii(fa, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fa.nii',[0, 1])
dp.writeNii(fe, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fe.nii')
dp.writeNii(trace, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/trace.nii')
dp.writeNii(mk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/mk.nii',[0, 2])
dp.writeNii(rk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rk.nii',[0, 2])
dp.writeNii(ak, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ak.nii',[0, 2])
dp.writeNii(viols, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/outliers.nii')

dp.writeNii(md, dwi.hdr, mdPath)
dp.writeNii(rd, dwi.hdr, rdPath)
dp.writeNii(ad, dwi.hdr, adPath)
dp.writeNii(fa, dwi.hdr, faPath)
dp.writeNii(fe, dwi.hdr, fePath)
dp.writeNii(trace, dwi.hdr, tracePath)
dp.writeNii(mk, dwi.hdr, mkPath)
dp.writeNii(rk, dwi.hdr, rkPath)
dp.writeNii(ak, dwi.hdr, akPath)
dp.writeNii(propviol, dwi.hdr, violPath)

med = dp.medianFilter(mk, viols, th=1, sz=3, conn='face')
reps = med.findReplacement(bias='rand')
md = med.applyReplacement(md)
rd = med.applyReplacement(rd)
ad = med.applyReplacement(ad)
fa = med.applyReplacement(fa)
mk = med.applyReplacement(rk)
rk = med.applyReplacement(mk)
ak = med.applyReplacement(ak)

dp.writeNii(md, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/md_m.nii',[0, 3])
dp.writeNii(rd, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rd_m.nii',[0, 3])
dp.writeNii(ad, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ad_m.nii',[0, 3])
dp.writeNii(fa, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fa_m.nii', [0, 1])
dp.writeNii(fe, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/fe_m.nii')
dp.writeNii(trace, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/trace_m.nii')
dp.writeNii(mk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/mk_m.nii',[0, 2])
dp.writeNii(rk, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/rk_m.nii',[0, 2])
dp.writeNii(ak, dwi.hdr, '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/PyDesigner/ak_m.nii',[0, 2])


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
