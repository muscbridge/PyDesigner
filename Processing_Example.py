#!/usr/bin/env python
# coding: utf-8

# # PyDesigner Example
# This notebook serves to educate on PyDesigner and it's usage. Currently, this example will only cover portions past preprocessing for stability testing of **dwipi**.

# ## Load Modules
# Start by loading modules necessary to execute this script

import nibabel as nib
import numpy as np
import dwipi as dp
import matplotlib.pyplot as plt
import time
import os


niiPath = 'D:\SystemFiles\siddh\Box Sync\Home-Work\PARAMAPS\dwi_designer.nii'
savePath = 'C:\\Users\siddh\Desktop\PyDesigner_Test'

myimage = dp.DWI(niiPath)

outliers, dt_hat = myimage.irlls()

myimage.fit(constraints=[0,1,0], reject=outliers)

md, rd, ad, fa, fe, trace, mk, rk, ak = myimage.extract()

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

dp.writeNii(md, myimage.hdr, mdPath)
dp.writeNii(rd, myimage.hdr, rdPath)
dp.writeNii(ad, myimage.hdr, adPath)
dp.writeNii(fa, myimage.hdr, faPath)
dp.writeNii(fe, myimage.hdr, fePath)
dp.writeNii(trace, myimage.hdr, tracePath)
dp.writeNii(mk, myimage.hdr, mkPath)
dp.writeNii(rk, myimage.hdr, rkPath)
dp.writeNii(ak, myimage.hdr, akPath)
dp.writeNii(outliers, myimage.hdr, violPath)

goodDirs = myimage.goodDirections(outliers)
dirPath = os.path.join(savePath, 'good_directions.nii')
dp.writeNii(goodDirs, myimage.hdr, dirPath)

medFilt = dp.medianFilter(img=mk,
                         violmask=goodDirs,
                         th=30)
medMask = os.path.join(savePath, 'median_mask.nii')
dp.writeNii(medFilt.Mask, myimage.hdr, medMask)

medFilt.findReplacement(bias='rand')

md = medFilt.applyReplacement(md)
rd = medFilt.applyReplacement(rd)
ad = medFilt.applyReplacement(ad)
fa = medFilt.applyReplacement(fa)
mk = medFilt.applyReplacement(mk)
rk = medFilt.applyReplacement(rk)
ak = medFilt.applyReplacement(ak)

mdPath = os.path.join(savePath, 'md_med.nii')
rdPath = os.path.join(savePath, 'rd_med.nii')
adPath = os.path.join(savePath, 'ad_med.nii')
faPath = os.path.join(savePath, 'fa_med.nii')
mkPath = os.path.join(savePath, 'mk_med.nii')
rkPath = os.path.join(savePath, 'rk_med.nii')
akPath = os.path.join(savePath, 'ak_med.nii')
dp.writeNii(md, myimage.hdr, mdPath)
dp.writeNii(rd, myimage.hdr, rdPath)
dp.writeNii(ad, myimage.hdr, adPath)
dp.writeNii(fa, myimage.hdr, faPath)
dp.writeNii(mk, myimage.hdr, mkPath)
dp.writeNii(rk, myimage.hdr, rkPath)
dp.writeNii(ak, myimage.hdr, akPath)


