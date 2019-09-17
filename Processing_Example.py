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

niiPath = '/Users/sid/Downloads/nii_test/DWI/PARAMAPS/dwi_designer.nii'
savePath = '/Users/sid/Downloads/PyDesigner_Test'

myimage = dp.DWI(niiPath)

outliers, dt_hat = myimage.irlls()
outlierPath = os.path.join(savePath, 'outliers.nii')
dp.writeNii(outliers, myimage.hdr, outlierPath)

myimage.fit(constraints=[0,1,0], reject=outliers)
DT, KT = myimage.tensorReorder(myimage.tensorType())

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
dtPath = os.path.join(savePath, 'DT.nii')
ktPath = os.path.join(savePath, 'KT.nii')

dp.writeNii(md, myimage.hdr, mdPath)
dp.writeNii(rd, myimage.hdr, rdPath)
dp.writeNii(ad, myimage.hdr, adPath)
dp.writeNii(fa, myimage.hdr, faPath)
dp.writeNii(fe, myimage.hdr, fePath)
dp.writeNii(trace, myimage.hdr, tracePath)
dp.writeNii(mk, myimage.hdr, mkPath)
dp.writeNii(rk, myimage.hdr, rkPath)
dp.writeNii(ak, myimage.hdr, akPath)
dp.writeNii(DT, myimage.hdr, dtPath)
dp.writeNii(KT, myimage.hdr, ktPath)

akc_out = myimage.akcoutliers()
akcPath = os.path.join(savePath, 'akc_out.nii')
dp.writeNii(akc_out, myimage.hdr, akcPath)
myimage.akccorrect(akc_out=akc_out)

DT, KT = myimage.tensorReorder(myimage.tensorType())
md, rd, ad, fa, fe, trace, mk, rk, ak = myimage.extract()

# goodDirs = myimage.goodDirections(outliers)
# dirPath = os.path.join(savePath, 'good_directions.nii')
# dp.writeNii(goodDirs, myimage.hdr, dirPath)
#
# medFilt = dp.medianFilter(img=mk,
#                          violmask=goodDirs,
#                           th=30)
#
# medMask = os.path.join(savePath, 'median_mask.nii')
# dp.writeNii(medFilt.Mask, myimage.hdr, medMask)
#
# medFilt.findReplacement(bias='rand')
#
# md = medFilt.applyReplacement(md)
# rd = medFilt.applyReplacement(rd)
# ad = medFilt.applyReplacement(ad)
# fa = medFilt.applyReplacement(fa)
# mk = medFilt.applyReplacement(mk)
# rk = medFilt.applyReplacement(rk)
# ak = medFilt.applyReplacement(ak)

mdPath = os.path.join(savePath, 'md_med.nii')
rdPath = os.path.join(savePath, 'rd_med.nii')
adPath = os.path.join(savePath, 'ad_med.nii')
faPath = os.path.join(savePath, 'fa_med.nii')
mkPath = os.path.join(savePath, 'mk_med.nii')
rkPath = os.path.join(savePath, 'rk_med.nii')
akPath = os.path.join(savePath, 'ak_med.nii')
dtPath = os.path.join(savePath, 'DT_med.nii')
ktPath = os.path.join(savePath, 'KT_med.nii')
dp.writeNii(md, myimage.hdr, mdPath)
dp.writeNii(rd, myimage.hdr, rdPath)
dp.writeNii(ad, myimage.hdr, adPath)
dp.writeNii(fa, myimage.hdr, faPath)
dp.writeNii(mk, myimage.hdr, mkPath)
dp.writeNii(rk, myimage.hdr, rkPath)
dp.writeNii(ak, myimage.hdr, akPath)
dp.writeNii(DT, myimage.hdr, dtPath)
dp.writeNii(KT, myimage.hdr, ktPath)