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

md, rd, ad, fa, fe, trace = myimage.extractDTI()

mdPath = os.path.join(savePath, 'md.nii')
rdPath = os.path.join(savePath, 'rd.nii')
adPath = os.path.join(savePath, 'ad.nii')
faPath = os.path.join(savePath, 'fa.nii')
fePath = os.path.join(savePath, 'fe.nii')
tracePath = os.path.join(savePath, 'trace.nii')

dp.writeNii(md, myimage.hdr, mdPath)
dp.writeNii(rd, myimage.hdr, rdPath)
dp.writeNii(ad, myimage.hdr, adPath)
dp.writeNii(fa, myimage.hdr, faPath)
dp.writeNii(fe, myimage.hdr, fePath)
dp.writeNii(trace, myimage.hdr, tracePath)


akc_out = myimage.akcoutliers()
akcPath = os.path.join(savePath, 'akc_out.nii')
dp.writeNii(akc_out, myimage.hdr, akcPath)

myimage.akccorrect(akc_out=akc_out)

DT, KT = myimage.tensorReorder(myimage.tensorType())
mk, rk, ak, trace = myimage.extractDKI()

tracePath = os.path.join(savePath, 'trace.nii')
mkPath = os.path.join(savePath, 'mk.nii')
rkPath = os.path.join(savePath, 'rk.nii')
akPath = os.path.join(savePath, 'ak.nii')
dtPath = os.path.join(savePath, 'DT.nii')
ktPath = os.path.join(savePath, 'KT.nii')
dp.writeNii(trace, myimage.hdr, tracePath)
dp.writeNii(mk, myimage.hdr, mkPath)
dp.writeNii(rk, myimage.hdr, rkPath)
dp.writeNii(ak, myimage.hdr, akPath)
dp.writeNii(DT, myimage.hdr, dtPath)
dp.writeNii(KT, myimage.hdr, ktPath)