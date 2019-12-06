#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

def vectorize(img, mask):
    """ Returns vectorized image based on brain mask, requires no input
    parameters
    If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
    If the input is 3D or 4D, vectorize it using a mask
    Classification: Method

    Usage
    -----
    vec = dwi.vectorize(img) if there's no mask
    vec = dwi.vectorize(img, mask) if there's a mask

    Returns
    -------
    vec: N X number_of_voxels vector or array, where N is the number of DWI
    volumes
    """
    if mask is None:
        mask = np.ones((img.shape[0],
                        img.shape[1],
                        img.shape[2]),
                       order='F')
    mask = mask.astype(bool)
    if img.ndim == 1:
        n = img.shape[0]
        s = np.zeros((mask.shape[0],
                      mask.shape[1],
                      mask.shape[2]),
                     order='F')
        s[mask] = img
    if img.ndim == 2:
        n = img.shape[0]
        s = np.zeros((mask.shape[0],
                      mask.shape[1],
                      mask.shape[2], n),
                     order='F')
        for i in range(0, n):
            s[mask, i] = img[i,:]
    if img.ndim == 3:
        maskind = np.ma.array(img, mask=np.logical_not(mask))
        s = np.ma.compressed(maskind)
    if img.ndim == 4:
        s = np.zeros((img.shape[-1], np.sum(mask).astype(int)), order='F')
        for i in range(0, img.shape[-1]):
            tmp = img[:,:,:,i]
            # Compressed returns non-masked area, so invert the mask first
            maskind = np.ma.array(tmp, mask=np.logical_not(mask))
            s[i,:] = np.ma.compressed(maskind)
    return np.squeeze(s)

class makesnr:
    """
    Class object that computes and prints SNR plots for any number of
    input DWIs

    Parameters
    ----------
    dwilist:    string list
                list of 4D DWI (nifti-format) paths to evaluate and plot
    outpath:    string
                path to save SNR plot
    """
    def __init__(self, dwilist, noisepath=None, maskPath=None):
        """
        Constructor for makesnr class
        """
        if noisepath is None:
            raise Exception('Please provide the path to noise map from '
                            '"dwidenoise"')

        self.nDWI = len(dwilist)     # Number of input DWIs
        # Open the first image in list
        self.hdr = nib.load(dwilist[0])
        sDWI = self.hdr.shape        # Shape of input DWIs
        self.nvox = sDWI[0] * sDWI[1] * sDWI[2]
        if self.hdr.ndim != 4:
            raise IOError('Input DWIs need are not 4D. Please ensure you '
                          'use 4D NifTi files only.')
            print('No brain mask supplied')
        # Load image into 2D array
        self.img = np.array(self.hdr.dataobj)
        # Load noise into a vector
        self.noise = np.array(nib.load(noisepath).dataobj)
        # Load BVAL
        fName = op.splitext(dwilist[0])[0]
        bvalPath = op.join(fName + '.bval')
        if op.exists(bvalPath):
            self.bval = np.rint(np.loadtxt(bvalPath) / 1000)
        else:
            raise IOError('BVAL file {} not found'.format(bvalPath))
        if maskPath is not None and op.exists(maskPath):
            self.mask = np.array(nib.load(maskPath).dataobj).astype(bool)
            self.maskStatus = True
        else:
            self.mask = np.ones((self.img.shape[0], self.img.shape[
                1], self.img.shape[2]), order='F')
            self.maskStatus = False
        # Vectorize images
        self.img = vectorize(self.img, self.mask)
        print(self.img.shape)
        self.noise = vectorize(self.noise, self.mask)
        if self.nDWI > 1:
            # From second image to last image
            for i in range(1, self.nDWI):
                try:
                    tmp = vectorize(np.array(nib.load(dwilist[i]).dataobj),
                                    self.mask)
                    print(tmp.shape)
                    self.img = np.dstack((self.img, tmp))
                except:
                    raise ValueError('all input DWIs must have the same '
                                     'shape.')
                try:
                    fName = op.splitext(dwilist[i])[0]
                    bvalPath = op.join(fName + '.bval')
                    self.bval = np.stack((self.bval,
                        np.rint(
                            np.loadtxt(bvalPath) / 1000)))
                except:
                    raise IOError('Unable to locate BVAL file for image: {'
                                  '}'.format(dwilist[i]))

    def computesnr(self):
        """
        Computes SNR of all DWIs in class object
        :return:
        """
