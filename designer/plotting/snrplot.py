#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

minZero = 1e-8
np.seterr(all='ignore')

def vectorize(img, mask):
    """
    Returns vectorized image based on brain mask, requires no input
    parameters
    If the input is 1D or 2D, unpatch it to 3D or 4D using a mask
    If the input is 3D or 4D, vectorize it using a mask
    Classification: Function

    Parameters
    ----------
    img : ndarray
        1D, 2D, 3D or 4D image array to vectorize
    mask : ndarray
        3D image array for masking

    Returns
    -------
    vec : N X number_of_voxels vector or array, where N is the number
        of DWI volumes

    Usage
    -----
    vec = vectorize(img) if there's no mask
    vec = vectorize(img, mask) if there's a mask
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
    dwilist : list of str
        List of 4D DWI (nifti-format) paths to evaluate and plot
    noisepath : str
        Path to noise map from "dwidenoise"
    maskpath : str, optional
        Path to brain mask

    Methods
    ----------
    __init__ : constructs makesnr class
    getuniquebval : creates a list of unique B-values for the purpose of
                    SNR computation
    computesnr : performs SNR computation
    histcount : bins SNR values
    makeplot : creates and saves SNR plot from bin counts

    """
    def __init__(self, dwilist, noisepath=None, maskpath=None):
        """
        Constructor for makesnr class

        Parameters
        ----------
        dwilist : list of str
            String list of nifti paths to plot
        noisepath : str
            Path to noisemap (Default: None)
        maskpath : str, optional
            Path to nifti brain mask (Default: None)
        """
        if noisepath is None:
            raise Exception('Please provide the path to noise map from '
                            '"dwidenoise"')

        self.nDWI = len(dwilist)     # Number of input DWIs
        self.DWInames = [op.split(i)[-1] for i in dwilist]
        # Open the first image in list
        self.hdr = nib.load(dwilist[0])
        sDWI = self.hdr.shape        # Shape of input DWIs
        if self.hdr.ndim != 4:
            raise IOError('Input DWIs need are not 4D. Please ensure you '
                          'use 4D NifTi files only.')
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
        if maskpath is not None and op.exists(maskpath):
            self.mask = np.array(nib.load(maskpath).dataobj).astype(bool)
            self.maskStatus = True
        else:
            self.mask = np.ones((self.img.shape[0], self.img.shape[
                1], self.img.shape[2]), order='F')
            self.maskStatus = False
        # Vectorize images
        self.img = vectorize(self.img, self.mask)
        self.nvox = self.img.shape[1]
        self.noise = vectorize(self.noise, self.mask)
        if self.nDWI > 1:
            # From second image to last image
            for i in range(1, self.nDWI):
                try:
                    tmp = vectorize(np.array(nib.load(dwilist[i]).dataobj),
                                    self.mask)
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
        truncateIdx = np.logical_or(np.isnan(self.img),
                                     (self.img < minZero))
        self.img[truncateIdx] = minZero

    def getuniquebval(self):
        """
        Creates a list of unique B-values for the purpose of SNR
        computation. In the calculation of SNR, B0 signal can be averaged
        becase they are not associated to any direction. This is not true
        for non-B0 values however, because every 3D volume represents a
        different direction. To compute SNR appropriately, differences in
        gradients have to be accounted. This function creates a list of
        B-values in the order they need to appear for the calculation of
        SNR.

        Returns
        -------
        b_list : ndarray
            Numpy vector containing list of B-values to be used in
            SNR calculation
        """

        b_list = []
        for i in range(self.nDWI):
            bvals = self.bval[i, :]
            unibvals = np.array(np.unique(bvals),dtype=int)
            bval_list = []
            for j in range(unibvals.size):
                bval = unibvals[j]
                # Index where entirety of bvals, given by variable
                # bvals, is equal to a single unique bval
                idx_bval = np.where(np.isin(bvals, bval))[-1]
                if bval != 0:
                    bval_list.append(str(bval))
                else:
                    # Appends '0' to bval_list n countb0 number of times
                    for countb in range(len(idx_bval)):
                        bval_list.append(str(bval))
            b_list.append(bval_list)
        return np.asarray(b_list, dtype=int)

    def computesnr(self):
        """
        Computes SNR of all DWIs in class object

        Returns
        -------
        snr_dwi : ndarray
            Numpy array of SNR across all DWI.
        """
        bval_list = self.getuniquebval()
        snr_dwi = np.zeros((self.nvox, bval_list.shape[1], self.nDWI))
        for i in range(self.nDWI):
            bvals = self.bval[i, :]
            unibvals = np.array(np.unique(bvals),dtype=int)
            print('Computing SNR: ' + self.DWInames[i])
            for j in range(unibvals.size):
                bval = unibvals[j]
                print('   * B' + str(bval * 1000) + '...')
                # Index where entirety of bvals, given by variable
                # bvals, is equal to a single unique bval
                idx_bval = np.where(np.isin(bvals, bval))[-1]
                idx_list = np.where(np.isin(bval_list[i, :], bval))[-1]
                img = self.img[idx_bval, :, i]
                if bval != 0:
                    snr_dwi[:, idx_list, i] = np.mean((img /self.noise),
                                                      axis=0).reshape((
                        self.nvox, 1))
                else:
                    # Appends '0' to bval_list n countb0 number of times
                    for countb in range(img.shape[0]):
                        snr_dwi[:, idx_list, i] = \
                            np.divide(img, self.noise).reshape((
                        self.nvox, idx_list.size))
        truncateIdx = np.logical_or(np.isnan(snr_dwi),
                                     (snr_dwi < minZero))
        snr_dwi[truncateIdx] = minZero
        return snr_dwi

    def histcount(self, nbins=100):
        """
        Bins SNR into nbins and returns various counting properties

        Parameters
        ----------
        nbins :  int
        Number of bins to plot

        Returns
        -------
        count : ndarray
            Array of count of voxels in bins
        binval : ndarray
            Array of bin values
        unibvals : ndarray
            Array containing all unique B-values detected
        """
        if not isinstance(nbins, int):
            raise ValueError('Number of bins (nbins) entered is not an '
                             'integer. Please specify and integer.')
        bval_list = self.getuniquebval()
        snr = self.computesnr()
        # Get min and max values of SNR
        snr_ = snr.reshape(-1)
        minVal = np.min(snr_[np.isfinite(snr_)])
        maxVal = np.max(snr_[np.isfinite(snr_)])
        unibvals = np.array(np.unique(bval_list), dtype=int)
        count = np.zeros((nbins, unibvals.size, self.nDWI))
        edges = np.zeros((nbins+1, unibvals.size, self.nDWI))
        for j in range(self.nDWI):
            for i in range(unibvals.size):
                bval = unibvals[i]
                idx_list = np.where(np.isin(bval_list[j, :], bval))[-1]
                vals = snr[:, idx_list, j]
                (count[:, i, j], edges[:, i, j]) = \
                np.histogram(vals,
                             bins=nbins,
                             range=(minVal, maxVal),
                             density=True)
        edges = np.unique(edges)
        if edges.size != nbins + 1:
            raise Exception('Number of binning edges across B-values and '
                            'DWIs is not consistent. Aborting SNR '
                            'binning.')
        binval = np.zeros((nbins))
        for i in range(binval.size):
            binval[i] = np.median([edges[i], edges[i + 1]])
        return count, binval, unibvals

    def makeplot(self, path, smooth=True, smoothfactor=5):
        """
        Creates and saves SNR plot to a path as SNR.png

        Parameters
        ----------
        path : str
            Directory to save the plot in
        smooth : bool, optional
            Specify whether to interpolate and smooth (Default: True)
        smoothfactor : int, optional
            Smoothing factor to apply (Default: 5)

        Returns
        -------
        None: Writes out image into directory as SNR.png
        """
        if not isinstance(smoothfactor, int):
            raise ValueError('Please specify an integer for smooth '
                             'factor.')
        if not op.isdir(path):
            raise IOError('Output path defined does not exist.')
        outpath = op.join(path, 'SNR.png')
        (count, binval, unibvals) = self.histcount()
        count = count * 100
        if smooth:
            smoothFactor = smoothfactor *  count.shape[0] # Number of
            # points to add between histogram counts for smoothing
            binval_interp = np.linspace(binval.min(), binval.max(),
                                   smoothFactor)
            (R, C, D) = count.shape
            count_interp = np.zeros((smoothFactor, C, D))
            for y in range(C):
                for z in range(D):
                    spl = make_interp_spline(binval, count[:, y, z], k=3)
                    count_interp[: , y, z] = spl(binval_interp)
            count = count_interp
            binval = binval_interp
        plt.style.use('seaborn')
        nplots = unibvals.size
        titles = ['B' + str(i * 1000) for i in unibvals]
        fig = plt.figure(figsize=(8, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.1)
        fig.suptitle('SNR of Acquisitions', fontsize=20)
        for i in range(nplots):
            ax = fig.add_subplot(nplots, 1, i + 1)
            ax.plot(binval, count[:, i , :])
            # Subplot Properties
            ax.grid(True)
            ax.set_title(titles[i],
                         loc='right')
            bval = unibvals[i]
            ax.set_xlabel('SNR')
            ax.set_ylabel('% of voxels')
            ax.set_ylim(count[np.isfinite(count)].min(),
                count[np.isfinite(count)].max())
            if bval == 0:
                ax.set_xlim(0, 200)
            elif bval == 1:
                ax.set_xlim(0, 100)
            elif bval == 2:
                ax.set_xlim(0, 80)
            else:
                ax.set_xlim(0, 60)
        # Plot Properties
        plt.legend(self.DWInames,
                         ncol=nplots,
                         loc='upper left',
                         frameon = False,
                         bbox_to_anchor=(0.25, -0.19))
        plt.savefig(outpath, dpi=300)
