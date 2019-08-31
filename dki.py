"""A Python implimentation of diffusion kurtosis parameter estimation using unconstrained weighted linear least squares.
    
    By default this class fits a model to the entire image, however the option to specify a mask is included.
    
    Voxelwise fitting and eigenvalue decomposition are performed in parallel over processes using joblib.
    
    Inputs are a 4D diffusion weighted image with dimentions (X x Y x Z x Ndwis) and a gradient file with dimentions (Ndwis x 4)
    
    Usage:
    from dki import WLLS
    import numpy as np
    
    dki = WLLS(dwi, grad)
    mask = np.ones((dwi.shape[0], dwi.shape[1], dwi.shape[2]))
    dt, s0, b = dki.fit()
    md, rd, ad, fa, fe, trace, mk, rk, ak = dki.extract(dt, b, mask)
    
    Benjamin Ades-Aron
        """

import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.optimize as opt
np.warnings.filterwarnings('ignore')

class WLLS(object):
    def __init__(self, dwi, grad):
        self.dwi = dwi
        self.grad = grad

    def vectorize(self, dwi, mask):
        # if the input is 1D or 2D, unpatch it to 3D or 4D using a mask
        # if the input is 3D or 4D, vectorize it using a mask
        if mask is None:
            mask = np.ones((self.dwi.shape[0], self.dwi.shape[1], self.dwi.shape[2]), order='F')
        if dwi.ndim == 1:
            dwi = np.expand_dims(dwi, axis=0)
        if dwi.ndim == 2:
            n = dwi.shape[0]
            s = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], n), order='F')
            for i in range(0, n):
                s[:,:,:,i] = np.reshape(dwi[i,:], (mask.shape), order='F')
        if dwi.ndim == 3:
            dwi = np.expand_dims(dwi, axis=-1)
        if dwi.ndim == 4:
            s = np.zeros((dwi.shape[-1], np.sum(mask).astype(int)), order='F')
            for i in range(0, dwi.shape[-1]):
                tmp = dwi[:,:,:,i]
                maskind = np.ma.array(tmp, mask=mask)
                s[i,:] = np.ma.ravel(maskind, order='F').data
        return np.squeeze(s)

    def createTensorOrder(self, order):
        if order == 2:
            cnt = np.array([1, 2, 2, 1, 2, 1], dtype=int)
            ind = np.array(([1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3])) - 1
        if order == 4:
            cnt = np.array([1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1], dtype=int)
            ind = np.array(([1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,2,2],[1,1,2,3],[1,1,3,3],\
                [1,2,2,2],[1,2,2,3],[1,2,3,3],[1,3,3,3],[2,2,2,2],[2,2,2,3],[2,2,3,3],[2,3,3,3],[3,3,3,3])) - 1
        return cnt, ind

    def diffusionCoeff(self, dt, dir):
        # compute ADC
        dcnt, dind = self.createTensorOrder(2)
        ndir = dir.shape[0]
        bD = np.tile(dcnt,(ndir, 1)) * dir[:,dind[:, 0]] * dir[:,dind[:, 1]]
        adc = np.matmul(bD, dt)
        return adc

    def kurtosisCoeff(self, dt, dir):
        # compute AKC
        wcnt, wind = self.createTensorOrder(4)
        ndir = dir.shape[0]
        adc = self.diffusionCoeff(dt[:6], dir)
        md = np.sum(dt[np.array([0,3,5])], 0)/3
        bW = np.tile(wcnt,(ndir, 1)) * dir[:,wind[:, 0]] * dir[:,wind[:, 1]] * dir[:,wind[:, 2]] * dir[:,wind[:, 3]]
        akc = np.matmul(bW, dt[6:])
        akc = (akc * np.tile(md**2, (adc.shape[0], 1)))/(adc**2)
        return akc

    def fibonacciSphere(self, samples=1, randomize=True):
        # generate "samples" evenly spaced points on a sphere
        import random
        rnd = 1
        if randomize:
            rnd = random.random() * samples
        points = []
        offset = 2/samples
        increment = np.pi * (3. - np.sqrt(5.))
        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - pow(y,2))
            phi = ((i + rnd) % samples) * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            points.append([x,y,z])
        return points

    def radialSampling(self, dir, n):
        # get the radial component of a set of directions
        dt = 2*np.pi/n
        theta = np.arange(0,2*np.pi-dt,dt)
        dirs = np.vstack((np.cos(theta), np.sin(theta), 0*theta))
        v = np.hstack((-dir[1], dir[0], 0))
        s = np.sqrt(np.sum(v**2))
        c = dir[2]
        V = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
        R = np.eye(3) + V + np.matmul(V,V) * (1-c)/(s**2)
        dirs = np.matmul(R, dirs)
        return dirs

    def wlls(self, shat, dwi, b):
        # compute a wlls fit using weights from inital fit shat
        w = np.diag(shat)
        dt = np.matmul(np.linalg.pinv(np.matmul(w, b)), np.matmul(w, np.log(dwi)))
        # for constrained fitting I'll need to modify this line. It is much slower than pinv so lets ignore for now.
        # dt = opt.lsq_linear(np.matmul(w, b), np.matmul(w, np.log(dwi)), \
        #     method='trf', tol=1e-12, max_iter=22000, lsq_solver='bvls')
        return dt

    def dtiTensorParams(self, nn):
        # compute dti tensor eigenvalues and eigenvectors and sort them
        values, vectors = np.linalg.eig(nn)
        idx = np.argsort(-values)
        values = -np.sort(-values)
        vectors = vectors[:, idx]
        return values, vectors

    def dkiTensorParams(self, v1, dt):
        # kurtosis tensor parameters use average directional
        # statistics to approximate ak and rk
        dirs = np.vstack((v1, -v1))
        akc = self.kurtosisCoeff(dt, dirs)
        ak = np.mean(akc)
        dirs = self.radialSampling(v1, 256).T
        akc = self.kurtosisCoeff(dt, dirs)
        rk = np.mean(akc)
        return ak, rk

    def extract(self, dt, b, mask):
        # extract all tensor parameters from dt
        num_cores = multiprocessing.cpu_count()

        print('...extracting dti parameters')
        DT = np.reshape(np.concatenate((dt[0,:], dt[1,:], dt[2,:], dt[1,:], dt[3,:], dt[4,:], dt[2,:], dt[4,:], dt[5,:])),(3,3,dt.shape[1]))
        
        # get the trace
        rdwi = np.exp(np.matmul(b[:,1:], dt))
        B = np.round(-(b[:,0]+b[:,3]+b[:,5])*1000)
        uB = np.unique(B)
        trace = np.zeros((dt.shape[1], uB.shape[0]))
        for ib in range(0, uB.shape[0]): 
            t = np.where(B==uB[ib])
            trace[:,ib] = np.mean(rdwi[t[0],:], axis=0)

        nvox = dt.shape[1]
        inputs = range(0, nvox)
        values, vectors = zip(*Parallel(n_jobs=num_cores,prefer='processes')\
            (delayed(self.dtiTensorParams)(DT[:,:,i]) for i in inputs))        
        values = np.reshape(np.abs(values), (nvox, 3))
        vectors = np.reshape(vectors, (nvox, 3, 3))

        print('...extracting dki parameters')
        dirs = np.array(self.fibonacciSphere(256, True))
        akc = self.kurtosisCoeff(dt, dirs)
        mk = np.mean(akc, 0)
        ak, rk = zip(*Parallel(n_jobs=num_cores,prefer='processes')\
            (delayed(self.dkiTensorParams)(vectors[i,:,0], dt[:,i]) for i in inputs))
        ak = np.reshape(ak, (nvox))
        rk = np.reshape(rk, (nvox))

        l1 = self.vectorize(values[:,0], mask)
        l2 = self.vectorize(values[:,1], mask)
        l3 = self.vectorize(values[:,2], mask)
        v1 = self.vectorize(vectors[:,:,0].T, mask)

        md = (l1+l2+l3)/3
        rd = (l2+l3)/2
        ad = l1
        fa = np.sqrt(1/2)*np.sqrt((l1-l2)**2+(l2-l3)**2+(l3-l1)**2)/np.sqrt(l1**2+l2**2+l3**2)
        trace = self.vectorize(trace.T, mask)
        fe = np.abs(np.stack((fa*v1[:,:,:,0], fa*v1[:,:,:,1], fa*v1[:,:,:,2]), axis=3))
        ak = self.vectorize(ak, mask)
        rk = self.vectorize(rk, mask)
        mk = self.vectorize(mk, mask)
        return md, rd, ad, fa, fe, trace, mk, ak, rk

    def fit(self):
        # run the fit
        order = np.floor(np.log(np.abs(np.max(self.grad[:,-1])+1))/np.log(10))
        if order >= 2:
            self.grad[:, -1] = self.grad[:, -1]/1000

        self.dwi.astype(np.double)
        self.dwi[self.dwi <= 0] = np.finfo(np.double).eps

        self.grad.astype(np.double)
        normgrad = np.sqrt(np.sum(self.grad[:,:3]**2, 1))
        normgrad[normgrad == 0] = 1

        self.grad[:,:3] = self.grad[:,:3]/np.tile(normgrad, (3,1)).T
        self.grad[np.isnan(self.grad)] = 0

        dcnt, dind = self.createTensorOrder(2)
        wcnt, wind = self.createTensorOrder(4)

        ndwis = self.dwi.shape[-1]
        bs = np.ones((ndwis, 1))
        bD = np.tile(dcnt,(ndwis, 1))*self.grad[:,dind[:, 0]]*self.grad[:,dind[:, 1]]
        bW = np.tile(wcnt,(ndwis, 1))*self.grad[:,wind[:, 0]]*self.grad[:,wind[:, 1]]*self.grad[:,wind[:, 2]]*self.grad[:,wind[:, 3]]
        b = np.concatenate((bs, (np.tile(-self.grad[:,-1], (6,1)).T*bD), np.squeeze(1/6*np.tile(self.grad[:,-1], (15,1)).T**2)*bW), 1)

        dwi_ = self.vectorize(self.dwi, None)
        init = np.matmul(np.linalg.pinv(b), np.log(dwi_))
        shat = np.exp(np.matmul(b, init))

        print('...fitting with wlls')
        inputs = tqdm(range(0, dwi_.shape[1]))
        num_cores = multiprocessing.cpu_count()
        dt = Parallel(n_jobs=num_cores,prefer='processes')\
            (delayed(self.wlls)(shat[:,i], dwi_[:,i], b) for i in inputs)
        dt = np.reshape(dt, (dwi_.shape[1], b.shape[1])).T

        s0 = np.exp(dt[0,:])
        dt = dt[1:,:]
        D_apprSq = 1/(np.sum(dt[(0,3,5),:], axis=0)/3)**2
        dt[6:,:] = dt[6:,:]*np.tile(D_apprSq, (15,1))
        return dt, s0, b

    def main(self):
        mask = np.ones((self.dwi.shape[0], self.dwi.shape[1], self.dwi.shape[2]))
        dt, s0, b = self.fit()
        md, rd, ad, fa, fe, trace, mk, rk, ak = self.extract(dt, b, mask)
        return md, rd, ad, fa, fe, trace, mk, rk, ak
