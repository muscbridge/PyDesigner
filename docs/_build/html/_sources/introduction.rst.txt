Introduction
^^^^^^^^^^^^

Pydesigner provides the code to estimate the diffusion kurtosis tensors from diffusion-weighted images. The (constrained) weighted linear least squares estimator is here preferred because of its accuracy and precision. See “Veraart, J., Sijbers, J., Sunaert, S., Leemans, A. & Jeurissen, B., Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage, 2013, 81, 335-346” for more details. Next, a set of diffusion and kurtosis parameters, including the white matter tract integrity metrics, can be calculated form the resulting kurtosis tensor.

Some important notes needs to be considered:

| 1. Since the apparent diffusion tensor has 6 independent elements and the kurtosis tensor has 15 elements, there is a total of 21 parameters to be estimated. As an additional degree of freedom is associated with the noise free nondiffusion-weighted signal, at least 22 diffusion-weighted images must be acquired for DKI. It can be further shown that there must be at least three distinct b-values, which only differ in the gradient magnitude. Furthermore, at least 15 distinct diffusion (gradient) directions are required (Jensen et al. 2005). Some additional consideration must be made. The maximal b-value should be chosen carefully and is a trade-off between accuracy and precision. While for DTI, diffusion-weighted images are typically acquired with rather low b-values, about 1000 s⁄mm^2 , somewhat stronger diffusion sensitizing gradients need to be applied for DKI as the quadratic term in the b-value needs to be apparent. It is shown that b-values of about 2000 s⁄mm^2 are sufficient to measure the degree of non-Gaussianity with an acceptable precision (Jensen & Helpern 2010).

| 2. Outliers, or “black voxels”, in kurtosis maps are not uncommon. They result from undesired signal fluctuations due to motion, Gibbs ringing, or noise, which can often only be reduced using sophisticated tools. Unfortunately, those outliers will interfere with the visual and statistical inspection of the kurtosis parameters maps. Smoothing is typically used to suppress those outliers. Use of smoothing must be done with care as image blur partial voluming effects might be introduced.


The PyDesigner Pipeline
=======================

There are three main stages involved in DTI/DKI: `image acquisition <acquisition.rst>`__ , preprocessing, and tensor estiamation. The scanner handles the first stage, while our PyDesigner pipeline handles the last two.

Preprocessing
-------------

The next step is to boost SNR of the acquired image through various preprocessing steps. These steps include:

   #. Denoising (MRTRIX3's :code:`dwidenoise`)
   #. Removal of Gibbs ringing artifact (MRTRIX3's :code:`mrdegibbs`)
   #. Rigid body alignment of multiple DWI series (MRTRIX3's :code:`mrregister` and :code:`mrtransform`)
   #. Distortion correction (FSL's :code:`eddy` and :code:`topup` via MRTRIX3's :code:`dwidenoise`)
   #. Brain mask extraction (FSL's :code:`bet`)
   #. Smoothing
   #. Rician correction (MRTRIX3's :code:`mrcalc`)

These corrections are performs with command-line executables from FSL and MRTRIX, making it mandatory to have these installed prior to running PyDesigner.

Tensor Estimation
-----------------

The third and final stage performs actual metric extraction using mathematical means entirely via Python dependencies. The basic tensor estiamtion pipeline flows something like this:

   #. IRLLS outlier detection and tensor estimation
   #. Precise tensor fitting with constraints
   #. DTI parameter extraction
   #. AKC outlier detection
   #. DKI parameter extraction

