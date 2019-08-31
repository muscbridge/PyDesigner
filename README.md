# PyDesigner
<p align="center">
  <img src="https://avatars1.githubusercontent.com/u/47674287?s=400&u=9ca45aeafe30730e26fb70865c14e736f3a4dabf&v=4" alt="MAMA Logo" width="256">
</p>


PyDesigner is a complete Python port of NYU's DESIGNER pipeline for preprocessing diffusion MRI images (dMRI). This work was motivated by:

* Minimization of **dependencies** to make it easier to deploy
  and understandable metric compared to the size in bytes.
* **Faster** dMRI preprocessing
* More **accurate** diffusion and kurtosis tensor estimation via cutting-edge algorithms
* **Cross-platform compatibility** between Windows, Mac and Linux
* **Ease-of-use** through Python classes so anyone can preprocess dMRI data
* **Docker** compatibility for enhanced deployment

This is a collaboration project between MUSC and NYU to bring easy-to-use dMRI preprocessing and diffusion and kurtosis tensor estimation to masses.


<p align="center">
  <a href="https://medicine.musc.edu/departments/centers/cbi/dki">
    <img src="https://tfcbt2.musc.edu/assets/musc_logo-69ee0f1483cd4d8772c5d114f89a0aace954f2f4a299d10f814fc532c7b3c719.png" alt="MUSC DKI Page" width="256">
</p>

<p align="center">
  <a href="https://github.com/NYU-DiffusionMRI">
    <img src="https://greatoakscharter.org/wp-content/uploads/2017/03/NYU-Logo.png"
         alt="Sponsored by Evil Martians" width="256">
  </a>
</p>

## Table of Contents
**[Abstract](#PyDesigner)**<br>
**[General Information](##general-information)**<br>
**[Meet the Team](##meet-the-team)**<br>

## General Information
We here provide the code to estimate the diffusion kurtosis tensors from diffusion-weighted images. The (constrained) weighted linear least squares estimator is here preferred because of its accuracy and precision. See “Veraart, J., Sijbers, J., Sunaert, S., Leemans, A. & Jeurissen, B.,  Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage, 2013, 81, 335-346” for more details. Next, a set of diffusion and kurtosis parameter, including the white matter tract integrity metrics, can be calculated form the resulting kurtosis tensor.

Some important notes needs to be considered:

1. Since the apparent diffusion tensor has 6 independent elements and the kurtosis tensor has 15 elements, there is a total of 21 parameters to be estimated. As an additional degree of freedom is associated with the noise free nondiffusion-weighted signal at least 22 diffusion-weighted images must be acquired for DKI. It can be further shown that there must be at least three distinct b-values, which only differ in the gradient magnitude. Furthermore, at least 15 distinct diffusion (gradient) directions are required (Jensen et al. 2005). Some additional consideration must be made.  The maximal b-value should be chosen carefully and is a trade-off between accuracy and precision. While for DTI, diffusion-weighted images are typically acquired with rather low b-values, about 1000 s⁄mm^2 , somewhat stronger diffusion sensitizing gradients need to be applied for DKI as the quadratic term in the b-value needs to be apparent. It is shown that b-values of about 2000 s⁄mm^2  are sufficient to measure the degree of non-Gaussianity with an acceptable precision (Jensen & Helpern 2010). 

2. Outliers, or “black voxels”, in kurtosis maps are not uncommon. They result from undesired signal fluctuations due to motion, Gibbs ringing, or noise, which can often only be reduced using sophisticated tools.  Unfortunately, those outliers will interfere with the visual and statistical inspection of the kurtosis parameters maps. Smoothing is typically used to suppress those outliers. Use of smoothing must be done with care as image blur partial voluming effects might be introduced.

## Meet the Team
PyDesigner is a join collarobation and as such consists of several developers.

### Developer
<img src="https://avatars0.githubusercontent.com/u/13654344?s=400&v=4" align="left"
     title="GitHub: Siddhartha Dhiman" height="163"> 

    Siddhartha Dhiman

    Research Specialist
    Department of Neuroscience
    Medical University of South Carolina<
    dhiman@musc.edu

### Developer
<img src="https://avatars2.githubusercontent.com/u/26722533?s=400&v=4" align="right"
     title="GitHub: Joshua Teves" height="163"> 

     Joshua Teves

     Systems Programmer
     Department of Neuroscience
     Medical University of South Carolina
     teves@musc.edu

### Advisor
<img src="https://muschealth.org/MUSCApps/HealthAssets/ProfileImages/jej50.jpg" align="left"
     title="MUSC: Jens Jensen" height="163">

     Jens Jensen, Ph.D.

     Professor
     Department of Neuroscience
     Medical University of South Carolina
     <email placeholder>
     




