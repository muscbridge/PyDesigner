# PyDesigner [UNDER CONSTRUCTION]
**Project is currently under construction and will not run. Official release will be marked by the removal of 'under construction' header.**

**_Disclaimer:_**
```
This project is in very early stages of development and most likely will not work as intended. 
We are not responsible for any data corruption, loss of valuable data, computer issues, you 
getting fired because you chose to run this, or a thermonuclear war. We strongle enocurage 
all potential users to wait for an official release.
```
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
    <img src="https://tfcbt2.musc.edu/assets/musc_logo-69ee0f1483cd4d8772c5d114f89a0aace954f2f4a299d10f814fc532c7b3c719.png" alt="MUSC DKI Page" width="128">
</p>

<p align="center">
  <a href="http://www.diffusion-mri.com/">
    <img src="https://greatoakscharter.org/wp-content/uploads/2017/03/NYU-Logo.png"
         alt="Sponsored by Evil Martians" width="128">
  </a>
</p>

## Table of Contents
**[Abstract](#pydesigner)**<br>
**[General Information](#general-information)**<br>
**[Introduction](#introduction)**<br>
**[L- The PyDesigner Pipeline](#the-pydesigner-pipeline)**<br>
**[L-- Image Acquisition](#image-acquisition)**<br>
**[L-- Preprocessing](#preprocessing)**<br>
**[L-- Tensor Estimation](#tensor-estimation)**<br>
**[Installation](#installation)**<br>
**[L- FSL](#fsl)**<br>
**[L- MRTRIX3](#mrtrix3)**<br>
**[L- Python](#python)**<br>
**[L- PyDesigner](#pydesigner)**<br>
**[Running PyDesigner](#running-pydesigner)**<br>
**[Meet the Team](#meet-the-team)**<br>

## General Information
### Introduction
We here provide the code to estimate the diffusion kurtosis tensors from diffusion-weighted images. The (constrained) weighted linear least squares estimator is here preferred because of its accuracy and precision. See “Veraart, J., Sijbers, J., Sunaert, S., Leemans, A. & Jeurissen, B.,  Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage, 2013, 81, 335-346” for more details. Next, a set of diffusion and kurtosis parameter, including the white matter tract integrity metrics, can be calculated form the resulting kurtosis tensor.

Some important notes needs to be considered:

1. Since the apparent diffusion tensor has 6 independent elements and the kurtosis tensor has 15 elements, there is a total of 21 parameters to be estimated. As an additional degree of freedom is associated with the noise free nondiffusion-weighted signal at least 22 diffusion-weighted images must be acquired for DKI. It can be further shown that there must be at least three distinct b-values, which only differ in the gradient magnitude. Furthermore, at least 15 distinct diffusion (gradient) directions are required (Jensen et al. 2005). Some additional consideration must be made.  The maximal b-value should be chosen carefully and is a trade-off between accuracy and precision. While for DTI, diffusion-weighted images are typically acquired with rather low b-values, about 1000 s⁄mm^2 , somewhat stronger diffusion sensitizing gradients need to be applied for DKI as the quadratic term in the b-value needs to be apparent. It is shown that b-values of about 2000 s⁄mm^2  are sufficient to measure the degree of non-Gaussianity with an acceptable precision (Jensen & Helpern 2010). 

2. Outliers, or “black voxels”, in kurtosis maps are not uncommon. They result from undesired signal fluctuations due to motion, Gibbs ringing, or noise, which can often only be reduced using sophisticated tools.  Unfortunately, those outliers will interfere with the visual and statistical inspection of the kurtosis parameters maps. Smoothing is typically used to suppress those outliers. Use of smoothing must be done with care as image blur partial voluming effects might be introduced.

### The PyDesigner Pipeline
There are three main stages involved in DTI/DKI: image acquisition, preprocessing, and tensor estiamation. The scanner handles the first stage, while our PyDesigner pipeline handles the last two.

#### Image acquisition
Like with any other bioimaging modalities, the first step is always acquiring imaging data. Depending on your institution, ensure that you are using the most recent protocol for either DTI or DKI.

#### Preprocessing
The next step is to boost SNR of the acquired image through various preprocessing steps. These steps include:

1. Denoising (MRTRIX3's `dwidenoise`)
2. Removal of Gibbs ringing artifact (MRTRIX3's `mrdegibbs`)
3. Rigid body alignment of multiple DWI series (MRTRIX3's `mrregister` and `mrtransform`)
4. Distortion correction (FSL's `eddy` and `topup` via MRTRIX3's `dwidenoise`)
5. Brain mask extraction (FSL's `bet`)
6. Smoothing
7. Rician correction (MRTRIX3's `mrcalc`)

These corrections are performs with command-line executables from FSL and MRTRIX, making it mandatory to have these installed prior to running PyDesigner.

#### Tensor Estimation
The third and final stage performs actual metric extraction using mathematical means entirely via Pyhton dependencies. The basic tensor estiamtion pipeline flows something like this:

1. IRLLS outlier detection and tensor estimation
2. Precise tensor fitting with constraints
3. DTI parameter extraction
4. AKC outlier detection
5. DKI parameter extraction

Performing these calculations require only Python dependencies which can easily be obtained via `pip install ...` or `conda install ...`. More information will be provided in the [installation](#installation) section.

## Installation
There are three dependencies for PyDesigner: 1) FSL, 2) MRTRIX3, and 3) Python. Configuration and means of obtaining said dependencies are listed below.

**Note**: All testing was conducted using Python 3.7, MRtrix3, and FSL 6.0.1. Usage of DESIGNER with alternative versions of these dependencies has not been tested and is not recommended. 

### FSL
FSL is a collection of tools and software used to process fMRI, MRI and DWI data. [Visit their installation page](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) for download and installation guide.

**FSL 6.0.1 is recommended.** All testing has been done using FSL 6.0.1. PyDesigner has not been tested with other versions of FSL.

To check your FSL version:

```
FLIRT -version
```

**If you are currently running an FSL version after v6.0.1:**

As of most recent FSL 6.0.3, `eddy` does not support CUDA 10, while `bedpost` and `probtrakx` do. Moreover, the version supplied after FSL v6.0.1 fails on certain datasets. If running on a CUDA system, users are advised to downgrade to CUDA 9.1 for maximum compatibility, and to do so prior to installing FSL.

After the installation of FSL, replace `eddy_cuda` with the one from [FSL v6.0.1](https://users.fmrib.ox.ac.uk/~thanayik/eddy_cuda9.1). Create a backup of original rename `eddy_cuda9.1` to `eddy_cuda` Then, make the file executable with `sudo chmod +x /path/to/eddy_cuda`.

Replace/Install [bedpostx for GPU](https://users.fmrib.ox.ac.uk/~moisesf/Bedpostx_GPU/Installation.html) for CUDA 9.1.

### MRTRIX3
MRTRIX3 is another software suite aimed at analysis of DWI data. Here are some of their helpful pages.
1. [Homepage](https://www.mrtrix.org/)
2. [Download and Install](https://www.mrtrix.org/download/)

To check your MRtrix version:

```
mrconvert - version
```

### Python
PyDesigner was built and tested on Python 3.7, so we enourage all users to adopt this version as well. While you may use the Python supplied by default on your OS, we highly enocurage users to adopt a Conda-based Python like [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/). Refer to either of these distributions' page for installation. This guide assumes a conda installation for setting up Python.

First, update conda with
```
conda update conda
```

Next, create a conda environment specifically for dMRI preprocessing, called `dmri`.

Creating a conda environment is recommended as this will keep all of the dependencies required for this project isolated to just the conda environment called `dmri`. For more information about conda environments, see [The Definitive Guide to Conda Environments](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533). 

If you prefer not to create this environment, skip to package installation. In addition, we'll be installing `pip` to this environment.

```
conda create -n dmri python=3.7
conda install -n dmri pip
```
Activate the new environment with:
```
conda activate dmri
```
**Note**: Environment activation (`conda activate dmri`) needs to be done each time a new terminal window is opened. If this behavior is undesired, you may set this environment as default python environment. 

Once the base environment is created and activated, proceed with the installation of all packages.

1. [NumPy](https://numpy.org/)
2. [SciPy](https://www.scipy.org/)
3. [CVXPY](https://www.cvxpy.org/)
4. [NiBabel](https://nipy.org/nibabel/)
5. [Multiprocessing](https://docs.python.org/3.4/library/multiprocessing.html?highlight=process)
6. [Joblib](https://joblib.readthedocs.io/en/latest/)
7. [TQDM](https://tqdm.github.io/)

Install necessary packages with the commands:
```
conda install -c anaconda numpy scipy joblib
conda install -c conda-forge tqdm nibabel multiprocess
pip install --upgrade setuptools
pip install cvxpy
```

If conda fails to install a package, use pip to install the package with:
```
pip install [package name]
```

Completion of this step will ready your system for dMRI processing. Let's go!

## PyDesigner

On the main PyDesigner Github page, click the green "Clone or download" button. Click "Download ZIP". When the download is complete, find the PyDesigner-master.zip in your Downloads folder and unzip. 

PyDesigner is located here: `/PyDesigner-master/designer/pydesigner.py`

## Running PyDesigner

**Before Running PyDesigner**

Ensure that all your DICOMS are converted to NifTi files and that all diffusion series have a valid `.json` file, as well as `.bvec` and `.bval` files where applicable. Dicom to nifti conversion can be done with [dcm2niix available for download here](https://github.com/rordenlab/dcm2niix). 

Ensure that none of your file or folder names contain a period (aside from the file extension; eg. DKI.nii). 

**To Run PyDesigner**

Switch to the appropriate conda environment; run `conda activate dmri` if you followed this guide. Then, for any given subject, call PyDesigner with the relevant flags:

```
python /Path/to/pydesigner.py --denoise --degibbs --smooth --rician --mask /Path/to/input_file.nii -o /Path/to/output/folder
```

**Note**: Flags can be added and removed as needed. It is recommended to always run PyDesigner with the `--mask` flag, as this flag utilizes a brain mask with excludes non-brain voxels and subsequently speeds up processing.

If your dataset contains more than one DKI average per subject, your file input may contain all relevant nifti files separated by a comma:

```
python /Path/to/pydesigner.py --denoise --degibbs --smooth --rician --mask /Path/to/DKI_avg_1.nii,/Path/to/DKI_avg_2.nii -o /Path/to/output/folder
```

**Note**: Multiple average inputs with additional interleved B0s can be given to PyDesigner but separate B0 sequences cannot.

If your dataset contains a top up sequence, you can use the `--topup` and `--undistort` flags:

```
python /Path/to/pydesigner.py --denoise --degibbs --smooth --rician --mask --topup /Path/to/reverse_phase.nii /Path/to/input_file.nii -o /Path/to/output/folder
```

**Note**: Using `--undistort` and `--topup` without supplying top up data will return an error.

**Basic PyDesigner Flags**

`--standard` - runs the standard pipeline (denoising, gibbs unringing, topup + eddy, b1 bias correction, CSF-excluded smoothing, rician bias correction, normalization to white matter in the first B0 image, IRWLLS, CWLLS DKI fit, outlier detection and removal)<br>
`--denoise` - performs denoising<br>
`--degibbs` - performs gibbs unringing correction<br>
`--smooth` - performs smoothing<br>
`--rician` - performs rician bias correction<br>
`--mask` - computes brain mask prior to tensor fitting; recommended<br>
`--undistort` - performs image undistortion via FSL eddy<br>
`--topup` - incorporates top up B0 series; required for `--undistort`<br>
`--o` - specifies output folder<br>
`--verbose` - prints out all output<br>
`--force` - overwrites existing files in output folder<br>

## Questions and Issues

For any questions not answered in the above documentation, see the contacts below.

To report any bugs or issues, see [the Issues tool on the PyDesigner GitHub page.](https://github.com/m-ama/PyDesigner/issues)

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

### 
<img src="https://avatars1.githubusercontent.com/u/47329645?s=460&v=4" align="left"
     title="GitHub: Kayti Keith" height="163">

     Kayti Keith

     Research Specialist
     Department of Neuroscience
     Medical University of South Carolina
     keithka@musc.edu

### Advisor
<img src="https://muschealth.org/MUSCApps/HealthAssets/ProfileImages/jej50.jpg" align="left"
     title="MUSC: Jens Jensen" height="163">

     Jens Jensen, Ph.D.

     Professor
     Department of Neuroscience
     Medical University of South Carolina
     <email placeholder>
     




