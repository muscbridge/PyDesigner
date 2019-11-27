# PyDesigner [DEVELOPMENTAL CYCLE]

[![Actions Status](https://github.com/m-ama/PyDesigner/workflows/Docker%20Build%20(Latest)/badge.svg)](https://github.com/m-ama/PyDesigner/commit/3b049c5f491ff33faf77116b135ce86e49189c27/checks?check_suite_id=332225619)
[![Actions Status](https://github.com/m-ama/PyDesigner/workflows/Docker%20Build%20(Release)/badge.svg)](https://github.com/m-ama/PyDesigner/actions?query=workflow%3A%22Docker+Build+%28Release%29%22)
[![Docker Pulls](https://img.shields.io/docker/pulls/dmri/neurodock?logo=docker)](https://hub.docker.com/r/dmri/neurodock)

**Project is currently under developmental cycle and is undergoing stability testing and debugging. Users are recommended to wait for a stable public release instead.**

<p align="center">
  <img src="https://i.imgur.com/Anc33XI.png" width="512">
</p>

**_Disclaimer:_**
```
This project is in early stages of development and most likely will not work as intended. 
We are not responsible for any data corruption, loss of valuable data, computer issues, you 
getting fired because you chose to run this, or a thermonuclear war. We strongly encourage 
all potential users to wait for an official release. In all seriousness, do NOT used this for
any data analysis, speculation, or writing papers: 1) not all citations have been incorporated, 
and 2) this is a developmental cycle with no papers to cite back on.
```

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
  <a href="http://www.diffusion-mri.com/">
    <img src="https://greatoakscharter.org/wp-content/uploads/2017/03/NYU-Logo.png"
         alt="NYU MRI Biophysics Group" width="256">
  </a>
</p>

## Table of Contents
**[Abstract](#pydesigner)**<br>
**[General Information](#general-information)**<br>
**[Introduction](#introduction)**<br>
[|__ The PyDesigner Pipeline](#the-pydesigner-pipeline)<br>
[|__ Image Acquisition](#image-acquisition)<br>
[|__ Preprocessing](#preprocessing)<br>
[|__ Tensor Estimation](#tensor-estimation)<br>
**[Installation](#installation)**<br>
[|__ FSL](#fsl)<br>
[|__ MRTRIX3](#mrtrix3)<br>
[|__ Python](#python)<br>
[|__ PyDesigner](#pydesigner)<br>
**[Running PyDesigner](#running-pydesigner)**<br>
[|__ Before Running PyDesigner](#before-running-pydesigner)<br>
[|__ To Run PyDesigner](#to-run-pydesigner)<br>
[|__ Basic PyDesigner Flags](#basic-pydesigner-flags)<br>
**[Docker Setup](#docker-setup)**<br>
[|__ Docker Installation](#docker-installation)<br>
[|__ Configure Docker](#configure-docker)<br>
**[Compiling Instructions](#compiling-instructions)**<br>
[|__ Via DockerHub](#via-dockerhub)<br>
[|__ Via GitHub](#via-github)<br>
**[Running NeuroDock](#running-neurodock)**<br>
[|__ SSH Mode](#ssh-mode)<br>
[|__ Command-basis mode](#command-basis-mode)<br>
[|__ Alias](#alias)<br>
**[Information for Developers](#information-for-developers)**<br>
[|__ General Pipeline Flow](#general-pipeline-flow)<br>
[|__ List of Files](#list-of-files)<br>
**[Future Plans](#future-plans)**<br>
**[Questions and Issues](#questions-and-issues)**<br>
**[Meet the Team](#meet-the-team)**<br>

## General Information
### Introduction
We here provide the code to estimate the diffusion kurtosis tensors from diffusion-weighted images. The (constrained) weighted linear least squares estimator is here preferred because of its accuracy and precision. See “Veraart, J., Sijbers, J., Sunaert, S., Leemans, A. & Jeurissen, B.,  Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage, 2013, 81, 335-346” for more details. Next, a set of diffusion and kurtosis parameters, including the white matter tract integrity metrics, can be calculated form the resulting kurtosis tensor.

Some important notes needs to be considered:

1. Since the apparent diffusion tensor has 6 independent elements and the kurtosis tensor has 15 elements, there is a total of 21 parameters to be estimated. As an additional degree of freedom is associated with the noise free nondiffusion-weighted signal, at least 22 diffusion-weighted images must be acquired for DKI. It can be further shown that there must be at least three distinct b-values, which only differ in the gradient magnitude. Furthermore, at least 15 distinct diffusion (gradient) directions are required (Jensen et al. 2005). Some additional consideration must be made.  The maximal b-value should be chosen carefully and is a trade-off between accuracy and precision. While for DTI, diffusion-weighted images are typically acquired with rather low b-values, about 1000 s⁄mm^2 , somewhat stronger diffusion sensitizing gradients need to be applied for DKI as the quadratic term in the b-value needs to be apparent. It is shown that b-values of about 2000 s⁄mm^2  are sufficient to measure the degree of non-Gaussianity with an acceptable precision (Jensen & Helpern 2010). 

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

**If you are currently running an FSL version after v6.0.1 and using NVIDIA CUDA:**

As of most recent FSL 6.0.3, `eddy` does not support CUDA 10, while `bedpost` and `probtrakx` do. Moreover, the version supplied after FSL v6.0.1 fails on certain datasets. If running on a CUDA system, users are advised to downgrade to CUDA 9.1 for maximum compatibility, and to do so prior to installing FSL.

After the installation of FSL, Create a backup of original `eddy_cuda9.1` by renaming it to to `eddy_cuda9.1.BAK`. Then, replace `eddy_cuda9.1` with the one from [FSL v6.0.1](https://users.fmrib.ox.ac.uk/~thanayik/eddy_cuda9.1). End by making the new file executable with `sudo chmod +x /path/to/eddy_cuda9.1`.

Replace/Install [bedpostx for GPU](https://users.fmrib.ox.ac.uk/~moisesf/Bedpostx_GPU/Installation.html) for CUDA 9.1.

### MRTRIX3
MRTRIX3 is another software suite aimed at analysis of DWI data. Here are some of their helpful pages.
1. [Homepage](https://www.mrtrix.org/)
2. [Download and Install](https://www.mrtrix.org/download/)

To check your MRtrix version:

```
mrconvert -version
```

### Python
PyDesigner was built and tested on Python 3.7, so we enourage all users to adopt this version as well. While you may use the Python supplied by default on your OS, we highly enocurage users to adopt a Conda-based Python like [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/). Conda is a command line tool that allows the creation of separated environments with different python versions and packages. This of it as running multiple virtual machines on the a single host - you can easily switch between any for different needs, or run them simultaneously.

Refer to either of these distributions' page for installation. This guide assumes a conda (Miniconda) installation for setting up Python. If you already have conda, or prefer using the default Python supplied by your OS, skip to package installation at the end of this subsection.

First, update conda with
```
conda update conda
```

Creating a conda environment is recommended as this will keep all of the dependencies required for this project isolated to just the conda environment called `dmri`. For more information about conda environments, see [The Definitive Guide to Conda Environments](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533). Next, create a conda environment specifically for dMRI preprocessing, called `dmri`. You can choose any name, but be sure to replace `dmri` in this guide with the name of your choice.

If you prefer not to create this environment, skip to package installation. In addition, we'll be installing `pip` to this environment, in the even `conda install <package name>` fails to compile and install a package.

```
conda create -n dmri python=3.7
conda install -n dmri pip
```
Activate the new environment with:
```
conda activate dmri
```
**Note**: Environment activation (`conda activate dmri`) needs to be done each time a new terminal window is opened. If this behavior is undesired, you may set this environment as default python environment. Refer to advanced conda user guides or Google search how to do this.

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

### PyDesigner
On the [main PyDesigner Github page](https://github.com/m-ama/PyDesigner), click the green "Clone or download" button to access the latest PyDesigner build. Click "Download ZIP". When the download is complete, find the PyDesigner-master.zip in your Downloads folder and unzip. 

PyDesigner is located here: `/PyDesigner-master/designer/pydesigner.py`

**Note:** If you need a stable and tested build, download the most recent release from the [Release tab](https://github.com/m-ama/PyDesigner/releases). Click on `Source code (zip)` link and decompress (unzip) to any folder you desire.

## Running PyDesigner
With PyDesigner installed and ready to run, let's floor the pedal.

### Before Running PyDesigner
Ensure that all your DICOMS are converted to NifTi files and that all diffusion series have a valid `.json` file, as well as `.bvec` and `.bval` files where applicable. Dicom to nifti conversion can be done with [dcm2niix available for download here](https://github.com/rordenlab/dcm2niix). 

Ensure that none of your file or folder names contain a period (aside from the file extension; eg. DKI.nii). 

### To Run PyDesigner
Switch to the appropriate conda environment; run `conda activate dmri` if you followed this guide. Then, for any given subject, call PyDesigner with the relevant flags:

```
python /Path/to/pydesigner.py \
--denoise \
--degibbs \
--smooth \
--rician \
--mask \
-o /Path/to/output/folder \
/Path/to/input_file.nii
```

**Note**: Flags can be added and removed as needed. It is recommended to always run PyDesigner with the `--mask` flag, as this flag utilizes a brain mask with excludes non-brain voxels and subsequently speeds up processing.

If your dataset contains more than one DKI average per subject, your file input may contain all relevant nifti files separated by a comma (no space superceding a comma):

```
python /Path/to/pydesigner.py \
--denoise \
--degibbs \
--smooth \
--rician \
--mask \
-o /Path/to/output/folder \
/Path/to/DKI_avg_1.nii,/Path/to/DKI_avg_2.nii
```

As long as all sequences come from the same acquisition with the same parameters (phase encoding direction, gradients, etc.), they can be combined to preprocess and produce DTI/DKI maps.

**Note**: Multiple average inputs with additional interleved B0s can be given to PyDesigner but suport for separate B0 sequences is experimental. See [PR #84](https://github.com/m-ama/PyDesigner/pull/84) for further information.

If your dataset contains a top up sequence, you can use the `--topup` and `--undistort` flags:

```
python /Path/to/pydesigner.py \
--denoise \
--degibbs \
--smooth \
--rician \
--mask \
--topup /Path/to/reverse_phase.nii \
-o /Path/to/output/folder \
/Path/to/input_file.nii
```

**Note**: Using `--undistort` and `--topup` without supplying top up data will return an error.

### Basic PyDesigner Flags
Flags are to be preceeded by `--`. For example, to parse a _denoise_ flag, one would type the flag as `--denoise`.

  | Flag        | Description |
  | :---------- | :- |
  |`standard` | runs the standard pipeline (denoising, gibbs unringing, topup + eddy, b1 bias correction, CSF-excluded smoothing, rician bias correction, normalization to white matter in the first B0 image, IRWLLS, CWLLS DKI fit, outlier detection and removal) |
  |`denoise`  |performs denoising|
  |`extent`   |Denoising extent formatted n,n,n; (forces denoising) is specified|
  |`degibbs`  |performs gibbs unringing correction|
  |`smooth`   |performs smoothing|
  |`rician`   |performs rician bias correction|
  |`mask`     |computes brain mask prior to tensor fitting; recommended|
  |`maskthr`  |FSL bet threshold used for brain masking; specify only when using `--mask`|
  |`undistort`|performs image undistortion via FSL eddy|
  |`topup`    | performs EPI correction byincorporating topup B0 series; required for `--undistort`|
  |`o`        |specifies output folder|
  |`force`    |overwrites existing files in output folder|
  |`resume`   |resumes processing from a previous state; only if same output folder|
  |`resume`   |resumes processing from a previous state; only if same output folder|
  |`nofit`    |preprocess only; does not perform tensor fitting and parameter extraction|
  |`noakc`    |disables outlier correction on kurtosis fitting metrics|
  |`nooutliers`|disables IRWLLS outlier detection (not recommended for DKI)|
  |`fit_constraints`|specifies constraints for WLLS fitting; formatted n,n,n|
  |`verbose`  |prints out all output: recommended for debugging|
  |`adv`      |disables safety checks for advanced users who want to force a preprocessing step. **WARNING: FOR ADVANCED USERS ONLY**|

## Docker Setup

<p align="center">
  <a href="https://github.com/m-ama/PyDesigner">
    <img src="https://i.imgur.com/ktvRd1Y.png" alt="MUSC DKI Page" width="256">
   </a>
</p>

**Before proceeding, please ensure that your hardware is capable of virtualization. Most modern Intel and AMD processors are compatible. 16 GB minimum RAM is recommended.**

### Docker Installation
The first step is actually installing Docker. One can easily download and install this from the [Docker Desktop](https://www.docker.com/products/docker-desktop) page by following instructions for respective platforms.

**Windows Users**: Please ensure that Hyper-V is enabled on your machine by following the guide [here](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/quick-start/enable-hyper-v).

### Configure Docker
Run docker after installation, then head over to docker preferences by opening the **Docker menu > Preferences...**, via the Docker icon in your taskbar.

<p align="center">
    <img src="https://docs.docker.com/docker-for-mac/images/menu/prefs.png"
         alt="Docker preferences button" width="256">
  </a>
</p>

Head over to the advanced tab and select the maximum number of CPU to make available to Docker. For dMRI processing, upward of 4 is recommended. Then allocate 16.0 GiB or more memory with 1.0 GiB of Swap.

<p align="center">
    <img src="https://docs.docker.com/docker-for-mac/images/menu/prefs-advanced.png"
         alt="Docker advanced preferences" width="640">
  </a>
</p>

Users are encouraged not to allocate all their physical CPUs to Docker. As a guideline, it is okay to allcate up to 75% of all physical cores.

### Compiling Instructions
There are two ways to build a the NeuroDock docker image, through the [GitHub repository](https://github.com/m-ama/PyDesigner) or via [DockerHub](https://hub.docker.com/r/dmri/neurodock), where the latter is highly recommended.

#### Via DockerHub
DockerHub is a repository of Docker images and containers for every occassion.
The greatest advantage to using DockerHub is the ease and speed of building a container, where a simple commands can fetch a precompiled image. One can easily build NeuroDocker with the [`docker pull`](https://docs.docker.com/engine/reference/commandline/pull/) command:

```
docker pull dmri/neurodock
```

Total estimated download for `dmri/neurodock:latest` is approximately 6.0 GB.

#### Via GitHub
To build a docker image via the GitHub repo, begin by cloning the repository to your local machine and take note of its path.

Then, in your command line, type:

**To build latest from master:**
```
docker build -t <image_name> -f <path_to_repo>/docker/Dockerfile_latest <path_to_repo>/docker
```
Builds from master may not be stable, so we highly recommend building from latest release.


**To build from latest release:**
```
docker build -t <image_name> -f <path_to_repo>/docker/Dockerfile_release <path_to_repo>/docker
```


Where `<image_name>` is the name given to the container, and `<path_to_repo>` is the absolute path where the NeuroDock repository is located. For additional building information, please see the [`docker build`](https://docs.docker.com/engine/reference/commandline/build/) documentation.

As a real example, here is the command parsed on one of the lab's computers to build this Docker image:
```
docker build -t neurodock -f /Users/sid/Repos/PyDesigner/docker/Dockerfile_release /Users/sid/Repos/PyDesigner/docker
```

Total buildtime on an 8-core/16-thread machine is approximately 45 minutes.

### Running NeuroDock
Congratulations on building NeuroDock! Let's proceed to container usage, it's fairly simple!

There are two ways to run NeuroDock, either in SSH mode or on a command-basis. SSH mode is akin to having a command-line window entirely to NeuroDock via an SSH. The latter allows NeuroDock to behave like an installed package which allows FSL, MRtrix3 and PyDesigner commands to be called from command line.

Both methods are initiated by the [`docker run`](https://docs.docker.com/engine/reference/run/) command.

#### SSH Mode
Simply execute the command below to run NeuroDock in SSH mode:
```
docker run -it --rm -v <source>:<destination> dmri/neurodock
```
 It is higly recommended to run NeuroDock with the cleanup flag `--rm`, which cleans up and removes the filesystem from Docker when it exits to prevent storage waste.

 The `-v ` allows one to [bind mount](https://docs.docker.com/storage/bind-mounts/) a local folder/drive to make it visible to the container. This folder is mounted at the `<destination>`.

 A real-world usage command to demonstrate this is given below. This command mounts the local folder `/Users/sid/Documents/Projects` to `/Data` within the container.

 ```
 docker run -it --rm -v /Users/sid/Documents/Projects:/Data dmri/neurodock
 ```

 Once in SSH mode, the terminal window will only accept commands from the NeuroDock container - these are FSL, MRtrix3 and PyDesigner commands. To leave exit the container and leave SSH mode, simply type `exit` to leave the container.

 #### Command-basis mode
 This mode of running a NeuroDock allows one to run FSL, MRtrix3, or PyDesigner commands easily without executing the entire container in SSH mode. The container will exit as soon as the command given terminates. The syntax is the same as running in SSH mode, with the addition of the command. For example, to call FSL's bet, one would run:

```
docker run -it --rm -v <source>:<destination> dmri/neurodock bet -m -t 0.25 <destnation/path_to_input> <destination/path_to_output>
```

Similarly, one can run PyDesigner with:
```
docker run -it --rm -v <source>:<destination> dmri/neurodock python3 pydesigner.py [options] inputs
```

Here's a real example:
```
docker run -it --rm \
-v /Users/sid/Documents/Projects/IAM/Dataset/NifTi/IAM_1122:/Proc \
python3 pydesigner.py  \
--denoise \
--rician \
--degibbs \
--smooth \
--undistort --rpe_none \
--mask \
--verbose \
-o /Proc/pyd \
/Proc/dwi/IAM_1122_15_DKI_BIPOLAR_2_5mm_64dir_50slices.nii
```

#### Alias
It is cumbersome to add long commands such those above each time. An alias acts as a shortcut to a command, thus allowing us to shorten a command drastically. To call PyDesigner in a docker environment, an alias can be set up like this:

```
alias pydesigner="docker run -it --rm -v <source>:<destination> dmri/neurodock python3 pydesigner.py"
```

Now, simply parsing the command `pydesigner` into your CLI will execute PyDesigner in a container. Running `pydesigner -h` will open up the PyDesigner help text.

Take note that the alias is assigned to a specific mounted volume defined by the flag `-v <source>:<destination>`. To process several subjects, it is advisable to mount an entire drive to the alias.

## Information for Developers
This sections covers information on the various files provided in this package. Users contributing to the project should refer to this section to understand the order of operations.

### General Pipeline Flow
The pipeline is designed to process NifTi acquisitions as a starting point, and end with DTI/DKI maps. This pipeline can be broken down into two important segments:

1. Preprocessing
2. Tensor estimation

Each segment is responsible for unique sets of computations to produce useful metrics.

### List of Files
There are several files in this package that allow the two segemnts to flow smoothly. The table in this section lists all these files and their purpose.

| File | Purpose |
| :---------- | :- |
|  **Main Script** |
| `pydesigner.py` | main PyDesigner script that controls preprocessing steps and tensor fitting |
| `Tensor_Fitting_Guide.ipynb` | a Jupyter Notebook that details functioning of `fitting/dwi.py` for tensor fitting only|
| **Preprocessing** | found in `PyDesigner/designer/preprocessing` |
| `preparation.py` | adds utilities for preparing the data for eddy and analysis |
| `rician.py` | performs Rician correction on a DWI with a noisemap from MRTRIX3's `dwipreproc` |
| `smoothing.py` | applies nan-smoothing to input DWI |
| `ulti.py` | utilities for the command-line interface and file I/O handling |
| **Tensor Estimation** | found in `PyDesigner/designer/fitting/`|
| `dirs30.csv` | csv file containing 30 static directions for constraint creation |
| `dirs256.csv` | csv file containing 250 static directions for parameter extraction  |
| `dirs10000.csv` | csv file containing 10,000 static directions for AKC correction and WMTI parameter extraction |
| `dwidirs.py` | handles loading of static directions into `np.array` |
| `dwipi.py` | main tensor fitting script to handle IRWLLS, WLLS, parameter extraction and filtering |
| **Extras**  |  found in 'PyDesigner/Extras'  |
| `des2dke.m` | legacy MATLAB script for converting PyDesigner processed file to DKE-compatible input for validation |
| `dke_parameters.txt` | DKE parameters file, used by `des2dke.m` to activate DKE compatibility |

## Future Plans
PyDesigner is still in early stages of development. Release of a stable build will allow us to explore extending the pipeline even further with the inclusion of (in no particular order of preference):

1. Publishing PyDesigner on [PyPi](https://pypi.org/) and [Conda](https://docs.conda.io/en/latest/)
2. ~~Docker container with FSL, MRTRIX3 and Python dependencies for deployment on HPC clusters and cross-platform compute capabilites across Linux, Mac OS and Microsoft Windows~~
3. Fiber ball imaging (FBI) for microstructural parameters. See [Fiber ball imaging](https://www.ncbi.nlm.nih.gov/pubmed/26432187), and [modeling white matter microstructure with fiber ball imaging](https://www.ncbi.nlm.nih.gov/pubmed/29660512) for more information
4. Deterministic and probabilistic tractography

## Questions and Issues
For any questions not answered in the above documentation, see the contacts below.

To report any bugs or issues, see [the Issues tool on the PyDesigner GitHub page.](https://github.com/m-ama/PyDesigner/issues)

## Meet the Team
PyDesigner is a joint collarobation and as such consists of several developers.

### Developers
<img src="https://avatars3.githubusercontent.com/u/13654344?s=400&u=c318d7dcc292486b87bc5c7e81bd8e02947d834e&v=4" align="left"
     title="GitHub: Siddhartha Dhiman" height="163"> 

    Siddhartha Dhiman, MSc

    Research Specialist
    MUSC Advanced Image Analysis
    Department of Neuroscience
    Medical University of South Carolina<

<img src="https://avatars2.githubusercontent.com/u/26722533?s=400&v=4" align="right"
     title="GitHub: Joshua Teves" height="163"> 

     Joshua Teves, BSc

     Systems Programmer
     MUSC Advanced Image Analysis
     Department of Neuroscience
     Medical University of South Carolina

<img src="https://avatars1.githubusercontent.com/u/47329645?s=460&v=4" align="left"
     title="GitHub: Kayti Keith" height="163">

     Kayti Keith, BSc

     Research Specialist
     MUSC Advanced Image Analysis
     Department of Neuroscience
     Medical University of South Carolina

<img src="http://www.diffusion-mri.com/sites/default/files/styles/medium/public/pictures/picture-48-1455813197.jpg?itok=B4goKbp-" align="right"
  title="NYU MRI Biophysics Group: Benjamin Ades-Aron" height="163">

    Benjamin Ades-Aron, MSc

    PhD Student
    MRI Biophysics Group
    NYU School of Medicine
    New York University

### Advisors
<img src="https://muschealth.org/MUSCApps/HealthAssets/ProfileImages/jej50.jpg" align="left"
     title="MUSC: Jens Jensen" height="163">

     Jens Jensen, PhD

     Professor
     MUSC Advanced Image Analysis
     Department of Neuroscience
     Medical University of South Carolina

<img src="http://www.diffusion-mri.com/sites/default/files/styles/medium/public/pictures/picture-2-1455646448.jpg?itok=ppO8NJs6" align="right"
     title="NYU MRI Biophysics Group: Els Fieremans" height="163">

     Els Fieremans, PhD

     Assistant Professor
     MRI Biophysics Group
     NYU School of Medicine
     New York University

<img src="http://www.diffusion-mri.com/sites/default/files/styles/medium/public/pictures/picture-45-1455813153.jpg?itok=jnWxCQol" align="left"
  title="NYU MRI Biophysics Group: Jelle Veraart" height="163">

    Jelle Veraart, PhD

    Postdoctoral Researcher
    MRI Biophysics Group
    NYU School of Medicine
    New York University

<img src="https://media.licdn.com/dms/image/C4D03AQHepgjpxgV2Uw/profile-displayphoto-shrink_800_800/0?e=1577318400&v=beta&t=k9IEREGye7VLB2FkNFzOVBFl1RJW1Rydt5JKh1V4oFk" align="right"
  title="MUSC: Vitria Adisetiyo" height="163">

    Vitria Adisetiyo, PhD

    Senior Scientist
    MUSC Advanced Image Analysis
    Department of Neuroscience
    Medical University of South Carolina