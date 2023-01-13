List of Flags
=============

PyDesigner is extremely flexible when it comes to dMRI processing.
Users can easily enable or disable various preprocessing steps without
changing the overall sequence.

The list below covers all these flags.

IO Control
---------------

These flags allow control of the pipeline's I/O handling

-o DIR, --output DIR    PyDesigner output directory


Preprocessing Control
---------------------

Preprocessing contol flags allow users to tweak certain parts of the
preprocessing pipeline, to accomodate all types of datasets.


-s, --standard      Runs the recommended preprocessing pipeline in order: denoise, degibbs, undistort, brain mask, smooth, rician

-n, --denoise       Denoises input DWI

--extent            Shape of denoising extent matrix, defaults to 5,5,5

--reslice xyz       Reslices DWI to voxel resolution specified in miliimeters (mm) or output dimensions. Performinh reslicing will skip plotting of SNR curves. Dimensions less than 9 will reslice in mm, image dimension otherwise. Specify argument as `--reslice x,y,z`.

--interp CHOICE     Set the interpolation to use when reslicing. Choices are linear (default), nearest, cubic, and sinc

-g, --degibbs       Corrects Gibbs ringing

-u, --undistort     Undistorts image using a suite of EPI distortion correction, eddy current correction, and co-registration. Does not run EPI correction if reverse phase encoding DWI is absent.

--rpe_pairs N       Speeds up TOPUP if a reverse PE is present; specify the number (integer) of reverse PE direction B0 pairs to use.

--mask              Computes a brain mask at 0.20 threshold by default

--maskthr           Specify FSL bet fractional intensity threshold for brain masking, defaults to 0.20

--user_mask         Provide path to user-generated brain mask in NifTi (.nii) format

-cf, --csf_fsl      Compute a CSF mask for CSF-excluded smoothing to minimize partial volume effects using FSL fit_constraints

-cd, --csf_adc      Compute CSF mask for CSF-excluded smoothing to minimize partial volume effects by thresholding a pseudo-ADC map computed as ln(S0/S1000)/b1000. N is the ADC thresold to use (default: 2).

-z, --smooth        Smooths DWI data at a default FWHM of 1.25

--fwhm              Specify the FWHM at which to smooth, defaults to 1.25

-r, --rician        Corrects Rician bias

-te, --multite      Enable multi-TE support. This mode preprocesses all concatenated DWIs together, but performs tensor fitting separately.


Diffusion Tensor Control
------------------------

Users may also tweak computations in estimating DTI or DKI parameters
with the following flags.

--nofit             Performs preprocessing only, disables DTI/DKI parameter extraction

--noakc             Performs brute forced kurtosis tensor outlier rejection

--nooutliers        Disables IRLLS outlier detection

--fit_constraints   Specify fitting constraints to use, defaults to 0,1,0

--noqc              Disables saving of quality control (QC) metrics

--median            Performs post processing median filter of final DTI/DKI maps. **WARNING: Use on a case-by-case basis for bad data only. When applied, the filter alters the values of most voxels, so it should be used with caution and avoided when data quality is otherwise adequate. While maps appear visually soother with this flag on, they may nonetheless be less accurate**

Fiber Ball Imaging (FBI) Control
--------------------------------

FBI parameters may be fine-tuned with the following flags.

--l_max n     Maximum spherical harmonic degree used in spherical harmonic expansion for fODF calculation

--no_rectify  Disable rectification of FBI fODFs in instances where it does more harm than good. In rare instances, fODFs computed from FBI acquisitions can be degraded from rectification - this flag disables rectification for such datasets.

Pipeline Control
----------------

These are more general pipeline flags that interface directly with the
user or machine.

--nthreads n    Specify number of CPU workers to use in processing, defaults to all physically available workers

--resume        Resumes preprocessing from an aborted or partial previous run

--force         Forces overwrite of existing output files

--verbose       Displays console output

--adv           Diables safety check to force run certain preprocessing steps **WARNING: This flag is for advanced users only who fully understand the MRI system and its outputs. Running with this flag could potentially yield inaccuracies in resulting DTI/DKI metrics**
