Usage
^^^^^

With PyDesigner installed and ready to run, let's floor the pedal.
Before Running PyDesigner

Ensure that all your DICOMS are converted to NifTi files and that all diffusion series have a valid .json file, as well as .bvec and .bval files where applicable. Dicom to nifti conversion can be done with dcm2niix available for download here.

Ensure that none of your file or folder names contain a period (aside from the file extension; eg. DKI.nii).
To Run PyDesigner

Switch to the appropriate conda environment; run conda activate dmri if you followed this guide. Then, for any given subject, call PyDesigner with the relevant flags:

python /Path/to/pydesigner.py \
--denoise \
--degibbs \
--smooth \
--rician \
--mask \
-o /Path/to/output/folder \
/Path/to/input_file.nii

Note: Flags can be added and removed as needed. It is recommended to always run PyDesigner with the --mask flag, as this flag utilizes a brain mask with excludes non-brain voxels and subsequently speeds up processing.

If your dataset contains more than one DKI average per subject, your file input may contain all relevant nifti files separated by a comma (no space superceding a comma):

python /Path/to/pydesigner.py \
--denoise \
--degibbs \
--smooth \
--rician \
--mask \
-o /Path/to/output/folder \
/Path/to/DKI_avg_1.nii,/Path/to/DKI_avg_2.nii

As long as all sequences come from the same acquisition with the same parameters (phase encoding direction, gradients, etc.), they can be combined to preprocess and produce DTI/DKI maps.

Note: Multiple average inputs with additional interleved B0s can be given to PyDesigner but suport for separate B0 sequences is experimental. See PR #84 for further information.

If your dataset contains a top up sequence, you can use the --topup and --undistort flags:

python /Path/to/pydesigner.py \
--denoise \
--degibbs \
--smooth \
--rician \
--mask \
--topup /Path/to/reverse_phase.nii \
-o /Path/to/output/folder \
/Path/to/input_file.nii

Note: Using --undistort and --topup without supplying top up data will return an error.
Basic PyDesigner Flags

Flags are to be preceeded by --. For example, to parse a denoise flag, one would type the flag as --denoise.
Flag 	Description
standard 	runs the standard pipeline (denoising, gibbs unringing, topup + eddy, b1 bias correction, CSF-excluded smoothing, rician bias correction, normalization to white matter in the first B0 image, IRWLLS, CWLLS DKI fit, outlier detection and removal)
denoise 	performs denoising
extent 	Denoising extent formatted n,n,n; (forces denoising) is specified
degibbs 	performs gibbs unringing correction
smooth 	performs smoothing
rician 	performs rician bias correction
mask 	computes brain mask prior to tensor fitting; recommended
maskthr 	FSL bet threshold used for brain masking; specify only when using --mask
undistort 	performs image undistortion via FSL eddy
topup 	performs EPI correction byincorporating topup B0 series; required for --undistort
o 	specifies output folder
force 	overwrites existing files in output folder
resume 	resumes processing from a previous state; only if same output folder
resume 	resumes processing from a previous state; only if same output folder
nofit 	preprocess only; does not perform tensor fitting and parameter extraction
noakc 	disables outlier correction on kurtosis fitting metrics
nooutliers 	disables IRWLLS outlier detection (not recommended for DKI)
fit_constraints 	specifies constraints for WLLS fitting; formatted n,n,n
verbose 	prints out all output: recommended for debugging
adv 	disables safety checks for advanced users who want to force a preprocessing step. WARNING: FOR ADVANCED USERS ONLY
