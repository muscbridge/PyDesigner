List of PyDesigner Flags
========================

PyDesigner is extremely flexible when it comes to dMRI processing.
Users can easily enable or disable various preocessing steps without
changing the overall sequence.

THe list below covers all these flags.

+--------------------------------+--------------------------------------------------------------------+
| Flag                           |                                                                    |
|                                | Usage                                                              |
+--------------------------------+--------------------------------------------------------------------+
|                                |                                                                    |
+--------------------------------+--------------------------------------------------------------------+
| :code:`-o`, :code:`--output`   | pydesigner output directory                                        |
+--------------------------------+--------------------------------------------------------------------+
| :code:`-s`, :code:`--standard` |                                                                    |
|                                | runs the recommended preprocessing pipeline in order:              |
|                                | denoise, degibbs, undistort, brain mask, smooth, rician            |
+--------------------------------+--------------------------------------------------------------------+
| **Preprocessing Control**      |                                                                    |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--denoise`              | denoises input DWI                                                 |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--extent`               | shape of denoising extent matrix; default: 5,5,5                   |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--degibbs`              | corrects Gibb's ringing                                            |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--undistort`            |                                                                    |
|                                | undistorts image using a suite of EPI distortion correction,       |
|                                |                                                                    |
|                                | eddy current correction, and co-registration. Does not run EPI     |
|                                | correction if reverse phase encoding DWI is absent                 |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--epiboost`             |                                                                    |
|                                | speeds up topup if a reverse PE is present; specify the integer(s) |
|                                | index of B0 volume to use                                          |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--mask`                 | computes a brain mask at 0.20 threshold                            |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--maskthr`              |                                                                    |
|                                | specify FSL bet fractional intensity threshold for brain masking;  |
|                                | default: 0.20                                                      |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--user_mask`            | provide path to user-generated brain mask in NifTi (.nii) format   |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--smooth`               | smooths DWI data at a default FWHM of 1.25                         |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--fwhm`                 | specify the FWHM at which to smooth; default 1.25                  |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--rician`               | corrects Rician bias                                               |
+--------------------------------+--------------------------------------------------------------------+
| **Postprocessing Control**     |                                                                    |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--nofit`                |                                                                    |
|                                | performs preprocessing only; disables DTI/DKI parameter extraction |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--noakc`                | disables brute forced kurtosis tensor outlier rejection            |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--nooutliers`           | disables IRLLS outlier detection                                   |
+--------------------------------+--------------------------------------------------------------------+
| :code:`-w`, :code:`--wmti`     | computes white matter tract integrity (WMTI) metrics               |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--fit_constraints`      | specify fitting constraints to use; default: 0,1,0                 |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--noqc`                 | disables saving of quality control (QC) metrics                    |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--median`               | performs post processing median filter of final DTI/DKI maps       |
+--------------------------------+--------------------------------------------------------------------+
| **System Control**             |                                                                    |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--nthreads`             | specify number of CPU workers to use in processing; default: all   |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--resume`               | resumes preprocessing from an aborted or partial previous run      |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--force`                | forces overwrite of existing output files                          |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--verbose`              | displays console output                                            |
+--------------------------------+--------------------------------------------------------------------+
| :code:`--adv`                  | disables safety check to force run certain preprocessing steps     |
|                                | THIS FLAG IS FOR ADVANCED USERS ONLY WHO FULLY ','UNDERSTAND       |
|                                | THE MRI SYSTEM AND ITS OUTPUTS. ','RUNNING WITH THIS FLAG          |
|                                | COULD POTENTIALLY ','RESULT IN IMPRECISE AND INACCURATE RESULTS.'  |
+--------------------------------+--------------------------------------------------------------------+