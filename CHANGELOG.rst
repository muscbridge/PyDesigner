Changelog
=========

All notable changes to this project will be documented in this file or
page

`v1.0.0`_
------------

Jan 13, 2023

**Added**

* Tractography modules `tractography.dsistudio`, `tractography.odf`,
  `tractography.sphericalsampling`
* ODF computations and spherical harmonic expansion for DTI and DKI
* Option to add user-defined map for tractography stopping criteria
* Added option to import multiple custom maps into DSI studio file
* PyDesigner can now be pulled from PyPI with `pip install pydesigner`

**Changed**

* Fixed a logic in multi-TE detection algorithm that prevented certain
  datasets from processing
* Overhauled how inputs paths are entered. Paths to input DWIs can now
  be provided to PyDesigner without comma separation
* Udpate `des2dke.m` for compatibility with nii_preprocess
* Replaced FBI SH with tesseral SH

**Removed**

* None

`v1.0-RC12`_
------------

Jan 14, 2022

**Added**

* None

**Changed**

* Fixed a logic in multi-TE detection algorithm that prevented certain
  datasets from processing
* Overhauled how inputs paths are entered. Paths to input DWIs can now
  be provided to PyDesigner without comma separation

**Removed**

* None

`v1.0-RC11`_
------------

Nov 30, 2021

**Added**:

* CSF excluded smoothing to minimize partial volume effect (PVE).
  Two methods to do this have been implemented - (1) `-cf or --csf_fsl`
  using FSL FAST segmentation, and (2) `-cd or --csf_adc n` using
  pseudo-ADC threshold of more than 2 (ADC > 2).
* Various other support functions such as `mrpreproc.csfmask()` and
  `mrinfoutil.shells()` to support CSF masking. These functions can
  also be used for other applications
* User provided brain mask is now rotated to the same orientation as
  input DWI to prevent fitting errors from incorrect masking.

**Changed**

* Overhaul of preprocessing.smoothing to enable NaN-smoothing
* B0 volumes are now excluded from IRLLS outlier detection to ensure
  there are some minimum volumes present in tensor fitting. This
  prevents various fitting errors.

**Removed**

* None

`v1.0-RC10`_
------------

Jun 29, 2021

**Added**:

* Support for multi echo time (TE) datasets. PyDesigner will now
  preprocess DWIs with multiple TEs together, but extract diffusion
  metrics for each TE separately. Users need to parse :code:`-te`
  flag to enable this feature.
* Added :code:`dwiextract` function to *mrpreproc.py* to allowing
  splitting of *.mif* files.
* Added function :code:`fit_regime` to *dwipy.py* to automatically run
  all tensor fitting steps in an appropriate manner.
* Added :code:`highprecisionpower` to *dwipy.py* to mitigate integer
  overflow error when performing FBI fODF calculation.
* Flag :code:`--no_rectify` to disable rectification of FBI fODFs. In
  some cases where FBI acquistion is excellent, rectification can
  degrade fODFs instead. This flag is intended to disable
  rectification of such datasets.


**Changed**

* Maximum DKI b-value threshold has been raised to 3,000 mm/s^2,
  thereby enabling DKI support for researchers using b-values higher
  than 2,000 mm/s^2 but less than 3,000 mm/s^2.
* IRLLS now also includes B0 volumes when evaluating goodness-of-fit
  to make outlier detection more robust and accurate.
* Various stability patches for FBI and FBWM to ensure error-free
  extraction of FBI/FBWM metrics.

**Removed**

* None

`v1.0-RC9`_
-----------

Mar 16, 2021

**Added**

* None

**Changed**

* B-values are first rounded to a float insted of integer directly to
  prevent errors in preprocessing

**Removed**

* None

`v1.0-RC8`_
-----------

Feb 15, 2021

**Added**

* Added missing Rician preprocessing to :code:`-s, --standard`
  preprocessing

**Changed**

* Potential sources of errors in FBWM have been mitigated
  with error-handling

**Removed**

* None

`v1.0-RC7`_
-----------

Feb 11, 2021

**Added**

* Missing Docker figures in RTD documentation

**Changed**

* Added error mitigation when FBI cost function fails to converge to
  a minimum cost
* Updated WMTI calculation to follow DKE outputs

**Removed**

* Unnecessary WMTI calculations


`v1.0-RC6`_
-----------

Dec 22, 2020

**Added**

* None

**Changed**

* Replaced ``preprocessing.util.bvec_is_fullsphere()`` and 
  ``preprocessing.util.vecs_are_fullsphere()`` with 
  ``preprocessing.mrinfoutil.is_fullsphere()``. Even though datasets
  may be half-shelled, it is inaccurate to label them as such because
  distortion relative to b-value is not linear. As such, the
  ``slm=linear`` makes no sense. This new method performs the proper
  checks required before labelling a DWI as fully-shelled. A DWI is
  half-shelled iff max B-value is less than 3000 AND the norm of the
  mean direction vector is more than 0.3.

**Removed**

* See above


`v1.0-RC5`_
-----------

Oct 26, 2020

**Added**

* Check for b-value scaling so .bval file so values
  specified as either 2.0 or 2000 can be processed.
* ``fitting.dwipy()`` can now be pointed to user-defined
  bvec and bval paths. It previously required bvec and
  bval files to have the same name and path as DWI.
* **DSI Studio tractography** for FBI. Processing FBI dataset now
  produces an ``fbi_tractography_dsi.fib`` file that can be loaded
  into DSI Studio to perform tractography.

**Changed**

* Fixed issue where eddy correction would attempt
  to QC and fail despite parsing the ``--noqc`` flag.
* SNR plotting works in very specific scenarious when
  input DWIs are of the same same dimensions. A try/except
  loop now ensure that the entire pipeline doesn't halt
  due to errors in plotting.

**Removed**:

* None

`v1.0-RC4`_
-----------

Sep 22, 2020

**Added**

* Reslicing compatibility udpated for new MRTrix3 version
  where ``mrrelice`` has been changed to ``mrgrid``.
  PyDesigner will work with either versions.

**Changed**

* Fixed a bad indent in tensor reordering function
  that produced an error in DTI protocols.

**Removed**

* None

`v1.0-RC3`_
-----------

Sep 21, 2020

**Added**

* FBI fODF map for FBI tractography. Users may use MRTrix3
  to further process this file.
* Variable maximum spherical harmonic degree to improve
  robustness of FBI fit. This was fixed at 6 previous, but has
  been defaulted to 6 now. Users may change l_max with the
  ``-l_max n`` flag. This is based on
  information found at https://mrtrix.readthedocs.io/en/dev/concepts/sh_basis_lmax.html

**Changed**

* None

**Removed**

* None

`v1.0-RC2`_
-----------

Aug 25, 2020

**Added**

* References to README.rst

**Changed**

* The minimum B-value required for FBI (4000) is now inclusive
  instead of exclusive. This would allow executiong of FBI/FBWM
  for datasets with b=4000 mm/s^2
* Convert variable ``nthreads`` to string so ``subproces.run``
  can recognize the flag
* Updated Slack permalink in README.rst

**Removed**

* None

`v1.0-RC1`_
-----------

Aug 19, 2020

**Added**

* Methods to perform tensor only with compatible B-values. PyDesigner
  previously use all B-values in a DWI to do so. This behavior has
  been updated to use only B-values less than 2500
* FBI and FBWM calculations
* Brief documentation on how to run PyDesigner

**Changed**

* Automatically issues ``dwipreproc`` or ``dwifslpreproc`` for
  compatibility with MRtrix3 >= 3.0.1
* Updated minimum version for required Python modules

**Removed**

* None

`v0.32`_
--------

Apr 21, 2020

**Added**

* Intrinsic inter-axonal and mean extra-axonal diffusivity
  calculation to WMTI

**Changed**

* Method ``json2fslgrad`` converted from class method to function
  definition
* ``json2fslgrad`` now transposes B0s in BVAL file in accordance with
  FSL's gradient scheme
* Documentation update
* ``Extras`` directory renamed to ``extras``
* DKE conversion scripts modified to correctly create ft and dke
  parameter files

**Removed**

* None

`v0.31`_
--------

Apr 9, 2020

**Added**

* NaN check in AWF calculculation that prevents further errors in intra-axonal
  and extra-axonal WMTI metrics computation

**Changed**

* ``designer.fitting.dwipy`` input file detection method
* ``Dockerfile_release`` now deletes the correct temporary file to prevent build
  error

**Removed**

* None

`v0.3`_
--------

Apr 8, 2020

**Added**

* Head motion plot from on eddy_qc outputs
* Outlier plot from IRRLS outlier detection
* Updated documentation
* Option to reslice DWI with ``--reslice [x,y,z]``

**Changed**

* Flag ``--epiboost [index]`` changed to ``--epi [n]``, where
  users can specify the number of reverse phase encoded B0 pairs to
  use in EPI correction. Non-indexed B0s were previously destructively
  removed from DWI, leading to incorrect weighing of B0s in tensor
  estimation. The new method now preserves all B0s, thereby allowing
  faster EPI distortion correction without degrading DTI/DKI maps.
* Documentation moved to ReadTheDocs
* Moved B0 production module from designer.preprocessing.brainmask to
  a separate function at ``designer.preprocessing.extractmeanbzero()`` 
  that gets called by PyDesigner main. This allows a B0.nii to be
  produced regardless of the ``--mask`` flag.

**Removed**

* Documentation inconsistencies

`v0.2 [The Cupid Release]`_
---------------------------

Feb 26, 2020

**Added**

* Installer for setup with ``pip install .``
* Multiple file support: *.nii*, *.nii.gz*, *.dcm*, *.mif*
* reStructuredText styled documentation
* Ability to use ``--resume`` flag for DWI concatenation
* SNR plot to depict signal changes before and after preprocessing
* Full utilization of AVX instruction set on AMD machines
* WMTI parameters

**Changed**

* Fixed topup series not being denoised

**Removed**

* CSF masking; feature failed to work consistently

`dev-0.11`_
------------

Dec 2, 2019


**Added**

* None

**Changed**

* Fixed bug in Dockerfile that prevented ``pydesigner.py`` from being
  found

**Removed**

* None

`0.1-dev`_
-----------

Oct 22, 2019

Initial port of MATLAB code to Python. 200,000,000,000 BCE


.. Links

.. _v1.0.0: https://github.com/m-ama/PyDesigner/releases/tag/v1.0.0
.. _v1.0-RC12: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC12
.. _v1.0-RC11: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC11
.. _v1.0-RC10: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC10
.. _v1.0-RC9: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC9
.. _v1.0-RC8: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC8
.. _v1.0-RC7: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC7
.. _v1.0-RC6: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC6
.. _v1.0-RC5: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC5
.. _v1.0-RC4: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC4
.. _v1.0-RC3: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC3
.. _v1.0-RC2: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC2
.. _v1.0-RC1: https://github.com/m-ama/PyDesigner/releases/tag/v1.0-RC1
.. _v0.32: https://github.com/m-ama/PyDesigner/releases/tag/v0.32
.. _v0.31: https://github.com/m-ama/PyDesigner/releases/tag/v0.31
.. _v0.3: https://github.com/m-ama/PyDesigner/releases/tag/v0.3
.. _v0.2 [The Cupid Release]: https://github.com/m-ama/PyDesigner/releases/tag/v0.2
.. _dev-0.11: https://github.com/m-ama/PyDesigner/releases/tag/dev-0.11
.. _0.1-dev: https://github.com/m-ama/PyDesigner/releases/tag/0.1-dev