Changelog
=========

All notable changes to this project will be documented in this file or
page

`v1.0_RC1`_
-----------

July, 2020

**Added**:
* Methods to perform tensor only with compatible B-values. PyDesigner
  previously use all B-values in a DWI to do so. This behavior has
  been updated to use only B-values less than 2500.

**Changed**:
* Automatically issues ``dwipreproc`` or ``dwifslpreproc`` for
  compatibility with MRtrix3 >= 3.0.1

**Removed**:



`v0.32`_
--------

Apr 21, 2020

**Added**:

* Intrinsic inter-axonal and mean extra-axonal diffusivity
  calculation to WMTI

**Changed**:

* Method ``json2fslgrad`` converted from class method to function
  definition
* ``json2fslgrad`` now transposes B0s in BVAL file in accordance with
  FSL's gradient scheme
* Documentation update
* ``Extras`` directory renamed to ``extras``
* DKE conversion scripts modified to correctly create ft and dke
  parameter files

**Removed**:

* None

`v0.31`_
--------

Apr 9, 2020

**Added**:

* NaN check in AWF calculculation that prevents further errors in intra-axonal
  and extra-axonal WMTI metrics computation

**Changed**:

* ``designer.fitting.dwipy`` input file detection method
* ``Dockerfile_release`` now deletes the correct temporary file to prevent build
  error

**Removed**:

* None

`v0.3`_
--------

Apr 8, 2020

**Added**:

* Head motion plot from on eddy_qc outputs
* Outlier plot from IRRLS outlier detection
* Updated documentation
* Option to reslice DWI with ``--reslice [x,y,z]``

**Changed**:

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

**Removed**:

* Documentation inconsistencies

`v0.2 [The Cupid Release]`_
---------------------------

Feb 26, 2020

**Added**:

* Installer for setup with ``pip install .``
* Multiple file support: *.nii*, *.nii.gz*, *.dcm*, *.mif*
* reStructuredText styled documentation
* Ability to use ``--resume`` flag for DWI concatenation
* SNR plot to depict signal changes before and after preprocessing
* Full utilization of AVX instruction set on AMD machines
* WMTI parameters

**Changed**:

* Fixed topup series not being denoised

**Removed**:

* CSF masking; feature failed to work consistently

`v0.11-dev`_
------------

Dec 2, 2019


**Added**:

* None

**Changed**:

* Fixed bug in Dockerfile that prevented ``pydesigner.py`` from being
  found

**Removed**:

* None

`v0.1-dev`_
-----------

Oct 22, 2019

Initial port of MATLAB code to Python. 200,000,000,000 BCE


.. Links
.. _v1.0_RC1: https://github.com/m-ama/PyDesigner/releases/tag/v1.0_RC1
.. _v0.32: https://github.com/m-ama/PyDesigner/releases/tag/v0.32
.. _v0.31: https://github.com/m-ama/PyDesigner/releases/tag/v0.31
.. _v0.3: https://github.com/m-ama/PyDesigner/releases/tag/v0.3
.. _v0.2 [The Cupid Release]: https://github.com/m-ama/PyDesigner/releases/tag/v0.2
.. _v0.11-dev: https://github.com/m-ama/PyDesigner/releases/tag/dev-0.11
.. _v0.2-dev: https://github.com/m-ama/PyDesigner/releases/tag/0.1-dev