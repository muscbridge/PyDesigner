Changelog
=========

All notable changes to this project will be documented in this file or
page

v0.3 [Upcoming]
---------------

TBA

**Added**:

* Head motion plot from on eddy_qc outputs
* Outlier plot from IRRLS outlier detection
* Updated documentation

**Changed**:

* Flag ``--epiboost [index]`` changed to ``--epi [n]``, where
  users can specify the number of reverse phase encoded B0 pairs to
  use in EPI correction. Non-indexed B0s were previously destructively
  removed from DWI, leading to incorrect weighing of B0s in tensor
  estimation. The new method now preserves all B0s, thereby allowing
  faster EPI distortion correction without degrading DTI/DKI maps.
* Documentation moved to ReadTheDocs

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
.. v0.2 [The Cupid Release]: https://github.com/m-ama/PyDesigner/releases/tag/v0.2
.. v0.11-dev: https://github.com/m-ama/PyDesigner/releases/tag/dev-0.11
.. v0.2-dev: https://github.com/m-ama/PyDesigner/releases/tag/0.1-dev