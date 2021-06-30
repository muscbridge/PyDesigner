
PyDesigner
==========

.. image:: https://img.shields.io/docker/pulls/dmri/neurodock?style=flat-square
   :target: https://hub.docker.com/r/dmri/neurodock
   :alt: Docker Pulls

.. image:: https://img.shields.io/docker/cloud/automated/dmri/neurodock?style=flat-square
   :target: https://hub.docker.com/r/dmri/neurodock/builds
   :alt: Docker Cloud Automated build

.. image:: https://img.shields.io/docker/cloud/build/dmri/neurodock?style=flat-square
   :target: https://hub.docker.com/r/dmri/neurodock/builds
   :alt: Docker Cloud Build Status

.. image:: https://img.shields.io/github/v/release/m-ama/PyDesigner?include_prereleases&style=flat-square\
   :target: https://github.com/m-ama/PyDesigner/releases/latest
   :alt: GitHub release (latest SemVer including pre-releases)

.. image:: https://img.shields.io/readthedocs/pydesigner?style=flat-square
   :target: https://pydesigner.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Welcome to the official PyDesigner project!
*******************************************

PyDesigner was inspired by `NYU's DESIGNER`_ dMRI preprocessing pipeline
to bring pre- and post- processing to every MRI imaging scientist. With PyDesigner,
users are no longer confined to specific file types, operating systems,
or complicated scripts just to extract DTI or DKI parameters –
PyDesigner makes this easy, and you will love it!

.. _NYU's DESIGNER: https://github.com/NYU-DiffusionMRI/DESIGNER

.. image:: https://i.imgur.com/YeCvz8s.png
   :width: 512pt
   :target: https://pydesigner.readthedocs.io/en/latest/
   :alt: Click here to view documentation

Notable Features
================

- **100% Python-based** scripts
- **Minimized package dependencies** for small package footprint
- Preprocessing designed to **boost SNR**
- **Accurate and fast** DTI and DKI metrics via cutting-edge algorithms
- **One-shot** preprocessing to parameter extraction
- **Cross-platform compatibility** between Windows, Mac and Linux using Docker
- Highly flexible and **easy to use**
- **Parallel processing** for quicker preprocessing and parameterization
- **Easy install** with `pip`
- Input **file-format agnostic** – works with .nii, .nii.gz, .mif and dicoms
- **Quality control metrics** to evaluate data integrity – SNR graphs, outlier voxels, and head motion
- Uses the **latest techniques** from DTI/DKI/FBI literature
- Works with **DTI**, **DKI**, **WMTI**, **FBI**, or **FBWM** datasets

We welcome all DTI/DKI researchers to evaluate this software and pass
on their feedback or issues through the `Issues`_ page of this
project’s GitHub repository. Additionally, you may join the `M-AMA
Slack channel`_ for live support.

.. _Issues: https://github.com/m-ama/PyDesigner/issues
.. _M-AMA Slack channel: https://m-ama.slack.com/

**System Requirements**
   Parallel processing in PyDesigner scales almost linearly with the
   nummber of CPU cores present. The application is also memory-intensive
   due to the number of parameter maps being computed.

   Based on this evaluation, for processing a single DWI using
   PyDesigner, we recommend the following minimum system specifications:

   - Ubuntu 18.04
   - Intel i7-9700 or AMD Ryzen 1800X [8 cores]
   - 16 GB RAM
   - 12 GB free storage
   - Nvidia CUDA-enabled GPU

References
==========

The PyDesigner software packages is based upon the the references
listed below. Please be sure to cite them if PyDesigner was used
in any publications.

1. Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K. Diffusional kurtosis imaging: the quantification of non-Gaussian water diffusion by means of MRI. Magn Reson Med 2005;53:1432-1440. doi: 10.1002/mrm.20508 
2. Jensen JH, Helpern JA. MRI Quantification of non-Gaussian water diffusion by kurtosis analysis. NMR Biomed 2010;23:698-710. doi: 10.1002/nbm.1518 
3. Fieremans E, Jensen JH, Helpern JA. White matter characterization with diffusional kurtosis imaging. Neuroimage 2011;58:177-188. doi: 10.1016/j.neuroimage.2011.06.006 
4. Tabesh A, Jensen JH, Ardekani BA, Helpern JA. Estimation of tensors and tensor-derived measures in diffusional kurtosis imaging. Magn Reson Med 2011;65:823-836. doi: 10.1002/mrm.22655 
5. Glenn GR, Helpern JA, Tabesh A, Jensen JH. Quantitative assessment of diffusional kurtosis anisotropy. NMR Biomed 2015;28:448-459. doi: 10.1002/nbm.3271 
6. Jensen JH, Glenn GR, Helpern JA. Fiber ball imaging. Neuroimage 2016; 124:824-833. doi: 10.1016/j.neuroimage.2015.09.049 
7. McKinnon ET, Helpern JA, Jensen JH. Modeling white matter microstructure with fiber ball imaging. Neuroimage 2018;176:11-21. doi: 10.1016/j.neuroimage.2018.04.025 
8. Ades-Aron B, Veraart J, Kochunov P, McGuire S, Sherman P, Kellner E, Novikov DS, Fieremans E. Evaluation of the accuracy and precision of the diffusion parameter EStImation with Gibbs and NoisE removal pipeline. Neuroimage. 2018;183:532-543. doi: 10.1016/j.neuroimage.2018.07.066 
9. Moss H, McKinnon ET, Glenn GR, Helpern JA, Jensen JH. Optimization of data acquisition and analysis for fiber ball imaging. Neuroimage 2019;200;690-703. doi: 10.1016/j.neuroimage.2019.07.005
10. Moss HG, Jensen JH. Optimized rectification of fiber orientation density function. Magn Reson Med. 2020 Jul 25. doi: 10.1002/mrm.28406. Online ahead of print. 
