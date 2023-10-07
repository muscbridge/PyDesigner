PyDesigner Requirements
=======================

PyDesigner, currently, only requires the following three dependencies:

1. Python *3.6*, or above
2. `FSL`_ *6.0.2*, or above
3. `MRtrix3`_, *3.0_RC3* or above

.. _FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
.. _MRtrix3: https://www.mrtrix.org/

Linux and Mac Users
-------------------

Unix-based system users are able to natively run all dependencies.
Please proceed with the installation steps to configure PyD.


Windows Users
-------------

FSL and MRtrix3 are currently **not** *available on the Microsoft Windows*
platform. Users running Windows are recommended to run the Docker image
`NeuroDock`_ these interdependencies at near-native speed.

.. _NeuroDock: https://hub.docker.com/repository/docker/dmri/neurodock

You may still proceed with the installation of PyDesigner Python
modules to perform tensor fitting and map extraction.
