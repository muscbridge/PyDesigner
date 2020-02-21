NeuroDock
=====================

Docker is a contaner technology desgned to package an application and
all its needs, such as libraried and other dependencies, into one
package. We adapted PyDesigner and its dependencies for compatibility
with the Docker Engine to bring DTI/DKI analyses to every one.

We bring you, `NeuroDock`_

.. _NeuroDock: https://hub.docker.com/r/dmri/neurodock

.. image:: /images/NeuroDock_Logo.png
   :align: center
   :width: 256pt

.. image:: /images/MUSC_TAG_00447c.png
   :align: center
   :width: 128pt

NeuroDock is a Docker image containing the most cutting-edge tools
required for diffusion and kurtosis imaging. This container was
designer for complete dMRI processing pipelines to be platform
agnostic. NeuroDock was inspired by the lack of easily-accessible
tools across various platforms. NeuroDock is 100% compatible across
Windows, Linux, and Mac - while making available the full suite of
FSL, MRtrix3 and PyDesigner commands.


Why Docker
----------

By packaging fixed versions of FSL, MRtrix3, and PyDesigner, we are
able to guarantee repeatbility and concistency across all platforms.
Regardless of whether researchers are running Linux, Windows, or Mac
OS, identical results can be replicated with Docker technology.

A side-effect to ensuring repeatiblity with Docker is that it becomes
host operating system (OS) agnostic. This allows users to run FSL,
MRtrix3, or PyDesigner commands at near-native speed, even on
Microsoft Windows.

Additionally, researchers can easily deploy Docker containers to HPCs
for rapid processing of large-cohort or longitudinal studies with
ease.

Docker vs Virtual Machines
--------------------------
Okay, so you may ask, "why not just load up a VM?". You have a point.
While the two technologies appear to be behaving the same way, at
least on the surface level, their inner mechanisms are differ vastly.

Unlike a VM, rather than creating a whole virtual OS loaded with
dependencies and other applications, Docker allows applications to
share the same OS kernel, thereby providing a significant performance
uplift while saving up storage space. With the removal of an entire
guest OS in VMs, Docker containers save tons of computational
resources that can be diverted towards better performance.

Now that you know some differences, it is time to move on to preparing
the Docker image!



