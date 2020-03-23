Docker Configuration
====================

Docker can be configured in a wide-varietry of ways based on hardware
resources available. Parameters such as CPU cores, RAM and storage can
be assigned to Docker for running NeuroDock.

For validations purposes, the NeuroDock image was tested to work as
intended on the following three systems:

.. csv-table::
    :header: "Part", "Machine A", "Machine B", "Machine C"

    "Build", "Apple iMac Pro", "Custom", "Custom"
    "OS", "Mac OS X Mojave", "Ubuntu 18.04", "Microsoft Windows 10 Pro"
    "CPU", "Intel Xeon W [8C/16T]", "AMD Ryzen R9 2700X [8C/16T]", "AMD Ryzen R9 2700X [8C/16T]"
    "Memory", 64 GB, 16 GB, 16 GB
    "Video", Raden Pro Vega 56 8 GB, Nvidia GTX 1080 8 GB, Nvidia GTX 1080 8 GB

We found identical results across the three operating systes on all
these configurations.

Docker Preferences
------------------

Based on Docker's system requirements, we recommend assigning the
following sysem resources to Docker:

.. csv-table::
    :header: "Parameter", "Value"

    "CPUs", 8
    "Memory", 16.00 GB
    "Disk image size", 32.00 GB

By default, Docker assigns itself half the number of available
CPU cores and 2 GB of memory. Considering that the entire NeuroDock
image is ~14.5 GB, we recommend at least double in disk image size.
You may configure your Docker Engine to run on this configuration, or
input your own values based on your processing needs. The following
sections detail how to set these parameters.

Linux
~~~~~

CPU and memory access to Docker containers on Linux machines is
manipulated via CFGS scheduler flags at run time. These flags are:

.. csv-table::
    :header: "Flag", "Description"

    :code:`--cpus=<value>`, specify how many CPU cores to use
    :code:`-m` or :code:`--memory`, specify the maximum amount of memory available to containers

For a more comprehensive list of manupulable system parameters for
for Linux, please visit the `Runtime options with Memory, CPUs, and GPUs`_
page on Docker documentation.

.. _Runtime options with Memory, CPUs, and GPUs: https://docs.docker.com/config/containers/resource_constraints/

Mac OS
~~~~~~

Manipulating these three variables is very simple on Mac OS because
these parameters are located in the GUI.

1. On the Docker icon in the status bar, right-click on the Docker
icon, then **Preferences**.

2. Click on the **Resources** tab on the left

.. figure:: /images/Docker_Mac_resources.png
    :scale: 75 %
    :align: center

    Docker Mac preferences GUI; click on resources

3. The **Resourcs** menu will show you the configuration, please
change them to desired valus. You may leave "Swap" at default.

.. figure:: /images/Docker_Mac_configs.png
    :scale: 75 %
    :align: center

    Docker Mac resources configuration


Windows
~~~~~~~

Similar to the Mac, the same sequence of steps apply for the Windows
platform.

1. Right-click on the Dpcker icon in the taskbar, then click on
**Preferences**.

2. Clock on the **Resources** tab on the left.

.. figure:: /images/Docker_Win_configs.png
    :scale: 75 %
    :align: center

    Docker Windows preferences GUI; configure as desired

3. The **Resourcs** menu will show you the configuration, please
change them to desired valus. You may leave "Swap" at default.

Setting the correct configuration will force Docker to not exceed
these constraints. By splitting up CPU and memory loads, researchers
can process multiple DWIs simultaneously.

GPU Support
-----------

At this time, there is no CUDA or ROCm GPU support. These feature are
planned for a later release. Please use the non-Docker, native Linux
configuration to utilize GPU for eddy and EPI correction.
