Python
======

PyDesigner was built and tested on Python 3.7, so we enourage all
users to adopt this version as well. While you may use the Python
supplied by default on your OS, we highly enocurage users to adopt a
Conda-based Python like `Miniconda`_ or `Anaconda`_. Conda is a command
line tool that allows the creation of separated environments with different
python versions and packages. This of it as running multiple virtual
machines on the a single host - you can easily switch between any for
different needs, or run them simultaneously.

.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda: https://www.anaconda.com/

Download and Insall
-------------------

Refer to either of these distributions' page for installation. This
guide assumes a conda (Miniconda) installation for setting up Python.
If you already have conda, or prefer using the default Python supplied
by your OS, skip PyDesigner installation.

Update Conda
------------
First, update conda with

.. code-block:: console
    
    $ conda update conda

Create new environment
----------------------
Creating a conda environment is recommended as this will keep all of
the dependencies required for this project isolated to just the conda
environment called dmri. For more information about conda
environments, see `The Definitive Guide to Conda Environments`_. Next,
create a conda environment specifically for dMRI preprocessing, called
:code:`dmri`. You can choose any name, but be sure to replace *dmri*
in this guide with any name of your choice.

.. _The Definitive Guide to Conda Environments: https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533

Next, execute the following two line to create a Python environment
ready for PyD installation.

.. code-block:: python

    $ conda create -n dmri python=3.7
    $ conda install -n dmri pip

The first line create an environment with Python v3.7, while the
second line installs the PyPi package manager.

Once this is all set, you may proceed with the installation of PyD.

