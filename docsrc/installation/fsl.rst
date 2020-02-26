FSL
===

FSL is a collection of tools and software used to process fMRI, MRI
and DWI data. `Visit their installation page`_ for download and
installation guide.

.. _Visit their installation page: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

**FSL 6.0.2 and above are recommended**. All testing has been done
with FSL 6.0.2. PyDesigner has not been tested with other versions of
FSL.

To check your FSL version:

.. code-block:: console
    
    $ flirt -version

A return value of at least :code:`FLIRT version 6.0` indicates
successful installation of FSL, and that meets the PyD requirement.
