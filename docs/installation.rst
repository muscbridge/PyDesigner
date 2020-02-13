Installation
^^^^^^^^^^^^

Quick Start Guide
-----------------

PyDesigner requires the following dependencies:

1. `FSL 6.0.1 <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`__
2. `MRTRIX3 <https://www.mrtrix.org/download/>`__
3. `Python 3.7 <https://www.anaconda.com/>`__

**Note**: All testing was conducted using Python 3.7, MRtrix3, and FSL 6.0.1. Usage of DESIGNER with alternative versions of these dependencies has not been tested and is not recommended.

To install PyDesigner, navigate to the `PyDesigner Github page <https://github.com/m-ama/PyDesigner>`__, click the green "Clone or download" button, and lick "Download ZIP". When the download is complete, find the PyDesigner-master.zip in your Downloads folder and unzip.
 
:code:`cd //PyDesigner-master/`

For more information, see the detailed sections below. 

Automated Install
-----------------

On the main PyDesigner Github page, click the green "Clone or download" button to access the latest PyDesigner build. Click "Download ZIP". When the download is complete, find the PyDesigner-master.zip in your Downloads folder and unzip.

PyDesigner can be automatically installed with all dependencies by opening a CLI and changing directory to root PyDesigner directory, followed by :code:`pip install .`

This will execute the setup.py script in root directory to automatically configure your Python environment for PyDesigner. When running the automated methods, PyDesigner can    simply be called with the commad pydesigner instead of specifying the python pydesigner.py prefix.

Note: If you need a stable and tested build, download the most recent release from the Release tab. Click on Source code (zip) link and decompress (unzip) to any folder you     desire.

FSL
---

FSL is a collection of tools and software used to process fMRI, MRI and DWI data. `Visit their installation page <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`__ for download and installation guide.

**FSL 6.0.1** is recommended. All testing has been done using FSL 6.0.1. PyDesigner has not been tested with other versions of FSL.

Check your FSL version with :code:`FLIRT -version`

**If you are currently running an FSL version after v6.0.1 and using NVIDIA CUDA:**

As of most recent FSL 6.0.3, :code:`eddy` does not support CUDA 10, while :code:`bedpost` and :code:`probtrakx` do. Moreover, the version supplied after FSL v6.0.1 fails on certain datasets. If running on a CUDA system, users are advised to downgrade to CUDA 9.1 for maximum compatibility, and to do so prior to installing FSL.

After the installation of FSL, create a backup of original :code:`eddy_cuda9.1` by renaming it to to :code:`eddy_cuda9.1.BAK`. Then, replace :code:`eddy_cuda9.1` with the one from `FSL v6.0.1 <https://users.fmrib.ox.ac.uk/~thanayik/eddy_cuda9.1>`__ . End by making the new file executable with :code:`sudo chmod +x /path/to/eddy_cuda9.1`.

Replace/Install `bedpostx for GPU <https://users.fmrib.ox.ac.uk/~moisesf/Bedpostx_GPU/Installation.html>`__ for CUDA 9.1.

MRTRIX3
-------

MRTRIX3 is another software suite aimed at analysis of DWI data. Here are some of their helpful pages.

    | `Homepage <https://www.mrtrix.org/>`__
    | `Download and Install <https://www.mrtrix.org/download/>`__

Check your MRtrix version :code:`mrconvert -version`

Python
------

PyDesigner was built and tested on Python 3.7, so we enourage all users to adopt this version as well. While you may use the Python supplied by default on your OS, we highly enocurage users to adopt a Conda-based Python like `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ or `Anaconda <https://www.anaconda.com/>`__. Conda is a command line tool that allows the creation of separated environments with different python versions and packages. This of it as running multiple virtual machines on the a single host - you can easily switch between any for different needs, or run them simultaneously.

Refer to either of these distributions' page for installation. This guide assumes a conda (Miniconda) installation for setting up Python. If you already have conda, or prefer using the default Python supplied by your OS, skip to package installation at the end of this subsection.

First, update conda with :code:`conda update conda`

Creating a conda environment is recommended as this will keep all of the dependencies required for this project isolated to just the conda environment called dmri. For more information about conda environments, see `The Definitive Guide to Conda Environments <https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533?gi=f526e7f5ec4b>`__. Next, create a conda environment specifically for dMRI preprocessing, called :code:`dmri`. You can choose any name, but be sure to replace dmri in this guide with the name of your choice.

If you prefer not to create this environment, skip to package installation. In addition, we'll be installing pip to this environment, in the even conda install <package name> fails to compile and install a package.

.. code-block:: sh

    conda create -n dmri python=3.7
    conda install -n dmri pip

Activate the new environment with:

.. code-block:: sh

   conda activate dmri

**Note**: Environment activation (conda activate dmri) needs to be done each time a new terminal window is opened. If this behavior is undesired, you may set this environment as default python environment. Refer to advanced conda user guides or Google search how to do this.

**Pepare Python for PyDesigner**

**Note**: Skip to automated install to configure pydesigner automatically

Once the base environment is created and activated, proceed with the installation of all packages.

    1. `NumPy <https://numpy.org/>`__
    2. `SciPy <https://www.scipy.org/>`__
    3. `CVXPY <https://www.cvxpy.org/>`__
    4. `NiBabel <https://nipy.org/nibabel/>`__
    5. `Multiprocessing <https://docs.python.org/3.4/library/multiprocessing.html?highlight=process>`__
    6. `Joblib <https://joblib.readthedocs.io/en/latest/>`__
    7. `TQDM <https://tqdm.github.io/>`__
    8. `py-cpuinfo <https://github.com/workhorsy/py-cpuinfo>`__
    9. `matplotlib <https://matplotlib.org/>`__

**Install necessary packages with the commands**:

.. code-block:: sh

   conda install -c anaconda numpy scipy joblib
   conda install -c conda-forge tqdm nibabel multiprocess matplotlib py-cpuinfo 
   pip install --upgrade setuptools
   pip install cvxpy

If conda fails to install a package, use pip to install the package with:

.. code-block:: sh

   pip install [package name]

Completion of this step will ready your system for dMRI processing. Let's go!

PyDesigner
----------

On the main `PyDesigner Github page <https://github.com/m-ama/PyDesigner>`__, click the green "Clone or download" button to access the latest PyDesigner build. Click "Download ZIP". When the download is complete, find the PyDesigner-master.zip in your Downloads folder and unzip.

PyDesigner is located here: /PyDesigner-master/designer/pydesigner.py

**Note**: If you need a stable and tested build, download the most recent release from the Release tab. Click on Source code (zip) link and decompress (unzip) to any folder you desire.
