PyDesigner
==========

PyD is an installable Python package deisgned to perform pre- and
post- processing of dMRI acquisitions. 

Easy Install
------------
PyDesigner can be installed using the PyPI package manager using the command

.. code-block:: console

    $ pip install pydesigner

**Note**:
    Remember to switch to your conda environement before parsing this
    command.

That's it, you're done!

Instructions on installing PyDesigner manually from the GitHub repository
are list below.

Download
--------
You may clone the main `PyDesigner repository`_ for the latest build,
or download the build version of your choice from the `Releases tab`_.

.. _PyDesigner repository: https://github.com/m-ama/PyDesigner
.. _Releases tab: https://github.com/m-ama/PyDesigner/releases

To clone the PyDesigner repository, in terminal, run:

.. code-block:: console

    $ git clone https://github.com/m-ama/PyDesigner.git

Install
-------
PyDesigner can be automatically installed with all dependencies by
opening a CLI and changing directory to root PyDesigner directory, 
followed by

.. code-block:: console

    $ pip install .

**Note**:
    Remember to switch to your conda environement before parsing this
    command.

This will execute the :code:`setup.py` script in root directory to
automatically configure your Python environment for PyDesigner. When
running the automated methods, PyDesigner can simply be called with
the commad :code:`pydesigner`.

**Note:** If you need a stable, tested and versioned build, download
the most recent release from the Release tab. Click on Source code
(zip) link and decompress (unzip) to any folder you desire.
