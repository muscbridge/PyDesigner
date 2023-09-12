PyDesigner Syntax
=================

PyDesigner has a simple syntax:

.. code-block:: console

    $ pydesigner [OPTIONS] DWI1,DWI2,DWI3...

Mutiple DWIs may be combined by separating their paths with just a comma.
For example, one may run PyDesigner with standard processing using the command:

.. code-block:: console

    $ pydesigner -s --verbose \
    ~/Dataset/DKI_B0.nii ~/Dataset/DKI_B1000.nii ~/Dataset/DKI_B2000.nii

Simple as that!

DTI, DKI, FBI, or FBWM?
-----------------------

Now that you are ready to process, you may be wondering how to get various metrics
from your DWIs. It's very simple, PyDesigner figures this out for you. It analyzes
input dataset's B-value shells to extract as much information as possible. All you,
the user, have to do is to just load your DWIs in and grab a drink. Cheers and welcome!
