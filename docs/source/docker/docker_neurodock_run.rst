Run NeuroDock
=============

Congratulations, you've come this far. You've installed Docker and
NeuroDock, and are probably wondering how what else to do...

**You're done. Not even kidding!** You can now start processing data
with PyDesigner and NeuroDock. It's almost as if FSL, MRtrix3 and
PyDesigner commands are built natively into your OS - be it Linux,
Mac OS, or even Windows!

Intro to Docker Run
-------------------

Use the following form of :code:`docker run` command to call all
command made availble by NeuroDock:

.. code-block:: console

    $ docker run [OPTIONS] IMAGE [COMMAND] [ARG...]

where,

.. csv-table::
    :header: "Flag", "Definition"

    :code:`[OPTIONS]`, "docker options to use when running the container; common options are :code:`-it`, :code:`-v`, :code:`-d`"
    :code:`IMAGE`, "image name to run; in this instance, this is :code:`dmri/neurodock`"
    :code:`[COMMAND]`, "specify which NeuroDock commands to run; these can be FSL, MRtrix3 or PyDesigner commands"
    :code:`[ARG]`, "arguments for :code:`[COMMAND]`"


Users are encouraged to visit the `Docker run reference`_ documentation
for more information on controlling the :code:`docker run ` command.

.. _Docker run reference: https://docs.docker.com/engine/reference/run/

Practical Run
-------------

The section above convered a generic way to use the :code:`docker run`
command. For actual data analysis, we use the following options.

1. :code:`-it --rm` to run docker in interative TTY mode. What this
implies is that your NeuroDock command will run like any other OS
commands such as :code:`ipconfig`, :code:`watch`, :code:`ls` etc.

2. :code:`-v` to mount the file system or folder to processing

Bind Mount
~~~~~~~~~~

The second flag. :code:`-v`, makes visible the host's local filesystem
to a Docker container, which otherwise runs in a completely isolated
system. By mounting a folder for NeuroDock, you are able to make it
process data in said folder. The general guideline is to mount one
subject folder at a time. It is advisable that users read through
`Docker's bind mounts`_ to understand how Docker containers handle
storage.

.. _Docker's bind mounts: https://docs.docker.com/storage/bind-mounts/

The correct syntax for the :code:`-v` flag is:

.. code-block:: console

    -v [HOST PATH TO MOUNT]:[v]

Suppose a subject folder :code:`bond_007` in need of processing is
structured the following way:

|   bond_007
|   │
|   ├── nifti
|   │   ├── bond_dwi.bval
|   │   ├── bond_dwi.bvec
|   │   ├── bond_dwi.bval
|   │   ├── bond_dwi.json
|   │   ├── bond_topup.json
|   │   └── bond_topup.nii
|   │
|   └── processed (empty dir)

This subject needs to be processed PyDesigner read the input nifti
files in the :code:`nifti` directory, and saves the outputs in the
:code:`processed` directory. Since both :code:`nifti` and
:code:`processed` folders belong to a common parent directory, the
:code:`bond_007` directory can be mounted to give NeurDock access to
both child directories simultaneously.

Here, the directory :code:`bond_007` is the :code:`[HOST PATH TO MOUNT]`,
the directory that NeuroDock will not be able to see.

Next, we need to define where within the container this directory is
mounted, :code:`[TARGET AT WHICH TO MOUNT]`. You may simply mount this
in the root NeuroDock directory at :code:`/data`.

The flag to reflect this would then be:

.. code-block:: console

    -v /Users/sid/Desktop/bond_007:/data

This would make the contents of host directory :code:`bond_007`
available in the NeuroDock at :code:`\data`. Say, for example, the
nifti file :code:`bond_dwi.nii`, is located in the host system at
:code:`/Users/sid/Desktop/bond_007/nifti/bond_dwi.nii`. If the above
mounting scheme is used, the NeuroDock container will see this file in
:code:`/data/nifti/bond_dwi.nii`

This filesystem transformation is particularly important when writing
scripts for automatic or batch processing of subject directories using
the NeuroDock container.

Put it all together
~~~~~~~~~~~~~~~~~~~

Considering everything on this page, it becomes incredibly easy to
process a subject using the NeuoDock container. Sticking to
:code:`bond_007` example above, and combining everthing so far, one
could process Mr. Bond's DWI with the command:

.. code-block:: console

    $ docker run -it --rm -v /Users/sid/Desktop/bond_007:/data \
        dmri/neurodock pydesigner --standard \
        --output /data/processed \
        /data/nifti/bond_dwi.nii,/data/nifti/bond_topup.nii

This command runs the :code:`--standard` PyDesigner pipeline on
the input files :code:`/Users/sid/Desktop/bond_007/nifti/bond_dwi.nii`
and :code:`/Users/sid/Desktop/bond_007/nifti/bond_topup.nii`, and
saves all outputs into the directory :code:`Users/sid/Desktop/bond_007/processed`
