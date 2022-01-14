Install NeuroDock
=================

Aftet successfully installing and configuring docker, you can install
the NeuroDock container in one of two ways:

1. Pulling pre-build image from Docker Hub with :code:`docker pull [image]`
2. Building the image yourself with :code:`docker build [path to image]`

The first option is the recommended method because prebuilt images are
guaranteed to work and enhance repatibility even further. In additon,
they are numbered version-controlled for referencing. Your copy
of NeuroDock will be configured exactly the same other another
person's.

The second option is intended for devopers who make frequent changes
to the PyDesigner source code and wish to test their changes in a
Docker environment. The Dockerfile script is designed to build a
Docker image using PyDesigner in the root directory of the repository.

Docker Hub
----------

Pulling pre-built NeuroDock is incredibly straight forward. Run the
following command to pull NeuroDock.

.. code-block:: console

    $ docker pull docker pull dmri/neurodock:tagname

where :code:`tagname` is the version you'd like to pull. To install
the latest NeuroDock, you would run the command

.. code-block:: console

    $ docker pull dmri/neurodock:latest

And that's it! All you have to do now is to wait for the NeuroDock
image to finish downloading.

Local Build
-----------

**Disclaimer**
    It must be reiterated that this option is preserved for developoers;
    regular users are encoruaged to stay away from this method because
    there is no sematic versioning to referece.

1. Open up a command line interface and change directory to your
PyDesigner repository

.. code-block:: console

    $ cd [PyDesginer Repo Path]

2. To build a Docker image using your local PyDesigner copy, run the
command:

.. code-block:: console

    $ docker build -t [tagname] .

Here, :code:`tagname` can be any name you wish to give this image. If
you wish to build an image called neurodock, run the command:

.. code-block:: console

    $ docker build -t neurodock .

This will build a Docker image called NeuroDock based on your local
Pyesigner repository.
