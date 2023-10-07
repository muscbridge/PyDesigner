Docker Installation
===================

Docker is relatively straightforward to install and run.
Windows and Mac users are able to install Docker like any other
GUI-based software package installtion. The installation is not
dependent on console arguments, like Linux.

Please efer to the instructions below for links and guide.

Linux
-----

Users may refer to the Docker Engine installation guide located
`here`_, for installation instructions on their Linux disribution. the
steps covered below are targeted for Debian-based or Ubuntu
distributions.

.. _here: https://docs.docker.com/install/linux/docker-ce/ubuntu/

Uninstall Docker
~~~~~~~~~~~~~~~~

1. Uninstall older version or any traces of existing Docker
installations

.. code-block:: console

    $ sudo apt-get remove docker docker-engine docker.io containerd runc

Don't panic if :code:`apt-get` returns an a warning about missing
packages. It's good they are missing, since we're trying to purge
existing installations of Docker

Install Docker Engine
~~~~~~~~~~~~~~~~~~~~~

Once all taces of existing Docker installation and dependencies have
been purged, you may proceed with the following steps to install the
Docker Engine - Community.

1. Update the debian package list with:

.. code-block:: console

    $ sudo apt-get Update

2. Install basic packages that enable installation of Docker Engine
and its dependencies with:

.. code-block:: console

    $ sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common

3. Add the Docker official GNU Privacy Guard (GPG) key to enable
encryption and decryption of communication with the Docker server:

.. code-block:: console

    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -


You may verify this key by following the full guide on official Docker
documentation, the link to which is located at the beginning of this
page.

4. Add the stable Docker Engine repository to your package list with
the command:

.. code-block:: console

    $ sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"

Adding a repository to your Linux distribution allows the OS to pull
software packages from the developers' servers. It directs the OS to
the location where these packages are stored.

Then, update your package manager repository with the command:

.. code-block:: console

    $ sudo apt-get update

This updates the list of softwares your OS can fetch from various
repositories.

5. Once your Debian-based system becomes aware of the Docker Engine,
you may install it simply via the command:

.. code-block:: console

    $ sudo apt-get install docker-ce docker-ce-cli containerd.io

6. Verify your Docker Engine installation with the command:

.. code-block:: console

    $ sudo docker run hello-world

If the following information prints in the console window, your Docker
Engine installation was sucessful.

.. code-block:: console
    :linenos:

    Hello from Docker!
    This message shows that your installation appears to be working correctly.

    To generate this message, Docker took the following steps:
        1. The Docker client contacted the Docker daemon.
        2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
           (amd64)
        3. The Docker daemon created a new container from that image which runs the
           executable that produces the output you are currently reading.
        4. The Docker daemon streamed that output to the Docker client, which sent it
           to your terminal.

    To try something more ambitious, you can run an Ubuntu container with:
    "$ docker run -it ubuntu bash"

    Share images, automate workflows, and more with a free Docker ID:
     https://hub.docker.com/

    For more examples and ideas, visit:
    https://docs.docker.com/get-started/

You may now proceed with the fetching of NeuroDock Docker image.


Mac OS
------

1. Download Docker `Docker Desktop for Mac`_.

.. _Docker Desktop for Mac: https://hub.docker.com/editions/community/docker-ce-desktop-mac/

2. Double-click on the downloaded `Docker.dmg` to start the install
process. Follw all on-screen instrcutions and prompts.

3. Docker should start automatically, indicated by the whale icon in
the status bar. Alternatively, you may verfiy whether Docker is running
by parsing the following command in Terminal:

.. code-block:: console

    $ docker version

Or you may run the `hello-world` container to verify the installation:

.. code-block::

    $ docker run hello-world

If you information text being printed into the PowerShell windows,
then Docker has been installed successfully.


Windows
-------

1. Download `Docker Desktop for Windows`_.

.. _Docker Desktop for Windows: https://hub.docker.com/editions/community/docker-ce-desktop-windows/

2. Double-click the `Docker for Windows Installer` to run the
installer.

3. Docker should start automatically, indicated by the whale icon in
the taskbar. Alternatively, you may verfiy whether Docker is running
by parsing the following command in PowerShell.

.. code-block:: console

    $ docker version

Or you may run the `hello-world` container to verify the installation:

.. code-block::

    $ docker run hello-world

If you information text being printed into the PowerShell windows,
then Docker has been installed successfully.
