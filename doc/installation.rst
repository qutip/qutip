.. This file can be edited using retext 6.1 https://github.com/retext-project/retext

.. _install:

**************
Installation
**************

.. _quick-start:

Quick Start
===========

From QuTiP version 4.6 onwards, you should be able to get a working version of QuTiP with the standard

.. code-block:: bash

   pip install qutip

It is not recommended to install any packages directly into the system Python environment; consider using ``pip`` or ``conda`` virtual environments to keep your operating system space clean, and to have more control over Python and other package versions.

You do not need to worry about the details on the rest of this page unless this command did not work, but do also read the next section for the list of optional dependencies.
The rest of this page covers `installation directly from conda <install-with-conda_>`_, `installation from source <install-from-source_>`_, and `additional considerations when working on Windows <install-on-windows_>`_.


.. _install-requires:

General Requirements
=====================

QuTiP depends on several open-source libraries for scientific computing in the Python programming language.
The following packages are currently required:

.. cssclass:: table-striped

+----------------+--------------+-----------------------------------------------------+
| Package        | Version      | Details                                             |
+================+==============+=====================================================+
| **Python**     | 3.9+         | 3.6+ for version 4.7                                |
+----------------+--------------+-----------------------------------------------------+
| **NumPy**      | 1.22+ <2.0   | 1.16+ for version 4.7                               |
+----------------+--------------+-----------------------------------------------------+
| **SciPy**      | 1.8+         | 1.0+ for version 4.7                                |
+----------------+--------------+-----------------------------------------------------+


In addition, there are several optional packages that provide additional functionality:

.. cssclass:: table-striped

+--------------------------+--------------+-----------------------------------------------------+
| Package                  | Version      | Details                                             |
+==========================+==============+=====================================================+
| ``matplotlib``           | 1.2.1+       | Needed for all visualisation tasks.                 |
+--------------------------+--------------+-----------------------------------------------------+
| ``cython``               | 0.29.20+     | Needed for compiling some time-dependent            |
| ``setuptools``           |              | Hamiltonians. Cython needs a working C++ compiler.  |
| ``filelock``             |              |                                                     |
+--------------------------+--------------+-----------------------------------------------------+
| ``cvxpy``                | 1.0+         | Needed to calculate diamond norms.                  |
+--------------------------+--------------+-----------------------------------------------------+
| ``pytest``,              | 5.3+         | For running the test suite.                         |
| ``pytest-rerunfailures`` |              |                                                     |
+--------------------------+--------------+-----------------------------------------------------+
| LaTeX                    | TeXLive 2009+| Needed if using LaTeX in matplotlib figures, or for |
|                          |              | nice circuit drawings in IPython.                   |
+--------------------------+--------------+-----------------------------------------------------+
| ``loky``, ``mpi4py``     |              | Extra parallel map back-ends.                       |
+--------------------------+--------------+-----------------------------------------------------+
| ``tqdm``                 |              | Extra progress bars back-end.                       |
+--------------------------+--------------+-----------------------------------------------------+

In addition, there are several additional packages that are not dependencies, but may give you a better programming experience.
`IPython <https://ipython.org/>`_ provides an improved text-based Python interpreter that is far more full-featured that the default interpreter, and runs in a terminal.
If you prefer a more graphical set-up, `Jupyter <https://jupyter.org/>`_ provides a notebook-style interface to mix code and mathematical notes together.
Alternatively, `Spyder <https://www.spyder-ide.org/>`_ is a free integrated development environment for Python, with several nice features for debugging code.
QuTiP will detect if it is being used within one of these richer environments, and various outputs will have enhanced formatting.

.. _install-with-conda:

Installing with conda
=====================

If you already have your conda environment set up, and have the ``conda-forge`` channel available, then you can install QuTiP using:

.. code-block:: bash

   conda install qutip

This will install the minimum set of dependences, but none of the optional packages.

.. _adding-conda-forge:

Adding the conda-forge channel
------------------------------

To install QuTiP from conda, you will need to add the conda-forge channel.
The following command adds this channel with lowest priority, so conda will still try and install all other packages normally:

.. code-block:: bash

   conda config --append channels conda-forge

If you want to change the order of your channels later, you can edit your ``.condarc`` (user home folder) file manually, but it is recommended to keep ``defaults`` as the highest priority.


.. _building-conda-environment:

New conda environments
----------------------

The default Anaconda environment has all the Python packages needed for running QuTiP installed already, so you will only need to add the ``conda-forge`` channel and then install the package.
If you have only installed Miniconda, or you want a completely clean virtual environment to install QuTiP in, the ``conda`` package manager provides a convenient way to do this.

To create a conda environment for QuTiP called ``qutip-env``:

.. code-block:: bash

   conda create -n qutip-env python qutip

This will automatically install all the necessary packages, and none of the optional packages.
You activate the new environment by running

.. code-block:: bash

   conda activate qutip-env

You can also install any more optional packages you want with ``conda install``, for example ``matplotlib``, ``ipython`` or ``jupyter``.


.. _install-from-source:

Installing from Source
======================

Official releases of QuTiP are available from the download section on `the project's web pages <https://qutip.org/download.html>`_, and the latest source code is available in `our GitHub repository <https://github.com/qutip/qutip>`_.
In general we recommend users to use the latest stable release of QuTiP, but if you are interested in helping us out with development or wish to submit bug fixes, then use the latest development version from the GitHub repository.

You can install from source by using the `Python-recommended PEP 517 procedure <build-pep517_>`_, or if you want more control or to have a development version, you can use the `low-level build procedure with setuptools <build-setuptools_>`_.

.. _build-pep517:

PEP 517 Source Builds
---------------------

The easiest way to build QuTiP from source is to use a PEP-517-compatible builder such as the ``build`` package available on ``pip``.
These will automatically install all build dependencies for you, and the ``pip`` installation step afterwards will install the minimum runtime dependencies.
You can do this by doing (for example)

.. code-block:: bash

   pip install build
   python -m build <path to qutip>
   pip install <path to qutip>/dist/qutip-<version>.whl

The first command installs the reference PEP-517 build tool, the second effects the build and the third uses ``pip`` to install the built package.
You will need to replace ``<path to qutip>`` with the actual path to the QuTiP source code.
The string ``<version>`` will depend on the version of QuTiP, the version of Python and your operating system.
It will look something like ``4.6.0-cp39-cp39-manylinux1_x86_64``, but there should only be one ``.whl`` file in the ``dist/`` directory, which will be the correct one.


.. _build-setuptools:

Direct Setuptools Source Builds
-------------------------------

This is the method to have the greatest amount of control over the installation, but it the most error-prone and not recommended unless you know what you are doing.
You first need to have all the runtime dependencies installed.
The most up-to-date requirements will be listed in ``pyproject.toml`` file, in the ``build-system.requires`` key.
As of the 5.0.0 release, the build requirements can be installed with

.. code-block:: bash

   pip install setuptools wheel packaging cython 'numpy<2.0.0' scipy

or similar with ``conda`` if you prefer.
You will also need to have a functional C++ compiler installed on your system.
This is likely already done for you if you are on Linux or macOS, but see the `section on Windows installations <install-on-windows_>`_ if that is your operating system.

To install QuTiP from the source code run:

.. code-block:: bash

   pip install .

If you wish to contribute to the QuTiP project, then you will want to create your own fork of `the QuTiP git repository <https://github.com/qutip/qutip>`_, clone this to a local folder, and install it into your Python environment using:

.. code-block:: bash

   python setup.py develop

When you do ``import qutip`` in this environment, you will then load the code from your local fork, enabling you to edit the Python files and have the changes immediately available when you restart your Python interpreter, without needing to rebuild the package.
Note that if you change any Cython files, you will need to rerun the build command.

You should not need to use ``sudo`` (or other superuser privileges) to install into a personal virtual environment; if it feels like you need it, there is a good chance that you are installing into the system Python environment instead.


.. _install-on-windows:

Installation on Windows
=======================

As with other operating systems, the easiest method is to use ``pip install qutip``, or use the ``conda`` procedure described above.
If you want to build from source or use runtime compilation with Cython, you will need to have a working C++ compiler.

You can `download the Visual Studio IDE from Microsoft <https://visualstudio.microsoft.com/downloads/>`_, which has a free Community edition containing a sufficient C++ compiler.
This is the recommended compiler toolchain on Windows.
When installing, be sure to select the following components:

- Windows "X" SDK (where "X" stands for your version: 7/8/8.1/10)
- Visual Studio C++ build tools

You can then follow the `installation from source <install-from-source_>`_ section as normal.

.. important::

   In order to prevent issues with the ``PATH`` environment variable not containing the compiler and associated libraries, it is recommended to use the developer command prompt in the Visual Studio installation folder instead of the built-in command prompt.

The Community edition of Visual Studio takes around 10GB of disk space.
If this is prohibitive for you, it is also possible to install `only the build tools and necessary SDKs <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ instead, which should save about 2GB of space.


.. _install-verify:

Verifying the Installation
==========================

QuTiP includes a collection of built-in test scripts to verify that an installation was successful.
To run the suite of tests scripts you must also have the ``pytest`` testing library.
After installing QuTiP, leave the installation directory and call:

.. code-block:: bash

   pytest qutip/qutip/tests

This will take between 10 and 30 minutes, depending on your computer.
At the end, the testing report should report a success; it is normal for some tests to be skipped, and for some to be marked "xfail" in yellow.
Skips may be tests that do not run on your operating system, or tests of optional components that you have not installed the dependencies for.
If any failures or errors occur, please check that you have installed all of the required modules.
See the next section on how to check the installed versions of the QuTiP dependencies.
If these tests still fail, then head on over to the `QuTiP Discussion Board <https://groups.google.com/g/qutip>`_ or `the GitHub issues page <https://github.com/qutip/qutip/issues>`_ and post a message detailing your particular issue.

If the ``mpi4py`` module is installed, the test suide will also run a set of tests checking the MPI capabilities of QuTiP.
If the MPI backend on your system is not configured correctly, these tests may sometimes cause the test suite to crash or hang.
Please make sure that you are using the latest versions of ``mpi4py`` and the MPI backend.
If the tests still crash or hang, try running pytest with the ``-s`` option to display any potential error or warning messages from the MPI backend.

.. _install-about:

Checking Version Information
============================

QuTiP includes an "about" function for viewing information about QuTiP and the important dependencies installed on your system.
To view this information:

.. code-block:: python

   import qutip
   qutip.about()
