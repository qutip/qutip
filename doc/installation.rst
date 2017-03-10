.. QuTiP 
   Copyright (C) 2011 and later, Paul D. Nation, Robert J. Johansson & Alexander Pitchford

.. This file can be edited using retext 6.1 https://github.com/retext-project/retext

.. _install:

**************
Installation
**************

.. _install-requires:

General Requirements
=====================

QuTiP depends on several open-source libraries for scientific computing in the Python
programming language.  The following packages are currently required:

.. cssclass:: table-striped

+----------------+--------------+-----------------------------------------------------+
| Package        | Version      | Details                                             |
+================+==============+=====================================================+
| **Python**     | 2.7+         | Version 3.4+ is highly recommended.                 |
+----------------+--------------+-----------------------------------------------------+
| **Numpy**      | 1.8+         | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **Scipy**      | 0.15+        | Lower versions have missing features.               |
+----------------+--------------+-----------------------------------------------------+
| **Matplotlib** | 1.2.1+       | Some plotting does not work on lower versions.      |
+----------------+--------------+-----------------------------------------------------+
| **Cython**     | 0.21+        | Needed for compiling some time-dependent            |
|                |              | Hamiltonians.                                       |
+----------------+--------------+-----------------------------------------------------+
| **GCC**        | 4.2+         | Needed for compiling Cython files.                  |
| **Compiler**   |              |                                                     |
+----------------+--------------+-----------------------------------------------------+
| **Python**     | 2.7+         | Linux only. Needed for compiling Cython files.      |
| **Headers**    |              |                                                     |
+----------------+--------------+-----------------------------------------------------+


In addition, there are several optional packages that provide additional functionality:

+----------------+--------------+-----------------------------------------------------+
| Package        | Version      | Details                                             |
+================+==============+=====================================================+
| gfortran       | 4.2+         | Needed for compiling the optional Fortran-based     |
|                |              | Monte Carlo solver.                                 |
+----------------+--------------+-----------------------------------------------------+
| BLAS           | 1.2+         | Optional, Linux & Mac only.                         |
| library        |              | Needed for installing Fortran Monte Carlo solver.   |
+----------------+--------------+-----------------------------------------------------+
| Mayavi         | 4.1+         | Needed for using the Bloch3d class.                 |
+----------------+--------------+-----------------------------------------------------+
| LaTeX          | TexLive 2009+| Needed if using LaTeX in matplotlib figures.        |    
+----------------+--------------+-----------------------------------------------------+
| nose           | 1.1.2+       | For running the test suite.                         |
+----------------+--------------+-----------------------------------------------------+


As of version 2.2, QuTiP includes an optional Fortran-based Monte Carlo solver that has some performance benefit over the Python-based solver when simulating small systems. In order to install this package you must have a Fortran compiler (for example gfortran) and BLAS development libraries.  At present, these packages are tested only on the Linux and OS X platforms.

We would not recommend installation into the system Python on Linux platforms, as it is likely that the required libraries will be difficult to update to sufficiently recent versions. The system Python on Linux is used for system things, changing its configuration could lead to highly undesirable results. We are recommending and supporting Anaconda / Miniconda Python environments for QuTiP on all platforms.

.. _install-platform-independent:

Platform-independent Installation
=================================

QuTiP is designed to work best when using the `Anaconda <https://www.continuum.io/downloads>`_ or `Intel <https://software.intel.com/en-us/python-distribution>`_ Python distributions that support the conda package management system.

If you aleady have your conda environment set up, and have the ``conda-forge`` channel available, then you can install QuTiP using:

.. code-block:: bash

   conda install qutip

Otherwise refer to building-conda-environment_

If you are using MS Windows, then you will probably want to refer to installation-on-MS-Windows_

.. _building-conda-environment:

Building your Conda environment
-------------------------------
The default Anaconda environment has all the Python packages needed for running QuTiP. 
You may however wish to install QuTiP in a Conda environment (env) other than the default Anaconda environment. 
You may wish to this for many reasons:

1. It is a good idea generally
2. You are using MS Windows and want to use Python 3
3. You are using `Miniconda <http://conda.pydata.org/miniconda.html>`_ because you do not have the disk space for full Anaconda.

To create a Conda env for QuTiP called ``qutip``:-

(note the ``python=3`` can be ommited if you want the default Python version, if you want to use Python 3 with MS Windows, then it must be ``python=3.4``)

recommended:

.. code-block:: bash

   conda create -n qutip python=3 mkl numpy scipy cython matplotlib nose jupyter notebook spyder

minimum (recommended):

.. code-block:: bash

   conda create -n qutip numpy scipy cython nose matplotlib

absolute mimimum:

.. code-block:: bash

   conda create -n qutip numpy scipy cython

The ``jupyter`` and ``notebook`` packages are for working with `Jupyter <http://jupyter.org/>`_ notebooks (fka IPython notebooks). 
`Spyder <https://pythonhosted.org/spyder/>`_ is an IDE for scientific development with Python.

Adding the conda-forge channel
------------------------------

If you have conda 4.1.0 or later then, add the conda-forge channel with lowest priority using:

.. code-block:: bash

   conda config --append channels conda-forge

Otherwise you should consider reinstalling Anaconda / Miniconda. In theory:

.. code-block:: bash

   conda update conda

will update your conda to the latest version, but this can lead to breaking your default Ananconda enviroment.

Alternatively, this will add ``conda-forge`` as the highest priority channel.

.. code-block:: bash

   conda config --add channels conda-forge

It is almost certainly better to have ``defaults`` as the highest priority channel.
You can edit your ``.condarc`` (user home folder) file manually, so that ``conda-forge`` is below ``defaults`` in the ``channels`` list.


Installing via pip
==================

For other types of installation, it is often easiest to use the Python package manager `pip <http://www.pip-installer.org/>`_.

.. code-block:: bash

   pip install qutip

Or, optionally, to also include the Fortran-based Monte Carlo solver:

.. code-block:: bash

   pip install qutip --install-option=--with-f90mc

More detailed platform-dependent installation alternatives are given below.

.. _install-get-it:

Installing from Source
======================

Official releases of QuTiP are available from the download section on the project's web pages

    http://www.qutip.org/download.html

and the latest source code is available in our Github repository

    http://github.com/qutip

In general we recommend users to use the latest stable release of QuTiP, but if you are interested in helping us out with development or wish to submit bug fixes, then use the latest development version from the Github repository.

Installing QuTiP from source requires that all the dependencies are satisfied.  To install QuTiP from the source code run:

.. code-block:: bash

   sudo python setup.py install

To also include the optional Fortran Monte Carlo solver, run:

.. code-block:: bash
    
   sudo python setup.py install --with-f90mc

On Windows, omit ``sudo`` from the commands given above.


.. _install-verify:

Verifying the Installation
==========================

QuTiP includes a collection of built-in test scripts to verify that an installation was successful. To run the suite of tests scripts you must have the nose testing library. After installing QuTiP, leave the installation directory, run Python (or iPython), and call:

.. code-block:: python

   import qutip.testing as qt
   qt.run()

If successful, these tests indicate that all of the QuTiP functions are working properly.  If any errors occur, please check that you have installed all of the required modules.  See the next section on how to check the installed versions of the QuTiP dependencies. If these tests still fail, then head on over to the `QuTiP Discussion Board <http://groups.google.com/group/qutip>`_ and post a message detailing your particular issue.

.. _install-about:

Checking Version Information using the About Function
=====================================================

QuTiP includes an "about" function for viewing information about QuTiP and the important dependencies installed on your system.  To view this information:

.. code-block:: python

   In [1]: from qutip import *

   In [2]: about()

.. _installation-on-MS-Windows:

Installation on MS Windows
==========================
We are recommending and supporting installation of QuTiP into a Conda environment. Other scientific Python implementations such as Python-xy may also work.

QuTiP uses dynamic compilation of C for some of its time-dependant dynamics solvers. For MS Windows users the additional challenge is the need for a ANSI C99 compliant C compiler. Unlike other platforms, no C compiler is provided with Windows by default. 
It is possible to install a Windows SDK that includes a C compiler, but ANSI C99 compliance is not 100%. 
The `mingw-w64 <https://mingw-w64.org>`_ project looks to help overcome this, and to some extent it is successful. 
The `conda-forge <https://conda-forge.github.io>`_ packages for QuTiP will also install the `Mingwpy <https://mingwpy.github.io>`_ package, which uses mingw-w64.

Currently we are only able get QuTiP working with Python <= 3.4. Python >= 3.5 is compiled with a newer version of the MSVC compiler, and there are currently license restrictions.

To specify the use of the mingw compiler you will need to create the following file: ::

   <path to my Python env>/Lib/distutils/distutils.cfg

with the following contents: ::

   [build]
   compiler=mingw32
   [build_ext]
   compiler=mingw32


