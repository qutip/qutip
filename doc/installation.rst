.. QuTiP 
   Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

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


.. _install-platform-independent:

Platform-independent Installation
=================================

QuTiP is designed to work best when using the `Anaconda <https://www.continuum.io/downloads>`_ or `Intel <https://software.intel.com/en-us/python-distribution>`_ Python distributions that support the conda package management system.  Once installed, the QuTiP library can be obtained using the conda-forge repository:

.. code-block:: bash
    
    conda config --add channels conda-forge 
    conda install qutip


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

.. ipython::

   In [1]: from qutip import *

   In [2]: about()
