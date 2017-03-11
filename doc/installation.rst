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
| **Python**     | 2.7+         | Version 3.5+ is highly recommended.                 |
+----------------+--------------+-----------------------------------------------------+
| **NumPy**      | 1.8+         | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **SciPy**      | 0.15+        | Lower versions have missing features.               |
+----------------+--------------+-----------------------------------------------------+
| **Matplotlib** | 1.2.1+       | Some plotting does not work on lower versions.      |
+----------------+--------------+-----------------------------------------------------+
| **Cython**     | 0.21+        | Needed for compiling some time-dependent            |
|                |              | Hamiltonians.                                       |
+----------------+--------------+-----------------------------------------------------+
| **GCC**        | 4.7+         | Needed for compiling Cython files.                  |
| **Compiler**   |              |                                                     |
+----------------+--------------+-----------------------------------------------------+
| **Python**     | 2.7+         | Linux only. Needed for compiling Cython files.      |
| **Headers**    |              |                                                     |
+----------------+--------------+-----------------------------------------------------+


In addition, there are several optional packages that provide additional functionality:

+----------------+--------------+-----------------------------------------------------+
| Package        | Version      | Details                                             |
+================+==============+=====================================================+
| LaTeX          | TexLive 2009+| Needed if using LaTeX in matplotlib figures.        |    
+----------------+--------------+-----------------------------------------------------+
| nose           | 1.1.2+       | For running the test suite.                         |
+----------------+--------------+-----------------------------------------------------+


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

1. It is a good idea generally.
2. You are using `Miniconda <http://conda.pydata.org/miniconda.html>`_ because you do not have the disk space for full Anaconda.

To create a Conda env for QuTiP called ``qutip``:-

(note the ``python=3`` can be ommited if you want the default Python version)

recommended:

.. code-block:: bash

   conda create -n qutip python=3 mkl numpy scipy cython matplotlib nose multiprocess jupyter notebook spyder

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
   
To install OPENMP support, if available, run:


.. code-block:: bash

   sudo python setup.py install --with-openmp


On Windows, omit ``sudo`` from the commands given above.


.. _installation-on-MS-Windows:

Installation on MS Windows
==========================

.. important:: Installation on Windows has changed substantially as of QuTiP 4.1.  The only supported installation configuration is using the Conda environment with Python 3.5+ and Visual Studio 2015. 

We are recommending and supporting installation of QuTiP into a Conda environment. Other scientific Python implementations such as Python-xy may also work, but are not supported.  As of QuTiP 4.1, installation on Windows requires Python 3.5+, as well as Visual Studio 2015.  With this configuration, one can install QuTiP using the above mentioned receipes.


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

