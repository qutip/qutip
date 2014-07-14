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
| **Python**     | 2.7+         | Version 3.3+ is highly recommended.                 |
+----------------+--------------+-----------------------------------------------------+
| **Numpy**      | 1.7+         | Not tested on lower versions.                       |
+----------------+--------------+-----------------------------------------------------+
| **Scipy**      | 0.13+        | Lower versions have missing features.               |
+----------------+--------------+-----------------------------------------------------+
| **Matplotlib** | 1.2.0+       | Some plotting does not work on lower versions.      |
+----------------+--------------+-----------------------------------------------------+
| **Cython**     | 0.15+        | Needed for compiling some time-dependent            |
|                |              | Hamiltonians.                                       |
+----------------+--------------+-----------------------------------------------------+
| **GCC**        | 4.2+         | Needed for compiling Cython files.                  |
| **Compiler**   |              |                                                     |
+----------------+--------------+-----------------------------------------------------+
| Fortran        | Fortran 90   | Needed for compiling the optional Fortran-based     |
| Compiler       |              | Monte Carlo solver.                                 |
+----------------+--------------+-----------------------------------------------------+
| BLAS           | 1.2+         | Optional, Linux & Mac only.                         |
| library        |              | Needed for installing Fortran Monte Carlo solver.   |
+----------------+--------------+-----------------------------------------------------+
| Mayavi         | 4.1+         | Optional. Needed for using the Bloch3d class.       |
+----------------+--------------+-----------------------------------------------------+
| Python         | 2.7+         | Linux only. Needed for compiling Cython files.      |
| Headers        |              |                                                     |
+----------------+--------------+-----------------------------------------------------+
| LaTeX          | TexLive 2009+| Optional. Needed if using LaTeX in figures.         |    
+----------------+--------------+-----------------------------------------------------+
| nose           | 1.1.2+       | Optional. For running tests.                        |
+----------------+--------------+-----------------------------------------------------+


As of version 2.2, QuTiP includes an optional Fortran-based Monte Carlo solver that has a substantial performance benefit when compared with the Python-based solver. In order to install this package you must have a Fortran compiler (for example gfortran) and BLAS development libraries.  At present, these packages are only tested on the Linux and OS X platforms.


.. _install-platform-independent:

Platform-independent installation
=================================

Often the easiest way is to install QuTiP is to use the Python package manager `pip <http://www.pip-installer.org/>`_.

.. code-block:: bash

    sudo pip install qutip

However, when installing QuTiP this way the Fortran-based Monte Carlo solver is not included.
More detailed platform-dependent installation alternatives are given below.

.. _install-get-it:

Get the source code
===================

Official releases of QuTiP are available from the download section on the project's web pages

    http://www.qutip.org/download.html

and the latest source code is available in our Github repository

    http://github.com/qutip

In general we recommend users to use the latest stable release of QuTiP, but if you are interested in helping us out with development or wish to submit bug fixes, then use the latest development version from the Github repository.

.. _install-it:

Installing from source
======================

Installing QuTiP from source requires that all the dependencies are satisfied. The installation of these dependencies is different on each platform, and detailed instructions for Linux (Ubuntu), Mac OS X and Windows are given below.

Regardless of platform, to install QuTiP from the source code run::

    sudo python setup.py install

To also include the optional Fortran Monte Carlo solver, run::

    sudo python setup.py install --with-f90mc

On Windows, omit ``sudo`` from the commands given above.

.. _install-linux:

Installation on Ubuntu Linux
============================

Using QuTiP's PPA
-------------------

The easiest way to install QuTiP in Ubuntu (12.04 and later) is to use the QuTiP PPA

.. code-block:: bash

    sudo add-apt-repository ppa:jrjohansson/qutip-releases
    sudo apt-get update
    sudo apt-get install python-qutip

With this method the most important dependencies are installed automatically, and when a new version of QuTiP is released it can be upgraded through the standard package management system. In addition to the required dependencies, it is also strongly recommended that you install the ``texlive-latex-extra`` package::

    sudo apt-get install texlive-latex-extra

Manual installation of dependencies
-----------------------------------

First install the required dependencies using:

.. code-block:: bash

    sudo apt-get install python-dev cython python-setuptoolspython-nose
    sudo apt-get install python-numpy python-scipy python-matplotlib

Then install QuTiP from source following the instructions given above.

Alternatively (or additionally), to install a Python 3 environment, use:

.. code-block:: bash

    sudo apt-get install python3-dev cython python3-setuptools python3-nose
    sudo apt-get install python3-numpy python3-scipy python3-matplotlib

and then do the installation from source using ``python3`` instead of ``python``.

Optional, but recommended, dependencies can be installed using:

.. code-block:: bash

    sudo apt-get install texlive-latex-extra # recommended
    sudo apt-get install mayavi2             # optional, for Bloch3d only
    sudo apt-get install libblas-dev         # optional, for Fortran Monte Carlo solver
    sudo apt-get install gfortran            # optional, for Fortran Monte Carlo solver

.. _install-mac:

Installation on Mac OS X (10.6+)
=================================

If you have not done so already, install the Apple Xcode developer tools from the Apple App Store.  After installation, open Xcode and go to: Preferences -> Downloads, and install the 'Command Line Tools'.

Setup Using Macports
---------------------

On the Mac OS, we recommended that you install the required libraries via `MacPorts <http://www.macports.org/ MacPorts>`_.  After installation, the necessary "ports" for QuTiP may be installed via

.. code-block:: bash

    sudo port install py34-scipy
    sudo port install py34-matplotlib +latex
    sudo port install py34-cython
    sudo port install py34-ipython +notebook+parallel


Optional, but highly recommended ports include

.. code-block:: bash

    sudo port install vtk5 +python27          #used for the Bloch3d class
    sudo port install py27-mayavi             #used for the Bloch3d class

Now, we want to tell OSX which Python and iPython we are going to use

.. code-block:: bash

    sudo port select python python34
    sudo port select ipython ipython34

To install QuTiP from Macports, run

.. code-block:: bash

    sudo port install py-qutip

.. note:: The next step is optional, but is necessary if you plan to use the string (Cython) based time-dependent format.  See :ref:`time`.

Finally, we want to set the macports compiler to the vanilla GCC version.  From the command line type::

    port select gcc

which will bring up a list of installed compilers, such as::

	Available versions for gcc:
		apple-gcc42
		gcc42
		llvm-gcc42
		mp-gcc47
		none (active)

We want to set the the compiler to the gcc4x compiler, where x is the highest number available, in this case ``mp-gcc47`` (the "mp-" does not matter).  To do this type::

    sudo port select gcc mp-gcc47

Running port select again should give::

	 Available versions for gcc:
	 	apple-gcc42
	 	gcc42
	 	llvm-gcc42
	 	mp-gcc47 (active)
	 	none


Installing QuTiP via Macports will take a long time as each of the QuTiP dependencies is build from source code. The advantage is that all dependencies are resolved automatically, and the result should be a consistent build.


Setup via SciPy Superpack
-------------------------

A second option is to install the required Python packages using the `SciPy Superpack <http://fonnesbeck.github.com/ScipySuperpack/>`_.  Further information on installing the superpack can be found on the `SciPy Downloads page <http://www.scipy.org/Download>`_. 


Anaconda CE Distribution
------------------------

Finally, one can also use the `Anaconda CE <https://store.continuum.io/cshop/anaconda>`_ package to install all of the QuTiP 


.. _install-win:

Installation on Windows
=======================

QuTiP is primarily developed for Unix-based platforms such as Linux an Mac OS X, but it can also be used on Windows. We have limited experience and ability to help troubleshoot problems on Windows, but the following installation steps have been reported to work:

1. Install the `Python(X,Y) <http://code.google.com/p/pythonxy/>`_ distribution (tested with version 2.7.3.1). Other Python distributions, such as `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ or `Anaconda CE <http://continuum.io/downloads.html>`_ have also been reported to work.

2. When installing Python(x,y), explicitly select to include the Cython package in the installation. This package is not selected by default.

3. Add the following content to the file `C:/Python27/Lib/distutils/distutils.cfg` (or create the file if it does not already exists)::

    [build]
    compiler = mingw32

    [build_ext]
    compiler = mingw32

The directory where the distutils.cfg file should be placed might be different if you have installed the Python environment in a different location than in the example above.

4. Obtain the QuTiP source code and installed it following the instructions given above.

.. note::

    In some cases, to get the dynamic compilation of Cython code to work, it
    might be necessary to edit the PATH variable and make sure that
    `C:\\MinGW32-xy\\bin` appears either *first* in the PATH list, or possibly
    *right after* `C:\\Python27\\Lib\\site-packages\\PyQt4`. This is to make sure
    that the right version of the MinGW compiler is used if more than one
    is installed (not uncommon under Windows, since many packages are
    distributed and installed with their own version of all dependencies).


.. _install-verify:

Verifying the Installation
==========================

QuTiP includes a collection of built-in test scripts to verify that an installation was successful. To run the suite of tests scripts you must have the nose testing library. After installing QuTiP, leave the installation directory, run Python (or iPython), and call:

>>> import qutip.testing as qt
>>> qt.run()

If successful, these tests indicate that all of the QuTiP functions are working properly.  If any errors occur, please check that your have installed all of the required modules.  See the next section on how to check the installed versions of the QuTiP dependencies. If these tests still fail, then head on over to the `QuTiP Discussion Board <http://groups.google.com/group/qutip>`_ and post a message detailing your particular issue.

.. _install-about:

Checking Version Information using the About Function
=====================================================

QuTiP includes an "about" function for viewing information about QuTiP and the important dependencies installed on your system.  To view this information:

>>> from qutip import *
>>> about()
