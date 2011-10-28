.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

Installation
*************

General Installation Requirements
=================================

QuTiP requires the following packages to run:

+------------+--------------+-----------------------------------------------------+
| Package    | Version      | Details                                             |
+============+==============+=====================================================+
| Python     | 2.6+ (not 3) | Requires multiprocessing (v2.6 and higher only).    |
|            |              | At present, Matplotlib does not work for Python 3.  |
+------------+--------------+-----------------------------------------------------+
| Numpy      | 1.5.1+       | Not tested on lower versions.                       |
+------------+--------------+-----------------------------------------------------+
| Scipy      | 0.8+         | Not tested on lower versions. Use 0.9+ is possible. |
+------------+--------------+-----------------------------------------------------+
| Matplotlib | 1.0.1+       | Some plotting does not work on lower versions.      |
+------------+--------------+-----------------------------------------------------+
| Qt         |  4.7.3+      | Optional.  For GUI elements only.                   |
+------------+--------------+-----------------------------------------------------+
| PySide     | 1.0.2+       | Optional, required only for GUI elements.           |
|            |              | PyQt4 may be used instead.                          |
+------------+--------------+-----------------------------------------------------+
| PyQt4      | 1.0.2+       | Optional, required only for GUI elements.           |
|            |              | PySide may be used instead (recommended).           |
+------------+--------------+-----------------------------------------------------+                      
| PyObjC     | 2.2+         | Mac only.  Very optional.  Needed only for a        |
|            |              | GUI Monte-Carlo progress bar.                       |
+------------+--------------+-----------------------------------------------------+
| GCC        | 4.2+         | Needed for compiling Cython files.                  |
| Compiler   |              |                                                     |
+------------+--------------+-----------------------------------------------------+
| Python     | 2.6+         | Linux only.  Needed for compiling Cython files.     |
| Headers    |              |                                                     |
+------------+--------------+-----------------------------------------------------+

On all platforms (Linux, Mac, Windows), QuTiP works "out-of-the-box" using the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ version 7.1 or higher.  This distribution is created by the developers of Numpy and Scipy, and is free for academic use.

Installation on Ubuntu Linux
++++++++++++++++++++++++++++

>>> sudo apt-get install python-scipy
>>> sudo apt-get install python-pyside or sudo apt-get install python-qt4
>>> sudo apt-get install python-setuptools
>>> sudo apt-get install python-dev

At present, Ubuntu 11.04 and lower do not have Matplotlib>=1.0 therefore we need to add the unofficial repository (in Ubuntu 11.10, skip this step)

>>> sudo add-apt-repository ppa:bgamari/matplotlib-unofficial
>>> sudo apt-get update

before running

>>> sudo apt-get install python-matplotlib


QuTiP installation:

>>> sudo python setup.py install


Installation on Mac OS X (10.6+)
++++++++++++++++++++++++++++++++

If you have not done so already, install the Apple XCode developer tools from the Apple App Store.

On the Mac, it is recommended that you install the required libraries via `MacPorts <http://www.macports.org/ MacPorts>`_.  After installing with MacPorts, you may need to change your matplotlib backend

>>> sudo open -a TextEdit /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc

on line #31 to read:

>>> backend      : MacOSX


One can also use the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ version 7.1 or higher to satisfy the QuTiP dependencies.  

A further possibility is to use the `Scipy Superpack installer script <http://stronginference.com/scipy-superpack/>`_.  In order to have GUI elements using this method, you need to further install the Qt4 and PySide packages at:

http://qt.nokia.com/downloads/qt-for-open-source-cpp-development-on-mac-os-x

http://developer.qt.nokia.com/wiki/PySide_Binaries_MacOSX

Installing QuTiP is the same as on linux.  From the QuTiP directory:

>>> sudo python setup.py install


Installation on Microsoft Windows
+++++++++++++++++++++++++++++++++

The developers of QuTiP have not touched Windows in several years, and will be continuing this trend for the foreseeable future.  Therefore we recommend the `Enthought Python Distribution <http://www.enthought.com/products/epd.php>`_ version 7.1 or higher to satisfy the QuTiP dependencies. QuTiP has also been reported to work out-of-the-box with `Python(x,y) <http://www.pythonxy.com>`_.  In Python(x,y), QuTiP may be installed using:

>>> python setup.py install build --compiler=mingw32

Does it work?
=============

DO NOT RUN QUTIP FROM THE INSTALLATION DIRECTORY


To verify that everything is installed properly, from the python command line call:

>>> from qutip import *

If nothing but another command prompt appears then you are ready to go.  To see if the GUI components are working, after the import statement type:

>>> about()

which will pop-up the about box for QuTiP which gives you information on the installed version of QuTiP and its dependencies.  If instead you get command-line output, then your graphics is not installed properly or unavailable.

.. _about: 
.. figure:: http://qutip.googlecode.com/svn/wiki/images/about.png
   :align: center
