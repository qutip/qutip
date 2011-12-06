.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

Change Log
**********

Version 1.1.3 [svn-1450] (November 21, 2011):
+++++++++++++++++++++++++++++++++++++++++++++

New Functions:
--------------

SVN-1347: Allow custom naming of Bloch sphere.

Bug Fixes:
----------
SVN-1450: Fixed text alignment issues in AboutBox.

SVN-1448: Added fix for SciPy V>0.10 where factorial was moved to scipy.misc module.

SVN-1447: Added tidyup function to tensor function output.

SVN-1442: Removed openmp flags from setup.py as new Mac Xcode compiler does not recognize them.

SVN-1435: Qobj diag method now returns real array if all imaginary parts are zero.

SVN-1434: Examples GUI now links to new documentation.

SVN-1415: Fixed zero-dimensional array output from metrics module.


Version 1.1.2 [svn-1218] (October 27, 2011)
+++++++++++++++++++++++++++++++++++++++++++

Bug Fixes
---------

SVN-1218: Fixed issue where Monte-Carlo states were not output properly.


Version 1.1.1 [svn-1210] (October 25, 2011)
+++++++++++++++++++++++++++++++++++++++++++

**THIS POINT-RELEASE INCLUDES VASTLY IMPROVED TIME-INDEPENDENT MCSOLVE AND ODESOLVE PERFORMANCE**

New Functions
---------------

SVN-1183: Added linear entropy function.

SVN-1179: Number of CPU's can now be changed.

Bug Fixes
---------

SVN-1184: Metrics no longer use dense matrices.

SVN-1184: Fixed Bloch sphere grid issue with matplotlib 1.1.

SVN-1183: Qobj trace operation uses only sparse matrices.

SVN-1168: Fixed issue where GUI windows do not raise to front.


Version 1.1.0 [svn-1097] (October 04, 2011)
+++++++++++++++++++++++++++++++++++++++++++

**THIS RELEASE NOW REQUIRES THE GCC COMPILER TO BE INSTALLED**

New Functions
---------------

SVN-1054: tidyup function to remove small elements from a Qobj.

SVN-1051: Added concurrence function.

SVN-1036: Added simdiag for simultaneous diagonalization of operators.

SVN-1032: Added eigenstates method returning eigenstates and eigenvalues to Qobj class.

SVN-1030: Added fileio for saving and loading data sets and/or Qobj's.

SVN-1029: Added hinton function for visualizing density matrices.

Bug Fixes
---------

SVN-1091: Switched Examples to new Signals method used in PySide 1.0.6+.

SVN-1090: Switched ProgressBar to new Signals method.

SVN-1075: Fixed memory issue in expm functions.

SVN-1069: Fixed memory bug in isherm.

SVN-1059: Made all Qobj data complex by default.

SVN-1053: Reduced ODE tolerance levels in Odeoptions.

SVN-1050: Fixed bug in ptrace where dense matrix was used instead of sparse.

SVN-1047: Fixed issue where PyQt4 version would not be displayed in about box.

SVN-1041: Fixed issue in Wigner where xvec was used twice (in place of yvec).


Version 1.0.0 [svn-1021] (July 29, 2011)
+++++++++++++++++++++++++++++++++++++++++

Initial release.