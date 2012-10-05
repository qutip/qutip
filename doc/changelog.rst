.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

**********
Change Log
**********

Version 2.1.0 [SVN-2679] (October 05, 2012):
++++++++++++++++++++++++++++++++++++++++++++++

New Features
-------------

- New method for generating Wigner functions based on Laguerre polynomials.

- coherent(), coherent_dm(), and thermal_dm() can now be expressed using analytic values.

- Unittests now use nose and can be run after installation.

- Added iswap and sqrt-iswap gates.

- Functions for quantum process tomography.

- Window icons are now set for Ubuntu application launcher.

- The propagator function can now take a list of times as argument, and returns a list of corresponding propagators.


Bug Fixes:
----------

SVN-2571: mesolver now correctly uses the user defined rhs_filename in Odeoptions().

SVN-2566: rhs_generate() now handles user defined filenames properly.

SVN-2565: Density matrix returned by propagator_steadystate is now Hermitian.

SVN-2548: eseries_value returns real list if all imag parts are zero.

SVN-2518: mcsolver now gives correct results for strong damping rates.

SVN-2513: Odeoptions now prints mc_avg correctly.

SVN-2516: Do not check for PyObj in mcsolve when gui=False. 

SVN-2514: Eseries now correctly handles purely complex rates.

SVN-2485: thermal_dm() function now uses truncated operator method.

SVN-2428: Cython based time-dependence now Python 3 compatible.

SVN-2391: Removed call to NSAutoPool on mac systems.

SVN-2389: Progress bar now displays the correct number of CPU's used.

SVN-2385: Qobj.diag() returns reals if operator is Hermitian.

SVN-2376: Text for progress bar on Linux systems is no longer cutoff.



Version 2.0.0 [SVN-2354] (June 01, 2012):
+++++++++++++++++++++++++++++++++++++++++

New Features
-------------

- QuTiP now includes solvers for both Floquet and Bloch-Redfield master equations.

- The Lindblad master equation and monte-carlo solvers allow for time-dependent collapse operators.

- It is possible to automatically compile time-dependent problems into c-code using Cython (if installed).

- Python functions can be used to create arbitrary time-dependent Hamiltonians and collapse operators.

- Solvers now return Odedata objects containing all simulation results and parameters, simplifying the saving of simulation results.

- mesolve and mcsolve can reuse Hamiltonian data when only the initial state, or time-dependent arguments, need to be changed.

- QuTiP includes functions for creating random quantum states and operators.

- The generation and manipulation of quantum objects is now more efficient.

- Quantum objects have basis transformation and matrix element calculations as built-in methods.

- The quantum object eigensolver can use sparse solvers.

- The partial-trace (ptrace) function is up to 20x faster.

- The Bloch sphere can now be used with the Matplotlib animation function, and embedded as a subplot in a figure.

- QuTiP has built-in functions for saving quantum objects and data arrays.

- The steady-state solver has been further optimized for sparse matrices, and can handle much larger system Hamiltonians.

- The steady-state solver can use the iterative bi-conjugate gradient method instead of a direct solver.

- There are three new entropy functions for concurrence, mutual information, and conditional entropy.

- Correlation functions have been combined under a single function.

- The operator norm can now be set to trace, Frobius, one, or max norm.

- Global QuTiP settings can now be modified.

- QuTiP includes a collection of unit tests for verifying the installation.

- Demos window now lets you copy and paste code from each example.



Version 1.1.4 [fixes backported to SVN-1450] (May 28, 2012):
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Bug Fixes:
----------

SVN-2101: Fixed bug pointed out by Brendan Abolins.

SVN-1796: Qobj.tr() returns zero-dim ndarray instead of float or complex.

SVN-1463: Updated factorial import for scipy version 0.10+


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

**Initial release.**
