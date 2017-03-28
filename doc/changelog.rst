.. QuTiP 
   Copyright (C) 2011-2013, Paul D. Nation, Robert J. Johansson & Alexander Pitchford

.. _changelog:

**********
Change Log
**********

Version 4.1.0 (March 10, 2017)
++++++++++++++++++++++++++++++

Improvements
------------

*Core libraries*

- **MAJOR FEATURE**: QuTiP now works for Python 3.5+ on Windows using Visual Studio 2015.

- **MAJOR FEATURE**: Cython and other low level code switched to C++ for MS Windows compatibility.

- **MAJOR FEATURE**: Can now use interpolating cubic splines as time-dependent coefficients.

- **MAJOR FEATURE**: Sparse matrix - vector multiplication now parallel using OPENMP.

- Automatic tuning of OPENMP threading threashold.

- Partial trace function is now up to 100x+ faster.

- Hermitian verification now up to 100x+ faster.

- Internal Qobj objects now created up to 60x faster.

- Inplace conversion from COO -> CSR sparse formats (e.g. Memory efficiency improvement.)

- Faster reverse Cuthill-Mckee and sparse one and inf norms.



Bug Fixes
---------

- Cleanup of temp. Cython files now more robust and working under Windows.



Version 4.0.2 (January 5, 2017)
+++++++++++++++++++++++++++++++

Bug Fixes
---------
- td files no longer left behind by correlation tests
- Various fast sparse fixes



Version 4.0.0 (December 22, 2016)
+++++++++++++++++++++++++++++++++

Improvements
------------
*Core libraries*

- **MAJOR FEATURE**: Fast sparse: New subclass of csr_matrix added that overrides commonly used methods to avoid certain checks that incurr execution cost. All Qobj.data now fast_csr_matrix
- HEOM performance enhancements
- spmv now faster
- mcsolve codegen further optimised

*Control modules*

- Time dependent drift (through list of pwc dynamics generators)
- memory optimisation options provided for control.dynamics

Bug Fixes
---------

- recompilation of pyx files on first import removed
- tau array in control.pulseoptim funcs now works

Version 3.2.0 (Never officially released)
+++++++++++++++++++++++++++++++++++++++++

New Features
------------

*Core libraries*

- **MAJOR FEATURE**: Non-Markovian solvers: Hierarchy (**Added by Neill Lambert**), Memory-Cascade, and Transfer-Tensor methods.
- **MAJOR FEATURE**: Default steady state solver now up to 100x faster using the Intel Pardiso library under the Anaconda and Intel Python distributions.
- The default Wigner function now uses a Clenshaw summation algorithm to evaluate a polynomial series that is applicable for any number of exciations (previous limitation was ~50 quanta), and is ~3x faster than before. (**Added by Denis Vasilyev**)
- Can now define a given eigen spectrum for random Hermitian and density operators. 
- The Qobj ``expm`` method now uses the equivilent SciPy routine, and performs a much faster ``exp`` operation if the matrix is diagonal.
- One can now build zero operators using the ``qzero`` function.

*Control modules*

- **MAJOR FEATURE**: CRAB algorithm added
  This is an alternative to the GRAPE algorithm, which allows for analytical control functions, which means that experimental constraints can more easily be added into optimisation.  
  See tutorial notebook for full information.


Improvements
------------
*Core libraries*

- Two-time correlation functions can now be calculated for fully time-dependent Hamiltonians and collapse operators. (**Added by Kevin Fischer**)
- The code for the inverse-power method for the steady state solver has been simplified.
- Bloch-Redfield tensor creation is now up to an order of magnitude faster. (**Added by Johannes Feist**)
- Q.transform now works properly for arrays directly from sp_eigs (or eig).
- Q.groundstate now checks for degeneracy.
- Added ``sinm`` and ``cosm`` methods to the Qobj class.
- Added ``charge`` and ``tunneling`` operators.
- Time-dependent Cython code is now easier to read and debug.


*Control modules*

- The internal state / quantum operator data type can now be either Qobj or ndarray
  Previous only ndarray was possible. This now opens up possibility of using Qobj methods in fidelity calculations
  The attributes and functions that return these operators are now preceded by an underscore, to indicate that the data type could change depending on the configuration options. 
  In most cases these functions were for internal processing only anyway, and should have been 'private'. 
  Accessors to the properties that could be useful outside of the library have been added. These always return Qobj. If the internal operator data type is not Qobj, then there could be signicant overhead in the conversion, and so this should be avoided during pulse optimisation.
  If custom sub-classes are developed that use Qobj properties and methods (e.g. partial trace), then it is very likely that it will be more efficient to set the internal data type to Qobj.
  The internal operator data will be chosen automatically based on the size and sparsity of the dynamics generator. It can be forced by setting ``dynamics.oper_dtype = <type>``
  Note this can be done by passing ``dyn_params={'oper_dtype':<type>}`` in any of the pulseoptim functions.
  
  Some other properties and methods were renamed at the same time. A full list is given here.
  
  - All modules 
    - function: ``set_log_level`` -> property: ``log_level``

  - dynamics functions
  
    - ``_init_lists`` now ``_init_evo``
    - ``get_num_ctrls`` now property: ``num_ctrls``
    - ``get_owd_evo_target`` now property: ``onto_evo_target``
    - ``combine_dyn_gen`` now ``_combine_dyn_gen`` (no longer returns a value)
    - ``get_dyn_gen`` now ``_get_phased_dyn_gen``
    - ``get_ctrl_den_gen`` now ``_get_phased_ctrl_dyn_gen``
    - ``ensure_decomp_curr`` now ``_ensure_decomp_curr``
    - ``spectral_decomp`` now ``_spectral_decomp``
    
  - dynamics properties
  
    - ``evo_init2t`` now ``_fwd_evo`` (``fwd_evo`` as Qobj)
    - ``evo_t2end`` now ``_onwd_evo`` (``onwd_evo`` as Qobj)
    - ``evo_t2targ`` now ``_onto_evo`` (``onto_evo`` as Qobj)

  - fidcomp properties
  
    - ``uses_evo_t2end`` now ``uses_onwd_evo``
    - ``uses_evo_t2targ`` now ``uses_onto_evo``
    - ``set_phase_option`` function now property ``phase_option``

  - propcomp properties
  
    - ``grad_exact`` (now read only)

  - propcomp functions
  
    - ``compute_propagator`` now ``_compute_propagator``
    - ``compute_diff_prop`` now ``_compute_diff_prop``
    - ``compute_prop_grad`` now ``_compute_prop_grad``

  - tslotcomp functions
  
    - ``get_timeslot_for_fidelity_calc`` now ``_get_timeslot_for_fidelity_calc``


*Miscellaneous*

- QuTiP Travis CI tests now use the Anaconda distribution.
- The ``about`` box and ipynb ``version_table`` now display addition system information.
- Updated Cython cleanup to remove depreciation warning in sysconfig.
- Updated ipynb_parallel to look for ``ipyparallel`` module in V4 of the notebooks.


Bug Fixes
---------
- Fixes for countstat and psuedo-inverse functions
- Fixed Qobj division tests on 32-bit systems.
- Removed extra call to Python in time-dependent Cython code.
- Fixed issue with repeated Bloch sphere saving.
- Fixed T_0 triplet state not normalized properly. (**Fixed by Eric Hontz**)
- Simplified compiler flags (support for ARM systems).
- Fixed a decoding error in ``qload``.
- Fixed issue using complex.h math and np.kind_t variables.
- Corrected output states mismatch for ``ntraj=1`` in the mcf90 solver.
- Qobj data is now copied by default to avoid a bug in multiplication. (**Fixed by Richard Brierley**)
- Fixed bug overwriting ``hardware_info`` in ``__init__``. (**Fixed by Johannes Feist**)
- Restored ability to explicity set Q.isherm, Q.type, and Q.superrep.
- Fixed integer depreciation warnings from NumPy.
- Qobj * (dense vec) would result in a recursive loop.
- Fixed args=None -> args={} in correlation functions to be compatible with mesolve.
- Fixed depreciation warnings in mcsolve.
- Fixed neagtive only real parts in ``rand_ket``.
- Fixed a complicated list-cast-map-list antipattern in super operator reps. (**Fixed by Stefan Krastanov**)
- Fixed incorrect ``isherm`` for ``sigmam`` spin operator.
- Fixed the dims when using ``final_state_output`` in ``mesolve`` and ``sesolve``.



Version 3.1.0 (January 1, 2015):
++++++++++++++++++++++++++++++++

New Features
-------------

- **MAJOR FEATURE**: New module for quantum control (qutip.control).
- **NAMESPACE CHANGE**: QuTiP no longer exports symbols from NumPy and matplotlib, so those modules must now be explicitly imported when required.
- New module for counting statistics.
- Stochastic solvers now run trajectories in parallel.
- New superoperator and tensor manipulation functions
  (super_tensor, composite, tensor_contract).
- New logging module for debugging (qutip.logging).
- New user-available API for parallelization (parallel_map).
- New enhanced (optional) text-based progressbar (qutip.ui.EnhancedTextProgressBar)
- Faster Python based monte carlo solver (mcsolve).
- Support for progress bars in propagator function.
- Time-dependent Cython code now calls complex cmath functions.
- Random numbers seeds can now be reused for successive calls to mcsolve.
- The Bloch-Redfield master equation solver now supports optional Lindblad type collapse operators.
- Improved handling of ODE integration errors in mesolve.
- Improved correlation function module (for example, improved support for time-dependent problems).
- Improved parallelization of mcsolve (can now be interrupted easily, support for IPython.parallel, etc.)
- Many performance improvements, and much internal code restructuring.

Bug Fixes
---------

- Cython build files for time-dependent string format now removed automatically.
- Fixed incorrect solution time from inverse-power method steady state solver.
- mcsolve now supports `Options(store_states=True)`
- Fixed bug in `hadamard` gate function.
- Fixed compatibility issues with NumPy 1.9.0.
- Progressbar in mcsolve can now be suppressed.
- Fixed bug in `gate_expand_3toN`.
- Fixed bug for time-dependent problem (list string format) with multiple terms in coefficient to an operator.

Version 3.0.1 (Aug 5, 2014):
++++++++++++++++++++++++++++

Bug Fixes
---------

- Fix bug in create(), which returned a Qobj with CSC data instead of CSR.
- Fix several bugs in mcsolve: Incorrect storing of collapse times and collapse
  operator records. Incorrect averaging of expectation values for different
  trajectories when using only 1 CPU.
- Fix bug in parsing of time-dependent Hamiltonian/collapse operator arguments
  that occurred when the args argument is not a dictionary.
- Fix bug in internal _version2int function that cause a failure when parsingthe version number of the Cython package.
- 


Version 3.0.0 (July 17, 2014):
++++++++++++++++++++++++++++++

New Features
-------------

- New module `qutip.stochastic` with stochastic master equation and stochastic
  Schrödinger equation solvers.

- Expanded steady state solvers. The function ``steady`` has been deprecated in
  favor of ``steadystate``. The steadystate solver no longer use umfpack by 
  default. New pre-processing methods for reordering and balancing the linear
  equation system used in direct solution of the steady state.

- New module `qutip.qip` with utilities for quantum information processing,
  including pre-defined quantum gates along with functions for expanding
  arbitrary 1, 2, and 3 qubit gates to N qubit registers, circuit
  representations, library of quantum algorithms, and basic physical models for
  some common QIP architectures. 

- New module `qutip.distributions` with unified API for working with
  distribution functions.

- New format for defining time-dependent Hamiltonians and collapse operators,
  using a pre-calculated numpy array that specifies the values of the
  Qobj-coefficients for each time step.

- New functions for working with different superoperator representations,
  including Kraus and Chi representation.

- New functions for visualizing quantum states using Qubism and Schimdt plots:
  ``plot_qubism`` and ``plot_schmidt``.

- Dynamics solver now support taking argument ``e_ops`` (expectation value
  operators) in dictionary form.

- Public plotting functions from the ``qutip.visualization`` module are now 
  prefixed with ``plot_`` (e.g., ``plot_fock_distribution``). The
  ``plot_wigner`` and ``plot_wigner_fock_distribution`` now supports 3D views
  in addition to contour views.

- New API and new functions for working with spin operators and states,
  including for example ``spin_Jx``, ``spin_Jy``, ``spin_Jz`` and
  ``spin_state``, ``spin_coherent``.

- The ``expect`` function now supports a list of operators, in addition to the
  previously supported list of states.

- Simplified creation of qubit states using ``ket`` function.

- The module ``qutip.cyQ`` has been renamed to ``qutip.cy`` and the sparse
  matrix-vector functions ``spmv`` and ``spmv1d`` has been combined into one
  function ``spmv``. New functions for operating directly on the underlaying
  sparse CSR data have been added (e.g., ``spmv_csr``). Performance
  improvements. New and improved Cython functions for calculating expectation
  values for state vectors, density matrices in matrix and vector form.

- The ``concurrence`` function now supports both pure and mixed states. Added
  function for calculating the entangling power of a two-qubit gate.
 
- Added function for generating (generalized) Lindblad dissipator
  superoperators.

- New functions for generating Bell states, and singlet and triplet states.

- QuTiP no longer contains the demos GUI. The examples are now available on the
  QuTiP web site. The ``qutip.gui`` module has been renamed to ``qutip.ui`` and
  does no longer contain graphical UI elements. New text-based and HTML-based
  progressbar classes.

- Support for harmonic oscillator operators/states in a Fock state basis that
  does not start from zero (e.g., in the range [M,N+1]). Support for
  eliminating and extracting states from Qobj instances (e.g., removing one
  state from a two-qubit system to obtain a three-level system).

- Support for time-dependent Hamiltonian and Liouvillian callback functions that
  depend on the instantaneous state, which for example can be used for solving
  master equations with mean field terms.

Improvements
-------------

- Restructured and optimized implementation of Qobj, which now has
  significantly lower memory footprint due to avoiding excessive copying of
  internal matrix data.

- The classes ``OdeData``, ``Odeoptions``, ``Odeconfig`` are now called 
  ``Result``, ``Options``, and ``Config``, respectively, and are available in
  the module `qutip.solver`.

- The ``squeez`` function has been renamed to ``squeeze``.

- Better support for sparse matrices when calculating propagators using the 
  ``propagator`` function.

- Improved Bloch sphere.

- Restructured and improved the module ``qutip.sparse``, which now only
  operates directly on sparse matrices (not on Qobj instances).

- Improved and simplified implement of the ``tensor`` function.

- Improved performance, major code cleanup (including namespace changes),
  and numerous bug fixes.

- Benchmark scripts improved and restructured.

- QuTiP is now using continuous integration tests (TravisCI).

Version 2.2.0 (March 01, 2013):
++++++++++++++++++++++++++++++++++++++++++++++


New Features
-------------

- **Added Support for Windows**

- New Bloch3d class for plotting 3D Bloch spheres using Mayavi.

- Bloch sphere vectors now look like arrows.

- Partial transpose function.

- Continuos variable functions for calculating correlation and covariance
  matrices, the Wigner covariance matrix and the logarithmic negativity for
  for multimode fields in Fock basis.

- The master-equation solver (mesolve) now accepts pre-constructed Liouvillian
  terms, which makes it possible to solve master equations that are not on
  the standard Lindblad form.
  
- Optional Fortran Monte Carlo solver (mcsolve_f90) by Arne Grimsmo.

- A module of tools for using QuTiP in IPython notebooks.

- Increased performance of the steady state solver.

- New Wigner colormap for highlighting negative values.

- More graph styles to the visualization module.


Bug Fixes:
----------

- Function based time-dependent Hamiltonians now keep the correct phase.

- mcsolve no longer prints to the command line if ntraj=1.


Version 2.1.0 (October 05, 2012):
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

- mesolver now correctly uses the user defined rhs_filename in Odeoptions().

- rhs_generate() now handles user defined filenames properly.

- Density matrix returned by propagator_steadystate is now Hermitian.

- eseries_value returns real list if all imag parts are zero.

- mcsolver now gives correct results for strong damping rates.

- Odeoptions now prints mc_avg correctly.

- Do not check for PyObj in mcsolve when gui=False. 

- Eseries now correctly handles purely complex rates.

- thermal_dm() function now uses truncated operator method.

- Cython based time-dependence now Python 3 compatible.

- Removed call to NSAutoPool on mac systems.

- Progress bar now displays the correct number of CPU's used.

- Qobj.diag() returns reals if operator is Hermitian.

- Text for progress bar on Linux systems is no longer cutoff.


Version 2.0.0 (June 01, 2012):
+++++++++++++++++++++++++++++++++++++++++

The second version of QuTiP has seen many improvements in the performance of the original code base, as well as the addition of several new routines supporting a wide range of functionality.  Some of the highlights of this release include:

New Features
-------------

- QuTiP now includes solvers for both Floquet and Bloch-Redfield master equations.

- The Lindblad master equation and Monte Carlo solvers allow for time-dependent collapse operators.

- It is possible to automatically compile time-dependent problems into c-code using Cython (if installed).

- Python functions can be used to create arbitrary time-dependent Hamiltonians and collapse operators.

- Solvers now return Odedata objects containing all simulation results and parameters, simplifying the saving of simulation results.

.. important:: This breaks compatibility with QuTiP version 1.x.

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


Version 1.1.4 (May 28, 2012):
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Bug Fixes:
----------

- Fixed bug pointed out by Brendan Abolins.

- Qobj.tr() returns zero-dim ndarray instead of float or complex.

- Updated factorial import for scipy version 0.10+


Version 1.1.3 (November 21, 2011):
+++++++++++++++++++++++++++++++++++++++++++++

New Functions:
--------------

- Allow custom naming of Bloch sphere.

Bug Fixes:
----------
- Fixed text alignment issues in AboutBox.

- Added fix for SciPy V>0.10 where factorial was moved to scipy.misc module.

- Added tidyup function to tensor function output.

- Removed openmp flags from setup.py as new Mac Xcode compiler does not recognize them.

- Qobj diag method now returns real array if all imaginary parts are zero.

- Examples GUI now links to new documentation.

- Fixed zero-dimensional array output from metrics module.


Version 1.1.2 (October 27, 2011)
+++++++++++++++++++++++++++++++++++++++++++

Bug Fixes
---------

- Fixed issue where Monte Carlo states were not output properly.


Version 1.1.1 (October 25, 2011)
+++++++++++++++++++++++++++++++++++++++++++

**THIS POINT-RELEASE INCLUDES VASTLY IMPROVED TIME-INDEPENDENT MCSOLVE AND ODESOLVE PERFORMANCE**

New Functions
---------------

- Added linear entropy function.

- Number of CPU's can now be changed.

Bug Fixes
---------

- Metrics no longer use dense matrices.

- Fixed Bloch sphere grid issue with matplotlib 1.1.

- Qobj trace operation uses only sparse matrices.

- Fixed issue where GUI windows do not raise to front.


Version 1.1.0 (October 04, 2011)
+++++++++++++++++++++++++++++++++++++++++++

**THIS RELEASE NOW REQUIRES THE GCC COMPILER TO BE INSTALLED**

New Functions
---------------

- tidyup function to remove small elements from a Qobj.

- Added concurrence function.

- Added simdiag for simultaneous diagonalization of operators.

- Added eigenstates method returning eigenstates and eigenvalues to Qobj class.

- Added fileio for saving and loading data sets and/or Qobj's.

- Added hinton function for visualizing density matrices.

Bug Fixes
---------

- Switched Examples to new Signals method used in PySide 1.0.6+.

- Switched ProgressBar to new Signals method.

- Fixed memory issue in expm functions.

- Fixed memory bug in isherm.

- Made all Qobj data complex by default.

- Reduced ODE tolerance levels in Odeoptions.

- Fixed bug in ptrace where dense matrix was used instead of sparse.

- Fixed issue where PyQt4 version would not be displayed in about box.

- Fixed issue in Wigner where xvec was used twice (in place of yvec).


Version 1.0.0 (July 29, 2011)
+++++++++++++++++++++++++++++++++++++++++

- **Initial release.**
