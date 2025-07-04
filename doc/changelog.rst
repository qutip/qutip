.. _changelog:

**********
Change Log
**********

.. towncrier release notes start

QuTiP 5.2.0 (2025-06-06)
========================

Features
--------

Improvements to Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Two 'DrudeLorentzPadeBath' can be added with complex conjugate 'gammas' to create a Shifted-DrudeLorentzPadeBath. (#2574, by Akhil Bhartiya)
- Adds extra fitting methods (power spectrum, prony, esprit, aaa, espira-I, espira-II) for bosonic environments (#2594, by Gerardo Jose Suarez)
- Add support for `FermionicEnvironment` in Bloch Redfield tensor and solver. (#2628)

New experimental solver
^^^^^^^^^^^^^^^^^^^^^^^
- Add a first version of Dysolve solver as ``dysolve_propagator``.
  A perturbation solver for Schrödinger equation optimized for driven systems.
  (#2648, #2679, by Mathis)

Better matrix format control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Add "default_dtype_scope" option.
  Give more control on matrix format used by operations. (#2519)
- Creating operator from dense ket.proj return CSR when appropriate (#2700)
- `|ket> @ <bra|` create sparse matrix per default. (#2611)
- Tensor product of sparse and dense matrices return sparse matrix per default. (#2391)

Other improvements
^^^^^^^^^^^^^^^^^^
- Add the calculation of the third cumulant to the functions of countstat.py.
  I use a result of "Electrons in nanostructures" C. Flindt, PhD Thesis (#2435, by Daniel Moreno Galán)
- Support pretty-printing of basis expansion for states (e.g. ``(0.5+0j) |0010> + ...``). (#2620, by Sola85)
- Added QuTiP family package information to qutip.about().
  The information is provided by a new QuTiP family package entry point. (#2604)
- Speed up ptrace for kets (by several orders of magnitude for large states). (#2657, by Sola85)
- Implement partial trace and tensor product for states in excitation-number restricted space. (#2696, magzpavz)

Bug Fixes
---------

- Fix arc behind hidden behind the wireframe. (#2467, by PositroniumJS)
- Update mkl finding to support the 'emscripten' sys.platform. (#2606)
- Make sure that dims of rand_ket output are set correctly. (#2613, reported by Mário Filho)
- Allow np.int as targets in expand_operator (#2634, by Sola85)
- Fix coefficient `args` not being properly updated
  when initiated from another coefficient. (#2646, reported by Kyle Wamer)
- Fix equality operators for diagonal format Qobj. (#2669, reported by Richard Dong)
- Enable again to plot a state as a point on the Bloch sphere with a RGBA tuple.
  Enable to do the same as a vector. (#2678, by PositroniumJS)
- Added missing abs when checking hermiticity (#2680, by Tanuj Rai, reported by tuureorell)
- Fix types and exceptions (#2701, Emmanuel Ferdman)

Documentation
-------------

- Docstrings, typing, guide page for distributions. (#2599, by Mathis)
- Guide for qutip.distributions (#2600, by Mathis)
- Removes the Quantum Optimal Control page in the Users Guide and adds Family guide page in Users guide. (#2616, by Kavya Rambhia)
- Fix small typo in docstring for `sesolve`. (#2677, by Sarang Joshi)


Miscellaneous
-------------

- Removed WignerDistribution, QDistribution from distributions with their tests.
  Added plot_qfunc to plot Husimi-Q function with tests and animation. (#2607, by Mathis)
- Specialized A @ B.dag() operation. (#2610)
- Set auto_tidyup_dims default to True (#2623)
- Change default ode method for mcsolve (#2643)
- Fix parameter name inconsistency in qsave (#2688, by Andrey Rakhubovsky)
- Add tests for qdiags and phasegate Hermitian fix (#2698, by Tanuj Rai)
- Changed type hints in function visualize in distributions.py (tests failed if mpl not installed). (#2607, by Mathis)


QuTiP 5.1.1 (2025-01-10)
=========================

Patch to add support for scipy 1.15.

Features
--------

- qutip.cite() now cites the QuTiP 5 paper, https://arxiv.org/abs/2412.04705.
- Added QuTiP family package information to qutip.about(). (#2604)


Bug Fixes
---------

- Fix support for calculating the eigenstates of ENR operators (#2595).
- Update various functions to use `sph_harm_y` when using scipy >= 1.15.
- Update mkl finding to support the 'emscripten' sys.platform. (#2606)


QuTiP 5.1.0 (2024-12-12)
========================

Features
--------

- It adds odd parity support to HEOM's fermionic solver (#2261, by Gerardo Jose Suarez)
- Create `SMESolver.run_from_experiment`, which allows to run stochastic evolution from know noise or measurements. (#2318)
- Add types hints. (#2327, #2473)
- Weighted trajectories in trajectory solvers (enables improved sampling for nm_mcsolve) (#2369, by Paul Menczel)
- Updated `qutip.core.metrics.dnorm` to have an efficient speedup when finding the difference of two unitaries. We use a result on page 18 of
  D. Aharonov, A. Kitaev, and N. Nisan, (1998). (#2416, by owenagnel)
- Allow mixed initial conditions for mcsolve and nm_mcsolve. (#2437, by Paul Menczel)
- Add support for `jit` and `grad` in qutip.core.metrics (#2461, by Rochisha Agarwal)
- Allow merging results from stochastic solvers. (#2474)
- Support measurement statistics for `jax` and `jaxdia` dtypes (#2493, by Rochisha Agarwal)
- Enable mcsolve with jax.grad using numpy_backend (#2499, by Rochisha Agarwal)
- Add propagator method to steadystate (#2508)
- Introduces the qutip.core.environment module, which contains classes that characterize bosonic and fermionic thermal environments. (#2534, by Gerardo Jose Suarez)
- Implements a `einsum` function for Qobj dimensions (Evaluates the Einstein summation convention on the operands.) (#2545, by Franco Mayo)
- Wave function calculations have been sped up with a Cython implementation.
  It optimizes the update method of the HarmonicOscillatorWaveFunction class in distribution.py. (#2553, by Matheus Gomes Cordeiro)
- Speed up `kraus_to_super` by adding a `sparse` option. (#2569, by Sola85)


Bug Fixes
---------

- Fix a dimension problem for the argument color of Bloch.add_states
  Clean-up of the code in  Bloch.add_state
  Add Bloch.add_arc and Bloch.add_line in the guide on Bloch class (#2445, by PositroniumJS)
- Fix HTMLProgressBar display (#2475)
- Make expm, cosm, sinm work with jax. (#2484, by Rochisha Agarwal)
- Fix stochastic solver step method (#2491)
- `clip` gives deprecation warning, that might be a problem in the future. Hence switch to `where` (#2507, by Rochisha Agarwal)
- Fix brmesolve detection of contant vs time-dependent system. (#2530)
- `propagator` now accepts list format `c_ops` like `mesolve` does. (#2532)
- Fix compatibility issue with matplotlib>=3.9 in matrix_histogram (#2544, by Andreas Maeder)
- Resolve incompatibility of TwoModeQuadratureCorrelation class (#2548, by quantum-menace)
- Fix sparse eigen solver issue with many degenerate eigen values. (#2586)
- Fix getting tensor permutation for uneven super operators. (#2561)


Documentation
-------------

- Improve guide-settings page. (#2403)
- Tidy up formatting of type aliases in the api documentation (#2436, by Paul Menczel)
- Update documentation
  - Update contributors
  - Improve apidoc readability (#2523)
- Fix error in simdiag docstring (#2585, by Sola85)


Miscellaneous
-------------

- Add auto_real_casting options. (#2329)
- Add dispatcher for sqrtm (#2453, by Rochisha Agarwal)
- Make `e_ops`, `args` and `options` keyword only.
  Solver were inconsistent with `e_ops` usually following `c_ops` but sometime
  preceding it. Setting it as keyword only remove the need to memorize the
  signature of each solver. (#2489)
- Introduces a new `NumpyBackend `class that enables dynamic selection of the numpy_backend used in `qutip`.
  The class facilitates switching between different numpy implementations ( `numpy` and `jax.numpy` mainly ) based on the configuration specified in `settings.core`. (#2490, by Rochisha Agarwal)
- Improve mkl lookup function. (#2497)
- Deterministic trajectories are not counted in ``ntraj``. (#2502)
- Allow tests to be executed multiple times in one Python session (#2538, by Zhang Maiyun)
- Improve performance of qutip.Qobj by using static numpy version check (#2557, by Pieter Eendebak)
- Fix towncrier check (#2542)



QuTiP 5.0.4 (2024-08-30)
========================

Micro release to add support for numpy 2.1

Bug Fixes
---------

- Fixed rounding error in dicke_trace_function that resulted in negative eigenvalues. (#2466, by  Andrey Nikitin)


QuTiP 5.0.3 (2024-06-20)
========================

Micro release to add support for numpy 2.

Bug Fixes
---------

- Bug Fix in Process Matrix Rendering. (#2400, by Anush Venkatakrishnan)
- Fix steadystate permutation being reversed. (#2443)
- Add parallelizing support for `vernN` methods with `mcsolve`. (#2454 by Utkarsh)


Documentation
-------------

- Added `qutip.core.gates` to apidoc/functions.rst and a Gates section to guide-states.rst. (#2441, by alan-nala)


Miscellaneous
-------------

- Add support for numpy 2 (#2421, #2457)
- Add support for scipy 1.14 (#2469)


QuTiP 5.0.2 (2024-05-16)
========================

Bug Fixes
---------

- Use CSR as the default for expand_operator (#2380, by BoxiLi)
- Fix import of the partial_transpose function.
  Ensures that the negativity function can handle both kets and density operators as input. (#2371, by vikas-chaudhary-2802)
- Ensure that end_condition of mcsolve result doesn't say target tolerance reached when it hasn't (#2382, by magzpavz)
- Fix two bugs in steadystate floquet solver, and adjust tests to be sensitive to this issue. (#2393, by Neill Lambert)


Documentation
-------------

- Correct a mistake in the doc (#2401, by PositroniumJS)
- Fix #2156: Correct a sample of code in the doc (#2409, by PositroniumJS)


Miscellaneous
-------------

- Better metadata management in operators creation functions (#2388)
- Implicitly set minimum python version to 3.9 (#2413)
- Qobj.__eq__ uses core's settings rtol. (#2425)
- Only normalize solver states when the initial state is already normalized. (#2427)


QuTiP 5.0.1 (2024-04-03)
========================


Patch update fixing small issues with v5.0.0 release

- Fix broken links in the documentation when migrating to readthedocs
- Fix readthedocs search feature
- Add setuptools to runtime compilation requirements
- Fix mcsolve documentation for open systems
- Fix OverFlowError in progress bars


QuTiP 5.0.0 (2024-03-26)
========================


QuTiP 5 is a redesign of many of the core components of QuTiP (``Qobj``,
``QobjEvo``, solvers) to make them more consistent and more flexible.

``Qobj`` may now be stored in either sparse or dense representations,
and the two may be mixed sensibly as needed. ``QobjEvo`` is now used
consistently throughout QuTiP, and the implementation has been
substantially cleaned up. A new ``Coefficient`` class is used to
represent the time-dependent factors inside ``QobjEvo``.

The solvers have been rewritten to work well with the new data layer
and the concept of ``Integrators`` which solve ODEs has been introduced.
In future, new data layers may provide their own ``Integrators``
specialized to their representation of the underlying data.

Much of the user-facing API of QuTiP remains familiar, but there have
had to be many small breaking changes. If we can make changes to
easy migrating code from QuTiP 4 to QuTiP 5, please let us know.

An extensive list of changes follows.

Contributors
------------

QuTiP 5 has been a large effort by many people over the last three years.

In particular:

- Jake Lishman led the implementation of the new data layer and coefficients.
- Eric Giguère led the implementation of the new QobjEvo interface and solvers.
- Boxi Li led the updating of QuTiP's QIP support and the creation of ``qutip_qip``.

Other members of the QuTiP Admin team have been heavily involved in reviewing,
testing and designing QuTiP 5:

- Alexander Pitchford
- Asier Galicia
- Nathan Shammah
- Shahnawaz Ahmed
- Neill Lambert
- Simon Cross
- Paul Menczel

Two Google Summer of Code contributors updated the tutorials and benchmarks to
QuTiP 5:

- Christian Staufenbiel updated many of the tutorials (`<https://github.com/qutip/qutip-tutorials/>`).
- Xavier Sproken update the benchmarks (`<https://github.com/qutip/qutip-benchmark/>`).

During an internship at RIKEN, Patrick Hopf created a new quantum control method and
improved the existing methods interface:

- Patrick Hopf created new quantum control package (`<https://github.com/qutip/qutip-qoc/>`).

Four experimental data layers backends were written either as part of Google Summer
of Code or as separate projects. While these are still alpha quality, they helped
significantly to test the data layer API:

- ``qutip-tensorflow``: a TensorFlow backend by Asier Galicia (`<https://github.com/qutip/qutip-tensorflow>`)
- ``qutip-cupy``: a CuPy GPU backend by Felipe Bivort Haiek (`<https://github.com/qutip/qutip-cupy/>`)`
- ``qutip-tensornetwork``: a TensorNetwork backend by Asier Galicia (`<https://github.com/qutip/qutip-tensornetwork>`)
- ``qutip-jax``: a JAX backend by Eric Giguère (`<https://github.com/qutip/qutip-jax/>`)

Finally, Yuji Tamakoshi updated the visualization function and added animation
functions as part of Google Summer of Code project.

We have also had many other contributors, whose specific contributions are
detailed below:

- Pieter Eendebak (updated the required SciPy to 1.5+, `#1982 <https://github.com/qutip/qutip/pull/1982>`).
- Pieter Eendebak (reduced import times by setting logger names, `#1981 <https://github.com/qutip/qutip/pull/1981>`)
- Pieter Eendebak (Allow scipy 1.12 to be used with qutip, `#2354 <https://github.com/qutip/qutip/pull/2354>`)
- Xavier Sproken (included C header files in the source distribution, `#1971 <https://github.com/qutip/qutip/pull/1971>`)
- Christian Staufenbiel (added support for multiple collapse operators to the Floquet solver, `#1962 <https://github.com/qutip/qutip/pull/1962>`)
- Christian Staufenbiel (fixed the basis used in the Floquet Master Equation solver, `#1952 <https://github.com/qutip/qutip/pull/1952>`)
- Christian Staufenbiel (allowed the ``bloch_redfield_tensor`` function to accept strings and callables for `a_ops`, `#1951 <https://github.com/qutip/qutip/pull/1951>`)
- Christian Staufenbiel (Add a guide on Superoperators, Pauli Basis and Channel Contraction, `#1984 <https://github.com/qutip/qutip/pull/1984>`)
- Henrique Silvéro (allowed ``qutip_qip`` to be imported as ``qutip.qip``, `#1920 <https://github.com/qutip/qutip/pull/1920>`)
- Florian Hopfmueller (added a vastly improved implementations of ``process_fidelity`` and ``average_gate_fidelity``, `#1712 <https://github.com/qutip/qutip/pull/1712>`, `#1748 <https://github.com/qutip/qutip/pull/1748>`, `#1788 <https://github.com/qutip/qutip/pull/1788>`)
- Felipe Bivort Haiek (fixed inaccuracy in docstring of the dense implementation of negation, `#1608 <https://github.com/qutip/qutip/pull/1608/>`)
- Rajath Shetty (added support for specifying colors for individual points, vectors and states display by `qutip.Bloch`, `#1335 <https://github.com/qutip/qutip/pull/1335>`)
- Rochisha Agarwal (Add dtype to printed ouput of qobj, `#2352 <https://github.com/qutip/qutip/pull/2352>`)
- Kosuke Mizuno (Add arguments of plot_wigner() and plot_wigner_fock_distribution() to specify parameters for wigner(), `#2057 <https://github.com/qutip/qutip/pull/2057>`)
- Matt Ord (Only pre-compute density matrices if keep_runs_results is False, `#2303 <https://github.com/qutip/qutip/pull/2303>`)
- Daniel Moreno Galán (Add the possibility to customize point colors as in V4 and fix point plot behavior for 'l' style, `#2303 <https://github.com/qutip/qutip/pull/2303>`)
- Sola85 (Fixed simdiag not returning orthonormal eigenvectors, `#2269 <https://github.com/qutip/qutip/pull/2269>`)
- Edward Thomas (Fix LaTeX display of Qobj state in Jupyter cell outputs, `#2272 <https://github.com/qutip/qutip/pull/2272>`)
- Bogdan Reznychenko (Rework `kraus_to_choi` making it faster, `#2284 <https://github.com/qutip/qutip/pull/2284>`)
- gabbence95 (Fix typos in `expect` documentation, `#2331 <https://github.com/qutip/qutip/pull/2331>`)
- lklivingstone (Added __repr__ to QobjEvo, `#2111 <https://github.com/qutip/qutip/pull/2111>`)
- Yuji Tamakoshi (Improve print(qutip.settings) by make it shorter, `#2113 <https://github.com/qutip/qutip/pull/2113>`)
- khnikhil (Added fermionic annihilation and creation operators, `#2166 <https://github.com/qutip/qutip/pull/2166>`)
- Daniel Weiss (Improved sampling algorithm for mcsolve, `#2218 <https://github.com/qutip/qutip/pull/2218>`)
- SJUW (Increase missing colorbar padding for matrix_histogram_complex() from 0 to 0.05, `#2181 <https://github.com/qutip/qutip/pull/2181>`)
- Valan Baptist Mathuranayagam (Changed qutip-notebooks to qutip-tutorials and fixed the typo in the link redirecting to the changelog section in the PR template, `#2107 <https://github.com/qutip/qutip/pull/2107>`)
- Gerardo Jose Suarez (Added information on sec_cutoff to the documentation, `#2136 <https://github.com/qutip/qutip/pull/2136>`)
- Cristian Emiliano Godinez Ramirez (Added inherited members to API doc of MESolver, SMESolver, SSESolver, NonMarkovianMCSolver, `#2167 <https://github.com/qutip/qutip/pull/2167>`)
- Andrey Rakhubovsky (Corrected grammar in Bloch-Redfield master equation documentation, `#2174 <https://github.com/qutip/qutip/pull/2174>`)
- Rushiraj Gadhvi (qutip.ipynbtools.version_table() can now be called without Cython installed, `#2110 <https://github.com/qutip/qutip/pull/2110>`)
- Harsh Khilawala (Moved HTMLProgressBar from qutip/ipynbtools.py to qutip/ui/progressbar.py, `#2112 <https://github.com/qutip/qutip/pull/2112>`)
- Avatar Srinidhi P V (Added new argument bc_type to take boundary conditions when creating QobjEvo, `#2114 <https://github.com/qutip/qutip/pull/2114>`)
- Andrey Rakhubovsky (Fix types in docstring of projection(), `#2363 <https://github.com/qutip/qutip/pull/2363>`)


Qobj changes
------------

Previously ``Qobj`` data was stored in a SciPy-like sparse matrix. Now the
representation is flexible. Implementations for dense and sparse formats are
included in QuTiP and custom implementations are possible. QuTiP's performance
on dense states and operators is significantly improved as a result.

Some highlights:

- The data is still acessible via the ``.data`` attribute, but is now an
  instance of the underlying data type instead of a SciPy-like sparse matrix.
  The operations available in ``qutip.core.data`` may be used on ``.data``,
  regardless of the data type.
- ``Qobj`` with different data types may be mixed in arithmetic and other
  operations. A sensible output type will be automatically determined.
- The new ``.to(...)`` method may be used to convert a ``Qobj`` from one data type
  to another. E.g. ``.to("dense")`` will convert to the dense representation and
  ``.to("csr")`` will convert to the sparse type.
- Many ``Qobj`` methods and methods that create ``Qobj`` now accepted a ``dtype``
  parameter that allows the data type of the returned ``Qobj`` to specified.
- The new ``&`` operator may be used to obtain the tensor product.
- The new ``@`` operator may be used to obtain the matrix / operator product.
  ``bar @ ket`` returns a scalar.
- The new ``.contract()`` method will collapse 1D subspaces of the dimensions of
  the ``Qobj``.
- The new ``.logm()`` method returns the matrix logarithm of an operator.
- The methods ``.set_data``, ``.get_data``, ``.extract_state``, ``.eliminate_states``,
  ``.evaluate`` and ``.check_isunitary`` have been removed.
- The property ``dtype`` return the representation of the data used.
- The new ``data_as`` allow to obtain the data as a common python formats:
  numpy array, scipy sparse matrix, JAX Array, etc.

QobjEvo changes
---------------

The ``QobjEvo`` type for storing time-dependent quantum objects has been
significantly expanded, standardized and extended. The time-dependent
coefficients are now represented using a new ``Coefficient`` type that
may be independently created and manipulated if required.

Some highlights:

- The ``.compile()`` method has been removed. Coefficients specified as
  strings are automatically compiled if possible and the compilation is
  cached across different Python runs and instances.
- Mixing coefficient types within a single ``Qobj`` is now supported.
- Many new attributes were added to ``QobjEvo`` for convenience. Examples
  include ``.dims``, ``.shape``, ``.superrep`` and ``.isconstant``.
- Many old attributes such as ``.cte``, ``.use_cython``, ``.type``, ``.const``,
  and ``.coeff_file`` were removed.
- A new ``Spline`` coefficient supports spline interpolations of different
  orders. The old ``Cubic_Spline`` coefficient has been removed.
- The new ``.arguments(...)`` method allows additional arguments to the
  underlying coefficient functions to be updated.
- The ``_step_func_coeff`` argument has been replaced by the ``order``
  parameter. ``_step_func_coeff=False`` is equivalent to ``order=3``.
  ``_step_func_coeff=True`` is equivalent to ``order=0``. Higher values
  of ``order`` gives spline interpolations of higher orders.
- The spline type can take ``bc_type`` to control the boundary conditions.
- QobjEvo can be creating from the multiplication of a Qobj with a coefficient:
  ``oper * qutip.coefficient(f, args=args)`` is equivalent to
  ``qutip.QobjEvo([[oper, f]], args=args)``.
- Coefficient function can be defined in a pythonic manner: ``def f(t, A, w)``.
  The dictionary ``args`` second argument is no longer needed.
  Function using the exact ``f(t, args)`` signature will use the old method for
  backward compatibility.

Solver changes
--------------

The solvers in QuTiP have been heavily reworked and standardized.
Under the hood solvers now make use of swappable ODE ``Integrators``.
Many ``Integrators`` are included (see the list below) and
custom implementations are possible. Solvers now consistently
accept a ``QobjEvo`` instance at the Hamiltonian or Liouvillian, or
any object which can be passed to the ``QobjEvo`` constructor.

A breakdown of highlights follows.

All solvers:

- Solver options are now supplied in an ordinary Python dict.
  ``qutip.Options`` is deprecated and returns a dict for backwards
  compatibility.
- A specific ODE integrator may be selected by supplying a
  ``method`` option.
- Each solver provides a class interface. Creating an instance
  of the class allows a solver to be run multiple times for the
  same system without having to repeatedly reconstruct the
  right-hand side of the ODE to be integrated.
- A ``QobjEvo`` instance is accepted for most operators, e.g.,
  ``H``, ``c_ops``, ``e_ops``, ``a_ops``.
- The progress bar is now selected using the ``progress_bar`` option.
  A new progess bar using the ``tqdm`` Python library is provided.
- Dynamic arguments, where the value of an operator depends on
  the current state of the evolution interface reworked. Now a property of the
  solver is to be used as an arguments:
  ``args={"state": MESolver.StateFeedback(default=rho0)}``

Integrators:

- The SciPy zvode integrator is available with the BDF and
  Adams methods as ``bdf`` and ``adams``.
- The SciPy dop853 integrator (an eighth order Runge-Kutta method by
  Dormand & Prince) is available as ``dop853``.
- The SciPy lsoda integrator is available as ``lsoda``.
- QuTiP's own implementation of Verner's "most efficient" Runge-Kutta methods
  of order 7 and 9 are available as ``vern7`` and ``vern9``. See
  http://people.math.sfu.ca/~jverner/ for a description of the methods.
- QuTiP's own implementation of a solver that directly diagonalizes the
  the system to be integrated is available as ``diag``. It only works on
  time-independent systems and is slow to setup, but once the diagonalization
  is complete, it generates solutions very quickly.
- QuTiP's own implementatoin of an approximate Krylov subspace integrator is
  available as ``krylov``. This integrator is only usable with ``sesolve``.

Result class:

- A new ``.e_data`` attribute provides expectation values as a dictionary.
  Unlike ``.expect``, the values are provided in a Python list rather than
  a numpy array, which better supports non-numeric types.
- The contents of the ``.stats`` attribute changed significantly and is
  now more consistent across solvers.

Monte-Carlo Solver (mcsolve):

- The system, H, may now be a super-operator.
- The ``seed`` parameter now supports supplying numpy ``SeedSequence`` or
  ``Generator`` types.
- The new ``timeout`` and ``target_tol`` parameters allow the solver to exit
  early if a timeout or target tolerance is reached.
- The ntraj option no longer supports a list of numbers of trajectories.
  Instead, just run the solver multiple times and use the class ``MCSolver``
  if setting up the solver uses a significant amount of time.
- The ``map_func`` parameter has been replaced by the ``map`` option.
- A loky based parallel map as been added.
- A mpi based parallel map as been added.
- The result returned by ``mcsolve`` now supports calculating photocurrents
  and calculating the steady state over N trajectories.
- The old ``parfor`` parallel execution function has been removed from
  ``qutip.parallel``. Use ``parallel_map``, ``loky_map`` or ``mpi_pmap`` instead.
- Added improved sampling options which converge much faster when the
  probability of collapse is small.

Non Markovian Monte-Carlo Solver (nm_mcsolve):

- New Monte-Carlo Solver supporting negative decay rates.
- Based on the influence martingale approach, Donvil et al., Nat Commun 13, 4140 (2022).
- Most of the improvements made to the regular Monte-Carlo solver are also available here.
- The value of the influence martingale is available through the ``.trace`` attribute of the result.

Stochastic Equation Solvers (ssesolve, smesolve)

- Function call greatly changed: many keyword arguments are now options.
- m_ops and dW_factors are now changed from the default from the new class interface only.
- Use the same parallel maps as mcsolve: support for loky and mpi map added.
- End conditions ``timeout`` and ``target_tol`` added.
- The ``seed`` parameter now supports supplying numpy ``SeedSequence``.
- Wiener function is now available as a feedback.

Bloch-Redfield Master Equation Solver (brmesolve):

- The ``a_ops`` and ``spectra`` support implementations been heavily reworked to
  reuse the techniques from the new Coefficient and QobjEvo classes.
- The ``use_secular`` parameter has been removed. Use ``sec_cutoff=-1`` instead.
- The required tolerance is now read from ``qutip.settings``.

Krylov Subspace Solver (krylovsolve):

- The Krylov solver is now implemented using ``SESolver`` and the ``krylov``
  ODE integrator. The function ``krylovsolve`` is maintained for convenience
  and now supports many more options.
- The ``sparse`` parameter has been removed. Supply a sparse ``Qobj`` for the
  Hamiltonian instead.

Floquet Solver (fsesolve and fmmesolve):

- The Floquet solver has been rewritten to use a new ``FloquetBasis`` class
  which manages the transformations from lab to Floquet basis and back.
- Many of the internal methods used by the old Floquet solvers have
  been removed. The Floquet tensor may still be retried using
  the function ``floquet_tensor``.
- The Floquet Markov Master Equation solver has had many changes and
  new options added. The environment temperature may be specified using
  ``w_th``, and the result states are stored in the lab basis and optionally
  in the Floquet basis using ``store_floquet_state``.
- The spectra functions supplied to ``fmmesolve`` must now be vectorized
  (i.e. accept and return numpy arrays for frequencies and densities) and
  must accept negative frequence (i.e. usually include a ``w > 0`` factor
  so that the returned densities are zero for negative frequencies).
- The number of sidebands to keep, ``kmax`` may only be supplied when using
  the ``FMESolver``
- The ``Tsteps`` parameter has been removed from both ``fsesolve`` and
  ``fmmesolve``. The ``precompute`` option to ``FloquetBasis`` may be used
  instead.

Evolution of State Solver (essovle):

- The function ``essolve`` has been removed. Use the ``diag`` integration
  method with ``sesolve`` or ``mesolve`` instead.

Steady-state solvers (steadystate module):

- The ``method`` parameter and ``solver`` parameters have been separated. Previously
  they were mixed together in the ``method`` parameter.
- The previous options are now passed as parameters to the steady state
  solver and mostly passed through to the underlying SciPy functions.
- The logging and statistics have been removed.

Correlation functions (correlation module):

- A new ``correlation_3op`` function has been added. It supports ``MESolver``
  or ``BRMESolver``.
- The ``correlation``, ``correlation_4op``, and ``correlation_ss`` functions have been
  removed.
- Support for calculating correlation with ``mcsolve`` has been removed.

Propagators (propagator module):

- A class interface, ``qutip.Propagator``, has been added for propagators.
- Propagation of time-dependent systems is now supported using ``QobjEvo``.
- The ``unitary_mode`` and ``parallel`` options have been removed.

Correlation spectra (spectrum module):

- The functions ``spectrum_ss`` and ``spectrum_pi`` have been removed and
  are now internal functions.
- The ``use_pinv`` parameter for ``spectrum`` has been removed and the
  functionality merged into the ``solver`` parameter. Use ``solver="pi"``
  instead.

Hierarchical Equation of Motion Solver (HEOM)

- Updated the solver to use the new QuTiP integrators and data layer.
- Updated all the HEOM tutorials to QuTiP 5.
- Added support for combining bosonic and fermionic baths.
- Sped up the construction of the RHS of the HEOM solver by a factor of 4x.
- As in QuTiP 4, the HEOM supports arbitrary spectral densities, bosonic and fermionic baths, Páde and Matsubara expansions of the correlation functions, calculating the Matsubara terminator and inspection of the ADOs (auxiliary density operators).


QuTiP core
----------

There have been numerous other small changes to core QuTiP features:

- ``qft(...)`` the function that returns the quantum Fourier
  transform operator was moved from ``qutip.qip.algorithm`` into ``qutip``.
- The Bloch-Redfield solver tensor, ``brtensor``, has been moved into
  ``qutip.core``. See the section above on the Bloch-Redfield solver
  for details.
- The functions ``mat2vec`` and ``vec2mat`` for transforming states to and
  from super-operator states have been renamed to ``stack_columns`` and
  ``unstack_columns``.
- The function ``liouvillian_ref`` has been removed. Used ``liouvillian``
  instead.
- The superoperator transforms ``super_to_choi``, ``choi_to_super``,
  ``choi_to_kraus``, ``choi_to_chi`` and ``chi_to_choi`` have been removed.
  Used ``to_choi``, ``to_super``, ``to_kraus`` and ``to_chi`` instead.
- All of the random object creation functions now accepted a
  numpy ``Generator`` as a seed.
- The ``dims`` parameter of all random object creation functions has
  been removed. Supply the dimensions as the first parameter if
  explicit dimensions are required.
- The function ``rand_unitary_haar`` has been removed. Use
  ``rand_unitary(distribution="haar")`` instead.
- The functions ``rand_dm_hs`` and ``rand_dm_ginibre`` have been removed.
  Use ``rand_dm(distribution="hs")`` and ``rand_dm(distribution="ginibre")``
  instead.
- The function ``rand_ket_haar`` has been removed. Use
  ``rand_ket(distribution="haar")`` instead.
- The measurement functions have had the ``target`` parameter for
  expanding the measurement operator removed. Used ``expand_operator``
  to expand the operator instead.
- ``qutip.Bloch`` now supports applying colours per-point, state or vector in
  ``add_point``, ``add_states``, and ``add_vectors``.
- Dimensions use a class instead of layered lists.
- Allow measurement functions to support degenerate operators.
- Add ``qeye_like`` and ``qzero_like``.
- Added fermionic annihilation and creation operators.

QuTiP settings
--------------

Previously ``qutip.settings`` was an ordinary module. Now ``qutip.settings`` is
an instance of a settings class. All the runtime modifiable settings for
core operations are in ``qutip.settings.core``. The other settings are not
modifiable at runtime.

- Removed ``load``. ``reset`` and ``save`` functions.
- Removed ``.debug``, ``.fortran``, ``.openmp_thresh``.
- New ``.compile`` stores the compilation options for compiled coefficients.
- New ``.core["rtol"]`` core option gives the default relative tolerance used by QuTiP.
- The absolute tolerance setting ``.atol`` has been moved to ``.core["atol"]``.

Visualization
-------------

- Added arguments to ``plot_wigner`` and ``plot_wigner_fock_distribution`` to specify parameters for ``wigner``.
- Removed ``Bloch3D``. The same functionality is provided by ``Bloch``.
- Added ``fig``, ``ax`` and ``cmap`` keyword arguments to all visualization functions.
- Most visualization functions now respect the ``colorblind_safe`` setting.
- Added new functions to create animations from a list of ``Qobj`` or directly from solver results with saved states.


Package reorganization
----------------------

- ``qutip.qip`` has been moved into its own package, qutip-qip. Once installed, qutip-qip is available as either ``qutip.qip`` or ``qutip_qip``. Some widely useful gates have been retained in ``qutip.gates``.
- ``qutip.control`` has been moved to qutip-qtrl and once installed qutip-qtrl is available as either ``qutip.control`` or ``qutip_qtrl``. Note that ``quitp_qtrl`` is provided primarily for backwards compatibility. Improvements to optimal control will take place in the new ``qutip_qoc`` package.
- ``qutip.lattice`` has been moved into its own package, qutip-lattice. It is available from `<https://github.com/qutip/qutip-lattice>`.
- ``qutip.sparse`` has been removed. It contained the old sparse matrix representation and is replaced by the new implementation in ``qutip.data``.
- ``qutip.piqs`` functions are no longer available from the ``qutip`` namespace. They are accessible from ``qutip.piqs`` instead.

Miscellaneous
-------------

- Support has been added for 64-bit integer sparse matrix indices, allowing
  sparse matrices with up to 2**63 rows and columns. This support needs to
  be enabled at compilation time by calling ``setup.py`` and passing
  ``--with-idxint-64``.

Feature removals
----------------

- Support for OpenMP has been removed. If there is enough demand and a good plan for how to organize it, OpenMP support may return in a future QuTiP release.
- The ``qutip.parfor`` function has been removed. Use ``qutip.parallel_map`` instead.
- ``qutip.graph`` has been removed and replaced by SciPy's graph functions.
- ``qutip.topology`` has been removed. It contained only one function ``berry_curvature``.
- The ``~/.qutip/qutiprc`` config file is no longer supported. It contained settings for the OpenMP support.
- Deprecate ``three_level_atom``
- Deprecate ``orbital``


Changes from QuTiP 5.0.0b1:
---------------------------

Features
--------

- Add dtype to printed ouput of qobj (#2352 by Rochisha Agarwal)


Miscellaneous
-------------

- Allow scipy 1.12 to be used with qutip. (#2354 by Pieter Eendebak)


QuTiP 5.0.0b1 (2024-03-04)
==========================

Features
--------

- Create a Dimension class (#1996)
- Add arguments of plot_wigner() and plot_wigner_fock_distribution() to specify parameters for wigner(). (#2057, by Kosuke Mizuno)
- Restore feedback to solvers (#2210)
- Added mpi_pmap, which uses the mpi4py module to run computations in parallel through the MPI interface. (#2296, by Paul)
- Only pre-compute density matrices if keep_runs_results is False (#2303, by Matt Ord)


Bug Fixes
---------

- Add the possibility to customize point colors as in V4 and fix point plot behavior for 'l' style (#1974, by Daniel Moreno Galán)
- Disabled broken "improved sampling" for `nm_mcsolve`. (#2234, by Paul)
- Fixed result objects storing a reference to the solver through options._feedback. (#2262, by Paul)
- Fixed simdiag not returning orthonormal eigenvectors. (#2269, by Sola85)
- Fix LaTeX display of Qobj state in Jupyter cell outputs (#2272, by Edward Thomas)
- Improved behavior of `parallel_map` and `loky_pmap` in the case of timeouts, errors or keyboard interrupts (#2280, by Paul)
- Ignore deprecation warnings from cython 0.29.X in tests. (#2288)
- Fixed two problems with the steady_state() solver in the HEOM method. (#2333)


Miscellaneous
-------------

- Improve fidelity doc-string (#2257)
- Improve documentation in guide/dynamics (#2271)
- Improve states and operator parameters documentation. (#2289)
- Rework `kraus_to_choi` making it faster (#2284, by Bogdan Reznychenko)
- Remove Bloch3D: redundant to Bloch (#2306)
- Allow tests to run without matplotlib and ipython. (#2311)
- Add too small step warnings in fixed dt SODE solver (#2313)
- Add `dtype` to `Qobj` and `QobjEvo` (#2325)
- Fix typos in `expect` documentation (#2331, by gabbence95)
- Allow measurement functions to support degenerate operators. (#2342)


QuTiP 5.0.0a2 (2023-09-06)
==========================

Features
--------

- Add support for different spectra types for bloch_redfield_tensor (#1951)
- Improve qutip import times by setting logger names explicitly. (#1981, by Pieter Eendebak)
- Change the order of parameters in expand_operator (#1991)
- Add `svn` and `solve` to dispatched (#2002)
- Added nm_mcsolve to provide support for Monte-Carlo simulations of master equations with possibly negative rates. The method implemented here is described in arXiv:2209.08958 [quant-ph]. (#2070 by pmenczel)
- Add support for combining bosonic and fermionic HEOM baths (#2089)
- Added __repr__ to QobjEvo (#2111 by lklivingstone)
- Improve print(qutip.settings) by make it shorter (#2113 by tamakoshi2001)
- Create the `trace_oper_ket` operation (#2126)
- Speed up the construction of the RHS of the HEOM solver by a factor of 4x by converting the final step to Cython. (#2128)
- Rewrite the stochastic solver to use the v5 solver interface. (#2131)
- Add `Qobj.get` to extract underlying data in original format. (#2141)
- Add qeye_like and qzero_like (#2153)
- Add capacity to dispatch on ``Data`` (#2157)
- Added fermionic annihilation and creation operators. (#2166 by khnikhil)
- Changed arguments and applied colorblind_safe to functions in visualization.py (#2170 by Yuji Tamakoshi)
- Changed arguments and applied colorblind_safe to plot_wigner_sphere and matrix_histogram in visualization.py (#2193 by Yuji Tamakoshi)
- Added Dia data layer which represents operators as multi-diagonal matrices. (#2196)
- Added support for animated plots. (#2203 by Yuji Tamakoshi)
- Improved sampling algorithm for mcsolve (#2218 by Daniel Weiss)
- Added support for early termination of map functions. (#2222)



Bug Fixes
---------

- Add missing state transformation to floquet_markov_mesolve (#1952 by christian512)
- Added default _isherm value (True) for momentum and position operators. (#2032 by Asier Galicia)
- Changed qutip-notebooks to qutip-tutorials and fixed the typo in the link redirecting to the changelog section in the PR template. (#2107 by Valan Baptist Mathuranayagam)
- Increase missing colorbar padding for matrix_histogram_complex() from 0 to 0.05. (#2181 by SJUW)
- Raise error on insufficient memory. (#2224)
- Fixed fallback to fsesolve call in fmmesolve (#2225)


Removals
--------

- Remove qutip.control and replace with qutip_qtrl. (#2116)
- Deleted _solve in countstat.py and used _data.solve. (#2120 by Yuji Tamakoshi)
- Deprecate three_level_atom (#2221)
- Deprecate orbital (#2223)


Documentation
-------------

- Add a guide on Superoperators, Pauli Basis and Channel Contraction. (#1984 by christian512)
- Added information on sec_cutoff to the documentation (#2136 by Gerardo Jose Suarez)
- Added inherited members to API doc of MESolver, SMESolver, SSESolver, NonMarkovianMCSolver (#2167 by Cristian Emiliano Godinez Ramirez)
- Corrected grammar in Bloch-Redfield master equation documentation (#2174 by Andrey Rakhubovsky)


Miscellaneous
-------------

- Update scipy version requirement to 1.5+ (#1982 by Pieter Eendebak)
- Added __all__ to qutip/measurements.py and qutip/core/semidefinite.py (#2103 by Rushiraj Gadhvi)
- Restore towncrier check (#2105)
- qutip.ipynbtools.version_table() can now be called without Cython installed (#2110 by Rushiraj Gadhvi)
- Moved HTMLProgressBar from qutip/ipynbtools.py to qutip/ui/progressbar.py (#2112 by Harsh Khilawala)
- Added new argument bc_type to take boundary conditions when creating QobjEvo (#2114 by Avatar Srinidhi P V )
- Remove Windows build warning suppression. (#2119)
- Optimize dispatcher by dispatching on positional only args. (#2135)
- Clean semidefinite (#2138)
- Migrate `transfertensor.py` to solver (#2142)
- Add a test for progress_bar (#2150)
- Enable cython 3 (#2151)
- Added tests for visualization.py (#2192 by Yuji Tamakoshi)
- Sorted arguments of sphereplot so that the order is similar to those of plot_spin_distribution (#2219 by Yuji Tamakoshi)


QuTiP 5.0.0a1 (2023-02-07)
==========================

QuTiP 5 is a redesign of many of the core components of QuTiP (``Qobj``,
``QobjEvo``, solvers) to make them more consistent and more flexible.

``Qobj`` may now be stored in either sparse or dense representations,
and the two may be mixed sensibly as needed. ``QobjEvo`` is now used
consistently throughout QuTiP, and the implementation has been
substantially cleaned up. A new ``Coefficient`` class is used to
represent the time-dependent factors inside ``QobjEvo``.

The solvers have been rewritten to work well with the new data layer
and the concept of ``Integrators`` which solve ODEs has been introduced.
In future, new data layers may provide their own ``Integrators``
specialized to their representation of the underlying data.

Much of the user-facing API of QuTiP remains familiar, but there have
had to be many small breaking changes. If we can make changes to
easy migrating code from QuTiP 4 to QuTiP 5, please let us know.

Any extensive list of changes follows.

Contributors
------------

QuTiP 5 has been a large effort by many people over the last three years.

In particular:

- Jake Lishman led the implementation of the new data layer and coefficients.
- Eric Giguère led the implementation of the new QobjEvo interface and solvers.
- Boxi Li led the updating of QuTiP's QIP support and the creation of ``qutip_qip``.

Other members of the QuTiP Admin team have been heavily involved in reviewing,
testing and designing QuTiP 5:

- Alexander Pitchford
- Asier Galicia
- Nathan Shammah
- Shahnawaz Ahmed
- Neill Lambert
- Simon Cross

Two Google Summer of Code contributors updated the tutorials and benchmarks to
QuTiP 5:

- Christian Staufenbiel updated many of the tutorials (`<https://github.com/qutip/qutip-tutorials/>`).
- Xavier Sproken update the benchmarks (`<https://github.com/qutip/qutip-benchmark/>`).

Four experimental data layers backends were written either as part of Google Summer
of Code or as separate projects. While these are still alpha quality, the helped
significantly to test the data layer API:

- ``qutip-tensorflow``: a TensorFlow backend by Asier Galicia (`<https://github.com/qutip/qutip-tensorflow>`)
- ``qutip-cupy``: a CuPy GPU backend by Felipe Bivort Haiek (`<https://github.com/qutip/qutip-cupy/>`)`
- ``qutip-tensornetwork``: a TensorNetwork backend by Asier Galicia (`<https://github.com/qutip/qutip-tensornetwork>`)
- ``qutip-jax``: a JAX backend by Eric Giguère (`<https://github.com/qutip/qutip-jax/>`)

We have also had many other contributors, whose specific contributions are
detailed below:

- Pieter Eendebak (updated the required SciPy to 1.4+, `#1982 <https://github.com/qutip/qutip/pull/1982>`).
- Pieter Eendebak (reduced import times by setting logger names, `#1981 <https://github.com/qutip/qutip/pull/1981>`)
- Xavier Sproken (included C header files in the source distribution, `#1971 <https://github.com/qutip/qutip/pull/1971>`)
- Christian Staufenbiel (added support for multiple collapse operators to the Floquet solver, `#1962 <https://github.com/qutip/qutip/pull/1962>`)
- Christian Staufenbiel (fixed the basis used in the Floquet Master Equation solver, `#1952 <https://github.com/qutip/qutip/pull/1952>`)
- Christian Staufenbiel (allowed the ``bloch_redfield_tensor`` function to accept strings and callables for `a_ops`, `#1951 <https://github.com/qutip/qutip/pull/1951>`)
- Henrique Silvéro (allowed ``qutip_qip`` to be imported as ``qutip.qip``, `#1920 <https://github.com/qutip/qutip/pull/1920>`)
- Florian Hopfmueller (added a vastly improved implementations of ``process_fidelity`` and ``average_gate_fidelity``, `#1712 <https://github.com/qutip/qutip/pull/1712>`, `#1748 <https://github.com/qutip/qutip/pull/1748>`, `#1788 <https://github.com/qutip/qutip/pull/1788>`)
- Felipe Bivort Haiek (fixed inaccuracy in docstring of the dense implementation of negation, `#1608 <https://github.com/qutip/qutip/pull/1608/>`)
- Rajath Shetty (added support for specifying colors for individual points, vectors and states display by `qutip.Bloch`, `#1335 <https://github.com/qutip/qutip/pull/1335>`)

Qobj changes
------------

Previously ``Qobj`` data was stored in a SciPy-like sparse matrix. Now the
representation is flexible. Implementations for dense and sparse formats are
included in QuTiP and custom implementations are possible. QuTiP's performance
on dense states and operators is significantly improved as a result.

Some highlights:

- The data is still acessible via the ``.data`` attribute, but is now an
  instance of the underlying data type instead of a SciPy-like sparse matrix.
  The operations available in ``qutip.core.data`` may be used on ``.data``,
  regardless of the data type.
- ``Qobj`` with different data types may be mixed in arithmetic and other
  operations. A sensible output type will be automatically determined.
- The new ``.to(...)`` method may be used to convert a ``Qobj`` from one data type
  to another. E.g. ``.to("dense")`` will convert to the dense representation and
  ``.to("csr")`` will convert to the sparse type.
- Many ``Qobj`` methods and methods that create ``Qobj`` now accepted a ``dtype``
  parameter that allows the data type of the returned ``Qobj`` to specified.
- The new ``&`` operator may be used to obtain the tensor product.
- The new ``@`` operator may be used to obtain the matrix / operator product.
  ``bar @ ket`` returns a scalar.
- The new ``.contract()`` method will collapse 1D subspaces of the dimensions of
  the ``Qobj``.
- The new ``.logm()`` method returns the matrix logarithm of an operator.
- The methods ``.set_data``, ``.get_data``, ``.extract_state``, ``.eliminate_states``,
  ``.evaluate`` and ``.check_isunitary`` have been removed.

QobjEvo changes
---------------

The ``QobjEvo`` type for storing time-dependent quantum objects has been
significantly expanded, standardized and extended. The time-dependent
coefficients are now represented using a new ``Coefficient`` type that
may be independently created and manipulated if required.

Some highlights:

- The ``.compile()`` method has been removed. Coefficients specified as
  strings are automatically compiled if possible and the compilation is
  cached across different Python runs and instances.
- Mixing coefficient types within a single ``Qobj`` is now supported.
- Many new attributes were added to ``QobjEvo`` for convenience. Examples
  include ``.dims``, ``.shape``, ``.superrep`` and ``.isconstant``.
- Many old attributes such as ``.cte``, ``.use_cython``, ``.type``, ``.const``,
  and ``.coeff_file`` were removed.
- A new ``Spline`` coefficient supports spline interpolations of different
  orders. The old ``Cubic_Spline`` coefficient has been removed.
- The new ``.arguments(...)`` method allows additional arguments to the
  underlying coefficient functions to be updated.
- The ``_step_func_coeff`` argument has been replaced by the ``order``
  parameter. ``_step_func_coeff=False`` is equivalent to ``order=3``.
  ``_step_func_coeff=True`` is equivalent to ``order=0``. Higher values
  of ``order`` gives spline interpolations of higher orders.

Solver changes
--------------

The solvers in QuTiP have been heavily reworked and standardized.
Under the hood solvers now make use of swappable ODE ``Integrators``.
Many ``Integrators`` are included (see the list below) and
custom implementations are possible. Solvers now consistently
accept a ``QobjEvo`` instance at the Hamiltonian or Liouvillian, or
any object which can be passed to the ``QobjEvo`` constructor.

A breakdown of highlights follows.

All solvers:

- Solver options are now supplied in an ordinary Python dict.
  ``qutip.Options`` is deprecated and returns a dict for backwards
  compatibility.
- A specific ODE integrator may be selected by supplying a
  ``method`` option.
- Each solver provides a class interface. Creating an instance
  of the class allows a solver to be run multiple times for the
  same system without having to repeatedly reconstruct the
  right-hand side of the ODE to be integrated.
- A ``QobjEvo`` instance is accepted for most operators, e.g.,
  ``H``, ``c_ops``, ``e_ops``, ``a_ops``.
- The progress bar is now selected using the ``progress_bar`` option.
  A new progess bar using the ``tqdm`` Python library is provided.
- Dynamic arguments, where the value of an operator depends on
  the current state of the evolution, have been removed. They
  may be re-implemented later if there is demand for them.

Integrators:

- The SciPy zvode integrator is available with the BDF and
  Adams methods as ``bdf`` and ``adams``.
- The SciPy dop853 integrator (an eighth order Runge-Kutta method by
  Dormand & Prince) is available as ``dop853``.
- The SciPy lsoda integrator is available as ``lsoda``.
- QuTiP's own implementation of Verner's "most efficient" Runge-Kutta methods
  of order 7 and 9 are available as ``vern7`` and ``vern9``. See
  http://people.math.sfu.ca/~jverner/ for a description of the methods.
- QuTiP's own implementation of a solver that directly diagonalizes the
  the system to be integrated is available as ``diag``. It only works on
  time-independent systems and is slow to setup, but once the diagonalization
  is complete, it generates solutions very quickly.
- QuTiP's own implementatoin of an approximate Krylov subspace integrator is
  available as ``krylov``. This integrator is only usable with ``sesolve``.

Result class:

- A new ``.e_data`` attribute provides expectation values as a dictionary.
  Unlike ``.expect``, the values are provided in a Python list rather than
  a numpy array, which better supports non-numeric types.
- The contents of the ``.stats`` attribute changed significantly and is
  now more consistent across solvers.

Monte-Carlo Solver (mcsolve):

- The system, H, may now be a super-operator.
- The ``seed`` parameter now supports supplying numpy ``SeedSequence`` or
  ``Generator`` types.
- The new ``timeout`` and ``target_tol`` parameters allow the solver to exit
  early if a timeout or target tolerance is reached.
- The ntraj option no longer supports a list of numbers of trajectories.
  Instead, just run the solver multiple times and use the class ``MCSolver``
  if setting up the solver uses a significant amount of time.
- The ``map_func`` parameter has been replaced by the ``map`` option. In
  addition to the existing ``serial`` and ``parallel`` values, the value
  ``loky`` may be supplied to use the loky package to parallelize trajectories.
- The result returned by ``mcsolve`` now supports calculating photocurrents
  and calculating the steady state over N trajectories.
- The old ``parfor`` parallel execution function has been removed from
  ``qutip.parallel``. Use ``parallel_map`` or ``loky_map`` instead.

Bloch-Redfield Master Equation Solver (brmesolve):

- The ``a_ops`` and ``spectra`` support implementaitons been heavily reworked to
  reuse the techniques from the new Coefficient and QobjEvo classes.
- The ``use_secular`` parameter has been removed. Use ``sec_cutoff=-1`` instead.
- The required tolerance is now read from ``qutip.settings``.

Krylov Subspace Solver (krylovsolve):

- The Krylov solver is now implemented using ``SESolver`` and the ``krylov``
  ODE integrator. The function ``krylovsolve`` is maintained for convenience
  and now supports many more options.
- The ``sparse`` parameter has been removed. Supply a sparse ``Qobj`` for the
  Hamiltonian instead.

Floquet Solver (fsesolve and fmmesolve):

- The Floquet solver has been rewritten to use a new ``FloquetBasis`` class
  which manages the transformations from lab to Floquet basis and back.
- Many of the internal methods used by the old Floquet solvers have
  been removed. The Floquet tensor may still be retried using
  the function ``floquet_tensor``.
- The Floquet Markov Master Equation solver has had many changes and
  new options added. The environment temperature may be specified using
  ``w_th``, and the result states are stored in the lab basis and optionally
  in the Floquet basis using ``store_floquet_state``.
- The spectra functions supplied to ``fmmesolve`` must now be vectorized
  (i.e. accept and return numpy arrays for frequencies and densities) and
  must accept negative frequence (i.e. usually include a ``w > 0`` factor
  so that the returned densities are zero for negative frequencies).
- The number of sidebands to keep, ``kmax`` may only be supplied when using
  the ``FMESolver``
- The ``Tsteps`` parameter has been removed from both ``fsesolve`` and
  ``fmmesolve``. The ``precompute`` option to ``FloquetBasis`` may be used
  instead.

Evolution of State Solver (essovle):

- The function ``essolve`` has been removed. Use the ``diag`` integration
  method with ``sesolve`` or ``mesolve`` instead.

Steady-state solvers (steadystate module):

- The ``method`` parameter and ``solver`` parameters have been separated. Previously
  they were mixed together in the ``method`` parameter.
- The previous options are now passed as parameters to the steady state
  solver and mostly passed through to the underlying SciPy functions.
- The logging and statistics have been removed.

Correlation functions (correlation module):

- A new ``correlation_3op`` function has been added. It supports ``MESolver``
  or ``BRMESolver``.
- The ``correlation``, ``correlation_4op``, and ``correlation_ss`` functions have been
  removed.
- Support for calculating correlation with ``mcsolve`` has been removed.

Propagators (propagator module):

- A class interface, ``qutip.Propagator``, has been added for propagators.
- Propagation of time-dependent systems is now supported using ``QobjEvo``.
- The ``unitary_mode`` and ``parallel`` options have been removed.

Correlation spectra (spectrum module):

- The functions ``spectrum_ss`` and ``spectrum_pi`` have been removed and
  are now internal functions.
- The ``use_pinv`` parameter for ``spectrum`` has been removed and the
  functionality merged into the ``solver`` parameter. Use ``solver="pi"``
  instead.

QuTiP core
----------

There have been numerous other small changes to core QuTiP features:

- ``qft(...)`` the function that returns the quantum Fourier
  transform operator was moved from ``qutip.qip.algorithm`` into ``qutip``.
- The Bloch-Redfield solver tensor, ``brtensor``, has been moved into
  ``qutip.core``. See the section above on the Bloch-Redfield solver
  for details.
- The functions ``mat2vec`` and ``vec2mat`` for transforming states to and
  from super-operator states have been renamed to ``stack_columns`` and
  ``unstack_columns``.
- The function ``liouvillian_ref`` has been removed. Used ``liouvillian``
  instead.
- The superoperator transforms ``super_to_choi``, ``choi_to_super``,
  ``choi_to_kraus``, ``choi_to_chi`` and ``chi_to_choi`` have been removed.
  Used ``to_choi``, ``to_super``, ``to_kraus`` and ``to_chi`` instead.
- All of the random object creation functions now accepted a
  numpy ``Generator`` as a seed.
- The ``dims`` parameter of all random object creation functions has
  been removed. Supply the dimensions as the first parameter if
  explicit dimensions are required.
- The function ``rand_unitary_haar`` has been removed. Use
  ``rand_unitary(distribution="haar")`` instead.
- The functions ``rand_dm_hs`` and ``rand_dm_ginibre`` have been removed.
  Use ``rand_dm(distribution="hs")`` and ``rand_dm(distribution="ginibre")``
  instead.
- The function ``rand_ket_haar`` has been removed. Use
  ``rand_ket(distribution="haar")`` instead.
- The measurement functions have had the ``target`` parameter for
  expanding the measurement operator removed. Used ``expand_operator``
  to expand the operator instead.
- ``qutip.Bloch`` now supports applying colours per-point, state or vector in
  ``add_point``, ``add_states``, and ``add_vectors``.

QuTiP settings
--------------

Previously ``qutip.settings`` was an ordinary module. Now ``qutip.settings`` is
an instance of a settings class. All the runtime modifiable settings for
core operations are in ``qutip.settings.core``. The other settings are not
modifiable at runtime.

- Removed ``load``. ``reset`` and ``save`` functions.
- Removed ``.debug``, ``.fortran``, ``.openmp_thresh``.
- New ``.compile`` stores the compilation options for compiled coefficients.
- New ``.core["rtol"]`` core option gives the default relative tolerance used by QuTiP.
- The absolute tolerance setting ``.atol`` has been moved to ``.core["atol"]``.

Package reorganization
----------------------

- ``qutip.qip`` has been moved into its own package, qutip-qip. Once installed, qutip-qip is available as either ``qutip.qip`` or ``qutip_qip``. Some widely useful gates have been retained in ``qutip.gates``.
- ``qutip.lattice`` has been moved into its own package, qutip-lattice. It is available from `<https://github.com/qutip/qutip-lattice>`.
- ``qutip.sparse`` has been removed. It contained the old sparse matrix representation and is replaced by the new implementation in ``qutip.data``.
- ``qutip.piqs`` functions are no longer available from the ``qutip`` namespace. They are accessible from ``qutip.piqs`` instead.

Miscellaneous
-------------

- Support has been added for 64-bit integer sparse matrix indices, allowing
  sparse matrices with up to 2**63 rows and columns. This support needs to
  be enabled at compilation time by calling ``setup.py`` and passing
  ``--with-idxint-64``.

Feature removals
----------------

- Support for OpenMP has been removed. If there is enough demand and a good plan for how to organize it, OpenMP support may return in a future QuTiP release.
- The ``qutip.parfor`` function has been removed. Use ``qutip.parallel_map`` instead.
- ``qutip.graph`` has been removed and replaced by SciPy's graph functions.
- ``qutip.topology`` has been removed. It contained only one function ``berry_curvature``.
- The ``~/.qutip/qutiprc`` config file is no longer supported. It contained settings for the OpenMP support.


QuTiP 4.7.5 (2024-01-29)
========================

Patch release for QuTiP 4.7. It adds support for SciPy 1.12.

Bug Fixes
---------

- Remove use of scipy.<numpy-func> in parallel.py, incompatible with scipy==1.12 (#2305 by Evan McKinney)


QuTiP 4.7.4 (2024-01-15)
========================

Bug Fixes
---------

- Adapt to deprecation from matplotlib 3.8 (#2243, reported by Bogdan Reznychenko)
- Fix name of temp files for removal after use. (#2251, reported by Qile Su)
- Avoid integer overflow in Qobj creation. (#2252, reported by KianHwee-Lim)
- Ignore DeprecationWarning from pyximport (#2287)
- Add partial support and tests for python 3.12. (#2294)


Miscellaneous
-------------

- Rework `choi_to_kraus`, making it rely on an eigenstates solver that can choose `eigh` if the Choi matrix is Hermitian, as it is more numerically stable. (#2276, by Bogdan Reznychenko)
- Rework `kraus_to_choi`, making it faster (#2283, by Bogdan Reznychenko and Rafael Haenel)


QuTiP 4.7.3 (2023-08-22)
========================

Bug Fixes
---------

- Non-oper qobj + scalar raise an error. (#2208 reported by vikramkashyap)
- Fixed issue where `extract_states` did not preserve hermiticity.
  Fixed issue where `rand_herm` did not set the private attribute _isherm to True. (#2214 by AGaliciaMartinez)
- ssesolve average states to density matrices (#2216 reported by BenjaminDAnjou)


Miscellaneous
-------------

- Exclude cython 3.0.0 from requirement (#2204)
- Run in no cython mode with cython >=3.0.0 (#2207)


QuTiP 4.7.2 (2023-06-28)
========================

This is a bugfix release for QuTiP 4.7.X. It adds support for
numpy 1.25 and scipy 1.11.

Bug Fixes
---------
- Fix setting of sso.m_ops in heterodyne smesolver and passing through of sc_ops to photocurrent solver. (#2081 by Bogdan Reznychenko and Simon Cross)
- Update calls to SciPy eigvalsh and eigsh to pass the range of eigenvalues to return using ``subset_by_index=``. (#2081 by Simon Cross)
- Fixed bug where some matrices were wrongly found to be hermitian. (#2082 by AGaliciaMartinez)

Miscellaneous
-------------
- Fixed typo in stochastic.py (#2049, by  eltociear)
- `ptrace` always return density matrix (#2185, issue by udevd)
- `mesolve` can support mixed callable and Qobj for `e_ops` (#2184 issue by balopat)


QuTiP 4.7.1 (2022-12-11)
========================

This is a bugfix release for QuTiP 4.7.X. In addition to the minor fixes
listed below, the release adds builds for Python 3.11 and support for
packaging 22.0.

Features
--------
- Improve qutip import times by setting logger names explicitly. (#1980)

Bug Fixes
---------
- Change floquet_master_equation_rates(...) to use an adaptive number of time steps scaled by the number of sidebands, kmax. (#1961)
- Change fidelity(A, B) to use the reduced fidelity formula for pure states which is more numerically efficient and accurate. (#1964)
- Change ``brmesolve`` to raise an exception when ode integration is not successful. (#1965)
- Backport fix for IPython helper Bloch._repr_svg_ from dev.major. Previously the print_figure function returned bytes, but since ipython/ipython#5452 (in 2014) it returns a Unicode string. This fix updates QuTiP's helper to match. (#1970)
- Fix correlation for case where only the collapse operators are time dependent. (#1979)
- Fix the hinton visualization method to plot the matrix instead of its transpose. (#2011)
- Fix the hinton visualization method to take into account all the matrix coefficients to set the squares scale, instead of only the diagonal coefficients. (#2012)
- Fix parsing of package versions in setup.py to support packaging 22.0. (#2037)
- Add back .qu suffix to objects saved with qsave and loaded with qload. The suffix was accidentally removed in QuTiP 4.7.0. (#2038)
- Add a default max_step to processors. (#2040)

Documentation
-------------
- Add towncrier for managing the changelog. (#1927)
- Update the version of numpy used to build documentation to 1.22.0. (#1940)
- Clarify returned objects from bloch_redfield_tensor(). (#1950)
- Update Floquet Markov solver docs. (#1958)
- Update the roadmap and ideas to show completed work as of August 2022. (#1967)

Miscellaneous
-------------
- Return TypeError instead of Exception for type error in sesolve argument. (#1924)
- Add towncrier draft build of changelog to CI tests. (#1946)
- Add Python 3.11 to builds. (#2041)
- Simplify version parsing by using packaging.version.Version. (#2043)
- Update builds to use cibuildwheel 2.11, and to build with manylinux2014 on Python 3.8 and 3.9, since numpy and SciPy no longer support manylinux2010 on those versions of Python. (#2047)


QuTiP 4.7.0 (2022-04-13)
========================

This release sees the addition of two new solvers -- ``qutip.krylovsolve`` based on the Krylov subspace approximation and ``qutip.nonmarkov.heom`` that reimplements the BoFiN HEOM solver.

Bloch sphere rendering gained support for drawing arcs and lines on the sphere, and for setting the transparency of rendered points and vectors, Hinton plots gained support for specifying a coloring style, and matrix histograms gained better default colors and more flexible styling options.

Other significant improvements include better scaling of the Floquet solver, support for passing ``Path`` objects when saving and loading files, support for passing callable functions as ``e_ops`` to ``mesolve`` and ``sesolve``, and faster state number enumeration and Husimi Q functions.

Import bugfixes include some bugs affecting plotting with matplotlib 3.5 and fixing support for qutrits (and other non-qubit) quantum circuits.

The many other small improvements, bug fixes, documentation enhancements, and behind the scenese development changes are included in the list below.

QuTiP 4.7.X will be the last series of releases for QuTiP 4. Patch releases will continue for the 4.7.X series but the main development effort will move to QuTiP 5.

The many, many contributors who filed issues, submitted or reviewed pull requests, and improved the documentation for this release are listed next to their contributions below. Thank you to all of you.

Improvements
------------
- **MAJOR** Added krylovsolve as a new solver based on krylov subspace approximation. (`#1739 <https://github.com/qutip/qutip/pull/1739>`_ by Emiliano Fortes)
- **MAJOR** Imported BoFiN HEOM (https://github.com/tehruhn/bofin/) into QuTiP and replaced the HEOM solver with a compatibility wrapper around BoFiN bosonic solver. (`#1601 <https://github.com/qutip/qutip/pull/1601>`_, `#1726 <https://github.com/qutip/qutip/pull/1726>`_, and `#1724 <https://github.com/qutip/qutip/pull/1724>`_ by Simon Cross, Tarun Raheja and Neill Lambert)
- **MAJOR** Added support for plotting lines and arcs on the Bloch sphere. (`#1690 <https://github.com/qutip/qutip/pull/1690>`_ by Gaurav Saxena, Asier Galicia and Simon Cross)
- Added transparency parameter to the add_point, add_vector and add_states methods in the Bloch and Bloch3d classes. (`#1837 <https://github.com/qutip/qutip/pull/1837>`_ by Xavier Spronken)
- Support ``Path`` objects in ``qutip.fileio``. (`#1813 <https://github.com/qutip/qutip/pull/1813>`_ by Adrià Labay)
- Improved the weighting in steadystate solver, so that the default weight matches the documented behaviour and the dense solver applies the weights in the same manner as the sparse solver. (`#1275 <https://github.com/qutip/qutip/pull/1275>`_ and `#1802 <https://github.com/qutip/qutip/pull/1802>`_ by NS2 Group at LPS and Simon Cross)
- Added a ``color_style`` option to the ``hinton`` plotting function. (`#1595 <https://github.com/qutip/qutip/issues/1595>`_ by Cassandra Granade)
- Improved the scaling of ``floquet_master_equation_rates`` and ``floquet_master_equation_tensor`` and fixed transposition and basis change errors in ``floquet_master_equation_tensor`` and ``floquet_markov_mesolve``. (`#1248 <https://github.com/qutip/qutip/pull/1248>`_ by Camille Le Calonnec, Jake Lishman and Eric Giguère)
- Removed ``linspace_with`` and ``view_methods`` from ``qutip.utilities``. For the former it is far better to use ``numpy.linspace`` and for the later Python's in-built ``help`` function or other tools. (`#1680 <https://github.com/qutip/qutip/pull/1680>`_ by Eric Giguère)
- Added support for passing callable functions as ``e_ops`` to ``mesolve`` and ``sesolve``. (`#1655 <https://github.com/qutip/qutip/pull/1655>`_ by Marek Narożniak)
- Added the function ``steadystate_floquet``, which returns the "effective" steadystate of a periodic driven system. (`#1660 <https://github.com/qutip/qutip/pull/1660>`_ by Alberto Mercurio)
- Improved mcsolve memory efficiency by not storing final states when they are not needed. (`#1669 <https://github.com/qutip/qutip/pull/1669>`_ by Eric Giguère)
- Improved the default colors and styling of matrix_histogram and provided additional styling options. (`#1573 <https://github.com/qutip/qutip/pull/1573>`_ and `#1628 <https://github.com/qutip/qutip/pull/1628>`_ by Mahdi Aslani)
- Sped up ``state_number_enumerate``, ``state_number_index``, ``state_index_number``, and added some error checking. ``enr_state_dictionaries`` now returns a list for ``idx2state``. (`#1604 <https://github.com/qutip/qutip/pull/1604>`_ by Johannes Feist)
- Added new Husimi Q algorithms, improving the speed for density matrices, and giving a near order-of-magnitude improvement when calculating the Q function for many different states, using the new ``qutip.QFunc`` class, instead of the ``qutip.qfunc`` function. (`#934 <https://github.com/qutip/qutip/pull/934>`_ and `#1583 <https://github.com/qutip/qutip/pull/1583>`_ by Daniel Weigand and Jake Lishman)
- Updated licence holders with regards to new governance model, and remove extraneous licensing information from source files. (`#1579 <https://github.com/qutip/qutip/pull/1579>`_ by Jake Lishman)
- Removed the vendored copy of LaTeX's qcircuit package which is GPL licensed. We now rely on the package being installed by user. It is installed by default with TexLive. (`#1580 <https://github.com/qutip/qutip/pull/1580>`_ by Jake Lishman)
- The signatures of rand_ket and rand_ket_haar were changed to allow N (the size of the random ket) to be determined automatically when dims are specified. (`#1509 <https://github.com/qutip/qutip/pull/1509>`_ by Purva Thakre)

Bug Fixes
---------
- Fix circuit index used when plotting circuits with non-reversed states. (`#1847 <https://github.com/qutip/qutip/pull/1847>`_ by Christian Staufenbiel)
- Changed implementation of ``qutip.orbital`` to use ``scipy.special.spy_harm`` to remove bugs in angle interpretation. (`#1844 <https://github.com/qutip/qutip/pull/1844>`_ by Christian Staufenbiel)
- Fixed ``QobjEvo.tidyup`` to use ``settings.auto_tidyup_atol`` when removing small elements in sparse matrices. (`#1832 <https://github.com/qutip/qutip/pull/1832>`_ by Eric Giguère)
- Ensured that tidyup's default tolerance is read from settings at each call. (`#1830 <https://github.com/qutip/qutip/pull/1830>`_ by Eric Giguère)
- Fixed ``scipy.sparse`` deprecation warnings raised by ``qutip.fast_csr_matrix``. (`#1827 <https://github.com/qutip/qutip/pull/1827>`_ by Simon Cross)
- Fixed rendering of vectors on the Bloch sphere when using matplotlib 3.5 and above. (`#1818 <https://github.com/qutip/qutip/pull/1818>`_ by Simon Cross)
- Fixed the displaying of ``Lattice1d`` instances and their unit cells. Previously calling them raised exceptions in simple cases. (`#1819 <https://github.com/qutip/qutip/pull/1819>`_, `#1697 <https://github.com/qutip/qutip/pull/1697>`_ and `#1702 <https://github.com/qutip/qutip/pull/1702>`_ by Simon Cross and Saumya Biswas)
- Fixed the displaying of the title for ``hinton`` and ``matrix_histogram`` plots when a title is given. Previously the supplied title was not displayed. (`#1707 <https://github.com/qutip/qutip/pull/1707>`_ by Vladimir Vargas-Calderón)
- Removed an incorrect check on the initial state dimensions in the ``QubitCircuit`` constructor. This allows, for example, the construction of qutrit circuits. (`#1807 <https://github.com/qutip/qutip/pull/1807>`_ by Boxi Li)
- Fixed the checking of ``method`` and ``offset`` parameters in ``coherent`` and ``coherent_dm``. (`#1469 <https://github.com/qutip/qutip/pull/1469>`_ and `#1741 <https://github.com/qutip/qutip/pull/1741>`_ by Joseph Fox-Rabinovitz and Simon Cross)
- Removed the Hamiltonian saved in the ``sesolve`` solver results. (`#1689 <https://github.com/qutip/qutip/pull/1689>`_ by Eric Giguère)
- Fixed a bug in rand_herm with ``pos_def=True`` and ``density>0.5`` where the diagonal was incorrectly filled. (`#1562 <https://github.com/qutip/qutip/pull/1562>`_ by Eric Giguère)

Documentation Improvements
--------------------------
- Added contributors image to the documentation. (`#1828 <https://github.com/qutip/qutip/pull/1828>`_ by Leonard Assis)
- Fixed the Theory of Quantum Information bibliography link. (`#1840 <https://github.com/qutip/qutip/pull/1840>`_ by Anto Luketina)
- Fixed minor grammar errors in the dynamics guide. (`#1822 <https://github.com/qutip/qutip/pull/1822>`_ by Victor Omole)
- Fixed many small documentation typos. (`#1569 <https://github.com/qutip/qutip/pull/1569>`_ by Ashish Panigrahi)
- Added Pulser to the list of libraries that use QuTiP. (`#1570 <https://github.com/qutip/qutip/pull/1570>`_ by Ashish Panigrahi)
- Corrected typo in the states and operators guide. (`#1567 <https://github.com/qutip/qutip/pull/1567>`_ by Laurent Ajdnik)
- Converted http links to https. (`#1555 <https://github.com/qutip/qutip/pull/1555>`_ by Jake Lishamn)

Developer Changes
-----------------
- Add GitHub actions test run on windows-latest. (`#1853 <https://github.com/qutip/qutip/pull/1853>`_ and `#1855 <https://github.com/qutip/qutip/pull/1855>`_ by Simon Cross)
- Bumped the version of pillow used to build documentation from 9.0.0 to 9.0.1. (`#1835 <https://github.com/qutip/qutip/pull/1835>`_ by dependabot)
- Migrated the ``qutip.superop_reps`` tests to pytest. (`#1825 <https://github.com/qutip/qutip/pull/1825>`_ by Felipe Bivort Haiek)
- Migrated the ``qutip.steadystates`` tests to pytest. (`#1679 <https://github.com/qutip/qutip/pull/1679>`_ by Eric Giguère)
- Changed the README.md CI badge to the GitHub Actions badge. (`#1581 <https://github.com/qutip/qutip/pull/1581>`_ by Jake Lishman)
- Updated CodeClimate configuration to treat our Python source files as Python 3. (`#1577 <https://github.com/qutip/qutip/pull/1577>`_ by Jake Lishman)
- Reduced cyclomatic complexity in ``qutip._mkl``. (`#1576 <https://github.com/qutip/qutip/pull/1576>`_ by Jake Lishman)
- Fixed PEP8 warnings in ``qutip.control``, ``qutip.mcsolve``, ``qutip.random_objects``, and ``qutip.stochastic``. (`#1575 <https://github.com/qutip/qutip/pull/1575>`_ by Jake Lishman)
- Bumped the version of urllib3 used to build documentation from 1.26.4 to 1.26.5. (`#1563 <https://github.com/qutip/qutip/pull/1563>`_ by dependabot)
- Moved tests to GitHub Actions. (`#1551 <https://github.com/qutip/qutip/pull/1551>`_ by Jake Lishman)
- The GitHub contributing guidelines were re-added and updated to point to the more complete guidelines in the documentation. (`#1549 <https://github.com/qutip/qutip/pull/1549>`_ by Jake Lishman)
- The release documentation was reworked after the initial 4.6.1 to match the actual release process. (`#1544 <https://github.com/qutip/qutip/pull/1544>`_ by Jake Lishman)


QuTiP 4.6.3 (2022-02-9)
=======================

This minor release adds support for numpy 1.22 and Python 3.10 and removes some blockers for running QuTiP on the Apple M1.

The performance of the ``enr_destroy``, ``state_number_enumerate`` and ``hadamard_transform`` functions was drastically improved (up to 70x or 200x faster in some common cases), and support for the drift Hamiltonian was added to the ``qutip.qip`` ``Processor``.

The ``qutip.hardware_info`` module was removed as part of adding support for the Apple M1. We hope the removal of this little-used module does not adversely affect many users -- it was largely unrelated to QuTiP's core functionality and its presence was a continual source of blockers to importing ``qutip`` on new or changed platforms.

A new check on the dimensions of ``Qobj``'s were added to prevent segmentation faults when invalid shape and dimension combinations were passed to Cython code.

In addition, there were many small bugfixes, documentation improvements, and improvements to our building and testing processes.


Improvements
------------
- The ``enr_destroy`` function was made ~200x faster in many simple cases. (`#1593 <https://github.com/qutip/qutip/pull/1593>`_ by Johannes Feist)
- The ``state_number_enumerate`` function was made significantly faster. (`#1594 <https://github.com/qutip/qutip/pull/1594>`_ by Johannes Feist)
- Added the missing drift Hamiltonian to the method run_analytically of ``Processor``. (`#1603 <https://github.com/qutip/qutip/pull/1603>`_ Boxi Li)
- The ``hadamard_transform`` was made much faster, e.g., ~70x faster for N=10. (`#1688 <https://github.com/qutip/qutip/pull/1688>`_ by Asier Galicia)
- Added support for computing the power of a scalar-like Qobj. (`#1692 <https://github.com/qutip/qutip/pull/1692>`_ by Asier Galicia)
- Removed the ``hardware_info`` module. This module wasn't used inside QuTiP and regularly broke when new operating systems were released, and in particular prevented importing QuTiP on the Apple M1. (`#1754 <https://github.com/qutip/qutip/pull/1754>`_, `#1758 <https://github.com/qutip/qutip/pull/1758>`_ by Eric Giguère)

Bug Fixes
---------
- Fixed support for calculating the propagator of a density matrix with collapse operators. QuTiP 4.6.2 introduced extra sanity checks on the dimensions of inputs to mesolve (Fix mesolve segfault with bad initial state `#1459 <https://github.com/qutip/qutip/pull/1459>`_), but the propagator function's calls to mesolve violated these checks by supplying initial states with the dimensions incorrectly set. ``propagator`` now calls mesolve with the correct dimensions set on the initial state. (`#1588 <https://github.com/qutip/qutip/pull/1588>`_ by Simon Cross)
- Fixed support for calculating the propagator for a superoperator without collapse operators. This functionality was not tested by the test suite and appears to have broken sometime during 2019. Tests have now been added and the code breakages fixed. (`#1588 <https://github.com/qutip/qutip/pull/1588>`_ by Simon Cross)
- Fixed the ignoring of the random number seed passed to ``rand_dm`` in the case where ``pure`` was set to true. (`#1600 <https://github.com/qutip/qutip/pull/1600>`_ Pontus Wikståhl)
- Fixed qutip.control.optimize_pulse support for sparse eigenvector decomposition with the Qobj oper_dtype (the Qobj oper_dtype is the default for large systems). (`#1621 <https://github.com/qutip/qutip/pull/1621>`_ by Simon Cross)
- Removed qutip.control.optimize_pulse support for scipy.sparse.csr_matrix and generic ndarray-like matrices. Support for these was non-functional. (`#1621 <https://github.com/qutip/qutip/pull/1621>`_ by Simon Cross)
- Fixed errors in the calculation of the Husimi spin_q_function and spin_wigner functions and added tests for them. (`#1632 <https://github.com/qutip/qutip/pull/1632>`_ by Mark Johnson)
- Fixed setting of OpenMP compilation flag on Linux. Previously when compiling the OpenMP functions were compiled without parallelization. (`#1693 <https://github.com/qutip/qutip/pull/1693>`_ by Eric Giguère)
- Fixed tracking the state of the Bloch sphere figure and axes to prevent exceptions during rendering. (`#1619 <https://github.com/qutip/qutip/pull/1619>`_ by Simon Cross)
- Fixed compatibility with numpy configuration in numpy's 1.22.0 release. (`#1752 <https://github.com/qutip/qutip/pull/1752>`_ by Matthew Treinish)
- Added dims checks for e_ops passed to solvers to prevent hanging the calling process when e_ops of the wrong dimensions were passed. (`#1778 <https://github.com/qutip/qutip/pull/1778>`_ by Eric Giguère)
- Added a check in Qobj constructor that the respective members of data.shape cannot be larger than what the corresponding dims could contain to prevent a segmentation fault caused by inconsistencies between dims and shapes. (`#1783 <https://github.com/qutip/qutip/pull/1783>`_, `#1785 <https://github.com/qutip/qutip/pull/1785>`_, `#1784 <https://github.com/qutip/qutip/pull/1784>`_ by Lajos Palanki & Eric Giguère)

Documentation Improvements
--------------------------
- Added docs for the num_cbits parameter of the QubitCircuit class. (`#1652 <https://github.com/qutip/qutip/pull/1652>`_ by  Jon Crall)
- Fixed the parameters in the call to fsesolve in the Floquet guide. (`#1675 <https://github.com/qutip/qutip/pull/1675>`_ by Simon Cross)
- Fixed the description of random number usage in the Monte Carlo solver guide. (`#1677 <https://github.com/qutip/qutip/pull/1677>`_ by Ian Thorvaldson)
- Fixed the rendering of equation numbers in the documentation (they now appear on the right as expected, not above the equation). (`#1678 <https://github.com/qutip/qutip/pull/1678>`_ by Simon Cross)
- Updated the installation requirements in the documentation to match what is specified in setup.py. (`#1715 <https://github.com/qutip/qutip/pull/1715>`_ by Asier Galicia)
- Fixed a typo in the ``chi_to_choi`` documentation. Previously the documentation mixed up chi and choi. (`#1731 <https://github.com/qutip/qutip/pull/1731>`_ by Pontus Wikståhl)
- Improved the documentation for the stochastic equation solvers. Added links to notebooks with examples, API doumentation and external references. (`#1743 <https://github.com/qutip/qutip/pull/1743>`_ by Leonardo Assis)
- Fixed a typo in ``qutip.settings`` in the settings guide. (`#1786 <https://github.com/qutip/qutip/pull/1786>`_ by Mahdi Aslani)
- Made numerous small improvements to the text of the QuTiP basics guide. (`#1768 <https://github.com/qutip/qutip/pull/1768>`_ by Anna Naden)
- Made a small phrasing improvement to the README. (`#1790 <https://github.com/qutip/qutip/pull/1790>`_ by Rita Abani)

Developer Changes
-----------------
- Improved test coverage of states and operators functions. (`#1578 <https://github.com/qutip/qutip/pull/1578>`_ by Eric Giguère)
- Fixed test_interpolate mcsolve use (`#1645 <https://github.com/qutip/qutip/pull/1645>`_ by Eric Giguère)
- Ensured figure plots are explicitly closed during tests so that the test suite passes when run headless under Xvfb. (`#1648 <https://github.com/qutip/qutip/pull/1648>`_ by Simon Cross)
- Bumped the version of pillow used to build documentation from 8.2.0 to 9.0.0. (`#1654 <https://github.com/qutip/qutip/pull/1654>`_, `#1760 <https://github.com/qutip/qutip/pull/1760>`_ by dependabot)
- Bumped the version of babel used to build documentation from 2.9.0 to 2.9.1. (`#1695 <https://github.com/qutip/qutip/pull/1695>`_ by dependabot)
- Bumped the version of numpy used to build documentation from 1.19.5 to 1.21.0. (`#1767 <https://github.com/qutip/qutip/pull/1767>`_ by dependabot)
- Bumped the version of ipython used to build documentation from 7.22.0 to 7.31.1. (`#1780 <https://github.com/qutip/qutip/pull/1780>`_ by dependabot)
- Rename qutip.bib to CITATION.bib to enable GitHub's citation support. (`#1662 <https://github.com/qutip/qutip/pull/1662>`_ by Ashish Panigrahi)
- Added tests for simdiags. (`#1681 <https://github.com/qutip/qutip/pull/1681>`_ by Eric Giguère)
- Added support for specifying the numpy version in the CI test matrix. (`#1696 <https://github.com/qutip/qutip/pull/1696>`_ by Simon Cross)
- Fixed the skipping of the dnorm metric tests if cvxpy is not installed. Previously all metrics tests were skipped by accident. (`#1704 <https://github.com/qutip/qutip/pull/1704>`_ by Florian Hopfmueller)
- Added bug report, feature request and other options to the GitHub issue reporting template. (`#1728 <https://github.com/qutip/qutip/pull/1728>`_ by Aryaman Kolhe)
- Updated the build process to support building on Python 3.10 by removing the build requirement for numpy < 1.20 and replacing it with a requirement on oldest-supported-numpy. (`#1747 <https://github.com/qutip/qutip/pull/1747>`_ by Simon Cross)
- Updated the version of cibuildwheel used to build wheels to 2.3.0. (`#1747 <https://github.com/qutip/qutip/pull/1747>`_, `#1751 <https://github.com/qutip/qutip/pull/1751>`_ by Simon Cross)
- Added project urls to linking to the source repository, issue tracker and documentation to setup.cfg. (`#1779 <https://github.com/qutip/qutip/pull/1779>`_ by Simon Cross)
- Added a numpy 1.22 and Python 3.10 build to the CI test matrix. (`#1777 <https://github.com/qutip/qutip/pull/1777>`_ by Simon Cross)
- Ignore deprecation warnings from SciPy 1.8.0 scipy.sparse.X imports in CI tests. (`#1797 <https://github.com/qutip/qutip/pull/1797>`_ by Simon Cross)
- Add building of wheels for Python 3.10 to the cibuildwheel job. (`#1796 <https://github.com/qutip/qutip/pull/1796>`_ by Simon Cross)


QuTiP 4.6.2 (2021-06-02)
========================

This minor release adds a function to calculate the quantum relative entropy, fixes a corner case in handling time-dependent Hamiltonians in ``mesolve`` and adds back support for a wider range of matplotlib versions when plotting or animating Bloch spheres.

It also adds a section in the README listing the papers which should be referenced while citing QuTiP.


Improvements
------------
- Added a "Citing QuTiP" section to the README, containing a link to the QuTiP papers. (`#1554 <https://github.com/qutip/qutip/pull/1554>`_)
- Added ``entropy_relative`` which returns the quantum relative entropy between two density matrices. (`#1553 <https://github.com/qutip/qutip/pull/1553>`_)

Bug Fixes
---------
- Fixed Bloch sphere distortion when using Matplotlib >= 3.3.0. (`#1496  <https://github.com/qutip/qutip/pull/1496>`_)
- Removed use of integer-like floats in math.factorial since it is deprecated as of Python 3.9. (`#1550 <https://github.com/qutip/qutip/pull/1550>`_)
- Simplified call to ffmpeg used in the the Bloch sphere animation tutorial to work with recent versions of ffmpeg. (`#1557 <https://github.com/qutip/qutip/pull/1557>`_)
- Removed blitting in Bloch sphere FuncAnimation example. (`#1558 <https://github.com/qutip/qutip/pull/1558>`_)
- Added a version checking condition to handle specific functionalities depending on the matplotlib version. (`#1556 <https://github.com/qutip/qutip/pull/1556>`_)
- Fixed ``mesolve`` handling of time-dependent Hamiltonian with a custom tlist and ``c_ops``. (`#1561 <https://github.com/qutip/qutip/pull/1561>`_)

Developer Changes
-----------------
- Read documentation version and release from the VERSION file.


QuTiP 4.6.1 (2021-05-04)
========================

This minor release fixes bugs in QIP gate definitions, fixes building from
the source tarball when git is not installed and works around an MKL
bug in versions of SciPy <= 1.4.

It also adds the ``[full]`` pip install target so that ``pip install qutip[full]``
installs qutip and all of its optional and developer dependencies.

Improvements
------------
- Add the ``[full]`` pip install target (by **Jake Lishman**)

Bug Fixes
---------
- Work around pointer MKL eigh bug in SciPy <= 1.4 (by **Felipe Bivort Haiek**)
- Fix berkeley, swapalpha and cz gate operations (by **Boxi Li**)
- Expose the CPHASE control gate (by **Boxi Li**)
- Fix building from the sdist when git is not installed (by **Jake Lishman**)

Developer Changes
-----------------
- Move the qutip-doc documentation into the qutip repository (by **Jake Lishman**)
- Fix warnings in documentation build (by **Jake Lishman**)
- Fix warnings in pytest runs and make pytest treat warnings as errors (by **Jake Lishman**)
- Add Simon Cross as author (by **Simon Cross**)


QuTiP 4.6.0 (2021-04-11)
========================

This release brings improvements for qubit circuits, including a pulse scheduler, measurement statistics, reading/writing OpenQASM and optimisations in the circuit simulations.

This is the first release to have full binary wheel releases on pip; you can now do ``pip install qutip`` on almost any machine to get a correct version of the package without needing any compilers set up.
The support for Numpy 1.20 that was first added in QuTiP 4.5.3 is present in this version as well, and the same build considerations mentioned there apply here too.
If building using the now-supported PEP 517 mechanisms (e.g. ``python -mbuild /path/to/qutip``), all build dependencies will be correctly satisfied.

Improvements
------------
- **MAJOR** Add saving, loading and resetting functionality to ``qutip.settings`` for easy re-configuration. (by **Eric Giguère**)
- **MAJOR** Add a quantum gate scheduler in ``qutip.qip.scheduler``, to help parallelise the operations of quantum gates.  This supports two scheduling modes: as late as possible, and as soon as possible. (by **Boxi Li**)
- **MAJOR** Improved qubit circuit simulators, including OpenQASM support and performance optimisations. (by **Sidhant Saraogi**)
- **MAJOR** Add tools for quantum measurements and their statistics. (by **Simon Cross** and **Sidhant Saraogi**)
- Add support for Numpy 1.20.  QuTiP should be compiled against a version of Numpy ``>= 1.16.6`` and ``< 1.20`` (note: does _not_ include 1.20 itself), but such an installation is compatible with any modern version of Numpy.  Source installations from ``pip`` understand this constraint.
- Improve the error message when circuit plotting fails. (by **Boxi Li**)
- Add support for parsing M1 Mac hardware information. (by **Xiaoliang Wu**)
- Add more single-qubit gates and controlled gates. (by **Mateo Laguna** and **Martín Sande Costa**)
- Support decomposition of ``X``, ``Y`` and ``Z`` gates in circuits. (by **Boxi Li**)
- Refactor ``QubitCircuit.resolve_gate()`` (by **Martín Sande Costa**)

Bug Fixes
---------
- Fix ``dims`` in the returns from ``Qobj.eigenstates`` on superoperators. (by **Jake Lishman**)
- Calling Numpy ufuncs on ``Qobj`` will now correctly raise a ``TypeError`` rather than returning a nonsense ``ndarray``. (by **Jake Lishman**)
- Convert segfault into Python exception when creating too-large tensor products. (by **Jake Lishman**)
- Correctly set ``num_collapse`` in the output of ``mesolve``. (by **Jake Lishman**)
- Fix ``ptrace`` when all subspaces are being kept, or the subspaces are passed in order. (by **Jake Lishman**)
- Fix sorting bug in ``Bloch3d.add_points()``. (by **pschindler**)
- Fix invalid string literals in docstrings and some unclosed files. (by **Élie Gouzien**)
- Fix Hermicity tests for matrices with values that are within the tolerance of 0. (by **Jake Lishman**)
- Fix the trace norm being incorrectly reported as 0 for small matrices. (by **Jake Lishman**)
- Fix issues with ``dnorm`` when using CVXPy 1.1 with sparse matrices. (by **Felipe Bivort Haiek**)
- Fix segfaults in ``mesolve`` when passed a bad initial ``Qobj`` as the state. (by **Jake Lishman**)
- Fix sparse matrix construction in PIQS when using Scipy 1.6.1. (by **Drew Parsons**)
- Fix ``zspmv_openmp.cpp`` missing from the pip sdist. (by **Christoph Gohlke**)
- Fix correlation functions throwing away imaginary components. (by **Asier Galicia Martinez**)
- Fix ``QubitCircuit.add_circuit()`` for SWAP gate. (by **Canoming**)
- Fix the broken LaTeX image conversion. (by **Jake Lishman**)
- Fix gate resolution of the FREDKIN gate. (by **Bo Yang**)
- Fix broken formatting in docstrings. (by **Jake Lishman**)

Deprecations
------------
- ``eseries``, ``essolve`` and ``ode2es`` are all deprecated, pending removal in QuTiP 5.0.  These are legacy functions and classes that have been left unmaintained for a long time, and their functionality is now better achieved with ``QobjEvo`` or ``mesolve``.

Developer Changes
-----------------
- **MAJOR** Overhaul of setup and packaging code to make it satisfy PEP 517, and move the build to a matrix on GitHub Actions in order to release binary wheels on pip for all major platforms and supported Python versions. (by **Jake Lishman**)
- Default arguments in ``Qobj`` are now ``None`` rather than mutable types. (by **Jake Lishman**)
- Fixed comsumable iterators being used to parametrise some tests, preventing the testing suite from being re-run within the same session. (by **Jake Lishman**)
- Remove unused imports, simplify some floats and remove unnecessary list conversions. (by **jakobjakobson13**)
- Improve Travis jobs matrix for specifying the testing containers. (by **Jake Lishman**)
- Fix coverage reporting on Travis. (by **Jake Lishman**)
- Added a ``pyproject.toml`` file. (by **Simon Humpohl** and **Eric Giguère**)
- Add doctests to documentation. (by **Sidhant Saraogi**)
- Fix all warnings in the documentation build. (by **Jake Lishman**)



QuTiP 4.5.3 (2021-02-19)
========================

This patch release adds support for Numpy 1.20, made necessary by changes to how array-like objects are handled. There are no other changes relative to version 4.5.2.

Users building from source should ensure that they build against Numpy versions >= 1.16.6 and < 1.20 (not including 1.20 itself), but after that or for those installing from conda, an installation will support any current Numpy version >= 1.16.6.

Improvements
------------
- Add support for Numpy 1.20.  QuTiP should be compiled against a version of Numpy ``>= 1.16.6`` and ``< 1.20`` (note: does _not_ include 1.20 itself), but such an installation is compatible with any modern version of Numpy.  Source installations from ``pip`` understand this constraint.



QuTiP 4.5.2 (2020-07-14)
========================

This is predominantly a hot-fix release to add support for Scipy 1.5, due to changes in private sparse matrix functions that QuTiP also used.

Improvements
------------
- Add support for Scipy 1.5. (by **Jake Lishman**)
- Improved speed of ``zcsr_inner``, which affects ``Qobj.overlap``. (by **Jake Lishman**)
- Better error messages when installation requirements are not satisfied. (by **Eric Giguère**)

Bug Fixes
---------
- Fix ``zcsr_proj`` acting on matrices with unsorted indices.  (by **Jake Lishman**)
- Fix errors in Milstein's heterodyne. (by **Eric Giguère**)
- Fix datatype bug in ``qutip.lattice`` module. (by **Boxi Li**)
- Fix issues with ``eigh`` on Mac when using OpenBLAS.  (by **Eric Giguère**)

Developer Changes
-----------------
- Converted more of the codebase to PEP 8.
- Fix several instances of unsafe mutable default values and unsafe ``is`` comparisons.



QuTiP 4.5.1 (2020-05-15)
========================

Improvements
------------
- ``husimi`` and ``wigner`` now accept half-integer spin (by **maij**)
- Better error messages for failed string coefficient compilation. (issue raised by **nohchangsuk**)

Bug Fixes
---------
- Safer naming for temporary files. (by **Eric Giguère**)
- Fix ``clebsch`` function for half-integer (by **Thomas Walker**)
- Fix ``randint``'s dtype to ``uint32`` for compatibility with Windows. (issue raised by **Boxi Li**)
- Corrected stochastic's heterodyne's m_ops (by **eliegenois**)
- Mac pool use spawn. (issue raised by **goerz**)
- Fix typos in ``QobjEvo._shift``. (by **Eric Giguère**)
- Fix warning on Travis CI. (by **Ivan Carvalho**)

Deprecations
------------
- ``qutip.graph`` functions will be deprecated in QuTiP 5.0 in favour of ``scipy.sparse.csgraph``.

Developer Changes
-----------------
- Add Boxi Li to authors. (by **Alex Pitchford**)
- Skip some tests that cause segfaults on Mac. (by **Nathan Shammah** and **Eric Giguère**)
- Use Python 3.8 for testing on Mac and Linux. (by **Simon Cross** and **Eric Giguère**)



QuTiP 4.5.0 (2020-01-31)
========================

Improvements
------------
- **MAJOR FEATURE**: Added `qip.noise`, a module with pulse level description of quantum circuits allowing to model various types of noise and devices (by **Boxi Li**).

- **MAJOR FEATURE**: Added `qip.lattice`, a module for the study of lattice dynamics in 1D (by **Saumya Biswas**).

- Migrated testing from Nose to PyTest (by **Tarun Raheja**).

- Optimized testing for PyTest and removed duplicated test runners (by **Jake Lishman**).

- Deprecated importing `qip` functions to the qutip namespace (by **Boxi Li**).

- Added the possibility to define non-square superoperators relevant for quantum circuits (by **Arne Grimsmo** and **Josh Combes**).

- Implicit tensor product for `qeye`, `qzero` and `basis` (by **Jake Lishman**).

- QObjEvo no longer requires Cython for string coefficient (by **Eric Giguère**).

- Added marked tests for faster tests in `testing.run()` and made faster OpenMP benchmarking in CI (by **Eric Giguère**).

- Added entropy and purity for Dicke density matrices, refactored into more general dicke_trace (by **Nathan Shammah**).

- Added option for specifying resolution in Bloch.save function (by **Tarun Raheja**).

- Added information related to the value of hbar in `wigner` and `continuous_variables` (by **Nicolas Quesada**).

- Updated requirements for `scipy 1.4` (by **Eric Giguère**).

- Added previous lead developers to the qutip.about() message (by **Nathan Shammah**).

- Added improvements to `Qobj` introducing the `inv` method and making the partial trace, `ptrace`, faster, keeping both sparse and dense methods (by **Eric Giguère**).

- Allowed general callable objects to define a time-dependent Hamiltonian (by **Eric Giguère**).

- Added feature so that `QobjEvo` no longer requires Cython for string coefficients (by **Eric Giguère**).

- Updated authors list on Github and added `my binder` link (by **Nathan Shammah**).


Bug Fixes
---------

- Fixed `PolyDataMapper` construction for `Bloch3d` (by **Sam Griffiths**).

- Fixed error checking for null matrix in essolve (by **Nathan Shammah**).

- Fixed name collision for parallel propagator (by **Nathan Shammah**).

- Fixed dimensional incongruence in `propagator` (by **Nathan Shammah**)

- Fixed bug by rewriting clebsch function based on long integer fraction (by **Eric Giguère**).

- Fixed bugs in QobjEvo's args depending on state and added solver tests using them (by **Eric Giguère**).

- Fixed bug in `sesolve` calculation of average states when summing the timeslot states (by **Alex Pitchford**).

- Fixed bug in `steadystate` solver by removing separate arguments for MKL and Scipy (by **Tarun Raheja**).

- Fixed `Bloch.add_ponts` by setting `edgecolor = None` in `plot_points` (by **Nathan Shammah**).

- Fixed error checking for null matrix in `essolve` solver affecting also `ode2es` (by **Peter Kirton**).

- Removed unnecessary shebangs in .pyx and .pxd files (by **Samesh Lakhotia**).

- Fixed `sesolve` and  import of `os` in `codegen` (by **Alex Pitchford**).

- Updated `plot_fock_distribution` by removing the offset value 0.4 in the plot (by **Rajiv-B**).


QuTiP 4.4.1 (2019-08-29)
========================

Improvements
------------

- QobjEvo do not need to start from 0 anymore (by **Eric Giguère**).

- Add a quantum object purity function (by **Nathan Shammah** and **Shahnawaz Ahmed**).

- Add step function interpolation for array time-coefficient (by **Boxi Li**).

- Generalize expand_oper for arbitrary dimensions, and new method for cyclic permutations of given target cubits (by **Boxi Li**).


Bug Fixes
---------

- Fixed the pickling but that made solver unable to run in parallel on Windows (Thank **lrunze** for reporting)

- Removed warning when mesolve fall back on sesolve (by **Michael Goerz**).

- Fixed dimension check and confusing documentation in random ket (by **Yariv Yanay**).

- Fixed Qobj isherm not working after using Qobj.permute (Thank **llorz1207** for reporting).

- Correlation functions call now properly handle multiple time dependant functions (Thank **taw181** for reporting).

- Removed mutable default values in mesolve/sesolve (by **Michael Goerz**).

- Fixed simdiag bug (Thank **Croydon-Brixton** for reporting).

- Better support of constant QobjEvo (by **Boxi Li**).

- Fixed potential cyclic import in the control module (by **Alexander Pitchford**).


QuTiP 4.4.0 (2019-07-03)
========================

Improvements
------------

- **MAJOR FEATURE**: Added methods and techniques to the stochastic solvers (by **Eric Giguère**) which allows to use a much broader set of solvers and much more efficiently.

- **MAJOR FEATURE**: Optimization of the montecarlo solver (by **Eric Giguère**). Computation are faster in many cases. Collapse information available to time dependant information.

- Added the QObjEvo class and methods (by **Eric Giguère**), which is used behind the scenes by the dynamical solvers, making the code more efficient and tidier. More built-in function available to string coefficients.

- The coefficients can be made from interpolated array with variable timesteps and can obtain state information more easily. Time-dependant collapse operator can have multiple terms.

- New wigner_transform and plot_wigner_sphere function. (by **Nithin Ramu**).

- ptrace is faster and work on bigger systems, from 15 Qbits to 30 Qbits.

- QIP module: added the possibility for user-defined gates, added the possibility to remove or add gates in any point of an already built circuit, added the molmer_sorensen gate, and fixed some bugs (by **Boxi Li**).

- Added the quantum Hellinger distance to qutip.metrics (by **Wojciech Rzadkowski**).

- Implemented possibility of choosing a random seed (by **Marek Marekyggdrasil**).

- Added a code of conduct to Github.


Bug Fixes
---------

- Fixed bug that made QuTiP incompatible with SciPy 1.3.


QuTiP 4.3.0 (2018-07-14)
========================

Improvements
------------

- **MAJOR FEATURE**: Added the Permutational Invariant Quantum Solver (PIQS) module (by **Nathan Shammah** and **Shahnawaz Ahmed**) which allows the simluation of large TLSs ensembles including collective and local Lindblad dissipation. Applications range from superradiance to spin squeezing.

- **MAJOR FEATURE**: Added a photon scattering module (by **Ben Bartlett**) which can be used to study scattering in arbitrary driven systems coupled to some configuration of output waveguides.

- Cubic_Spline functions as time-dependent arguments for the collapse operators in mesolve are now allowed.

- Added a faster version of bloch_redfield_tensor, using components from the time-dependent version. About 3x+ faster for secular tensors, and 10x+ faster for non-secular tensors.

- Computing Q.overlap() [inner product] is now ~30x faster.

- Added projector method to Qobj class.

- Added fast projector method, ``Q.proj()``.

- Computing matrix elements, ``Q.matrix_element`` is now ~10x faster.

- Computing expectation values for ket vectors using ``expect`` is now ~10x faster.

- ``Q.tr()`` is now faster for small Hilbert space dimensions.

- Unitary operator evolution added to sesolve

- Use OPENMP for tidyup if installed.


Bug Fixes
---------

- Fixed bug that stopped simdiag working for python 3.

- Fixed semidefinite cvxpy Variable and Parameter.

- Fixed iterative lu solve atol keyword issue.

- Fixed unitary op evolution rhs matrix in ssesolve.

- Fixed interpolating function to return zero outside range.

- Fixed dnorm complex casting bug.

- Fixed control.io path checking issue.

- Fixed ENR fock dimension.

- Fixed hard coded options in propagator 'batch' mode

- Fixed bug in trace-norm for non-Hermitian operators.

- Fixed bug related to args not being passed to coherence_function_g2

- Fixed MKL error checking dict key error


QuTiP 4.2.0 (2017-07-28)
========================

Improvements
------------

- **MAJOR FEATURE**: Initial implementation of time-dependent Bloch-Redfield Solver.

- Qobj tidyup is now an order of magnitude faster.

- Time-dependent codegen now generates output NumPy arrays faster.

- Improved calculation for analytic coefficients in coherent states (Sebastian Kramer).

- Input array to correlation FFT method now checked for validity.

- Function-based time-dependent mesolve and sesolve routines now faster.

- Codegen now makes sure that division is done in C, as opposed to Python.

- Can now set different controls for a each timeslot in quantum optimization.
  This allows time-varying controls to be used in pulse optimisation.


Bug Fixes
---------

- rcsolve importing old Odeoptions Class rather than Options.

- Non-int issue in spin Q and Wigner functions.

- Qobj's should tidyup before determining isherm.

- Fixed time-dependent RHS function loading on Win.

- Fixed several issues with compiling with Cython 0.26.

- Liouvillian superoperators were hard setting isherm=True by default.

- Fixed an issue with the solver safety checks when inputing a list with Python functions as time-dependence.

- Fixed non-int issue in Wigner_cmap.

- MKL solver error handling not working properly.



QuTiP 4.1.0 (2017-03-10)
========================

Improvements
------------

*Core libraries*

- **MAJOR FEATURE**: QuTiP now works for Python 3.5+ on Windows using Visual Studio 2015.

- **MAJOR FEATURE**: Cython and other low level code switched to C++ for MS Windows compatibility.

- **MAJOR FEATURE**: Can now use interpolating cubic splines as time-dependent coefficients.

- **MAJOR FEATURE**: Sparse matrix - vector multiplication now parallel using OPENMP.

- Automatic tuning of OPENMP threading threshold.

- Partial trace function is now up to 100x+ faster.

- Hermitian verification now up to 100x+ faster.

- Internal Qobj objects now created up to 60x faster.

- Inplace conversion from COO -> CSR sparse formats (e.g. Memory efficiency improvement.)

- Faster reverse Cuthill-Mckee and sparse one and inf norms.



Bug Fixes
---------

- Cleanup of temp. Cython files now more robust and working under Windows.



QuTiP 4.0.2 (2017-01-05)
========================

Bug Fixes
---------
- td files no longer left behind by correlation tests
- Various fast sparse fixes



QuTiP 4.0.0 (2016-12-22)
========================

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

QuTiP 3.2.0
===========

(Never officially released)

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



QuTiP 3.1.0 (2015-01-01)
========================

New Features
------------

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


QuTiP 3.0.1 (2014-08-05)
========================

Bug Fixes
---------

- Fix bug in create(), which returned a Qobj with CSC data instead of CSR.
- Fix several bugs in mcsolve: Incorrect storing of collapse times and collapse
  operator records. Incorrect averaging of expectation values for different
  trajectories when using only 1 CPU.
- Fix bug in parsing of time-dependent Hamiltonian/collapse operator arguments
  that occurred when the args argument is not a dictionary.
- Fix bug in internal _version2int function that cause a failure when parsingthe version number of the Cython package.


QuTiP 3.0.0 (2014-07-17)
========================

New Features
------------

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
------------

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


QuTiP 2.2.0 (2013-03-01)
========================


New Features
------------

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


Bug Fixes
---------

- Function based time-dependent Hamiltonians now keep the correct phase.

- mcsolve no longer prints to the command line if ntraj=1.


QuTiP 2.1.0 (2012-10-05)
========================


New Features
------------

- New method for generating Wigner functions based on Laguerre polynomials.

- coherent(), coherent_dm(), and thermal_dm() can now be expressed using analytic values.

- Unittests now use nose and can be run after installation.

- Added iswap and sqrt-iswap gates.

- Functions for quantum process tomography.

- Window icons are now set for Ubuntu application launcher.

- The propagator function can now take a list of times as argument, and returns a list of corresponding propagators.


Bug Fixes
---------

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


QuTiP 2.0.0 (2012-06-01)
========================

The second version of QuTiP has seen many improvements in the performance of the original code base, as well as the addition of several new routines supporting a wide range of functionality.  Some of the highlights of this release include:

New Features
------------

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


QuTiP 1.1.4 (2012-05-28)
========================

Bug Fixes
---------

- Fixed bug pointed out by Brendan Abolins.

- Qobj.tr() returns zero-dim ndarray instead of float or complex.

- Updated factorial import for scipy version 0.10+


QuTiP 1.1.3 (2011-11-21)
========================

New Functions
-------------

- Allow custom naming of Bloch sphere.

Bug Fixes
---------
- Fixed text alignment issues in AboutBox.

- Added fix for SciPy V>0.10 where factorial was moved to scipy.misc module.

- Added tidyup function to tensor function output.

- Removed openmp flags from setup.py as new Mac Xcode compiler does not recognize them.

- Qobj diag method now returns real array if all imaginary parts are zero.

- Examples GUI now links to new documentation.

- Fixed zero-dimensional array output from metrics module.


QuTiP 1.1.2 (2011-10-27)
========================

Bug Fixes
---------

- Fixed issue where Monte Carlo states were not output properly.


QuTiP 1.1.1 (2011-10-25)
========================

**THIS POINT-RELEASE INCLUDES VASTLY IMPROVED TIME-INDEPENDENT MCSOLVE AND ODESOLVE PERFORMANCE**

New Functions
-------------

- Added linear entropy function.

- Number of CPU's can now be changed.

Bug Fixes
---------

- Metrics no longer use dense matrices.

- Fixed Bloch sphere grid issue with matplotlib 1.1.

- Qobj trace operation uses only sparse matrices.

- Fixed issue where GUI windows do not raise to front.


QuTiP 1.1.0 (2011-10-04)
========================

**THIS RELEASE NOW REQUIRES THE GCC COMPILER TO BE INSTALLED**

New Functions
-------------

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


QuTiP 1.0.0 (2011-07-29)
========================

- **Initial release.**
