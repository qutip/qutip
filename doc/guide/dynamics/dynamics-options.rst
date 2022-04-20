.. _options:

*********************************************
Setting Options for the Dynamics Solvers
*********************************************

.. testsetup:: [dynamics_options]

   from qutip import SolverOptions

   import numpy as np

Occasionally it is necessary to change the built in parameters of the dynamics solvers used by for example the :func:`qutip.mesolve` and :func:`qutip.mcsolve` functions.  The options for all dynamics solvers may be changed by using the SolverOptions class :class:`qutip.solve.solver.SolverOptions`.

.. testcode:: [dynamics_options]

   options = SolverOptions()

the properties and default values of this class can be view via the `print` function:

.. testcode:: [dynamics_options]

   print(options)

**Output**:

.. testoutput:: [dynamics_options]
  :options: +NORMALIZE_WHITESPACE

  Options:
  -----------
  atol:              1e-08
  rtol:              1e-06
  method:            adams
  order:             12
  nsteps:            1000
  first_step:        0
  min_step:          0
  max_step:          0
  tidy:              True
  num_cpus:          2
  norm_tol:          0.001
  norm_steps:        5
  rhs_filename:      None
  rhs_reuse:         False
  seeds:             0
  rhs_with_state:    False
  average_expect:    True
  average_states:    False
  ntraj:             500
  store_states:      False
  store_final_state: False

These properties are detailed in the following table.  Assuming ``options = SolverOptions()``:

.. cssclass:: table-striped

+-----------------------------+-----------------+----------------------------------------------------------------+
| Property                    | Default setting | Description                                                    |
+=============================+=================+================================================================+
| options.atol                | 1e-8            | Absolute tolerance                                             |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.rtol                | 1e-6            | Relative tolerance                                             |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.method              | 'adams'         | Solver method.  Can be 'adams' (non-stiff) or 'bdf' (stiff)    |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.order               | 12              | Order of solver.  Must be <=12 for 'adams' and <=5 for 'bdf'   |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.nsteps              | 1000            | Max. number of steps to take for each interval                 |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.first_step          | 0               | Size of initial step.  0 = determined automatically by solver. |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.min_step            | 0               | Minimum step size.  0 = determined automatically by solver.    |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.max_step            | 0               | Maximum step size.  0 = determined automatically by solver.    |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.tidy                | True            | Whether to run tidyup function on time-independent Hamiltonian.|
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.store_final_state   | False           | Whether or not to store the final state of the evolution.      |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.store_states        | False           | Whether or not to store the state vectors or density matrices. |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.rhs_filename        | None            | RHS filename when using compiled time-dependent Hamiltonians.  |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.rhs_reuse           | False           | Reuse compiled RHS function.  Useful for repetitive tasks.     |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.rhs_with_state      | False           | Whether or not to include the state in the Hamiltonian         |
|                             |                 | function callback signature.                                   |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.num_cpus            | installed num   | Integer number of cpus used by mcsolve.                        |
|                             | of processors   |                                                                |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.seeds               | None            | Array containing random number seeds for mcsolver.             |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.norm_tol            | 1e-6            | Tolerance used when finding wavefunction norm in mcsolve.      |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.norm_steps          | 5               | Max. number of steps used to find wavefunction's norm to within|
|                             |                 | norm_tol in mcsolve.                                           |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.steady_state_average| False           | Include an estimation of the steady state  in mcsolve.         |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.ntraj               | 500             | Number of trajectories in stochastic solvers.                  |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.average_expect      | True            | Average expectation values over trajectories.                  |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.average_states      | False           | Average of the states over trajectories.                       |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.openmp_threads      | installed num   | Number of OPENMP threads to use.                               |
|                             | of processors   |                                                                |
+-----------------------------+-----------------+----------------------------------------------------------------+
| options.use_openmp          | None            | Use OPENMP for sparse matrix vector multiplication.            |
+-----------------------------+-----------------+----------------------------------------------------------------+

As an example, let us consider changing the number of processors used, turn the GUI off, and strengthen the absolute tolerance.  There are two equivalent ways to do this using the SolverOptions class.  First way,

.. testcode:: [dynamics_options]

    options = SolverOptions()
    options.num_cpus = 3
    options.atol = 1e-10

or one can use an inline method,

.. testcode:: [dynamics_options]

    options = SolverOptions(num_cpus=4, atol=1e-10)

Note that the order in which you input the options does not matter.  Using either method, the resulting `options` variable is now:

.. testcode:: [dynamics_options]

  print(options)

**Output**:

.. testoutput:: [dynamics_options]
  :options: +NORMALIZE_WHITESPACE

  Options:
  -----------
  atol:              1e-10
  rtol:              1e-06
  method:            adams
  order:             12
  nsteps:            1000
  first_step:        0
  min_step:          0
  max_step:          0
  tidy:              True
  num_cpus:          4
  norm_tol:          0.001
  norm_steps:        5
  rhs_filename:      None
  rhs_reuse:         False
  seeds:             0
  rhs_with_state:    False
  average_expect:    True
  average_states:    False
  ntraj:             500
  store_states:      False
  store_final_state: False



To use these new settings we can use the keyword argument ``options`` in either the func:`qutip.mesolve` and :func:`qutip.mcsolve` function.  We can modify the last example as::

    >>> mesolve(H0, psi0, tlist, c_op_list, [sigmaz()], options=options)
    >>> mesolve(hamiltonian_t, psi0, tlist, c_op_list, [sigmaz()], H_args, options=options)

or::

    >>> mcsolve(H0, psi0, tlist, ntraj,c_op_list, [sigmaz()], options=options)
    >>> mcsolve(hamiltonian_t, psi0, tlist, ntraj, c_op_list, [sigmaz()], H_args, options=options)
