.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _options:

*********************************************
Setting Options for the Dynamics Solvers
*********************************************

Occasionally it is necessary to change the built in parameters of the dynamics solvers used by for example the :func:`qutip.mesolve` and :func:`qutip.mcsolve` functions.  The options for all dynamics solvers may be changed by using the Options class :class:`qutip.solver.Options`.

>>> options = Options()

the properties and default values of this class can be view via the `print` function::

    >>> print(options)
	Options properties:
	----------------------
	atol:          1e-08
	rtol:          1e-06
	method:        adams
	order:         12
	nsteps:        1000
	first_step:    0
	min_step:      0
	max_step:      0
	tidy:          True
	num_cpus:      8
	rhs_filename:  None
	rhs_reuse:     False
	gui:           True
	mc_avg:    	   True

These properties are detailed in the following table.  Assuming ``options = Options()``:

+---------------------+-----------------+----------------------------------------------------------------+
| Property            | Default setting | Description                                                    |
+=====================+=================+================================================================+
| options.atol        | 1e-8            | Absolute tolerance                                             |
+---------------------+-----------------+----------------------------------------------------------------+
| options.rtol        | 1e-6            | Relative tolerance                                             |
+---------------------+-----------------+----------------------------------------------------------------+
| options.method      | 'adams'         | Solver method.  Can be 'adams' (non-stiff) or 'bdf' (stiff)    |
+---------------------+-----------------+----------------------------------------------------------------+
| options.order       | 12              | Order of solver.  Must be <=12 for 'adams' and <=5 for 'bdf'   |
+---------------------+-----------------+----------------------------------------------------------------+
| options.nsteps      | 1000            | Max. number of steps to take for each interval                 |
+---------------------+-----------------+----------------------------------------------------------------+
| options.first_step  | 0               | Size of initial step.  0 = determined automatically by solver. |
+---------------------+-----------------+----------------------------------------------------------------+
| options.min_step    | 0               | Minimum step size.  0 = determined automatically by solver.    |
+---------------------+-----------------+----------------------------------------------------------------+
| options.max_step    | 0               | Maximum step size.  0 = determined automatically by solver.    |
+---------------------+-----------------+----------------------------------------------------------------+
| options.tidy        | True            | Whether to run tidyup function on time-independent Hamiltonian.| 
+---------------------+-----------------+----------------------------------------------------------------+
| options.num_cpus    | installed num   |  Integer number of cpu's used by mcsolve.                      |
|                     | of processors   |                                                                |
+---------------------+-----------------+----------------------------------------------------------------+
| options.rhs_filename| None            | RHS filename when using compiled time-dependent Hamiltonians.  |
+---------------------+-----------------+----------------------------------------------------------------+
| options.rhs_reuse   | False           | Reuse compiled RHS function.  Useful for repeatative tasks.    |
+---------------------+-----------------+----------------------------------------------------------------+
| options.gui         | True (if GUI)   | Use the mcsolve progessbar. Defaults to False on Windows.      |
+---------------------+-----------------+----------------------------------------------------------------+
| options.mc_avg      | True            | Average over trajectories for expectation values from mcsolve. |
+---------------------+-----------------+----------------------------------------------------------------+


As an example, let us consider changing the number of processors used, turn the GUI off, and strengthen the absolute tolerance.  There are two equivalent ways to do this using the Options class.  First way,

    >>> options = Options()
    >>> options.num_cpus = 3
    >>> options.gui = False
    >>> options.atol = 1e-10

or one can use an inline method,

	>>> options = Options(num_cpus=3, gui=False, atol=1e-10)

Note that the order in which you input the options does not matter.  Using either method, the resulting `options` variable is now::

	>>> print options
	Options properties:
	----------------------
	atol:          1e-10
	rtol:          1e-06
	method:        adams
	order:         12
	nsteps:        1000
	first_step:    0
	min_step:      0
	max_step:      0
	tidy:          True
	num_cpus:      3
	rhs_filename:  None
	rhs_reuse:     False
	gui:           False
	mc_avg:    True

To use these new settings we can use the keyword argument ``options`` in either the func:`qutip.mesolve` and :func:`qutip.mcsolve` function.  We can modify the last example as::

    >>> mesolve(H0, psi0, tlist, c_op_list, [sigmaz()], options=options)
    >>> mesolve(hamiltonian_t, psi0, tlist, c_op_list, [sigmaz()], H_args,
    >>>         options=options)

or::
    
    >>> mcsolve(H0, psi0, tlist, ntraj,c_op_list, [sigmaz()], options=options)
    >>> mcsolve(hamiltonian_t, psi0, tlist, ntraj, c_op_list, [sigmaz()], H_args,
    >>>         options=options)


