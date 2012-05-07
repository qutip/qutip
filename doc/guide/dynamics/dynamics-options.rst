.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _odeoptions:

*********************************************
Setting Options for the Dynamics ODE Solvers
*********************************************

Occasionally it is necessary to change the built in parameters of the ODE solvers used by both the mesolve and mcsolve functions.  The ODE options for either of these functions may be changed by calling the Odeoptions class :class:`qutip.Odeoptions`

>>> opts=Odeoptions()

the properties and default values of this class can be view via the `print` command::

    >>> print opts
	Odeoptions properties:
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
	expect_avg:    True

These properties are detailed in the following table.  Assuming ``opts=Odeoptions()``:

+-------------------+-----------------+----------------------------------------------------------------+
| Property          | Default setting | Description                                                    |
+===================+=================+================================================================+
| opts.atol         | 1e-8            | Absolute tolerance                                             |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.rtol         | 1e-6            | Relative tolerance                                             |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.method       | 'adams'         | Solver method.  Can be 'adams' (non-stiff) or 'bdf' (stiff)    |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.order        | 12              | Order of solver.  Must be <=12 for 'adams' and <=5 for 'bdf'   |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.nsteps       | 1000            | Max. number of steps to take for each interval                 |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.first_step   | 0               | Size of initial step.  0 = determined automatically by solver. |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.min_step     | 0               | Minimum step size.  0 = determined automatically by solver.    |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.max_step     | 0               | Maximum step size.  0 = determined automatically by solver.    |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.tidy         | True            | Whether to run tidyup function on time-independent Hamiltonian.| 
+-------------------+-----------------+----------------------------------------------------------------+
| opts.num_cpus     | installed num   |  Integer number of cpu's used by mcsolve.                      |
|                   | of processors   |                                                                |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.rhs_filename | None            | RHS filename when using compiled time-dependent Hamiltonians.  |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.rhs_reuse    | False           | Reuse compiled RHS function.  Useful for repeatative tasks.    |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.gui          | True (if GUI)   | Use the mcsolve progessbar. Defaults to False on Windows.      |
+-------------------+-----------------+----------------------------------------------------------------+
| opts.expect_avg   | True            | Average over trajectories for expectation values from mcsolve. |
+-------------------+-----------------+----------------------------------------------------------------+


As an example, let us consider changing the number of processors used, turn the GUI off, and strengthen the absolute tolerance.  There are two equivalent ways to do this using the Odeoptions class.  First way,

    >>> opts=Odeoptions()
    >>> opts.num_cpus=3
    >>> opts.gui=False
    >>> opts.atol=1e-10

or one can use an inline method,

	>>> opts=Odeoptions(num_cpus=3,gui=False,atol=1e-10)

Note that the order in which you input the options does not matter.  Using either method, the resulting `opts` variable is now::

	>>> print opts
	Odeoptions properties:
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
	expect_avg:    True

To use these new settings we can use the keyword argument `options` in either the `mesolve` or `mcsolve` function.  We can modify the last example as::

    >>> mesolve(H0, psi0, tlist, c_op_list, [sigmaz()],options=opts)
    >>> mesolve(hamiltonian_t, psi0, tlist, c_op_list, [sigmaz()], H_args,options=opts)

or::
    
    >>> mcsolve(H0, psi0, tlist, ntraj,c_op_list, [sigmaz()],options=opts)
    >>> mcsolve(hamiltonian_t, psi0, tlist, ntraj, c_op_list, [sigmaz()], H_args,options=opts)


