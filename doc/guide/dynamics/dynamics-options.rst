.. _options:

*********************************************
Setting Options for the Dynamics Solvers
*********************************************

.. testsetup:: [dynamics_options]

   from qutip.solver.mesolve import MeSolver, mesolve
   import numpy as np

Occasionally it is necessary to change the built in parameters of the dynamics solvers used by for example the :func:`qutip.mesolve` and :func:`qutip.mcsolve` functions.
The options for all dynamics solvers may be changed by using the dictionaries.

.. testcode:: [dynamics_options]

   options = {"store_states": True, "atol": 1e-12}

Supported items come from 2 sources, the solver and the ODE integration method.
Supported solver options and their default can be seen with the class interface:

.. testcode:: [dynamics_options]

   help(MeSolver.options)

Options supported by the ODE integration depend on the "method" options of the solver, they can be listed through the integrator method of the solvers:

.. testcode:: [dynamics_options]

   help(MeSolver.integrator("adams"))

See `Integrator <../../apidoc/classes.html#classes-ode>`_ for a list of supported methods.


As an example, let us consider changing the number of processors used, turn the GUI off, and strengthen the absolute tolerance.  There are two equivalent ways to do this using the SolverOptions class.  First way,

.. testcode:: [dynamics_options]

    options = {method="bdf", "atol": 1e-10}

To use these new settings we can use the keyword argument ``options`` in either the func:`qutip.mesolve` and :func:`qutip.mcsolve` function.  We can modify the last example as::

    >>> mesolve(H0, psi0, tlist, c_op_list, [sigmaz()], options=options)
    >>> MeSolver(hamiltonian_t, c_op_list, options=options)

or::

    >>> mcsolve(H0, psi0, tlist, ntraj,c_op_list, [sigmaz()], options=options)
    >>> mcsolve(hamiltonian_t, psi0, tlist, ntraj, c_op_list, [sigmaz()], H_args, options=options)
