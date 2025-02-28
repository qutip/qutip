.. _options:

*********************************************
Setting Options for the Dynamics Solvers
*********************************************

.. testsetup:: [dynamics_options]

   from qutip.solver.mesolve import MESolver, mesolve
   import numpy as np

Occasionally it is necessary to change the built in parameters of the dynamics
solvers used by for example the :func:`.mesolve` and :func:`.mcsolve` functions.
The options for all dynamics solvers may be changed by using the dictionaries.

.. testcode:: [dynamics_options]

   options = {"store_states": True, "atol": 1e-12}

Supported items come from 2 sources, the solver and the ODE integration method.
Supported solver options and their default can be seen using the class interface:

.. testcode:: [dynamics_options]

   help(MESolver.options)

Options supported by the ODE integration depend on the "method" options of the solver,
they can be listed through the integrator method of the solvers:

.. testcode:: [dynamics_options]

   help(MESolver.integrator("adams").options)

See :ref:`api-ode` for a list of supported methods.


As an example, let us consider changing the integrator, turn the GUI off, and
strengthen the absolute tolerance.

.. testcode:: [dynamics_options]

    options = {method="bdf", "atol": 1e-10, "progress_bar": False}

To use these new settings we can use the keyword argument ``options`` in either
the :func:`.mesolve` and :func:`.mcsolve` function::

    >>> mesolve(H0, psi0, tlist, c_op_list, [sigmaz()], options=options)

or::

    >>> MCSolver(H0, c_op_list, options=options)
