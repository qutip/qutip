.. _solver_result:

********************************************************
Dynamics Simulation Results
********************************************************

.. _solver_result-class:

The solver.Result Class
=======================

Before embarking on simulating the dynamics of quantum systems, we will first
look at the data structure used for returning the simulation results. This
object is a :func:`~qutip.solver.result.Result` class that stores all the
crucial data needed for analyzing and plotting the results of a simulation.
A generic ``Result`` object ``result`` contains the following properties for
storing simulation data:

.. cssclass:: table-striped

+------------------------+-----------------------------------------------------------------------+
| Property               | Description                                                           |
+========================+=======================================================================+
| ``result.solver``      | String indicating which solver was used to generate the data.         |
+------------------------+-----------------------------------------------------------------------+
| ``result.times``       | List/array of times at which simulation data is calculated.           |
+------------------------+-----------------------------------------------------------------------+
| ``result.expect``      | List/array of expectation values, if requested.                       |
+------------------------+-----------------------------------------------------------------------+
| ``result.e_data``      | Dictionary of expectation values, if requested.                       |
+------------------------+-----------------------------------------------------------------------+
| ``result.states``      | List/array of state vectors/density matrices calculated at ``times``, |
|                        | if requested.                                                         |
+------------------------+-----------------------------------------------------------------------+
| ``result.final_state`` | State vector or density matrix at the last time of the evolution.     |
+------------------------+-----------------------------------------------------------------------+
| ``result.stats``       | Various statistics about the evolution.                               |
+------------------------+-----------------------------------------------------------------------+


.. _odedata-access:

Accessing Result Data
======================

To understand how to access the data in a Result object we will use an example
as a guide, although we do not worry about the simulation details at this stage.
Like all solvers, the Master Equation solver used in this example returns an
Result object, here called simply ``result``. To see what is contained inside
``result`` we can use the print function:

.. doctest::
  :options: +SKIP

  >>> print(result)
  <Result
    Solver: mesolve
    Solver stats:
      method: 'scipy zvode adams'
      init time: 0.0001876354217529297
      preparation time: 0.007544517517089844
      run time: 0.001268625259399414
      solver: 'Master Equation Evolution'
      num_collapse: 1
    Time interval: [0, 1.0] (2 steps)
    Number of e_ops: 1
    State not saved.
  >

The first line tells us that this data object was generated from the Master
Equation solver :func:`.mesolve`. Next we have the statistics including the ODE
solver used, setup time, number of collpases. Then the integration interval is
described, followed with the number of expectation value computed. Finally, it
says whether the states are stored.

Now we have all the information needed to analyze the simulation results.
To access the data for the two expectation values one can do:


.. testcode::
  :skipif: True

  expt0 = result.expect[0]
  expt1 = result.expect[1]

Recall that Python uses C-style indexing that begins with zero (i.e.,
[0] => 1st collapse operator data).
Alternatively, expectation values can be obtained as a dictionary:

.. testcode::
  :skipif: True

  e_ops = {"sx": sigmax(), "sy": sigmay(), "sz": sigmaz()}
  ...
  expt_sx = result.e_data["sx"]

When ``e_ops`` is a list, ``e_data`` ca be used with the list index. Together
with the array of times at which these expectation values are calculated:

.. testcode::
  :skipif: True

  times = result.times

we can plot the resulting expectation values:

.. testcode::
  :skipif: True

  plot(times, expt0)
  plot(times, expt1)
  show()


State vectors, or density matrices, are accessed in a similar manner, although
typically one does not need an index (i.e [0]) since there is only one list for
each of these components. Some other solver can have other output,
:func:`.heomsolve`'s results can have ``ado_states`` output if the options
``store_ados`` is set, similarly, :func:`.fmmesolve` can return
``floquet_states``.


Multiple Trajectories Solver Results
====================================


Solver which compute multiple trajectories such as the Monte Carlo Equations
Solvers or the Stochastics Solvers result will differ depending on whether the
trajectories are flags to be saved.
For example:

.. doctest::
  :options: +SKIP

  >>> mcsolve(H, psi, np.linspace(0, 1, 11), c_ops, e_ops=[num(N)], ntraj=25, options={"keep_runs_results": False})
  >>> np.shape(result.expect)
  (1, 11)

  >>> mcsolve(H, psi, np.linspace(0, 1, 11), c_ops, e_ops=[num(N)], ntraj=25, options={"keep_runs_results": True})
  >>> np.shape(result.expect)
  (1, 25, 11)


When the runs are not saved, the expectation values and states are averaged
over all trajectories, while a list over the runs are given when they are stored.
For a fix output format, ``average_expect`` return the average, while
``runs_states`` return the list over trajectories.  The ``runs_`` output will
return ``None`` when the trajectories are not saved. Standard derivation of the
expectation values is also available:

+-------------------------+----------------------+------------------------------------------------------------------------+
| Reduced result          | Trajectories results | Description                                                            |
+=========================+======================+========================================================================+
| ``average_states``      | ``runs_states``      | State vectors or density matrices calculated at each times of tlist    |
+-------------------------+----------------------+------------------------------------------------------------------------+
| ``average_final_state`` | ``runs_final_state`` | State vectors or density matrices calculated at the last time of tlist |
+-------------------------+----------------------+------------------------------------------------------------------------+
| ``average_expect``      | ``runs_expect``      | List/array of expectation values, if requested.                        |
+-------------------------+----------------------+------------------------------------------------------------------------+
| ``std_expect``          |                      | List/array of standard derivation of the expectation values.           |
+-------------------------+----------------------+------------------------------------------------------------------------+
| ``average_e_data``      | ``runs_e_data``      | Dictionary of expectation values, if requested.                        |
+-------------------------+----------------------+------------------------------------------------------------------------+
| ``std_e_data``          |                      | Dictionary of standard derivation of the expectation values.           |
+-------------------------+----------------------+------------------------------------------------------------------------+

Multiple trajectories results also keep the trajectories ``seeds`` to allows
recomputing the results.

.. testcode::
  :skipif: True

  seeds = result.seeds

One last feature specific to multi-trajectories results is the addition operation
that can be used to merge sets of trajectories.


.. code-block::

    >>> run1 = smesolve(H, psi, np.linspace(0, 1, 11), c_ops, e_ops=[num(N)], ntraj=25)
    >>> print(run1.num_trajectories)
    25
    >>> run2 = smesolve(H, psi, np.linspace(0, 1, 11), c_ops, e_ops=[num(N)], ntraj=25)
    >>> print(run2.num_trajectories)
    25
    >>> merged = run1 + run2
    >>> print(merged.num_trajectories)
    50

This allows one to improve statistics while keeping previous computations.
