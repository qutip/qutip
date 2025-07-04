.. _intro:

************
Introduction
************

Although in some cases, we want to find the stationary states of
a quantum system, often we are interested in the dynamics:
how the state of a system or an ensemble of systems evolves with time.
QuTiP provides many ways to model dynamics.

There are two kinds of quantum systems: open systems that interact
with a larger environment and closed systems that do not.
In a closed system, the state can be described by a state vector.
When we are modeling an open system, or an ensemble of systems,
the use of the density matrix is mandatory.

The following table lists of the solvers QuTiP provides for dynamic
quantum systems and indicates the type of object returned by the solver:

.. list-table:: QuTiP Solvers
   :widths: 50 25 25 25
   :header-rows: 1

   * - Equation
     - Function
     - Class
     - Returns
   * - Unitary evolution, Schrödinger equation.
     - :func:`~qutip.solver.sesolve.sesolve`
     - :obj:`~qutip.solver.sesolve.SESolver`
     - :obj:`~qutip.solver.result.Result`
   * - Periodic Schrödinger equation.
     - :func:`~qutip.solver.floquet.fsesolve`
     - None
     - :obj:`~qutip.solver.result.Result`
   * - Schrödinger equation using Krylov method
     - :func:`~qutip.solver.krylovsolve.krylovsolve`
     - None
     - :obj:`~qutip.solver.result.Result`
   * - Lindblad master eqn. or Von Neuman eqn.
     - :func:`~qutip.solver.mesolve.mesolve`
     - :obj:`~qutip.solver.mesolve.MESolver`
     - :obj:`~qutip.solver.result.Result`
   * - Monte Carlo evolution
     - :func:`~qutip.solver.mcsolve.mcsolve`
     - :obj:`~qutip.solver.mcsolve.MCSolver`
     - :obj:`~qutip.solver.multitrajresult.McResult`
   * - Non-Markovian Monte Carlo
     - :func:`~qutip.solver.nm_mcsolve.nm_mcsolve`
     - :obj:`~qutip.solver.nm_mcsolve.NonMarkovianMCSolver`
     - :obj:`~qutip.solver.multitrajresult.NmmcResult`
   * - Bloch-Redfield master equation
     - :func:`~qutip.solver.brmesolve.brmesolve`
     - :obj:`~qutip.solver.brmesolve.BRSolver`
     - :obj:`~qutip.solver.result.Result`
   * - Floquet-Markov master equation
     - :func:`~qutip.solver.floquet.fmmesolve`
     - :obj:`~qutip.solver.floquet.FMESolver`
     - :obj:`~qutip.solver.floquet.FloquetResult`
   * - Stochastic Schrödinger equation
     - :func:`~qutip.solver.stochastic.ssesolve`
     - :obj:`~qutip.solver.stochastic.SSESolver`
     - :obj:`~qutip.solver.multitrajresult.MultiTrajResult`
   * - Stochastic master equation
     - :func:`~qutip.solver.stochastic.smesolve`
     - :obj:`~qutip.solver.stochastic.SMESolver`
     - :obj:`~qutip.solver.multitrajresult.MultiTrajResult`
   * - Transfer Tensor Method time-evolution
     - :func:`~qutip.solver.nonmarkov.transfertensor.ttmsolve`
     - None
     - :obj:`~qutip.solver.result.Result`
   * - Hierarchical Equations of Motion evolution
     - :func:`~qutip.solver.heom.bofin_solvers.heomsolve`
     - :obj:`~qutip.solver.heom.bofin_solvers.HEOMSolver`
     - :obj:`~qutip.solver.heom.bofin_solvers.HEOMResult`
