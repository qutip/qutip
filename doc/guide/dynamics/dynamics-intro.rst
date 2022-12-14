.. _intro:

************
Introduction
************

Although in some cases, we want to find the stationary states of 
a quantum system, often we are interested in the dynamics: 
how the state of a system or an ensemble of systems evolves with time. QuTiP provides 
many ways to model dynamics. 

Broadly speaking, there are two categories 
of dynamical models: unitary and non-unitary. In unitary evolution, 
the state of the system remains normalized. In non-unitary, or 
dissipative, systems, it does not. 

There are two kinds of quantum systems: open systems that interact 
with a larger environment and closed systems that do not. 
In a closed system, the state can be described by a state vector, 
although when there is entanglement a density matrix may be 
needed instead. When we are modeling an open system, or an ensemble 
of systems, the use of the density matrix is mandatory.

Collapse operators are used to model the collapse of the state vector 
that can occur when a measurement is performed.

The following tables lists some of the solvers QuTiP provides for dynamic quantum systems and indicates the type of object 
returned by the solver:

.. list-table:: QuTiP Solvers
   :widths: 25 25 50
   :header-rows: 1

   * - Solver
     - Returns
     - Remarks
   * - sesolve()
     - :func:`qutip.solver.Result`
     - Unitary evolution, single system
   * - mesolve()
     - :func:`qutip.solver.Result`
     - Lindblad master eqn. or Von Neuman eqn. Density matrix.
   * - mcsolve()
     - :func:`qutip.solver.Result`
     - Monte Carlo with collapse operators
   * - essolve()
     - Array of expectation values
     - Exponential series with collapse operators
   * - bloch_redfield_solve()
     - :func:`qutip.solver`
     -
   * - floquet_markov_solve()
     - :func:`qutip.solver.Result`
     - Floquet-Markov master equation
   * - fmmesolve()
     - :func:`qutip.solver`
     - Floquet-Markov master equation
   * - smesolve()
     - :func:`qutip.solver.Result`
     - Stochastic master equation
   * - ssesolve()
     - :func:`qutip.solver.Result`
     - Stochastic Schrödinger equation