.. _solver_terminology:

Terminology
===========

This is a short terminology glossary for the solver and related classes.
See the rest of the guide for more in depth information.

Quantum system
    Set of operators and physical parameters used to build the dynamic equations.
    Common components are the Hamiltonian, collapse or jump operators,
    bath coupling operators and environment/spectral noise configurations.

method
    The identifier string used to select the numerical integration technique for a time-evolution run.
    Can refer to ODE or SDE integration algorithm, depending on the solver.

ODE
    Ordinary Differential Equation.

      $$ \frac{dX}{dt} = f(t, X) $$

SDE
    Stochastic Differential Equation:

      $$ dX = f(t, X) dt + \sum_i g_i(t, X) dW_i $$

    where $dW_i$ represent independent Wiener processes modeling Gaussian white noise.

RHS
    Right-Hand Side
    The representation describing the derivative in a differential equation
    (i.e., the $f(t, X)$ term in an ODE).

Feedback
    The term we use for quantum dynamics where the underlying quantum system parameters
    depend explicitly on the instantaneous state of the system, introducing a form of non-linearity.
    For example, when its Hamiltonian depends on the expectation value of an operator or the state itself: $H = H(t, \rho)$.
    Different :class:`Solver` provide can provide different feedback format.

Deterministic / Non-deterministic
    A categorization based on whether randomness is involved in the time-evolution.

    * Deterministic evolution algorithms (like standard master equations or Schrödinger equations)
      always yield identical results (up to numerical imprecision) across separate
      runs given identical initial conditions.
    * Non-deterministic evolution introduces pseudo-random variables,
      yielding structurally unique trajectories on successive invocations.

Stochastic
    While broadly synonymous with non-deterministic behavior, in this section,
    "stochastic" specifically denotes simulations driven by true Stochastic Differential Equations (SDEs)
    containing Gaussian noise channels (e.g., :class:`SMESolver`).
    Conversely, while Monte Carlo simulations (:class:`MCSolver`) are non-deterministic due to discrete,
    random quantum jumps, their trajectories are piecewise-deterministic and are typically
    excluded when discussing "stochastic evolution" within this architecture framework.

Trajectory
    A single, continuous realization of a non-deterministic quantum evolution process.
    For instance, an individual run of a Monte Carlo simulation contains a unique
    sequence of jump events and timings that defines one distinct trajectory.
