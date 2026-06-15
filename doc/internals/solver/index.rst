.. _solver_internals:

Quantum Solvers and Integrators Architecture
############################################

The solver framework in QuTiP is responsible for simulating the time-evolution of quantum systems.
Built directly on top of the core data layer,
the architecture enforces a separation of concerns between physical model and numerical solvers.

This document serves as an overview of the internal architecture of QuTiP's solvers,
guiding core developers and contributors on how the system is structured,
how data flows through it, and how to extend it.


Core Architecture and Separation of Concerns
============================================

The framework decouples physical models from their numerical integration backends, dividing
tasks between two primary base classes:

1. **The Physics Layer (:class:`Solver`)**: This layer represents the physical equation
   governing the system's dynamics. Each distinct quantum equation—such as the Schrödinger
   equation, Lindblad Master Equation, or Bloch-Redfield equations—maps to a unique
   subclass inheriting from the abstract base :class:`Solver` class (e.g., :class:`SESolver`,
   :class:`MESolver`, :class:`BRSolver`). The solver class is responsible for setting up
   the system operators and defining the physical derivative.

2. **The Numerical Layer (:class:`Integrator`)**: This layer handles the low-level
   numerical implementation of Ordinary Differential Equations (ODEs) or Stochastic
   Differential Equations (SDEs). The solver communicates the physical state and derivative
   rules down to the :class:`Integrator`. The selection of which numerical scheme to run
   is entirely dictated by the solver's configuration options through the ``"method"`` parameter.

Usually, integrators are general-purpose ODE solvers. However, to maximize performance,
certain integrators are specialized to specific physical representations,
where the boundary between physical assumptions and numerical routines is intentionally blended.


User-Facing Interfaces vs. Internal Infrastructure
==================================================

To maintain an intuitive user experience, QuTiP hides internal architectural complexity
behind high-level functional wrappers:

- Functions such as :func:`sesolve` or :func:`mesolve` serve as convenience entry points for the end user.
  In a single call, these helper functions instantiate the appropriate underlying :class:`Solver` class,
  run the evolution, and return a completed results container.
- Multiple functional entry points can map back to the same physical equations.
  For instance, :func:`sesolve`, :func:`fsesolve`, and :func:`krylovsolve`
  all represent closed-system Schrödinger dynamics, using :class:`SESolver` but different integrators.
- End users typically interact exclusively with top-level functions or the core :class:`Solver` class interface.
  The numerical :class:`Integrator` and the configuration parsing via :class:`SolverOptions` operate silently under the hood.
  Once execution concludes, the resulting :class:`Result` instance is returned
  to the user and expected to be used as a read-only data container.


.. toctree::
   :maxdepth: 2
   :caption: Sections

   terminology
   integrator

..
  Sections to add later:
  motivation
  solver
  integrator_stochastic
  result
  solveroptions
  feedback
