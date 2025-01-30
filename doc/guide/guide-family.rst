QuTiP Family Packages
=====================

QuTiP has several companion packages that extend its functionality and provide specialized tools for quantum computing and control.

qutip-qip: Quantum Information Processing
-----------------------------------------

The `qutip-qip <https://qutip-qip.readthedocs.io/en/stable/>`_ package aims to provide fundamental tools for quantum computing simulation. It offers:

- Two approaches for simulating quantum circuits:

  * ``QubitCircuit``: Calculates unitary evolution under quantum gates using matrix product
  * ``Processor``: Uses open system solvers to simulate noisy quantum devices

- Strong emphasis on the physics layer and seamless integration with QuTiP
- Supports both simple quantum algorithm design and experimental realization

qutip-qoc: Quantum Optimal Control
-----------------------------------

The `qutip-qoc <https://qutip-qoc.readthedocs.io/latest/>`_ package is an advanced quantum control optimization toolkit that:

- Builds upon the deprecated qutip-qtrl package
- Introduces two new control optimization algorithms:

  * Extended Gradient Optimization of Analytic conTrols (GOAT)
  * JOPT: Utilizing QuTiP 5's diffrax for automatic differentiation

- Supports multiple control algorithms:

  * GOAT
  * JOPT
  * GRAPE
  * CRAB

- Combines global and local optimization methods
- Provides physics-focused tools for quantum device control

qutip-jax: JAX Integration
--------------------------

The `qutip-jax <https://qutip-jax.readthedocs.io/en/latest/>`_ package integrates QuTiP with JAX, offering:

- Efficient automatic differentiation for quantum systems
- Hardware acceleration capabilities
- Optimization of quantum operations and control functions
- Seamless combination of QuTiP's physics-driven framework with JAX's computational efficiency

QuTiP Ecosystem
---------------

The QuTiP family of packages represents a modular approach to quantum simulation and control. By breaking functionality into specialized packages, QuTiP:

- Enables more focused development
- Allows users to install only the components they need
- Facilitates easier maintenance and updates
- Provides flexible tools for diverse quantum computing research
- Supports extensibility through domain-specific modules

These packages collectively aim to provide a comprehensive, physics-driven toolkit for quantum information processing, optimal control, and computational simulation.