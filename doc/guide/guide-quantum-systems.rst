.. _quantum_system_class:

**********************
QuantumSystem Class
**********************

.. _quantum_system_overview:

Overview
========

The :class:`.QuantumSystem` class provides a unified interface for working with
quantum systems in QuTiP. It serves as a container that encapsulates all the
components needed to define and simulate a quantum system, including the
Hamiltonian, operators, collapse operators, and system parameters.

This class acts as the foundation for all quantum systems in the framework,
ensuring consistent interfaces and making it easy to switch between different
types of systems or create custom ones.

.. _quantum_system_structure:

Class Structure
===============

The QuantumSystem class stores the following components:

**Core Components:**

- **Hamiltonian**: The system's time evolution operator (:class:`.Qobj`)
- **Operators**: Dictionary of system operators (creation, annihilation, Pauli, etc.)
- **Collapse operators**: List of operators describing dissipation and decoherence
- **Name**: Human-readable identifier for the system
- **Parameters**: Dictionary of system-specific parameters

**Display and Analysis:**

- **LaTeX representation**: Mathematical description for pretty printing
- **Dimension**: Hilbert space dimensions
- **Eigenvalues/Eigenstates**: Spectral properties of the Hamiltonian

.. _quantum_system_creation:

Creating QuantumSystem Objects
==============================

There are two main ways to create QuantumSystem objects, each suited for different use cases.

**1. Using Factory Functions (Recommended)**

Factory functions provide the most convenient and robust way to create quantum systems. 
These functions handle all the complex details of building Hamiltonians, operators, and 
collapse operators, while providing sensible defaults and parameter validation.

.. jupyter-execute::

    >>> from qutip.quantum_systems import jaynes_cummings, qubit
    >>> 
    >>> # Factory functions provide convenient interfaces
    >>> jc = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1)
    >>> q = qubit(omega=2.0, decay_rate=0.1)
    >>> 
    >>> jc.pretty_print()
    >>> q.pretty_print()

The factory functions automatically construct all necessary operators for the system. 
For example, the Jaynes-Cummings factory creates cavity operators (``a``, ``a_dag``, ``n_c``) 
and atomic operators (``sigma_plus``, ``sigma_minus``, ``sigma_z``), while properly handling their tensor product structure.

**2. Direct Construction**

For custom systems or when you need complete control over every component, 
you can construct QuantumSystem objects directly by manually building all operators and the Hamiltonian.

.. jupyter-execute::

    >>> from qutip.quantum_systems import QuantumSystem
    >>> from qutip import sigmaz, sigmam, sigmax, sigmay
    >>> 
    >>> # Manual creation for custom systems
    >>> H = 0.5 * sigmaz()
    >>> ops = {
    ...     'sigma_z': sigmaz(),
    ...     'sigma_x': sigmax(), 
    ...     'sigma_y': sigmay(),
    ...     'sigma_minus': sigmam()
    ... }
    >>> c_ops = [0.1 * sigmam()]  # Decay
    >>> 
    >>> custom_system = QuantumSystem(
    ...     hamiltonian=H,
    ...     name="Custom Qubit",
    ...     operators=ops,
    ...     c_ops=c_ops,
    ...     latex=r"H = \frac{\omega}{2}\sigma_z",
    ...     omega=1.0,  # Custom parameters
    ...     decay_rate=0.1
    ... )
    >>> 
    >>> custom_system.pretty_print()

This approach is useful when implementing novel quantum systems not covered by the existing factory functions, 
or when you need to understand exactly how each component is constructed.

.. _quantum_system_properties:

Properties and Methods
======================

The QuantumSystem class provides several useful properties for analysis:

**Dimensional Properties:**

.. jupyter-execute::

    >>> from qutip.quantum_systems import jaynes_cummings
    >>> 
    >>> # Create a Jaynes-Cummings system
    >>> jc = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1, n_cavity=5)
    >>> 
    >>> print(f"Hilbert space dimension: {jc.dimension}")

The ``dimension`` property shows the structure of the composite Hilbert space. For the Jaynes-Cummings system, 
this returns ``[5, 2]``, indicating a 5-dimensional cavity space tensored with a 2-dimensional atomic space.

**System Properties:**

.. jupyter-execute::

    >>> # Access key properties
    >>> eigenvals = jc.eigenvalues
    >>> print(f"Ground state energy: {eigenvals[0]:.3f}")
    >>> print(f"Available operators: {list(jc.operators.keys())}")
    >>> print(f"Number of collapse operators: {len(jc.c_ops)}")

These properties give you access to the system's spectral information and available operators. 
The ``operators`` dictionary contains all the operators you need for setting up measurements and calculations, 
while ``eigenvalues`` provides the energy spectrum of the system.

.. _quantum_system_display:

Display and Usage
=================

**Pretty Printing:**

.. jupyter-execute::

    >>> # Display comprehensive system information
    >>> jc.pretty_print()

The ``pretty_print()`` method provides a complete overview of the system, including its name, Hilbert space structure, 
parameters, available operators, number of collapse operators and the mathematical form of the Hamiltonian.

**Using with QuTiP Solvers:**

QuantumSystem objects are designed to work seamlessly with QuTiP's time evolution solvers. 
Here's how to simulate the classic vacuum Rabi oscillations in the Jaynes-Cummings model:

.. jupyter-execute::

    >>> from qutip.quantum_systems import jaynes_cummings
    >>> from qutip import tensor, basis, mesolve
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # Create a simple JC system for studying Rabi oscillations
    >>> jc = jaynes_cummings(
    ...     omega_c=1.0,  # Cavity frequency
    ...     omega_a=1.0,  # Atomic frequency (resonant)
    ...     g=0.1,        # Coupling strength
    ...     n_cavity=5,   # Small Hilbert space for clarity
    ... )
    >>> 
    >>> # Create initial state: atom excited, cavity empty
    >>> n_cavity = 5
    >>> psi0 = tensor(basis(n_cavity, 0), basis(2, 1))  # |0,e⟩
    >>> 
    >>> # Time evolution
    >>> tlist = np.linspace(0, 50, 1000)
    >>> 
    >>> # Define measurement operators
    >>> measure_ops = [
    ...     jc.operators["n_c"],  # Cavity photon number
    ...     jc.operators["sigma_plus"] * jc.operators["sigma_minus"],  # Atomic excitation
    ... ]
    >>> 
    >>> # Solve time evolution
    >>> result = mesolve(jc.hamiltonian, psi0, tlist, [], e_ops=measure_ops)
    >>> 
    >>> n_c = result.expect[0]
    >>> n_a = result.expect[1]
    >>> 
    >>> # Plot Rabi oscillations
    >>> fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    >>> axes.plot(tlist, n_c, 'b-', linewidth=2, label="Cavity photons")
    >>> axes.plot(tlist, n_a, 'r-', linewidth=2, label="Atom excited state")
    >>> axes.legend(loc='upper right')
    >>> axes.set_xlabel("Time")
    >>> axes.set_ylabel("Occupation probability")
    >>> axes.set_title("Vacuum Rabi Oscillations")
    >>> axes.grid(True, alpha=0.3)
    >>> plt.show()

This example demonstrates the key workflow: we use the system's Hamiltonian (``jc.hamiltonian``) directly in the solver, 
access the operators we need for measurements through the ``jc.operators`` dictionary and can also use collapse operator list by using ``jc.c_ops`` to model dissipation. 
The resulting plot shows the characteristic energy exchange between the atom and cavity mode.

.. _quantum_system_time_dependence:

Time-Dependent Parameters
=========================

One of the most powerful features of the ``QuantumSystem`` framework is support for time-dependent parameters through QuTiP's :class:`Coefficient` class. 
This allows you to model realistic experimental scenarios where system parameters vary in time.

**Understanding the Coefficient Class**

The ``Coefficient`` class represents a function of time that can be used in place of any constant parameter. 
Instead of passing a number, you pass a function that describes how the parameter changes over time.

.. jupyter-execute::

    >>> from qutip import coefficient
    >>> 
    >>> # Time-dependent frequency
    >>> def varying_omega(t, args):
    ...     return 0.1 * np.sin(0.5 * t)
    >>> 
    >>> omega_t = coefficient(varying_omega, args={})
    >>> q_td = qubit(omega=omega_t)
    >>> 
    >>> q_td.pretty_print()

The ``coefficient`` function takes two arguments: the time-dependent function and an ``args`` dictionary for passing additional parameters.
The function should accept ``t`` (time) and ``args`` as arguments and return the parameter value at that time.

Time-dependent systems work seamlessly with all QuTiP solvers. The solvers automatically detect time dependence and handle it appropriately during the evolution, 
making it straightforward to model complex experimental protocols like adiabatic parameter sweeps, pulsed driving, or realistic decay processes.

.. plot::
    :context: reset
    :include-source: false
    :nofigs: