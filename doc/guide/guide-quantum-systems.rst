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

There are two main ways to create QuantumSystem objects:

**1. Using Factory Functions (Recommended)**

.. jupyter-execute::

    >>> from qutip.quantum_systems import jaynes_cummings, qubit
    >>> 
    >>> # Factory functions provide convenient interfaces
    >>> jc = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1)
    >>> q = qubit(omega=2.0, decay_rate=0.1)
    >>> 
    >>> jc.pretty_print()
    >>> q.pretty_print()

**2. Direct Construction**

.. plot::
    :context: close-figs

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

.. _quantum_system_properties:

Properties and Methods
======================

The QuantumSystem class provides several useful properties for analysis:

**Dimensional Properties:**

.. plot::
    :context: close-figs

    >>> from qutip.quantum_systems import jaynes_cummings
    >>> 
    >>> # Create a Jaynes-Cummings system
    >>> jc = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1, n_cavity=5)
    >>> 
    >>> print(f"Hilbert space dimension: {jc.dimension}")

**System Properties:**

.. plot::
    :context: close-figs

    >>> # Access key properties
    >>> eigenvals = jc.eigenvalues
    >>> print(f"Ground state energy: {eigenvals[0]:.3f}")
    >>> print(f"Available operators: {list(jc.operators.keys())}")
    >>> print(f"Number of collapse operators: {len(jc.c_ops)}")

.. _quantum_system_display:

Display and Usage
=================

**Pretty Printing:**

.. plot::
    :context: close-figs

    >>> # Display comprehensive system information
    >>> jc.pretty_print()

**Using with QuTiP Solvers:**

.. plot::
    :context: close-figs

    >>> from qutip.quantum_systems import jaynes_cummings
    >>> from qutip import tensor,basis
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

.. _quantum_system_time_dependence:

Time-Dependent Parameters
=========================

.. plot::
    :context: close-figs

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

.. plot::
    :context: reset
    :include-source: false
    :nofigs: