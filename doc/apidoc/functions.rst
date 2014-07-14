.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _functions:

***************
Functions
***************

Manipulation and Creation of States and Operators
=================================================

Quantum States
----------------

.. automodule:: qutip.states
    :members: basis, coherent, coherent_dm, fock, fock_dm, ket2dm, qutrit_basis, thermal_dm, state_number_enumerate, state_number_index, state_index_number, state_number_qobj, phase_basis


Quantum Operators
---------------------

.. automodule:: qutip.operators
    :members: create, destroy, displace, jmat, num, qeye, identity, qutrit_ops, sigmam, sigmap, sigmax, sigmay, sigmaz, squeeze, squeezing, phase


.. _functions-rand:

Random Operators and States
---------------------------

.. automodule:: qutip.random_objects
    :members: rand_dm, rand_herm, rand_ket, rand_unitary


Three-Level Atoms
-------------------

.. automodule:: qutip.three_level_atom
    :members: three_level_basis, three_level_ops
    :undoc-members:


Superoperators and Liouvillians
-------------------------------

.. automodule:: qutip.superoperator
    :members: operator_to_vector, vector_to_operator, liouvillian, spost, spre, lindblad_dissipator
    
Superoperator Representations
-----------------------------

.. automodule:: qutip.superop_reps
    :members: to_choi, to_super, to_kraus
    :undoc-members:


Functions acting on states and operators
=================================================

Tensor
-------

.. automodule:: qutip.tensor
    :members: tensor


Expectation Values
--------------------

.. automodule:: qutip.expect
    :members: expect, variance

Partial Transpose
-----------------

.. automodule:: qutip.partial_transpose
    :members: partial_transpose


.. _functions-entropy:

Entropy Functions
-----------------

.. automodule:: qutip.entropy
    :members: concurrence, entropy_conditional, entropy_linear, entropy_mutual, entropy_vn


Density Matrix Metrics
----------------------

.. automodule:: qutip.metrics
    :members: fidelity, tracedist


Continous Variables
-------------------

.. automodule:: qutip.continuous_variables
    :members: correlation_matrix, covariance_matrix, correlation_matrix_field, correlation_matrix_quadrature, wigner_covariance_matrix, logarithmic_negativity


Dynamics and Time-Evolution
=============================

Schrödinger Equation
--------------------

.. automodule:: qutip.sesolve
    :members: sesolve

Master Equation
-----------------

.. automodule:: qutip.mesolve
    :members: mesolve

Monte Carlo Evolution
-----------------------

.. automodule:: qutip.mcsolve
    :members: mcsolve
    
.. automodule:: qutip.fortran.mcsolve_f90
    :members: mcsolve_f90


Exponential Series
------------------

.. automodule:: qutip.essolve
    :members: essolve, ode2es


Bloch-Redfield Master Equation
------------------------------

.. automodule:: qutip.bloch_redfield
    :members: brmesolve, bloch_redfield_tensor, bloch_redfield_solve


Floquet States and Floquet-Markov Master Equation
-------------------------------------------------

.. automodule:: qutip.floquet
    :members: fmmesolve, floquet_modes, floquet_modes_t, floquet_modes_table, floquet_modes_t_lookup, floquet_states_t, floquet_wavefunction_t, floquet_state_decomposition, fsesolve


Stochastic Schrödinger Equation and Master Equation
---------------------------------------------------

.. automodule:: qutip.stochastic
    :members: smesolve, ssesolve, smepdpsolve, ssepdpsolve


Correlation Functions
-----------------------

.. automodule:: qutip.correlation
    :members: correlation, correlation_ss, correlation_2op_1t, correlation_2op_2t, correlation_4op_1t, correlation_4op_2t, spectrum_ss, spectrum_pi, spectrum_correlation_fft, coherence_function_g1, coherence_function_g2


Steady-state Solvers
--------------------

.. automodule:: qutip.steadystate
    :members: steadystate
    :undoc-members:

Propagators
-----------

.. automodule:: qutip.propagator
    :members: propagator, propagator_steadystate
    :undoc-members:


Time-dependent problems
-----------------------

.. automodule:: qutip
    :members: rhs_generate, rhs_clear

Visualization
===============

	
Pseudoprobability Functions
----------------------------

.. automodule:: qutip.wigner
    :members: qfunc, wigner


Graphs and Visualization
------------------------

.. automodule:: qutip.visualization
    :members: hinton, matrix_histogram, matrix_histogram_complex, plot_energy_levels, wigner_cmap, plot_fock_distribution, plot_wigner_fock_distribution, plot_wigner, sphereplot, plot_schmidt, plot_qubism, plot_expectation_values
    :undoc-members:

.. automodule:: qutip
    :members: orbital


Quantum Process Tomography
--------------------------

.. automodule:: qutip.tomography
    :members: qpt, qpt_plot, qpt_plot_combined
    :undoc-members:



.. _functions-qip:

Quantum Information Processing
==============================

Gates
-----

.. automodule:: qutip.qip.gates
    :members: rx, ry, rz, sqrtnot, snot, phasegate, cphase, cnot, csign, berkeley, swapalpha, swap, iswap, sqrtswap, sqrtiswap, fredkin, toffoli, rotation, controlled_gate, globalphase, hadamard_transform, gate_sequence_product, gate_expand_1toN, gate_expand_2toN, gate_expand_3toN

Qubits
------

.. automodule:: qutip.qip.qubits
    :members: qubit_states

Algorithms
----------

.. automodule:: qutip.qip.algorithms.qft
    :members: qft, qft_steps, qft_gate_sequence

.. _functions-metrics:



Utilitiy Functions
==================


Graph Theory Routines
----------------------

.. automodule:: qutip.graph
    :members: breadth_first_search, graph_degree, symrcm


.. _functions-utilities:

Utility Functions
------------------

.. automodule:: qutip.utilities
    :members: n_thermal, linspace_with, clebsch, convert_unit


.. _functions-fileio:

File I/O Functions
------------------

.. automodule:: qutip.fileio
    :members: file_data_read, file_data_store, qload, qsave

.. _functions-misc:

IPython Notebook Tools
----------------------

.. automodule:: qutip.ipynbtools
    :members: parfor, version_table

Miscellaneous
--------------

.. automodule:: qutip
    :members: parfor, about, simdiag
