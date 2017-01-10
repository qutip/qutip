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
    :members: basis, bell_state, bra, coherent, coherent_dm, enr_state_dictionaries, enr_thermal_dm, enr_fock, fock, fock_dm, ghz_state, maximally_mixed_dm, ket, ket2dm, phase_basis, projection, qutrit_basis, singlet_state, spin_state, spin_coherent, state_number_enumerate, state_number_index, state_index_number, state_number_qobj, thermal_dm, triplet_states. w_state, zero_ket


Quantum Operators
---------------------

.. automodule:: qutip.operators
    :members: charge, commutator, create, destroy, displace, enr_destroy, enr_identity, jmat, num, qeye, identity, momentum, phase, position, qdiags, qutrit_ops, qzero, sigmam, sigmap, sigmax, sigmay, sigmaz, spin_Jx, spin_Jy, spin_Jz, spin_Jm, spin_Jp, squeeze, squeezing, tunneling


.. _functions-rand:

Random Operators and States
---------------------------

.. automodule:: qutip.random_objects
    :members: rand_dm, rand_dm_ginibre, rand_dm_hs, rand_herm, rand_ket, rand_ket_haar, rand_unitary, rand_unitary_haar, rand_super, rand_super_bcsz


Three-Level Atoms
-------------------

.. automodule:: qutip.three_level_atom
    :members: three_level_basis, three_level_ops
    :undoc-members:


Superoperators and Liouvillians
-------------------------------

.. automodule:: qutip.superoperator
    :members: operator_to_vector, vector_to_operator, liouvillian, spost, spre, sprepost, lindblad_dissipator
    
Superoperator Representations
-----------------------------

.. automodule:: qutip.superop_reps
    :members: to_choi, to_super, to_kraus
    :undoc-members:


Functions acting on states and operators
========================================

Tensor
-------

.. automodule:: qutip.tensor
    :members: tensor, super_tensor, composite, tensor_contract


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
    :members: fidelity, tracedist, bures_dist, bures_angle, hilbert_dist, average_gate_fidelity, process_fidelity


Continous Variables
-------------------

.. automodule:: qutip.continuous_variables
    :members: correlation_matrix, covariance_matrix, correlation_matrix_field, correlation_matrix_quadrature, wigner_covariance_matrix, logarithmic_negativity


Dynamics and Time-Evolution
===========================

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

.. ignore f90 stuff for now    
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
    :members: correlation, correlation_ss, correlation_2op_1t, correlation_2op_2t, correlation_3op_1t, correlation_3op_2t, correlation_4op_1t, correlation_4op_2t, spectrum, spectrum_ss, spectrum_pi, spectrum_correlation_fft, coherence_function_g1, coherence_function_g2


Steady-state Solvers
--------------------

.. automodule:: qutip.steadystate
    :members: steadystate, build_preconditioner
    :undoc-members:

Propagators
-----------

.. automodule:: qutip.propagator
    :members: propagator, propagator_steadystate
    :undoc-members:


Time-dependent problems
-----------------------

.. automodule:: qutip.rhs_generate
    :members: rhs_generate, rhs_clear

Visualization
===============

	
Pseudoprobability Functions
----------------------------

.. automodule:: qutip.wigner
    :members: qfunc, spin_q_function, spin_wigner, wigner


Graphs and Visualization
------------------------

.. automodule:: qutip.visualization
    :members: hinton, matrix_histogram, matrix_histogram_complex, plot_energy_levels, wigner_cmap, plot_fock_distribution, plot_wigner_fock_distribution, plot_wigner, sphereplot, plot_schmidt, plot_qubism, plot_expectation_values, plot_spin_distribution_2d, plot_spin_distribution_3d
    :undoc-members:

.. automodule:: qutip.orbital
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

.. _functions-non_markov:

non-Markovian Solvers
=====================

.. automodule:: qutip.nonmarkov.transfertensor
    :members: ttmsolve

.. _functions-control:

Optimal control
===============

.. automodule:: qutip.control.pulseoptim
    :members: optimize_pulse, optimize_pulse_unitary, create_pulse_optimizer, opt_pulse_crab, opt_pulse_crab_unitary

.. automodule:: qutip.control.pulsegen
    :members: create_pulse_gen

Utilitiy Functions
==================

.. _functions-graph:

Graph Theory Routines
----------------------

.. automodule:: qutip.graph
    :members: breadth_first_search, graph_degree, reverse_cuthill_mckee, maximum_bipartite_matching, weighted_bipartite_matching


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


.. _functions-parallel:

Parallelization
---------------

.. automodule:: qutip.parallel
    :members: parfor, parallel_map, serial_map


.. _functions-ipython:

IPython Notebook Tools
----------------------

.. automodule:: qutip.ipynbtools
    :members: parfor, parallel_map, version_table

.. _functions-misc:

Miscellaneous
--------------

.. automodule:: qutip
    :members: about, simdiag
