.. _functions:

***************
Functions
***************

Manipulation and Creation of States and Operators
=================================================

Quantum States
--------------

.. automodule:: qutip.core.states
    :members: basis, bell_state, bra, coherent, coherent_dm, enr_state_dictionaries, enr_thermal_dm, enr_fock, fock, fock_dm, ghz_state, maximally_mixed_dm, ket, ket2dm, phase_basis, projection, qutrit_basis, singlet_state, spin_state, spin_coherent, state_number_enumerate, state_number_index, state_index_number, state_number_qobj, thermal_dm, triplet_states, w_state, zero_ket


Quantum Operators
-----------------

.. automodule:: qutip.core.operators
    :members: charge, commutator, create, destroy, displace, enr_destroy, enr_identity, jmat, num, qeye, identity, momentum, phase, position, qdiags, qutrit_ops, qzero, sigmam, sigmap, sigmax, sigmay, sigmaz, spin_Jx, spin_Jy, spin_Jz, spin_Jm, spin_Jp, squeeze, squeezing, tunneling


.. _functions-rand:

Quantum Objects
---------------

.. automodule:: qutip.core.qobj
    :members: ptrace, dag, isequal, issuper, isoper, isoperket, isoperbra, isket, isbra, isherm, shape, dims


Random Operators and States
---------------------------

.. automodule:: qutip.random_objects
    :members: rand_dm, rand_dm_ginibre, rand_dm_hs, rand_herm, rand_ket, rand_ket_haar, rand_stochastic, rand_unitary, rand_unitary_haar, rand_super, rand_super_bcsz


Three-Level Atoms
-----------------

.. automodule:: qutip.three_level_atom
    :members: three_level_basis, three_level_ops
    :undoc-members:


Superoperators and Liouvillians
-------------------------------

.. automodule:: qutip.core.superoperator
    :members: operator_to_vector, vector_to_operator, liouvillian, spost, spre, sprepost, lindblad_dissipator

Superoperator Representations
-----------------------------

.. automodule:: qutip.core.superop_reps
    :members: super_to_choi, choi_to_super, choi_to_kraus, kraus_to_choi, kraus_to_super, choi_to_chi, chi_to_choi, to_choi, to_chi, to_super, to_kraus, to_stinespring
    :undoc-members:

Operators and Superoperator Dimensions
--------------------------------------

.. automodule:: qutip.core.dimensions
    :members: is_scalar, is_vector, is_vectorized_oper, type_from_dims, flatten, deep_remove, unflatten, collapse_dims_oper, collapse_dims_super, enumerate_flat, deep_map, dims_to_tensor_perm, dims_to_tensor_shape, dims_idxs_to_tensor_idxs


Functions acting on states and operators
========================================

Expectation Values
------------------

.. automodule:: qutip.core.expect
    :members: expect, variance


Tensor
------

.. automodule:: qutip.core.tensor
    :members: tensor, super_tensor, composite, tensor_contract



Partial Transpose
-----------------

.. automodule:: qutip.partial_transpose
    :members: partial_transpose


.. _functions-entropy:

Entropy Functions
-----------------

.. automodule:: qutip.entropy
    :members: concurrence, entropy_conditional, entropy_linear, entropy_mutual, entropy_relative, entropy_vn


Density Matrix Metrics
----------------------

.. automodule:: qutip.core.metrics
    :members: fidelity, tracedist, bures_dist, bures_angle, hilbert_dist, average_gate_fidelity, process_fidelity


Continuous Variables
--------------------

.. automodule:: qutip.continuous_variables
    :members: correlation_matrix, covariance_matrix, correlation_matrix_field, correlation_matrix_quadrature, wigner_covariance_matrix, logarithmic_negativity


Measurement
===========

Measurement of quantum states
-----------------------------

.. automodule:: qutip.measurement
    :members: measure, measure_observable, measurement_statistics, measurement_statistics_observable


Dynamics and Time-Evolution
===========================

Schrödinger Equation
--------------------

.. automodule:: qutip.solver.sesolve
    :members: sesolve

Master Equation
---------------

.. automodule:: qutip.solver.mesolve
    :members: mesolve

Monte Carlo Evolution
---------------------

.. automodule:: qutip.solve.mcsolve
    :members: mcsolve

.. ignore f90 stuff for now
    .. automodule:: qutip.fortran.mcsolve_f90
        :members: mcsolve_f90


Krylov Subspace Solver
----------------------

.. automodule:: qutip.krylovsolve
    :members: krylovsolve


Bloch-Redfield Master Equation
------------------------------

.. automodule:: qutip.solver.brmesolve
    :members: brmesolve


Floquet States and Floquet-Markov Master Equation
-------------------------------------------------

.. automodule:: qutip.solve.floquet
    :members: fmmesolve, floquet_modes, floquet_modes_t, floquet_modes_table, floquet_modes_t_lookup, floquet_states, floquet_states_t, floquet_wavefunction, floquet_wavefunction_t, floquet_state_decomposition, fsesolve, floquet_master_equation_rates, floquet_master_equation_steadystate, floquet_basis_transform, floquet_markov_mesolve


Stochastic Schrödinger Equation and Master Equation
---------------------------------------------------

.. automodule:: qutip.solve.stochastic
    :members: ssesolve, photocurrent_sesolve, smepdpsolve, smesolve, photocurrent_mesolve, ssepdpsolve, stochastic_solvers, general_stochastic


Correlation Functions
---------------------

.. automodule:: qutip.solve.correlation
    :members: correlation, correlation_ss, correlation_2op_1t, correlation_2op_2t, correlation_3op_1t, correlation_3op_2t, correlation_4op_1t, correlation_4op_2t, spectrum, spectrum_ss, spectrum_pi, spectrum_correlation_fft, coherence_function_g1, coherence_function_g2


Steady-state Solvers
--------------------

.. automodule:: qutip.solve.steadystate
    :members: steadystate, build_preconditioner
    :undoc-members:

Propagators
-----------

.. automodule:: qutip.solve.propagator
    :members: propagator, propagator_steadystate
    :undoc-members:

Scattering in Quantum Optical Systems
-------------------------------------

.. automodule:: qutip.solve.scattering
    :members: temporal_basis_vector, temporal_scattered_state, scattering_probability
    :undoc-members:

Permutational Invariance
------------------------

.. automodule:: qutip.solve.piqs
    :members: num_dicke_states, num_dicke_ladders, num_tls, isdiagonal, dicke_blocks, dicke_blocks_full, dicke_function_trace, purity_dicke, entropy_vn_dicke, state_degeneracy, m_degeneracy, energy_degeneracy, ap, am, spin_algebra, jspin, collapse_uncoupled, dicke_basis, dicke, excited, superradiant, css, ghz, ground, identity_uncoupled, block_matrix, tau_column,


Visualization
===============

Pseudoprobability Functions
---------------------------

.. automodule:: qutip.wigner
    :members: qfunc, spin_q_function, spin_wigner, wigner


Graphs and Visualization
------------------------

.. automodule:: qutip.visualization
    :members: hinton, matrix_histogram, matrix_histogram_complex, plot_energy_levels, plot_fock_distribution, plot_wigner_fock_distribution, plot_wigner, sphereplot, plot_schmidt, plot_qubism, plot_expectation_values, plot_spin_distribution_2d, plot_spin_distribution_3d, plot_wigner_sphere
    :undoc-members:

.. automodule:: qutip.orbital
    :members: orbital

.. automodule:: qutip.matplotlib_utilities
   :members: wigner_cmap, complex_phase_cmap


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

.. automodule:: qutip.qip.operations.gates
    :members: rx, ry, rz, sqrtnot, snot, phasegate, cphase, cnot, csign, berkeley, swapalpha, swap, iswap, sqrtswap, sqrtiswap, fredkin, toffoli, rotation, controlled_gate, globalphase, hadamard_transform, gate_sequence_product, gate_expand_1toN, gate_expand_2toN, gate_expand_3toN, expand_operator

Qubits
------

.. automodule:: qutip.qip.qubits
    :members: qubit_states

Algorithms
----------

.. automodule:: qutip.qip.algorithms.qft
    :members: qft, qft_steps, qft_gate_sequence


Circuit
-------

.. automodule:: qutip.qip.qasm
    :members: read_qasm, save_qasm, print_qasm, circuit_to_qasm_str

.. _functions-non_markov:

Non-Markovian Solvers
=====================

.. automodule:: qutip.solve.nonmarkov.transfertensor
    :members: ttmsolve

.. _functions-control:

Optimal control
===============

.. automodule:: qutip.control.pulseoptim
    :members: optimize_pulse, optimize_pulse_unitary, create_pulse_optimizer, opt_pulse_crab, opt_pulse_crab_unitary

.. automodule:: qutip.control.pulsegen
    :members: create_pulse_gen

Utility Functions
=================


.. _functions-utilities:

Utility Functions
-----------------

.. automodule:: qutip.utilities
    :members: n_thermal, clebsch, convert_unit


.. _functions-fileio:

File I/O Functions
------------------

.. automodule:: qutip.fileio
    :members: file_data_read, file_data_store, qload, qsave


.. _functions-parallel:

Parallelization
---------------

.. automodule:: qutip.solver.parallel
    :members: parallel_map, serial_map


.. _functions-ipython:

Semidefinite Programming
------------------------

.. Was this removed
    .. automodule:: qutip.semidefinite
        :members: complex_var, herm, pos_noherm, pos, dens, kron, conj, bmat, bmat, memoize, qudit_swap, dnorm_problem


.. _functions-semidefinite:

IPython Notebook Tools
----------------------

.. automodule:: qutip.ipynbtools
    :members: parfor, parallel_map, version_table

.. _functions-misc:

Miscellaneous
-------------

.. automodule:: qutip
    :members: about, simdiag
