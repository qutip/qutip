.. _api_qobj:

***************
Quantum Objects
***************

.. _classes-qobj:

Qobj
----

.. autoclass:: qutip.core.qobj.Qobj
    :members:
    :special-members: __call__

.. automodule:: qutip.core.properties
    :members: issuper, isoper, isoperket, isoperbra, isket, isbra, isherm

CoreOptions
-----------

.. autoclass:: qutip.core.options.CoreOptions
    :members:

********************************
Creation of States and Operators
********************************

Quantum States
--------------

.. automodule:: qutip.core.states
    :members: basis, bell_state, bra, coherent, coherent_dm, fock, fock_dm, ghz_state, maximally_mixed_dm, ket, ket2dm, phase_basis, projection, qutrit_basis, singlet_state, spin_state, spin_coherent, state_number_enumerate, state_number_index, state_index_number, state_number_qobj, thermal_dm, triplet_states, w_state, zero_ket


Quantum Operators
-----------------

.. automodule:: qutip.core.operators
    :members: charge, commutator, create, destroy, displace, fcreate, fdestroy, jmat, num, qeye, identity, momentum, phase, position, qdiags, qutrit_ops, qzero, sigmam, sigmap, sigmax, sigmay, sigmaz, spin_Jx, spin_Jy, spin_Jz, spin_Jm, spin_Jp, squeeze, squeezing, tunneling, qeye_like, qzero_like


Quantum Gates
-------------

.. automodule:: qutip.core.gates
    :members: rx, ry, rz, sqrtnot, snot, phasegate, qrot, cy_gate, cz_gate, s_gate, t_gate, cs_gate, ct_gate, cphase, cnot, csign, berkeley, swapalpha, swap, iswap, sqrtswap, sqrtiswap, fredkin, molmer_sorensen, toffoli, hadamard_transform, qubit_clifford_group, globalphase


Energy Restricted Operators
---------------------------

.. automodule:: qutip.core.energy_restricted
    :members: enr_state_dictionaries, enr_thermal_dm, enr_fock, enr_destroy, enr_identity


.. _api-rand:

Random Operators and States
---------------------------

.. automodule:: qutip.random_objects
    :members: rand_dm, rand_herm, rand_ket, rand_stochastic, rand_unitary, rand_super, rand_super_bcsz, rand_kraus_map


********************
Manipulation of Qobj
********************

Tensor
------

.. automodule:: qutip.core.tensor
    :members: tensor, super_tensor, composite, tensor_contract

.. automodule:: qutip.core.qobj
    :members: ptrace

.. automodule:: qutip.partial_transpose
    :members: partial_transpose

Superoperators and Liouvillians
-------------------------------

.. automodule:: qutip.core.superoperator
    :members: operator_to_vector, vector_to_operator, liouvillian, spost, spre, sprepost, lindblad_dissipator

Superoperators and Liouvillians
-------------------------------

.. automodule:: qutip.core.blochredfield
    :members: bloch_redfield_tensor, brterm, brcrossterm

Superoperator Representations
-----------------------------

.. automodule:: qutip.core.superop_reps
    :members: kraus_to_choi, kraus_to_super, to_choi, to_chi, to_super, to_kraus, to_stinespring
    :undoc-members:

Operators and Superoperator Dimensions
--------------------------------------

.. automodule:: qutip.core.dimensions
    :members: to_tensor_rep, from_tensor_rep

Miscellaneous
-------------

.. automodule:: qutip.simdiag
    :members: simdiag


*************************
Extracting data from Qobj
*************************

Expectation Values
------------------

.. automodule:: qutip.core.expect
    :members: expect, variance

Entropy Functions
-----------------

.. automodule:: qutip.entropy
    :members: concurrence, entropy_conditional, entropy_linear, entropy_mutual, entropy_relative, entropy_vn

Density Matrix Metrics
----------------------

.. automodule:: qutip.core.metrics
    :members: fidelity, tracedist, bures_dist, bures_angle, hellinger_dist, hilbert_dist, average_gate_fidelity, process_fidelity, unitarity, dnorm

Measurement of quantum states
-----------------------------

.. automodule:: qutip.measurement
    :members: measure, measure_povm, measure_observable, measurement_statistics, measurement_statistics_observable, measurement_statistics_povm
