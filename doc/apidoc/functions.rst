.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _functions:

********************
QuTiP Function List
********************

Manipulation and Creation of States and Operators
=================================================

Quantum States
----------------

.. automodule:: qutip.states
    :members: basis, qutrit_basis, coherent, coherent_dm, fock, fock_dm, thermal_dm, ket2dm

.. automodule:: qutip.qstate
    :members: qstate 

Quantum Operators
---------------------

.. automodule:: qutip.operators
    :members: jmat, sigmap, sigmam, sigmax, sigmay, sigmaz, destroy, create, qeye, num, squeez, displace, qutrit_ops

Qobj Check Functions
--------------------

.. automodule:: qutip.istests
    :members: isket, isbra, isoper, issuper, isequal, isherm

Liouvillian
-----------

.. automodule:: qutip.superoperator
    :members: liouvillian, spost, spre

Tensor
-------

.. automodule:: qutip.tensor
    :members: tensor


Expectation Values
--------------------

.. automodule:: qutip.expect
    :members: expect
	
Pseudoprobability Functions
----------------------------

.. automodule:: qutip.wigner
    :members: qfunc, wigner


Three-level atoms
-------------------

.. automodule:: qutip.three_level_atom
    :members: three_level_basis, three_level_ops
    :undoc-members:

Dynamics and time-evolution
=============================

Master Equation
-----------------

.. automodule:: qutip.mesolve
    :members: mesolve, odesolve

Monte Carlo Evolution
-----------------------

.. automodule:: qutip.mcsolve
    :members: mcsolve

Bloch-Redfield Master Equation
------------------------------

.. automodule:: qutip.bloch_redfield
    :members: brmesolve


Floquet States and Floquet-Markov Master Equation
-------------------------------------------------

.. automodule:: qutip.floquet
    :members: fmmesolve, floquet_modes, floquet_modes_t, floquet_modes_table, floquet_modes_t_lookup, floquet_states_t, floquet_wavefunction_t, floquet_state_decomposition

Correlation Functions
-----------------------

.. automodule:: qutip.correlation
    :members: correlation, correlation_ss, spectrum_ss

Exponential Series
------------------

.. automodule:: qutip.essolve
    :members: essolve, ode2es

Steady-state Solvers
--------------------

.. automodule:: qutip.steady
    :members: steady, steadystate
    :undoc-members:

Other Functions
===============

.. _functions-entropy:

Entropy Functions
-----------------

.. automodule:: qutip.entropy
    :members: entropy_linear, entropy_vn

.. _functions-gates:

Quantum Computing Gates
-----------------------

.. automodule:: qutip.gates
    :members: cnot, fredkin, phasegate, snot, swap, toffoli 

.. _functions-metrics:

Density Matrix Metrics
----------------------

.. automodule:: qutip.metrics
    :members: fidelity, tracedist


.. _functions-rand:

Random Operators and States
---------------------------

.. automodule:: qutip.rand
    :members: rand_dm, rand_herm, rand_ket, rand_unitary

.. _functions-misc:

Miscellaneous
--------------

.. automodule:: qutip
    :members: about, clebsch, demos, hinton, orbital, parfor, simdiag, sphereplot
    :undoc-members:

