.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

.. _functions:

QuTiP Functions
================

.. toctree::
   :maxdepth: 2

Manipulation and Creation of States and Operators
----------------------------------------------------

Quantum States
****************

.. automodule:: qutip.states
    :members: basis, qutrit_basis, coherent, coherent_dm, fock, fock_dm, thermal_dm, ket2dm

.. automodule:: qutip.qstate
    :members: qstate 

Quantum Operators
********************

.. automodule:: qutip.operators
    :members: jmat, sigmap, sigmam, sigmax, sigmay, sigmaz, destroy, create, qeye, num, squeez, displace, qutrit_ops

Qobj Check Functions
***********************

.. automodule:: qutip.istests
    :members: isket, isbra, isoper, issuper, isequal, isherm

Liouvillian
************

.. automodule:: qutip.superoperator
    :members: liouvillian, spost, spre

Tensor
**********

.. automodule:: qutip.tensor
    :members: tensor

Partial Trace
**************
.. automodule:: qutip.ptrace
    :members: ptrace

Expectation Values
********************
.. automodule:: qutip.expect
    :members: expect
	
Pseudoprobability Functions
****************************
.. automodule:: qutip.wigner
    :members: qfunc, wigner


Three-level atoms
*******************

.. automodule:: qutip.three_level_atom
    :members: three_level_basis, three_level_ops
    :undoc-members:

Dynamics and time-evolution
----------------------------

Master Equation
******************

.. automodule:: qutip
    :members: mesolve

Monte Carlo Evolution
************************

.. automodule:: qutip
    :members: mcsolve

Correlation Functions
************************

.. automodule:: qutip.correlation
    :members: correlation, correlation_ss

Exponential Series
*********************

.. automodule:: qutip.essolve
    :members: essolve, ode2es

Steady-state Solvers
**********************

.. automodule:: qutip.steady
    :members: steady, steadystate
    :undoc-members:

Other Functions
------------------

Entropy Functions
*******************

.. automodule:: qutip.entropy
    :members: entropy_linear, entropy_vn

Quantum Computing Gates
************************

.. automodule:: qutip.gates
    :members: cnot, fredkin, phasegate, snot, swap, toffoli 

Density Matrix Metrics
***********************

.. automodule:: qutip.metrics
    :members: fidelity, tracedist

Random Operators and States
****************************
.. automodule:: qutip.rand
    :members: rand_dm, rand_herm, rand_ket, rand_unitary


Miscellaneous
**************

.. automodule:: qutip
    :members: about, clebsch, demos, hinton, orbital, parfor, simdiag, sphereplot
    :undoc-members:

