.. QuTiP 
   Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson

QuTiP functions
==================

.. toctree::
   :maxdepth: 2

States and operators
--------------------

.. automodule:: qutip.states
    :members: basis, qutrit_basis, coherent, coherent_dm, fock, fock_dm, thermal_dm, ket2dm

.. automodule:: qutip.qstate
    :members: qstate 

.. automodule:: qutip.operators
    :members: jmat, sigmap, sigmam, sigmax, sigmay, sigmaz, destroy, create, qeye, num, squeez, displace, qutrit_ops

.. automodule:: qutip.istests
    :members: isket, isbra, isoper, issuper, isequal, isherm

.. automodule:: qutip.superoperator
    :members: liouvillian, spost, spre

.. automodule:: qutip.tensor
    :members: tensor

.. automodule:: qutip.ptrace
    :members: ptrace

.. automodule:: qutip.expect
    :members: expect


Three-level atoms
+++++++++++++++++

.. automodule:: qutip.three_level_atom
    :members: three_level_basis, three_level_ops
    :undoc-members:

Dynamics and time-evolution
---------------------------
.. automodule:: qutip
    :members: mcsolve, odesolve
    
.. automodule:: qutip.correlation
    :members: correlation_es, correlation_ode, correlation_mc, correlation_ss_es, correlation_ss_ode, correlation_ss_mc, spectrum_ss

.. automodule:: qutip.essolve
    :members: essolve, ode2es

.. automodule:: qutip.propagator
    :members: propagator, propagator_steadystate

.. automodule:: qutip.steady
    :members: steady, steadystate
    :undoc-members:

Other
-----

.. automodule:: qutip.entropy
    :members: entropy_linear, entropy_vn

.. automodule:: qutip.gates
    :members: cnot, fredkin, phasegate, snot, swap, toffoli 

.. automodule:: qutip.metrics
    :members: fidelity, tracedist

.. automodule:: qutip.wigner
    :members: qfunc, wigner


.. automodule:: qutip
    :members: about, clebsch, demos, hinton, orbital, parfor, qfunc, simdiag, sphereplot
    :undoc-members:

