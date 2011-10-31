.. QuTiP 
   Copyright (C) 2011, Paul D. Nation & Robert J. Johansson

QuTiP functions
===============

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
    :members: 
    :undoc-members:

Dynamics and time-evolution
---------------------------

.. automodule:: qutip
    :members: correlation, essolve, mcsolve, odesolve, propagator, steady
    :undoc-members:

Other
-----

.. automodule:: qutip
    :members: about, clebsch, demos, entropy, fileio, gates, graph, metrics, orbital, parfor, rotation, simdiag, sphereplot, wigner
    :undoc-members:

