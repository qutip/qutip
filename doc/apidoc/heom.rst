********************************
Hierarchical Equations of Motion
********************************

HEOM Solvers
------------

.. automodule:: qutip.solver.heom
    :members: heomsolve

.. autoclass:: qutip.solver.heom.HEOMSolver
    :members:

.. autoclass:: qutip.solver.heom.HSolverDL
    :members:

PETSc Backend
~~~~~~~~~~~~~

.. note::
   :class:`PETScHEOMSolver` requires ``petsc4py`` and ``mpi4py`` to be
   installed and scripts to be launched via an MPI runner (e.g.
   ``mpirun``). It supports **time-independent** systems only.
   See :ref:`heom-petsc` for the full usage guide, option reference, and
   worked examples.

.. autoclass:: qutip.solver.heom.backend_petsc.PETScHEOMSolver
    :members:
    :show-inheritance:

.. autoclass:: qutip.solver.heom.HierarchyADOs
    :members:

.. autoclass:: qutip.solver.heom.HierarchyADOsState
    :members:

.. autoclass:: qutip.solver.heom.HEOMResult
    :members:

Baths
-----

.. autoclass:: qutip.solver.heom.BathExponent
    :members:

.. autoclass:: qutip.solver.heom.Bath
    :members:

.. autoclass:: qutip.solver.heom.BosonicBath
    :members:

.. autoclass:: qutip.solver.heom.DrudeLorentzBath
    :members:

.. autoclass:: qutip.solver.heom.DrudeLorentzPadeBath
    :members:

.. autoclass:: qutip.solver.heom.UnderDampedBath
    :members:

.. autoclass:: qutip.solver.heom.FermionicBath
    :members:

.. autoclass:: qutip.solver.heom.LorentzianBath
    :members:

.. autoclass:: qutip.solver.heom.LorentzianPadeBath
    :members:
