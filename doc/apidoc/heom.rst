************
Environments
************

Bosonic Environments
--------------------

.. autoclass:: qutip.core.BosonicEnvironment
    :members:

.. autoclass:: qutip.core.DrudeLorentzEnvironment
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: from_correlation_function, from_power_spectrum, from_spectral_density

.. autoclass:: qutip.core.UnderDampedEnvironment
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: from_correlation_function, from_power_spectrum, from_spectral_density

.. autoclass:: qutip.core.OhmicEnvironment
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: from_correlation_function, from_power_spectrum, from_spectral_density

.. autoclass:: qutip.core.CFExponent
    :members:

.. autoclass:: qutip.core.ExponentialBosonicEnvironment
    :members:
    :show-inheritance:

.. autofunction:: qutip.core.environment.system_terminator


Fermionic Environments
----------------------

.. autoclass:: qutip.core.FermionicEnvironment
    :members:
    :exclude-members: from_correlation_function, from_power_spectrum, from_spectral_density

.. autoclass:: qutip.core.LorentzianEnvironment
    :members:
    :show-inheritance:

.. autoclass:: qutip.core.ExponentialFermionicEnvironment
    :members:
    :show-inheritance:



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
