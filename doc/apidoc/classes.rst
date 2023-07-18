.. _classes:

***************
Classes
***************

.. _classes-qobj:

Qobj
--------------

.. autoclass:: qutip.core.qobj.Qobj
    :members:

.. _classes-qobjevo:

QobjEvo
--------------

.. autoclass:: qutip.core.cy.qobjevo.QobjEvo
    :members:


.. _classes-bloch:

Bloch sphere
---------------

.. autoclass:: qutip.bloch.Bloch
    :members:

.. autoclass:: qutip.bloch3d.Bloch3d
    :members:

Distributions
-------------

.. autoclass:: qutip.QFunc
    :members:


.. _classes-solver:

Solvers
-------

.. autoclass:: qutip.solver.sesolve.SESolver
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: qutip.solver.mesolve.MESolver
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: qutip.solver.brmesolve.BRSolver
    :members:
    :inherited-members:
    :show-inheritance:


.. autoclass:: qutip.solver.stochastic.SMESolver
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: qutip.solver.stochastic.SSESolver
    :members:
    :inherited-members:
    :show-inheritance:



.. _classes-monte-carlo-solver:

Monte Carlo Solvers
-------------------

.. autoclass:: qutip.solver.mcsolve.MCSolver
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: qutip.solver.nm_mcsolve.NonMarkovianMCSolver
    :members:
    :inherited-members:
    :show-inheritance:


.. _classes-non_markov_heom:

Non-Markovian HEOM Solver
-------------------------

.. autoclass:: qutip.solver.heom.HEOMSolver
    :members:

.. autoclass:: qutip.solver.heom.HSolverDL
    :members:

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

.. autoclass:: qutip.solver.heom.HierarchyADOs
    :members:

.. autoclass:: qutip.solver.heom.HierarchyADOsState
    :members:

.. autoclass:: qutip.solver.heom.HEOMResult
    :members:


.. _classes-ode:

Integrator
----------

.. autoclass:: qutip.solver.integrator.scipy_integrator.IntegratorScipyAdams
    :members: options

.. autoclass:: qutip.solver.integrator.scipy_integrator.IntegratorScipyBDF
    :members: options

.. autoclass:: qutip.solver.integrator.scipy_integrator.IntegratorScipylsoda
    :members: options

.. autoclass:: qutip.solver.integrator.scipy_integrator.IntegratorScipyDop853
    :members: options

.. autoclass:: qutip.solver.integrator.qutip_integrator.IntegratorVern7
    :members: options

.. autoclass:: qutip.solver.integrator.qutip_integrator.IntegratorVern9
    :members: options

.. autoclass:: qutip.solver.integrator.qutip_integrator.IntegratorDiag
    :members: options

.. autoclass:: qutip.solver.integrator.krylov.IntegratorKrylov
    :members: options


.. _classes-sode:

Stochastic Integrator
---------------------

.. autoclass:: qutip.solver.sode.rouchon.RouchonSODE
    :members: options

.. autoclass:: qutip.solver.sode.itotaylor.EulerSODE
    :members: options

.. autoclass:: qutip.solver.sode.itotaylor.Milstein_SODE
    :members: options

.. autoclass:: qutip.solver.sode.itotaylor.Taylor1_5_SODE
    :members: options

.. autoclass:: qutip.solver.sode.itotaylor.Implicit_Milstein_SODE
    :members: options

.. autoclass:: qutip.solver.sode.itotaylor.Implicit_Taylor1_5_SODE
    :members: options

.. autoclass:: qutip.solver.sode.sode.PlatenSODE
    :members: options

.. autoclass:: qutip.solver.sode.itotaylor.Explicit1_5_SODE
    :members: options

.. autoclass:: qutip.solver.sode.sode.PredCorr_SODE
    :members: options


.. _classes-odeoptions:

Solver Options and Results
---------------------------

.. autoclass:: qutip.solver.result.Result
    :members:

.. _classes-piqs:

Permutational Invariance
------------------------

.. autoclass:: qutip.piqs.piqs.Dicke
    :members:

.. autoclass:: qutip.piqs.piqs.Pim
    :members:

.. _classes-distributions:

Distribution functions
----------------------------

.. autoclass:: qutip.distributions.Distribution
    :members:

.. autoclass:: qutip.distributions.WignerDistribution
    :members:

.. autoclass:: qutip.distributions.QDistribution
    :members:

.. autoclass:: qutip.distributions.TwoModeQuadratureCorrelation
    :members:

.. autoclass:: qutip.distributions.HarmonicOscillatorWaveFunction
    :members:

.. autoclass:: qutip.distributions.HarmonicOscillatorProbabilityFunction
    :members:
