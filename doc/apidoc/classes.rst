.. _classes:

***************
Classes
***************

.. _classes-qobj:

Qobj
----

.. autoclass:: qutip.core.qobj.Qobj
    :members:
    :special-members: __call__

.. _classes-qobjevo:

QobjEvo
-------

.. autoclass:: qutip.core.cy.qobjevo.QobjEvo
    :members:
    :special-members: __call__

.. _classes-coreoptions:

CoreOptions
-----------

.. autoclass:: qutip.core.options.CoreOptions
    :members:


.. _classes-bloch:

Bloch sphere
------------

.. autoclass:: qutip.bloch.Bloch
    :members:

Distributions
-------------

.. autoclass:: qutip.QFunc
    :members:

Environments
------------

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


.. _classes-solver:

Solvers
-------

.. autoclass:: qutip.solver.sesolve.SESolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.mesolve.MESolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.brmesolve.BRSolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.floquet.FMESolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.floquet.FloquetBasis
    :members:

.. autoclass:: qutip.solver.propagator.Propagator
    :members:
    :inherited-members:
    :special-members: __call__


.. _classes-monte-carlo-solver:

Monte Carlo Solvers
-------------------

.. autoclass:: qutip.solver.mcsolve.MCSolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.nm_mcsolve.NonMarkovianMCSolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator


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
    :show-inheritance:
    :members:

.. autoclass:: qutip.solver.heom.DrudeLorentzPadeBath
    :show-inheritance:
    :members:

.. autoclass:: qutip.solver.heom.UnderDampedBath
    :show-inheritance:
    :members:

.. autoclass:: qutip.solver.heom.FermionicBath
    :members:

.. autoclass:: qutip.solver.heom.LorentzianBath
    :show-inheritance:
    :members:

.. autoclass:: qutip.solver.heom.LorentzianPadeBath
    :show-inheritance:
    :members:

.. autoclass:: qutip.solver.heom.HierarchyADOs
    :members:

.. autoclass:: qutip.solver.heom.HierarchyADOsState
    :members:

.. autoclass:: qutip.solver.heom.HEOMResult
    :members:


.. _classes-stochastic:

Stochastic Solver
-----------------

.. autoclass:: qutip.solver.stochastic.SMESolver
    :members:
    :inherited-members:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.stochastic.SSESolver
    :members:
    :inherited-members:
    :exclude-members: add_integrator


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
    :inherited-members:
    :exclude-members: add_processor, add

.. autoclass:: qutip.solver.multitrajresult.MultiTrajResult
    :members:
    :inherited-members:
    :exclude-members: add_processor, add, add_end_condition

.. autoclass:: qutip.solver.multitrajresult.McResult
    :show-inheritance:
    :members:

.. autoclass:: qutip.solver.multitrajresult.NmmcResult
    :show-inheritance:
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
----------------------

.. autoclass:: qutip.distributions.Distribution
    :members:

CompilationOptions
------------------

.. autoclass:: qutip.core.coefficient.CompilationOptions


..
  Docstrings are empty...

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
