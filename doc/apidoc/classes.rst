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

Solver
------

.. autoclass:: qutip.solver.sesolve.SeSolver
    :members:

.. autoclass:: qutip.solver.mesolve.MeSolver
    :members:

.. autoclass:: qutip.solver.brmesolve.BRSolver
    :members:


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

.. autoclass:: qutip.solver.ode.scipy_integrator.IntegratorScipyAdams
    :members: options

.. autoclass:: qutip.solver.ode.scipy_integrator.IntegratorScipyBDF
    :members: options

.. autoclass:: qutip.solver.ode.scipy_integrator.IntegratorScipylsoda
    :members: options

.. autoclass:: qutip.solver.ode.scipy_integrator.IntegratorScipyDop853
    :members: options

.. autoclass:: qutip.solver.ode.qutip_integrator.IntegratorVern7
    :members: options

.. autoclass:: qutip.solver.ode.qutip_integrator.IntegratorVern9
    :members: options

.. autoclass:: qutip.solver.ode.qutip_integrator.IntegratorDiag
    :members: options


.. _classes-non_markov_mc_and_tt:

Non-Markovian Memory Cascade and Transfer Tensor Solvers
--------------------------------------------------------

.. autoclass:: qutip.solve.nonmarkov.memorycascade.MemoryCascade
    :members:

.. autoclass:: qutip.solve.nonmarkov.transfertensor.TTMSolverOptions
    :members:


.. _classes-odeoptions:

Solver Options and Results
---------------------------

.. autoclass:: qutip.solve.solver.ExpectOps
    :members:

.. autoclass:: qutip.solve.solver.Result
    :members:

.. autoclass:: qutip.solve.solver.SolverConfiguration
    :members:

.. autoclass:: qutip.solve.solver.Stats
    :members:

.. autoclass:: qutip.solve.stochastic.StochasticSolverOptions
    :members:

.. _classes-piqs:

Permutational Invariance
------------------------

.. autoclass:: qutip.solve.piqs.Dicke
    :members:

.. autoclass:: qutip.solve.piqs.Pim
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


.. _classes-control:

Optimal control
---------------

.. autoclass:: qutip.control.optimizer.Optimizer
    :members:

.. autoclass:: qutip.control.optimizer.OptimizerBFGS
    :members:

.. autoclass:: qutip.control.optimizer.OptimizerLBFGSB
    :members:

.. autoclass:: qutip.control.optimizer.OptimizerCrab
    :members:

.. autoclass:: qutip.control.optimizer.OptimizerCrabFmin
    :members:

.. autoclass:: qutip.control.optimizer.OptimIterSummary
    :members:

.. autoclass:: qutip.control.termcond.TerminationConditions
    :members:

.. autoclass:: qutip.control.optimresult.OptimResult
    :members:

.. autoclass:: qutip.control.dynamics.Dynamics
    :members:

.. autoclass:: qutip.control.dynamics.DynamicsGenMat
    :members:

.. autoclass:: qutip.control.dynamics.DynamicsUnitary
    :members:

.. autoclass:: qutip.control.dynamics.DynamicsSymplectic
    :members:

.. autoclass:: qutip.control.propcomp.PropagatorComputer
    :members:

.. autoclass:: qutip.control.propcomp.PropCompApproxGrad
    :members:

.. autoclass:: qutip.control.propcomp.PropCompDiag
    :members:

.. autoclass:: qutip.control.propcomp.PropCompFrechet
    :members:

.. autoclass:: qutip.control.fidcomp.FidelityComputer
    :members:

.. autoclass:: qutip.control.fidcomp.FidCompUnitary
    :members:

.. autoclass:: qutip.control.fidcomp.FidCompTraceDiff
    :members:

.. autoclass:: qutip.control.fidcomp.FidCompTraceDiffApprox
    :members:

.. autoclass:: qutip.control.tslotcomp.TimeslotComputer
    :members:

.. autoclass:: qutip.control.tslotcomp.TSlotCompUpdateAll
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGen
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenRandom
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenZero
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenLinear
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenPeriodic
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenSine
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenSquare
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenSaw
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenTriangle
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenGaussian
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenGaussianEdge
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenCrab
    :members:

.. autoclass:: qutip.control.pulsegen.PulseGenCrabFourier
    :members:

.. autoclass:: qutip.control.stats.Stats
    :members:

.. autoclass:: qutip.control.dump.Dump
    :members:

.. autoclass:: qutip.control.dump.OptimDump
    :members:

.. autoclass:: qutip.control.dump.DynamicsDump
    :members:

.. autoclass:: qutip.control.dump.DumpItem
    :members:

.. autoclass:: qutip.control.dump.EvoCompDumpItem
    :members:

.. autoclass:: qutip.control.dump.DumpSummaryItem
    :members:
