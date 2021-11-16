.. _classes:

***************
Classes
***************

.. _classes-qobj:

Qobj
--------------

.. autoclass:: qutip.Qobj
    :members:

.. _classes-qobjevo:

QobjEvo
--------------

.. autoclass:: qutip.QobjEvo
    :members:

.. _classes-eseries:

eseries
-----------------

.. autoclass:: qutip.eseries
    :members:

.. _classes-bloch:

Bloch sphere
---------------

.. autoclass:: qutip.bloch.Bloch
    :members:


Distributions
-------------

.. autoclass:: qutip.QFunc
    :members:


Cubic Spline
---------------

.. autoclass:: qutip.interpolate.Cubic_Spline
    :members:


.. _classes-non_markov:

Non-Markovian Solvers
---------------------

.. autoclass:: qutip.nonmarkov.heom.HEOMSolver
    :members:

.. autoclass:: qutip.nonmarkov.heom.BosonicHEOMSolver
    :members:

.. autoclass:: qutip.nonmarkov.heom.FermionicHEOMSolver
    :members:

.. autoclass:: qutip.nonmarkov.heom.HSolverDL
    :members:

.. autoclass:: qutip.nonmarkov.heom.BathExponent
    :members:

.. autoclass:: qutip.nonmarkov.heom.Bath
    :members:

.. autoclass:: qutip.nonmarkov.heom.BosonicBath
    :members:

.. autoclass:: qutip.nonmarkov.heom.DrudeLorentzBath
    :members:

.. autoclass:: qutip.nonmarkov.heom.DrudeLorentzPadeBath
    :members:

.. autoclass:: qutip.nonmarkov.heom.UnderDampedBath
    :members:

.. autoclass:: qutip.nonmarkov.heom.FermionicBath
    :members:

.. autoclass:: qutip.nonmarkov.dlheom_solver.HSolverDL
    :members:

.. autoclass:: qutip.nonmarkov.dlheom_solver.HEOMSolver
    :members:

.. autoclass:: qutip.nonmarkov.memorycascade.MemoryCascade
    :members:

.. autoclass:: qutip.nonmarkov.transfertensor.TTMSolverOptions
    :members:


.. _classes-odeoptions:

Solver Options and Results
---------------------------

.. autoclass:: qutip.solver.ExpectOps
    :members:

.. autoclass:: qutip.solver.Options
    :members:

.. autoclass:: qutip.solver.Result
    :members:

.. autoclass:: qutip.solver.SolverConfiguration
    :members:

.. autoclass:: qutip.solver.Stats
    :members:

.. autoclass:: qutip.stochastic.StochasticSolverOptions
    :members:

.. _classes-piqs:

Permutational Invariance
------------------------

.. autoclass:: qutip.piqs.Dicke
    :members:

.. autoclass:: qutip.piqs.Pim
    :members:

.. _classes-distributions:

One-Dimensional Lattice
-----------------------

.. autoclass:: qutip.lattice.Lattice1d
    :members:

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

.. _classes-qip:

Quantum information processing
------------------------------

.. autoclass:: qutip.qip.Gate
    :members:

.. autoclass:: qutip.qip.circuit.Measurement
    :members:

.. autoclass:: qutip.qip.circuit.QubitCircuit
    :members:

.. autoclass:: qutip.qip.circuit.CircuitResult
    :members:

.. autoclass:: qutip.qip.circuit.CircuitSimulator
    :members:

.. autoclass:: qutip.qip.device.Processor
    :members:

.. autoclass:: qutip.qip.device.OptPulseProcessor
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.device.ModelProcessor
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.device.SpinChain
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.device.LinearSpinChain
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.device.CircularSpinChain
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.device.DispersiveCavityQED
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.noise.Noise
    :members:

.. autoclass:: qutip.qip.noise.DecoherenceNoise
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.noise.RelaxationNoise
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.noise.ControlAmpNoise
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.noise.RandomNoise
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.pulse.Pulse
    :members:

.. autoclass:: qutip.qip.compiler.GateCompiler
    :members:

.. autoclass:: qutip.qip.compiler.CavityQEDCompiler
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.compiler.SpinChainCompiler
    :members:
    :inherited-members:

.. autoclass:: qutip.qip.compiler.Scheduler
    :members:

.. autoclass:: qutip.qip.compiler.Instruction
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
