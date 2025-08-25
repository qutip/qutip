***************************
Dynamics and Time-Evolution
***************************

Schrödinger Equation
--------------------

.. automodule:: qutip.solver.sesolve
    :members: sesolve

.. automodule:: qutip.solver.krylovsolve
    :members: krylovsolve

.. autoclass:: qutip.solver.sesolve.SESolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

Master Equation
---------------

.. automodule:: qutip.solver.mesolve
    :members: mesolve

.. autoclass:: qutip.solver.mesolve.MESolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.result.Result
    :members:
    :inherited-members:
    :exclude-members: add_processor, add

Monte Carlo Evolution
---------------------

.. automodule:: qutip.solver.mcsolve
    :members: mcsolve

.. autoclass:: qutip.solver.mcsolve.MCSolver
    :members:
    :inherited-members:
    :member-order: bysource
    :show-inheritance:
    :exclude-members: add_integrator

.. automodule:: qutip.solver.nm_mcsolve
    :members: nm_mcsolve

.. autoclass:: qutip.solver.nm_mcsolve.NonMarkovianMCSolver
    :members:
    :inherited-members:
    :member-order: bysource
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.multitrajresult.McResult
    :members: steady_state, merge

.. autoclass:: qutip.solver.multitrajresult.NmmcResult
    :members: steady_state, merge


Bloch-Redfield Master Equation
------------------------------

.. automodule:: qutip.solver.brmesolve
    :members: brmesolve

.. autoclass:: qutip.solver.brmesolve.BRSolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

Floquet States and Floquet-Markov Master Equation
-------------------------------------------------

.. automodule:: qutip.solver.floquet
    :members: fmmesolve, fsesolve, floquet_tensor

.. autoclass:: qutip.solver.floquet.FMESolver
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.floquet.FloquetBasis
    :members:

Stochastic Schrödinger Equation and Master Equation
---------------------------------------------------

.. automodule:: qutip.solver.stochastic
    :members: ssesolve, smesolve

.. autoclass:: qutip.solver.stochastic.SMESolver
    :members:
    :inherited-members:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.stochastic.SSESolver
    :members:
    :inherited-members:
    :exclude-members: add_integrator

.. autoclass:: qutip.solver.multitrajresult.MultiTrajResult
    :members:
    :inherited-members:
    :exclude-members: add_processor, add, add_end_condition

Non-Markovian Solvers
---------------------

.. automodule:: qutip.solver.nonmarkov.transfertensor
    :members: ttmsolve


.. _api-ode:

Integrator
----------
Different ODE solver from many sources (scipy, diffrax, home made, etc.) used
by qutip solvers. Their options are added to the solver options:

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

Parallelization
---------------

.. automodule:: qutip.solver.parallel
    :members: parallel_map, serial_map, loky_pmap, mpi_pmap


***********
Propagators
***********

.. automodule:: qutip.solver.propagator
    :members: propagator, propagator_steadystate
    :undoc-members:

.. autoclass:: qutip.solver.propagator.Propagator
    :members:
    :inherited-members:
    :special-members: __call__

Dysolve
-------

.. automodule:: qutip.solver.dysolve_propagator
    :members: dysolve_propagator, DysolvePropagator


************************
Other dynamics functions
************************


Correlation Functions
---------------------

.. automodule:: qutip.solver.correlation
    :members: correlation_2op_1t, correlation_2op_2t, correlation_3op_1t, correlation_3op_2t, correlation_3op, coherence_function_g1, coherence_function_g2

.. automodule:: qutip.solver.spectrum
    :members: spectrum, spectrum_correlation_fft


Steady-state Solvers
--------------------

.. automodule:: qutip.solver.steadystate
    :members: steadystate, pseudo_inverse, steadystate_floquet
    :undoc-members:


Scattering in Quantum Optical Systems
-------------------------------------

.. automodule:: qutip.solver.scattering
    :members: temporal_basis_vector, temporal_scattered_state, scattering_probability
    :undoc-members:
