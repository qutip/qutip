.. _integrator_stochastic_internal:

Stochastic Integrator Architecture
##################################


Stochastic differential equations are also common in quantum systems.
In QuTiP, SDE solvers derive from SIntegrator, similar to the base Integrator, but a few differences:


Attributes:

The ``SIntegrator``'s possible ``rhs_format`` are:

- "SDESystem": A class deriving from :class:`StochasticSystem` with the ``drift``
  and ``diffusion`` method, referencing to the deterministic and stochastic part
  of the equation respectively. This is used for explicit methods.

- "SDETaylorSystem":  class deriving from :class:`TaylorStochasticSystem`.
  This class provide the previous, ``drift`` and ``diffusion`` and derivative from the
  Ito-Taylor expansion. The interface provide interface for the taylor method up to order 1.5.

- "system"
  A small class having the ``H``, ``c_ops``, ``sc_ops``


_support_measurement_noise: a boolean that indicate if measurement noise is supported.
Measurement noise are case where the wiener noise is the output of a measurement from and experiment.
default: False

N_dw: int, default: 1
Number of random noise of each step per diffusion operator needed.
Usually only one is needed, but high order Taylor method may need more.



Method:

``set_state`` signature is now ``(t, state, wiener, is_measurement)``,
  ``wiener`` is a Wiener object to represent the noise, saved in ``self.wiener``.
  ``is_measurement`` is a bool that indicate whether the wiener noise is a measurement.
  Stored in ``self._wiener_is_measurement``


``get_state`` returns the tuple (t, state, wiener) where ``wiener`` is the object passed to ``set_state``.


Helper classes
##############

:class:`Wiener`
===============
Represent the noise.
The only method used by SIntegrator is :method:`Wiener.dW(t, N)`:

Return an array of noise for N steps starting at ``t``.
The noise is of shape ``(N, N_dw, num_diffusion)``.
The noise Gaussian noise scaled with ``dt**0.5``.
The ``t`` is expected to be a multiple of the ``dt`` options and is the time at the start of the step.

:class:`StochasticSystem`
=========================
...

:class:`TaylorStochasticSystem`
===============================
...

:class:`_StochasticRHS`
=======================
...
