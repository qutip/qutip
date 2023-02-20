import numpy as np
from . import _sode
from .sode import _Explicit_Simple_Integrator
from ..stochastic import StochasticSolver, SMESolver


__all__ = ["EulerSODE", "Milstein_SODE", "Taylor1_5_SODE", "Explicit1_5_SODE"]


class EulerSODE(_Explicit_Simple_Integrator):
    """
    A simple generalization of the Euler method for ordinary
    differential equations to stochastic differential equations.  Only
    solver which could take non-commuting ``sc_ops``.

    - Order: 0.5
    """
    stepper = _sode.Euler
    N_dw = 1


class Milstein_SODE(_Explicit_Simple_Integrator):
    """
    An order 1.0 strong Taylor scheme.  Better approximate numerical
    solution to stochastic differential equations.  See eq. (2.9) of
    chapter 12.2 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*..

    - Order strong 1.0
    """
    stepper = _sode.Milstein
    N_dw = 1


class Taylor1_5_SODE(_Explicit_Simple_Integrator):
    """
    Order 1.5 strong Taylor scheme.  Solver with more terms of the
    Ito-Taylor expansion.  Default solver for :obj:`~smesolve` and
    :obj:`~ssesolve`.  See eq. (4.6) of chapter 10.4 of Peter E. Kloeden and
    Exkhard Platen, *Numerical Solution of Stochastic Differential Equations*.

    - Order strong 1.5
    """
    stepper = _sode.Taylor15
    N_dw = 2


class Explicit1_5_SODE(_Explicit_Simple_Integrator):
    """
    Explicit order 1.5 strong schemes.  Reproduce the order 1.5 strong
    Taylor scheme using finite difference instead of derivatives.
    Slower than ``taylor15`` but usable when derrivatives cannot be
    analytically obtained.
    See eq. (2.13) of chapter 11.2 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations.*

    - Order: strong 1.5
    """
    stepper = _sode.Explicit15
    N_dw = 2


StochasticSolver.add_integrator(EulerSODE, "euler")
StochasticSolver.add_integrator(Explicit1_5_SODE, "explicit1.5")
SMESolver.add_integrator(Taylor1_5_SODE, "taylor1.5")
SMESolver.add_integrator(Milstein_SODE, "milstein")
