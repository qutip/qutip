from . import _sode
from .sode import _Explicit_Simple_Integrator, _Implicit_Simple_Integrator
from ..stochastic import StochasticSolver, SMESolver


__all__ = [
    "EulerSODE", "Milstein_SODE", "Taylor1_5_SODE", "Explicit1_5_SODE",
    "Implicit_Milstein_SODE", "Implicit_Taylor1_5_SODE"
]


class EulerSODE(_Explicit_Simple_Integrator):
    """
    A simple generalization of the Euler method for ordinary
    differential equations to stochastic differential equations.  Only
    solver which could take non-commuting ``sc_ops``.

    - Order: 0.5
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
    }
    stepper = _sode.Euler
    N_dw = 1
    _stepper_options = ["measurement_noise"]


class Milstein_SODE(_Explicit_Simple_Integrator):
    """
    An order 1.0 strong Taylor scheme.  Better approximate numerical
    solution to stochastic differential equations.  See eq. (3.12) of
    chapter 10.3 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*..

    - Order strong 1.0
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
    }
    stepper = _sode.Milstein
    N_dw = 1
    _stepper_options = ["measurement_noise"]


class Taylor1_5_SODE(_Explicit_Simple_Integrator):
    """
    Order 1.5 strong Taylor scheme.  Solver with more terms of the
    Ito-Taylor expansion. See eq. (4.6) of chapter 10.4 of Peter E. Kloeden and
    Exkhard Platen, *Numerical Solution of Stochastic Differential Equations*.

    - Order strong 1.5
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "derr_dt": 1e-6,
    }
    stepper = _sode.Taylor15
    N_dw = 2

    @property
    def options(self):
        """
        Supported options by Order 1.5 strong Taylor Stochastic Integrators:

        dt : float, default: 0.001
            Internal time step.

        tol : float, default: 1e-10
            Relative tolerance.

        derr_dt : float, default: 1e-6
            Finite time difference used to compute the derrivative of the
            hamiltonian and ``sc_ops``.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        _Explicit_Simple_Integrator.options.fset(self, new_options)


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


class Implicit_Milstein_SODE(_Implicit_Simple_Integrator):
    """
    An order 1.0 implicit strong Taylor scheme.  Implicit Milstein
    scheme for the numerical simulation of stiff stochastic
    differential equations.  Eq. (2.11) with alpha=0.5 of
    chapter 12.2 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*.

    - Order strong 1.0
    """
    stepper = _sode.Milstein_imp
    N_dw = 1


class Implicit_Taylor1_5_SODE(_Implicit_Simple_Integrator):
    """
    Order 1.5 implicit strong Taylor scheme.  Solver with more terms of the
    Ito-Taylor expansion.  Eq. (2.18) with ``alpha=0.5`` of chapter 12.2 of
    Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*.

    - Order strong 1.5
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "solve_method": None,
        "solve_options": {},
        "deff_dt": 1e-6
    }
    stepper = _sode.Taylor15_imp
    N_dw = 2

    @property
    def options(self):
        """
        Supported options by Implicit Order 1.5 strong Taylor Stochastic
        Integrators:

        dt : float, default: 0.001
            Internal time step.

        tol : float, default: 1e-10
            Tolerance for the time steps.

        solve_method : str, default: None
            Method used for solver the ``Ax=b`` of the implicit step.
            Accept methods supported by :func:`qutip.core.data.solve`.
            When the system is constant, the inverse of the matrix ``A`` can be
            used by entering ``inv``.

        solve_options : dict, default: {}
            Options to pass to the call to :func:`qutip.core.data.solve`.

        derr_dt : float, default: 1e-6
            Finite time difference used to compute the derrivative of the
            hamiltonian and ``sc_ops``.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        _Implicit_Simple_Integrator.options.fset(self, new_options)


StochasticSolver.add_integrator(EulerSODE, "euler")
StochasticSolver.add_integrator(Explicit1_5_SODE, "explicit1.5")
SMESolver.add_integrator(Taylor1_5_SODE, "taylor1.5")
SMESolver.add_integrator(Milstein_SODE, "milstein")
SMESolver.add_integrator(Implicit_Milstein_SODE, "milstein_imp")
SMESolver.add_integrator(Implicit_Taylor1_5_SODE, "taylor1.5_imp")
