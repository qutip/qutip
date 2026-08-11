import numpy as np
import warnings
from . import _sode
from ..integrator.integrator import Integrator
from ..stochastic import StochasticSolver, SMESolver, _StochasticRHS
from .ssystem import BaseStochasticSystem, TaylorStochasticSystem

__all__ = ["SIntegrator", "PlatenSODE", "PredCorr_SODE"]


class SIntegrator(Integrator):
    """
    A wrapper around stochastic ODE solvers.

    Parameters
    ----------
    system: qutip.StochasticSystem
        Quantum system in which states evolve.

    options: dict
        Options for the integrator.

    Class Attributes
    ----------------
    name : str
        The name of the integrator.

    rhs_format : {"SDESystem", "SDETaylorSystem", "Solver"}
        Which format the SDE integrator rhs is used by the integration method.
        - "SDESystem": Instance of a subclass of :class:"BaseStochasticSystem".
        - "SDETaylorSystem": Instance of a child class of
          :class:"TaylorStochasticSystem". Depending on the integration method,
          not all derivative may need to be defined.
        - "system": An instance of `_StochasticRHS`. For cases where the
          integrator build the RHS operator ifself. These are limited to
          integration method that mixes the physics of the problem and the
          numerics.

    integrator_options : dict
        A dictionary of options used by the integrator and their default
        values. Once initiated, ``self.options`` will be a dict with the same
        keys, not the full options object passed to the solver. Options' keys
        included here will be supported by the :cls:SolverOdeOptions.
    """
    _is_set = False
    _support_measurement_noise = False
    _wiener_is_measurement = False
    # How the rhs is passed to the integrator.
    # "SDESystem", "SDETaylorSystem", "system"
    rhs_format = "SDESystem"
    N_dw = 1

    def __init__(self, rhs, options):
        expected_type = {
            "SDESystem": BaseStochasticSystem,
            "SDETaylorSystem": TaylorStochasticSystem,
            "system": _StochasticRHS,
        }[self.rhs_format]
        if not isinstance(rhs, expected_type):
            raise TypeError(f"Got {type(rhs)}, expected {expected_type}.")
        self._options = self.integrator_options.copy()
        self.options = options
        self.rhs = rhs

    def set_state(self, t, state0, wiener):
        """
        Set the state of the SODE solver.

        Parameters
        ----------
        t : float
            Initial time

        state0 : qutip.Data
            Initial state.

        generator : numpy.random.generator
            Random number generator.
        """
        self.t = t
        self.state = state0
        self.wiener = wiener
        self.wiener._prepare(self.N_dw)
        if self.wiener.is_measurement and not self._support_measurement_noise:
            raise NotImplementedError(
                f"{type(self).__name__} does not support running"
                " the evolution from measurements."
            )
        self._is_set = True

    def get_state(self, copy=True):
        return self.t, self.state, self.wiener

    def integrate(self, t, copy=True):
        """
        Evolve to t.

        Before calling `integrate` for the first time, the initial state should
        be set with `set_state`.

        Parameters
        ----------
        t : float
            Time to integrate to, should be larger than the previous time.

        copy : bool [True]
            Whether to return a copy of the state or the state itself.

        Returns
        -------
        (t, state, noise) : (float, qutip.Data, np.ndarray)
            The state of the solver at ``t``.
        """
        raise NotImplementedError

    def mcstep(self, t, copy=True):
        raise NotImplementedError

    def reset(self, hard=False):
        if self._is_set:
            state = self.get_state()
            self.set_state(*state)
        if hard:
            raise NotImplementedError(
                "Changing stochastic integrator "
                "options is not supported."
            )


class _Cython_SIntegrator(SIntegrator):
    stepper = None
    _stepper_options = []

    def set_state(self, t, state0, wiener):
        stepper_opt = {
            key: self.options[key]
            for key in self._stepper_options
            if key in self.options
        }
        if "measurement_noise" in self._stepper_options:
            stepper_opt["measurement_noise"] = wiener.is_measurement
        super().set_state(t, state0, wiener)
        self.step_func = self.stepper(
            self.rhs,
            **stepper_opt
        ).run


class _Explicit_Simple_Integrator(_Cython_SIntegrator):
    """
    Stochastic evolution solver
    """

    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
    }
    stepper = None
    N_dw = 0

    def integrate(self, t, copy=True):
        delta_t = t - self.t
        dt = self.options["dt"]
        if delta_t < 0:
            raise ValueError("Integration time, can't be negative.")
        elif delta_t < 0.5 * dt:
            warnings.warn(
                f"Step under minimum step ({dt}), skipped.",
                RuntimeWarning
            )
            return self.t, self.state, np.zeros(self.N_dw)

        N, extra = np.divmod(delta_t, dt)
        N = int(N)
        if extra > 0.5 * dt:
            # Not a whole number of steps, round to higher
            N += 1
        dW = self.wiener.dW(self.t, N)

        self.state = self.step_func(self.t, self.state, dt, dW, N)
        self.t += dt * N

        return self.t, self.state, np.sum(dW[:, 0, :], axis=0)

    @property
    def options(self):
        """
        Supported options by Explicit Stochastic Integrators:

        dt : float, default: 0.001
            Internal time step.

        tol : float, default: 1e-10
            Tolerance for the time steps.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


class _Implicit_Simple_Integrator(_Explicit_Simple_Integrator):
    """
    Stochastic evolution solver
    """

    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "solve_method": None,
        "solve_options": {},
    }
    _stepper_options = ["solve_method", "solve_options"]
    stepper = None
    N_dw = 0

    @property
    def options(self):
        """
        Supported options by Implicit Stochastic Integrators:

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
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


class PlatenSODE(_Explicit_Simple_Integrator):
    """
    Explicit scheme, creates the Milstein using finite differences
    instead of analytic derivatives. Also contains some higher order
    terms, thus converges better than Milstein while staying strong
    order 1.0.  Does not require derivatives. See eq. (7.47) of chapter 7 of
    H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems*.

    - Order: strong 1, weak 2
    """
    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
    }
    stepper = _sode.Platen
    N_dw = 1
    _support_measurement_noise = True
    _stepper_options = ["measurement_noise"]
    rhs_format = "SDESystem"


class PredCorr_SODE(_Explicit_Simple_Integrator):
    """
    Generalization of the trapezoidal method to stochastic differential
    equations. More stable than explicit methods.  See eq. (5.4) of
    chapter 15.5 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*.

    - Order strong 0.5, weak 1.0
    - Codes to only correct the stochastic part (:math:`\\alpha=0`,
      :math:`\\eta=1/2`): ``'pred-corr'``, ``'predictor-corrector'`` or
      ``'pc-euler'``
    - Codes to correct both the stochastic and deterministic parts
      (:math:`\\alpha=1/2`, :math:`\\eta=1/2`): ``'pc-euler-imp'``,
      ``'pc-euler-2'`` or ``'pred-corr-2'``
    """

    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
        "alpha": 0.0,
        "eta": 0.5,
    }
    stepper = _sode.PredCorr
    N_dw = 1
    _support_measurement_noise = True
    _stepper_options = ["alpha", "eta", "measurement_noise"]
    rhs_format = "SDETaylorSystem"

    @property
    def options(self):
        """
        Supported options by Explicit Stochastic Integrators:

        dt : float, default: 0.001
            Internal time step.

        tol : float, default: 1e-10
            Tolerance for the time steps.

        alpha : float, default: 0.
            Implicit factor to the drift.
            eff_drift ~= drift(t) * (1-alpha) + drift(t+dt) * alpha

        eta : float, default: 0.5
            Implicit factor to the diffusion.
            eff_diffusion ~= diffusion(t) * (1-eta) + diffusion(t+dt) * eta
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


StochasticSolver.add_integrator(PlatenSODE, "platen")
SMESolver.add_integrator(PredCorr_SODE, "pred_corr")
