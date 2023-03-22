import numpy as np
from . import _sode
from ..integrator.integrator import Integrator
from ..stochastic import StochasticSolver, SMESolver


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

    supports_blackbox : bool
        If True, then the integrator calls only ``system.matmul``,
        ``system.matmul_data``, ``system.expect``, ``system.expect_data`` and
        ``isconstant``, ``isoper`` or ``issuper``. This allows the solver using
        the integrator to modify the system in creative ways. In particular,
        the solver may modify the system depending on *both* the time ``t``
        *and* the current ``state`` the system is being applied to.

        If the integrator calls any other methods, set to False.

    supports_time_dependent : bool
        If True, then the integrator supports time dependent systems. If False,
        ``supports_blackbox`` should usually be ``False`` too.

    integrator_options : dict
        A dictionary of options used by the integrator and their default
        values. Once initiated, ``self.options`` will be a dict with the same
        keys, not the full options object passed to the solver. Options' keys
        included here will be supported by the :cls:SolverOdeOptions.
    """

    def set_state(self, t, state0, generator):
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
        self.generator = generator

    def get_state(self, copy=True):
        return self.t, self.state, self.generator

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


class _Explicit_Simple_Integrator(SIntegrator):
    """
    Stochastic evolution solver
    """

    integrator_options = {
        "dt": 0.001,
        "tol": 1e-10,
    }
    stepper = None
    N_dw = 0

    def __init__(self, rhs, options):
        self._options = self.integrator_options.copy()
        self.options = options
        self.system = rhs(self.options)
        self.step_func = self.stepper(self.system).run

    def integrate(self, t, copy=True):
        delta_t = t - self.t
        if delta_t < 0:
            raise ValueError("Stochastic integration time")
        elif delta_t == 0:
            return self.t, self.state, np.zeros(self.N_dw)

        dt = self.options["dt"]
        N, extra = np.divmod(delta_t, dt)
        N = int(N)
        if extra > self.options["tol"]:
            # Not a whole number of steps.
            N += 1
            dt = delta_t / N
        dW = self.generator.normal(
            0, np.sqrt(dt), size=(N, self.N_dw, self.system.num_collapse)
        )

        self.state = self.step_func(self.t, self.state, dt, dW, N)
        self.t += dt * N

        return self.t, self.state, np.sum(dW[:, 0, :], axis=0)

    @property
    def options(self):
        """
        Supported options by Explicit Stochastic Integrators:

        dt : float, default=0.001
            Internal time step.

        tol : float, default=1e-10
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
    stepper = None
    N_dw = 0

    def __init__(self, rhs, options):
        self._options = self.integrator_options.copy()
        self.options = options
        self.system = rhs(self.options)
        self.step_func = self.stepper(
            self.system,
            self.options["solve_method"],
            self.options["solve_options"],
        ).run

    @property
    def options(self):
        """
        Supported options by Implicit Stochastic Integrators:

        dt : float, default=0.001
            Internal time step.

        tol : float, default=1e-10
            Tolerance for the time steps.

        solve_method : str, default=None
            Method used for solver the ``Ax=b`` of the implicit step.
            Accept methods supported by :func:`qutip.core.data.solve`.
            When the system is constant, the inverse of the matrix ``A`` can be
            used by entering ``inv``.

        solve_options : dict, default={}
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

    stepper = _sode.Platen
    N_dw = 1


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

    def __init__(self, rhs, options):
        self._options = self.integrator_options.copy()
        self.options = options
        self.system = rhs(self.options)
        self.step_func = self.stepper(
            self.system, self.options["alpha"], self.options["eta"]
        ).run

    @property
    def options(self):
        """
        Supported options by Explicit Stochastic Integrators:

        dt : float, default=0.001
            Internal time step.

        tol : float, default=1e-10
            Tolerance for the time steps.

        alpha : float, default=0.
            Implicit factor to the drift.
            eff_drift ~= drift(t) * (1-alpha) + drift(t+dt) * alpha

        eta : float, default=0.5
            Implicit factor to the diffusion.
            eff_diffusion ~= diffusion(t) * (1-eta) + diffusion(t+dt) * eta
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


StochasticSolver.add_integrator(PlatenSODE, "platen")
SMESolver.add_integrator(PredCorr_SODE, "pred_corr")
