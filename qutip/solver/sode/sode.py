import numpy as np
from . import _sode
from ..integrator.integrator import Integrator
from ..stochastic import StochasticSolver


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
        raise NotImplementedError


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
        "tol": 1e-7,
    }
    stepper = None
    N_dw = 0

    def __init__(self, system, options):
        self.system = system
        self._options = self.integrator_options.copy()
        self.options = options
        self.dt = self.options["dt"]
        self.tol = self.options["tol"]
        self.N_drift = system.num_collapse

    def _step(self, dt, dW):
        new_state = self.stepper(self.system, self.t, self.state, dt, dW)
        self.state = new_state
        self.t += dt

    def set_state(self, t, state0, generator):
        self.t = t
        self.state = state0
        self.generator = generator

    def integrate(self, t, copy=True):
        delta_t = (t - self.t)
        if delta_t < 0:
            raise ValueError("Stochastic integration time")
        elif delta_t == 0:
            return self.t, self.state, np.zeros((0, self.N_drift, self.N_dw))

        dt = self.dt
        N, extra = np.divmod(delta_t, dt)
        N = int(N)
        if extra > self.tol:
            # Not a whole number of steps.
            N += 1
            dt = delta_t / N
        dW = self.generator.normal(
            0,
            np.sqrt(dt),
            size=(N, self.N_drift, self.N_dw)
        )

        for i in range(N):
            self._step(dt, dW[i, :])

        return self.t, self.state, np.sum(dW, axis=0) / (N * dt)

    def get_state(self, copy=True):
        return self.t, self.state, self.generator

    @property
    def options(self):
        """
        Supported options by Explicit Stochastic Integrators:

        dt : float, default=0.001
            Internal time step.

        tol : float, default=1e-7
            Relative tolerance.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)


class EulerSODE(_Explicit_Simple_Integrator):
    """
    A simple generalization of the Euler method for ordinary
    differential equations to stochastic differential equations.  Only
    solver which could take non-commuting ``sc_ops``.

    - Order: 0.5
    """
    stepper = _sode.euler
    N_dw = 1


class PlatenSODE(_Explicit_Simple_Integrator):
    """
    Explicit scheme, creates the Milstein using finite differences
    instead of analytic derivatives. Also contains some higher order
    terms, thus converges better than Milstein while staying strong
    order 1.0.  Does not require derivatives. See eq. (7.47) of chapter 7 of
    H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum Systems*.

    - Order: strong 1, weak 2
    """
    stepper = _sode.platen
    N_dw = 1


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
    stepper = _sode.explicit15
    N_dw = 2


class Taylor1_5_SODE(_Explicit_Simple_Integrator):
    """
    Order 1.5 strong Taylor scheme.  Solver with more terms of the
    Ito-Taylor expansion.  Default solver for :obj:`~smesolve` and
    :obj:`~ssesolve`.  See eq. (4.6) of chapter 10.4 of Peter E. Kloeden and
    Exkhard Platen, *Numerical Solution of Stochastic Differential Equations*.

    - Order strong 1.5
    """
    stepper = _sode.taylor15
    N_dw = 2


class Milstein_SODE(_Explicit_Simple_Integrator):
    """
    An order 1.0 strong Taylor scheme.  Better approximate numerical
    solution to stochastic differential equations.  See eq. (2.9) of
    chapter 12.2 of Peter E. Kloeden and Exkhard Platen,
    *Numerical Solution of Stochastic Differential Equations*..

    - Order strong 1.0
    """
    stepper = _sode.milstein
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
    stepper = _sode.pred_corr
    N_dw = 1


StochasticSolver.add_integrator(EulerSODE, "euler")
StochasticSolver.add_integrator(EulerSODE, "euler-maruyama")
StochasticSolver.add_integrator(PlatenSODE, "platen")
StochasticSolver.add_integrator(Explicit1_5_SODE, "explicit1.5")
StochasticSolver.add_integrator(Taylor1_5_SODE, "taylor15")
StochasticSolver.add_integrator(PredCorr_SODE, "pred_corr")
StochasticSolver.add_integrator(Milstein_SODE, "milstein")
