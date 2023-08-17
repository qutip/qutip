# -*- coding: utf-8 -*-

import numpy as np
from qutip.cy.stochastic import (
    SSESolver, SMESolver, PcSSESolver, PcSMESolver, PmSMESolver,
    GenericSSolver,
)
from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.states import ket2dm
from qutip.solver import Result
from qutip.qobjevo import QobjEvo
from qutip.superoperator import spre, spost, mat2vec, liouvillian
from qutip.solver import Options
from qutip.parallel import serial_map
from qutip.ui.progressbar import TextProgressBar
from qutip.pdpsolve import main_ssepdpsolve, main_smepdpsolve

__all__ = ['ssesolve', 'photocurrent_sesolve', 'smepdpsolve',
           'smesolve', 'photocurrent_mesolve', 'ssepdpsolve',
           'stochastic_solvers', 'general_stochastic']


def stochastic_solvers():
    # This docstring contains several literal backslash characters inside LaTeX
    # blocks, but it cannot be declared as a raw string because we also need to
    # use a line continuation.  At one point we need a restructured text
    # "definition list", where the heading _must_ be entirely on one line,
    # however it will violate our line-length reporting if we do that.
    """
    This function is purely a reference point for documenting the available
    stochastic solver methods, and takes no actions.

    Notes
    -----
    Available solvers for :obj:`~ssesolve` and :obj:`~smesolve`
        euler-maruyama
            A simple generalization of the Euler method for ordinary
            differential equations to stochastic differential equations.  Only
            solver which could take non-commuting ``sc_ops``. *not tested*

            - Order 0.5
            - Code: ``'euler-maruyama'``, ``'euler'`` or ``0.5``

        milstein
            An order 1.0 strong Taylor scheme.  Better approximate numerical
            solution to stochastic differential equations.  See eq. (2.9) of
            chapter 12.2 of [1]_.

            - Order strong 1.0
            - Code: ``'milstein'`` or ``1.0``

        milstein-imp
            An order 1.0 implicit strong Taylor scheme.  Implicit Milstein
            scheme for the numerical simulation of stiff stochastic
            differential equations.

            - Order strong 1.0
            - Code: ``'milstein-imp'``

        predictor-corrector
            Generalization of the trapezoidal method to stochastic differential
            equations. More stable than explicit methods.  See eq. (5.4) of
            chapter 15.5 of [1]_.

            - Order strong 0.5, weak 1.0
            - Codes to only correct the stochastic part (:math:`\\alpha=0`,
              :math:`\\eta=1/2`): ``'pred-corr'``, ``'predictor-corrector'`` or
              ``'pc-euler'``
            - Codes to correct both the stochastic and deterministic parts
              (:math:`\\alpha=1/2`, :math:`\\eta=1/2`): ``'pc-euler-imp'``,
              ``'pc-euler-2'`` or ``'pred-corr-2'``

        platen
            Explicit scheme, creates the Milstein using finite differences
            instead of analytic derivatives. Also contains some higher order
            terms, thus converges better than Milstein while staying strong
            order 1.0.  Does not require derivatives, therefore usable by
            :func:`~general_stochastic`.  See eq. (7.47) of chapter 7 of [2]_.

            - Order strong 1.0, weak 2.0
            - Code: ``'platen'``, ``'platen1'`` or ``'explicit1'``

        rouchon
            Scheme keeping the positivity of the density matrix
            (:obj:`~smesolve` only).  See eq. (4) with :math:`\\eta=1` of [3]_.

            - Order strong 1.0?
            - Code: ``'rouchon'`` or ``'Rouchon'``

        taylor1.5
            Order 1.5 strong Taylor scheme.  Solver with more terms of the
            Ito-Taylor expansion.  Default solver for :obj:`~smesolve` and
            :obj:`~ssesolve`.  See eq. (4.6) of chapter 10.4 of [1]_.

            - Order strong 1.5
            - Code: ``'taylor1.5'``, ``'taylor15'``, ``1.5``, or ``None``

        taylor1.5-imp
            Order 1.5 implicit strong Taylor scheme.  Implicit Taylor 1.5
            (:math:`\\alpha = 1/2`, :math:`\\beta` doesn't matter).  See eq.
            (2.18) of chapter 12.2 of [1]_.

            - Order strong 1.5
            - Code: ``'taylor1.5-imp'`` or ``'taylor15-imp'``

        explicit1.5
            Explicit order 1.5 strong schemes.  Reproduce the order 1.5 strong
            Taylor scheme using finite difference instead of derivatives.
            Slower than ``taylor15`` but usable by
            :func:`~general_stochastic`.  See eq. (2.13) of chapter 11.2 of
            [1]_.

            - Order strong 1.5
            - Code: ``'explicit1.5'``, ``'explicit15'`` or ``'platen15'``

        taylor2.0
            Order 2 strong Taylor scheme.  Solver with more terms of the
            Stratonovich expansion.  See eq. (5.2) of chapter 10.5 of [1]_.

            - Order strong 2.0
            - Code: ``'taylor2.0'``, ``'taylor20'`` or ``2.0``

        All solvers, except taylor2.0, are usable in both smesolve and ssesolve
        and for both heterodyne and homodyne. taylor2.0 only works for 1
        stochastic operator independent of time with the homodyne method.
        :func:`~general_stochastic` only accepts the derivative-free
        solvers: ``'euler'``, ``'platen'`` and ``'explicit1.5'``.

    Available solvers for :obj:`~photocurrent_sesolve` and \
:obj:`~photocurrent_mesolve`
        Photocurrent use ordinary differential equations between
        stochastic "jump/collapse".

        euler
            Euler method for ordinary differential equations between jumps.
            Only one jump per time interval.  Default solver.  See eqs. (4.19)
            and (4.4) of chapter 4 of [4]_.

            - Order 1.0
            - Code: ``'euler'``

        predictor–corrector
            predictor–corrector method (PECE) for ordinary differential
            equations.  Uses the Poisson distribution to obtain the number of
            jumps at each timestep.

            - Order 2.0
            - Code: ``'pred-corr'``

    References
    ----------
    .. [1] Peter E. Kloeden and Exkhard Platen, *Numerical Solution of
       Stochastic Differential Equations*.
    .. [2] H.-P. Breuer and F. Petruccione, *The Theory of Open Quantum
       Systems*.
    .. [3] Pierre Rouchon and Jason F. Ralpha, *Efficient Quantum Filtering for
       Quantum Feedback Control*, `arXiv:1410.5345 [quant-ph]
       <https://arxiv.org/abs/1410.5345>`_, Phys. Rev. A 91, 012118,
       (2015).
    .. [4] Howard M. Wiseman, Gerard J. Milburn, *Quantum measurement and
       control*.
    """


class StochasticSolverOptions:
    """Class of options for stochastic solvers such as
    :func:`qutip.stochastic.ssesolve`, :func:`qutip.stochastic.smesolve`, etc.

    The stochastic solvers :func:`qutip.stochastic.general_stochastic`,
    :func:`qutip.stochastic.ssesolve`, :func:`qutip.stochastic.smesolve`,
    :func:`qutip.stochastic.photocurrent_sesolve` and
    :func:`qutip.stochastic.photocurrent_mesolve`
    all take the same keyword arguments as
    the constructor of these class, and internally they use these arguments to
    construct an instance of this class, so it is rarely needed to explicitly
    create an instance of this class.

    Within the attribute list, a ``time_dependent_object`` is either

    - :class:`~qutip.Qobj`: a constant term
    - 2-element list of ``[Qobj, time_dependence]``: a time-dependent term
      where the ``Qobj`` will be multiplied by the time-dependent scalar.

    For more details on all allowed time-dependent objects, see the
    documentation for :class:`~qutip.QobjEvo`.

    Attributes
    ----------
    H : time_dependent_object or list of time_dependent_object
        System Hamiltonian in standard time-dependent list format.  This is the
        same as the argument that (e.g.) :func:`~qutip.mesolve` takes.
        If this is a list of elements, they are summed.

    state0 : :class:`qutip.Qobj`
        Initial state vector (ket) or density matrix.

    times : array_like of float
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of time_dependent_object
        List of deterministic collapse operators.  Each element of the list is
        a separate operator; unlike the Hamiltonian, there is no implicit
        summation over the terms.

    sc_ops : list of time_dependent_object
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the equation of motion according to how the d1 and d2 functions
        are defined.  Each element of the list is a separate operator, like
        ``c_ops``.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    m_ops : list of :class:`qutip.Qobj`
        List of operators representing the measurement operators. The expected
        format is a nested list with one measurement operator for each
        stochastic increament, for each stochastic collapse operator.

    args : dict
        Dictionary of parameters for time dependent systems.

    tol : float
        Tolerance of the solver for implicit methods.

    ntraj : int
        Number of trajectors.

    nsubsteps : int
        Number of sub steps between each time-spep given in `times`.

    dW_factors : array
        Array of length len(sc_ops), containing scaling factors for each
        measurement operator in m_ops.

    solver : string
        Name of the solver method to use for solving the stochastic
        equations. Valid values are:

        - order 1/2 algorithms: 'euler-maruyama', 'pc-euler', 'pc-euler-imp'
        - order 1 algorithms: 'milstein', 'platen', 'milstein-imp', 'rouchon'
        - order 3/2 algorithms: 'taylor1.5', 'taylor1.5-imp', 'explicit1.5'
        - order 2 algorithms: 'taylor2.0'

        See the documentation of :func:`~qutip.stochastic.stochastic_solvers`
        for a description of the solvers.  Implicit methods can adjust
        tolerance via the kw 'tol'. Default is {'tol': 1e-6}

    method : string ('homodyne', 'heterodyne')
        The name of the type of measurement process that give rise to the
        stochastic equation to solve.

    store_all_expect : bool (default False)
        Whether or not to store the e_ops expect values for all paths.

    store_measurement : bool (default False)
        Whether or not to store the measurement results in the
        :class:`qutip.solver.Result` instance returned by the solver.

    noise : int, or 1D array of int, or 4D array of float
        - int : seed of the noise
        - 1D array : length = ntraj, seeds for each trajectories.
        - 4D array : ``(ntraj, len(times), nsubsteps, len(sc_ops)*[1|2])``.
          Vector for the noise, the len of the last dimensions is doubled for
          solvers of order 1.5. This corresponds to results.noise.

    noiseDepth : int
        Number of terms kept of the truncated series used to create the
        noise used by taylor2.0 solver.

    normalize : bool
        (default True for (photo)ssesolve, False for (photo)smesolve)
        Whether or not to normalize the wave function during the evolution.
        Normalizing density matrices introduce numerical errors.

    options : :class:`qutip.solver.Options`
        Generic solver options. Only options.average_states and
        options.store_states are used.

    map_func: function
        A map function or managing the calls to single-trajactory solvers.

    map_kwargs: dictionary
        Optional keyword arguments to the map_func function function.

    progress_bar : :class:`qutip.ui.BaseProgressBar`
        Optional progress bar class instance.
    """
    def __init__(self, me, H=None, c_ops=[], sc_ops=[], state0=None,
                 e_ops=[], m_ops=None, store_all_expect=False,
                 store_measurement=False, dW_factors=None,
                 solver=None, method="homodyne", normalize=None,
                 times=None, nsubsteps=1, ntraj=1, tol=None,
                 generate_noise=None, noise=None,
                 progress_bar=None, map_func=None, map_kwargs=None,
                 args={}, options=None, noiseDepth=20):

        if options is None:
            options = Options()

        if progress_bar is None:
            progress_bar = TextProgressBar()

        # System
        # Cast to QobjEvo so the code has only one version for both the
        # constant and time-dependent case.
        self.me = me

        if H is not None:
            msg = "The Hamiltonian format is not valid. "
            try:
                self.H = QobjEvo(H, args=args, tlist=times,
                                 e_ops=e_ops, state0=state0)
            except Exception as e:
                raise ValueError(msg + str(e)) from e
        else:
            self.H = H

        if sc_ops:
            msg = ("The sc_ops format is not valid. Options are "
                   "[ Qobj / QobjEvo / [Qobj, coeff]]. ")
            try:
                self.sc_ops = [QobjEvo(op, args=args, tlist=times,
                                       e_ops=e_ops, state0=state0)
                               for op in sc_ops]
            except Exception as e:
                raise ValueError(msg + str(e)) from e
        else:
            self.sc_ops = sc_ops

        if c_ops:
            msg = ("The c_ops format is not valid. Options are "
                   "[ Qobj / QobjEvo / [Qobj, coeff]]. ")
            try:
                self.c_ops = [QobjEvo(op, args=args, tlist=times,
                                      e_ops=e_ops, state0=state0)
                              for op in c_ops]
            except Exception as e:
                raise ValueError(msg + str(e)) from e
        else:
            self.c_ops = c_ops

        self.state0 = state0
        self.rho0 = mat2vec(state0.full()).ravel()

        # Observation

        for e_op in e_ops:
            if (
                isinstance(e_op, Qobj)
                and self.H is not None
                and e_op.dims[1] != self.H.cte.dims[0]
            ):
                raise TypeError(f"e_ops dims ({e_op.dims}) are not compatible "
                                f"with the system's ({self.H.cte.dims})")
        self.e_ops = e_ops
        self.m_ops = m_ops
        self.store_measurement = store_measurement
        self.store_all_expect = store_all_expect
        self.store_states = options.store_states
        self.dW_factors = dW_factors

        # Solver
        self.solver = solver
        self.method = method
        if normalize is None and me:
            self.normalize = 0
        elif normalize is None and not me:
            self.normalize = 1
        elif normalize:
            self.normalize = 1
        else:
            self.normalize = 0

        self.times = times
        self.nsubsteps = nsubsteps
        self.dt = (times[1] - times[0]) / self.nsubsteps
        self.ntraj = ntraj
        if tol is not None:
            self.tol = tol
        elif "tol" in args:
            self.tol = args["tol"]
        else:
            self.tol = 1e-7

        # Noise
        if noise is not None:
            if isinstance(noise, int):
                # noise contain a seed
                np.random.seed(noise)
                noise = np.random.randint(0, 2**32, ntraj, dtype=np.uint32)
            noise = np.array(noise)
            if len(noise.shape) == 1:
                if noise.shape[0] < ntraj:
                    raise ValueError("'noise' does not have enought seeds " +
                                     "len(noise) >= ntraj")
                # numpy seed must be between 0 and 2**32-1
                # 'u4': unsigned 32bit int
                self.noise = noise.astype("u4")
                self.noise_type = 0

            elif len(noise.shape) == 4:
                # taylor case not included
                dw_len = (2 if method == "heterodyne" else 1)
                dw_len_str = (" * 2" if method == "heterodyne" else "")
                msg = "Incorrect shape for 'noise': "
                if noise.shape[0] < ntraj:
                    raise ValueError(msg + "shape[0] >= ntraj")
                if noise.shape[1] < len(times):
                    raise ValueError(msg + "shape[1] >= len(times)")
                if noise.shape[2] < nsubsteps:
                    raise ValueError(msg + "shape[2] >= nsubsteps")
                if noise.shape[3] < len(self.sc_ops) * dw_len:
                    raise ValueError(msg + "shape[3] >= len(self.sc_ops)" +
                                     dw_len_str)
                self.noise_type = 1
                self.noise = noise

        else:
            self.noise = np.random.randint(0, 2**32, ntraj, dtype=np.uint32)
            self.noise_type = 0

        # Map
        self.progress_bar = progress_bar
        if self.ntraj > 1 and map_func:
            self.map_func = map_func
        else:
            self.map_func = serial_map
        self.map_kwargs = map_kwargs if map_kwargs is not None else {}

        # Other
        self.options = options
        self.args = args
        self.set_solver()
        self.p = noiseDepth

    def set_solver(self):
        if self.solver in ['euler-maruyama', 'euler', 50, 0.5]:
            self.solver_code = 50
            self.solver = 'euler-maruyama'
        elif self.solver in ['platen', 'platen1', 'explicit1', 100]:
            self.solver_code = 100
            self.solver = 'platen'
        elif self.solver in ['pred-corr', 'predictor-corrector',
                             'pc-euler', 101]:
            self.solver_code = 101
            self.solver = 'pred-corr'
        elif self.solver in ['milstein', 102, 1.0]:
            self.solver_code = 102
            self.solver = 'milstein'
        elif self.solver in ['milstein-imp', 103]:
            self.solver_code = 103
            self.solver = 'milstein-imp'
        elif self.solver in ['pred-corr-2', 'pc-euler-2', 'pc-euler-imp', 104]:
            self.solver_code = 104
            self.solver = 'pred-corr-2'
        elif self.solver in ['Rouchon', 'rouchon', 120]:
            self.solver_code = 120
            self.solver = 'rouchon'
            if not all((op.const for op in self.sc_ops)):
                raise ValueError("Rouchon only works with constant sc_ops")
        elif self.solver in ['platen15', 'explicit1.5', 'explicit15', 150]:
            self.solver_code = 150
            self.solver = 'explicit1.5'
        elif self.solver in ['taylor15', 'taylor1.5', None, 1.5, 152]:
            self.solver_code = 152
            self.solver = 'taylor1.5'
        elif self.solver in ['taylor15-imp', 'taylor1.5-imp', 153]:
            self.solver_code = 153
            self.solver = 'taylor1.5-imp'
        elif self.solver in ['taylor2.0', 'taylor20', 2.0, 202]:
            self.solver_code = 202
            self.solver = 'taylor2.0'
            if not len(self.sc_ops) == 1 or \
                    not self.sc_ops[0].const or \
                    not self.method == "homodyne":
                raise ValueError(
                    "Taylor2.0 only works with 1 constant sc_ops and for"
                    " homodyne method"
                )
        else:
            known = [
                None, 'euler-maruyama', 'platen', 'pc-euler', 'pc-euler-imp',
                'milstein', 'milstein-imp', 'rouchon', 'taylor1.5',
                'taylor1.5-imp', 'explicit1.5', 'taylor2.0',
            ]
            raise ValueError("The solver should be one of {!r}".format(known))


class StochasticSolverOptionsPhoto(StochasticSolverOptions):
    """
    Attributes
    ----------

    solver : string
        Name of the solver method to use for solving the evolution
        of the system.*
        order 1 algorithms: 'euler'
        order 2 algorithms: 'pred-corr'
        In photocurrent evolution
    """
    def set_solver(self):
        if self.solver in [None, 'euler', 1, 60]:
            self.solver_code = 60
            self.solver = 'euler'
        elif self.solver in ['pred-corr', 'predictor-corrector', 110, 2]:
            self.solver_code = 110
            self.solver = 'pred-corr'
        else:
            raise Exception("The solver should be one of " +
                            "[None, 'euler', 'predictor-corrector']")


def smesolve(H, rho0, times, c_ops=[], sc_ops=[], e_ops=[],
             _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, or time dependent system.
        System Hamiltonian.
        Can depend on time, see StochasticSolverOptions help for format.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.
        Can depend on time, see StochasticSolverOptions help for format.

    sc_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.
        Can depend on time, see StochasticSolverOptions help for format.

    e_ops : list of :class:`qutip.Qobj`
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.

    """
    if "method" in kwargs and kwargs["method"] == "photocurrent":
        print("stochastic solver with photocurrent method has been moved to "
              "it's own function: photocurrent_mesolve")
        return photocurrent_mesolve(H, rho0, times, c_ops=c_ops, sc_ops=sc_ops,
                                    e_ops=e_ops, _safe_mode=_safe_mode,
                                    args=args, **kwargs)
    if isket(rho0):
        rho0 = ket2dm(rho0)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(True, H=H, state0=rho0, times=times,
                                  c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                                  args=args, **kwargs)

    if _safe_mode:
        _safety_checks(sso)

    if sso.solver_code == 120:
        return _positive_map(sso, e_ops_dict)

    sso.LH = liouvillian(sso.H, c_ops=sso.sc_ops + sso.c_ops) * sso.dt
    if sso.method == 'homodyne' or sso.method is None:
        if sso.m_ops is None:
            sso.m_ops = [op + op.dag() for op in sso.sc_ops]
        sso.sops = [spre(op) + spost(op.dag()) for op in sso.sc_ops]
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [1] * len(sso.m_ops)
        elif len(sso.dW_factors) != len(sso.m_ops):
            raise Exception("The len of dW_factors is not the same as m_ops")

    elif sso.method == 'heterodyne':
        if sso.m_ops is None:
            m_ops = []
        elif len(sso.m_ops) != 2 * len(sc_ops):
            raise ValueError(
                "When using the heterodyne method there should be two"
                " measurement operators (m_ops) for each collapse operator"
                " (sc_ops)."
            )
        sso.sops = []
        for c in sso.sc_ops:
            if sso.m_ops is None:
                m_ops += [c + c.dag(), -1j * (c - c.dag())]
            sso.sops += [(spre(c) + spost(c.dag())) / np.sqrt(2),
                         (spre(c) - spost(c.dag())) * -1j / np.sqrt(2)]
        if sso.m_ops is None:
            sso.m_ops = m_ops
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [np.sqrt(2)] * len(sso.sops)
        elif len(sso.dW_factors) == len(sso.m_ops):
            pass
        elif len(sso.dW_factors) == len(sso.sc_ops):
            dW_factors = []
            for fact in sso.dW_factors:
                dW_factors += [np.sqrt(2) * fact, np.sqrt(2) * fact]
            sso.dW_factors = dW_factors
        elif len(sso.dW_factors) != len(sso.m_ops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == "photocurrent":
        raise NotImplementedError("Moved to 'photocurrent_mesolve'")

    else:
        raise Exception("The method must be one of None, homodyne, heterodyne")

    sso.ce_ops = [QobjEvo(spre(op)) for op in sso.e_ops]
    sso.cm_ops = [QobjEvo(spre(op)) for op in sso.m_ops]

    sso.LH.compile()
    [op.compile() for op in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    if sso.solver_code in [103, 153]:
        sso.imp = 1 - sso.LH * 0.5
        sso.imp.compile()

    sso.solver_obj = SMESolver
    sso.solver_name = "smesolve_" + sso.solver

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    return res


def ssesolve(H, psi0, times, sc_ops=[], e_ops=[],
             _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic schrodinger equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, or time dependent system.
        System Hamiltonian.
        Can depend on time, see StochasticSolverOptions help for format.

    psi0 : :class:`qutip.Qobj`
        State vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    sc_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.
        Can depend on time, see StochasticSolverOptions help for format.

    e_ops : list of :class:`qutip.Qobj`
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
    if "method" in kwargs and kwargs["method"] == "photocurrent":
        print("stochastic solver with photocurrent method has been moved to "
              "it's own function: photocurrent_sesolve")
        return photocurrent_sesolve(H, psi0, times, sc_ops=sc_ops,
                                    e_ops=e_ops, _safe_mode=_safe_mode,
                                    args=args, **kwargs)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(False, H=H, state0=psi0, times=times,
                                  sc_ops=sc_ops, e_ops=e_ops,
                                  args=args, **kwargs)

    if _safe_mode:
        _safety_checks(sso)

    if sso.solver_code == 120:
        raise Exception("rouchon only work with smesolve")

    if sso.method == 'homodyne' or sso.method is None:
        if sso.m_ops is None:
            sso.m_ops = [op + op.dag() for op in sso.sc_ops]
        sso.sops = [[op, op + op.dag()] for op in sso.sc_ops]
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [1] * len(sso.sops)
        elif len(sso.dW_factors) != len(sso.sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == 'heterodyne':
        if sso.m_ops is None:
            m_ops = []
        sso.sops = []
        for c in sso.sc_ops:
            if sso.m_ops is None:
                m_ops += [c + c.dag(), -1j * (c - c.dag())]
            c1 = c / np.sqrt(2)
            c2 = c * (-1j / np.sqrt(2))
            sso.sops += [[c1, c1 + c1.dag()],
                         [c2, c2 + c2.dag()]]
        sso.m_ops = m_ops
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [np.sqrt(2)] * len(sso.sops)
        elif len(sso.dW_factors) == len(sso.sc_ops):
            dW_factors = []
            for fact in sso.dW_factors:
                dW_factors += [np.sqrt(2) * fact, np.sqrt(2) * fact]
            sso.dW_factors = dW_factors
        elif len(sso.dW_factors) != len(sso.sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == "photocurrent":
        NotImplementedError("Moved to 'photocurrent_sesolve'")

    else:
        raise Exception("The method must be one of None, homodyne, heterodyne")

    sso.LH = sso.H * (-1j*sso.dt)
    for ops in sso.sops:
        sso.LH -= ops[0]._cdc()*0.5*sso.dt

    sso.ce_ops = [QobjEvo(op) for op in sso.e_ops]
    sso.cm_ops = [QobjEvo(op) for op in sso.m_ops]

    sso.LH.compile()
    [[op.compile() for op in ops] for ops in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    sso.solver_obj = SSESolver
    sso.solver_name = "ssesolve_" + sso.solver

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def _positive_map(sso, e_ops_dict):
    if sso.method == 'homodyne' or sso.method is None:
        sops = sso.sc_ops
        if sso.m_ops is None:
            sso.m_ops = [op + op.dag() for op in sso.sc_ops]
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [1] * len(sops)
        elif len(sso.dW_factors) != len(sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == 'heterodyne':
        if sso.m_ops is None:
            m_ops = []
        sops = []
        for c in sso.sc_ops:
            if sso.m_ops is None:
                m_ops += [c + c.dag(), -1j * (c - c.dag())]
            sops += [c / np.sqrt(2), -1j / np.sqrt(2) * c]
        sso.m_ops = m_ops
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [np.sqrt(2)] * len(sops)
        elif len(sso.dW_factors) == len(sso.sc_ops):
            dW_factors = []
            for fact in sso.dW_factors:
                dW_factors += [np.sqrt(2) * fact, np.sqrt(2) * fact]
            sso.dW_factors = dW_factors
        elif len(sso.dW_factors) != len(sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")
    else:
        raise Exception("The method must be one of homodyne or heterodyne")

    LH = 1 - (sso.H * 1j * sso.dt)
    sso.pp = spre(sso.H) * 0
    sso.sops = []
    sso.preops = []
    sso.postops = []
    sso.preops2 = []
    sso.postops2 = []

    def _prespostdag(op):
        return spre(op) * spost(op.dag())

    for op in sso.c_ops:
        LH -= op._cdc() * sso.dt * 0.5
        sso.pp += op.apply(_prespostdag)._f_norm2() * sso.dt

    for i, op in enumerate(sops):
        LH -= op._cdc() * sso.dt * 0.5
        sso.sops += [(spre(op) + spost(op.dag())) * sso.dt]
        sso.preops += [spre(op)]
        sso.postops += [spost(op.dag())]
        for op2 in sops[i:]:
            sso.preops2 += [spre(op * op2)]
            sso.postops2 += [spost(op.dag() * op2.dag())]

    sso.ce_ops = [QobjEvo(spre(op)) for op in sso.e_ops]
    sso.cm_ops = [QobjEvo(spre(op)) for op in sso.m_ops]
    sso.preLH = spre(LH)
    sso.postLH = spost(LH.dag())

    sso.preLH.compile()
    sso.postLH.compile()
    sso.pp.compile()
    [op.compile() for op in sso.sops]
    [op.compile() for op in sso.preops]
    [op.compile() for op in sso.postops]
    [op.compile() for op in sso.preops2]
    [op.compile() for op in sso.postops2]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    sso.solver_obj = PmSMESolver
    sso.solver_name = "smesolve_" + sso.solver
    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def photocurrent_mesolve(H, rho0, times, c_ops=[], sc_ops=[], e_ops=[],
                         _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic master equation using the photocurrent method.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, or time dependent system.
        System Hamiltonian.
        Can depend on time, see StochasticSolverOptions help for format.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.
        Can depend on time, see StochasticSolverOptions help for format.

    sc_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.
        Can depend on time, see StochasticSolverOptions help for format.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
    if isket(rho0):
        rho0 = ket2dm(rho0)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptionsPhoto(True, H=H, state0=rho0, times=times,
                                       c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                                       args=args, **kwargs)

    if _safe_mode:
        _safety_checks(sso)

    if sso.m_ops is None:
        sso.m_ops = [op * 0 for op in sso.sc_ops]
    if not isinstance(sso.dW_factors, list):
        sso.dW_factors = [1] * len(sso.sc_ops)
    elif len(sso.dW_factors) != len(sso.sc_ops):
        raise Exception("The len of dW_factors is not the same as sc_ops")

    sso.solver_obj = PcSMESolver
    sso.solver_name = "photocurrent_mesolve"
    sso.LH = liouvillian(sso.H, c_ops=sso.c_ops) * sso.dt

    def _prespostdag(op):
        return spre(op) * spost(op.dag())

    sso.sops = [[spre(op._cdc()) + spost(op._cdc()),
                 spre(op._cdc()),
                 op.apply(_prespostdag)._f_norm2()] for op in sso.sc_ops]
    sso.ce_ops = [QobjEvo(spre(op)) for op in sso.e_ops]
    sso.cm_ops = [QobjEvo(spre(op)) for op in sso.m_ops]

    sso.LH.compile()
    [[op.compile() for op in ops] for ops in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)
    res.num_collapse = [np.count_nonzero(noise) for noise in res.noise]

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def photocurrent_sesolve(H, psi0, times, sc_ops=[], e_ops=[],
                         _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic schrodinger equation using the photocurrent method.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, or time dependent system.
        System Hamiltonian.
        Can depend on time, see StochasticSolverOptions help for format.

    psi0 : :class:`qutip.Qobj`
        Initial state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    sc_ops : list of :class:`qutip.Qobj`, or time dependent Qobjs.
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.
        Can depend on time, see StochasticSolverOptions help for format.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.
    """
    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = list(e_ops.values())
    else:
        e_ops_dict = None

    sso = StochasticSolverOptionsPhoto(False, H=H, state0=psi0, times=times,
                                       sc_ops=sc_ops, e_ops=e_ops,
                                       args=args, **kwargs)

    if _safe_mode:
        _safety_checks(sso)

    if sso.m_ops is None:
        sso.m_ops = [op * 0 for op in sso.sc_ops]
    if not isinstance(sso.dW_factors, list):
        sso.dW_factors = [1] * len(sso.sc_ops)
    elif len(sso.dW_factors) != len(sso.sc_ops):
        raise Exception("The len of dW_factors is not the same as sc_ops")

    sso.solver_obj = PcSSESolver
    sso.solver_name = "photocurrent_sesolve"
    sso.sops = [[op, op._cdc()] for op in sso.sc_ops]
    sso.LH = sso.H * (-1j*sso.dt)
    for ops in sso.sops:
        sso.LH -= ops[0]._cdc()*0.5*sso.dt
    sso.ce_ops = [QobjEvo(op) for op in sso.e_ops]
    sso.cm_ops = [QobjEvo(op) for op in sso.m_ops]

    sso.LH.compile()
    [[op.compile() for op in ops] for ops in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)
    res.num_collapse = [np.count_nonzero(noise) for noise in res.noise]

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def general_stochastic(state0, times, d1, d2, e_ops=[], m_ops=[],
                       _safe_mode=True, len_d2=1, args={}, **kwargs):
    """
    Solve stochastic general equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.


    Parameters
    ----------

    state0 : :class:`qutip.Qobj`
        Initial state vector (ket) or density matrix as a vector.

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    d1 : function, callable class
        Function representing the deterministic evolution of the system.

        def d1(time (double), state (as a np.array vector)):
            return 1d np.array

    d2 : function, callable class
        Function representing the stochastic evolution of the system.

        def d2(time (double), state (as a np.array vector)):
            return 2d np.array (N_sc_ops, len(state0))

    len_d2 : int
        Number of output vector produced by d2

    e_ops : list of :class:`qutip.Qobj`
        single operator or list of operators for which to evaluate
        expectation values.
        Must be a superoperator if the state vector is a density matrix.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`
        An instance of the class :class:`qutip.solver.Result`.
    """

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if "solver" not in kwargs:
        kwargs["solver"] = 50

    sso = StochasticSolverOptions(False, H=None, state0=state0, times=times,
                                  e_ops=e_ops, args=args, **kwargs)
    if sso.solver_code not in [50, 100, 150]:
        raise ValueError("Only Euler, platen, platen15 can be " +
                         "used for the general stochastic solver.")

    sso.d1 = d1
    sso.d2 = d2
    if _safe_mode:
        # This state0_vec is computed as mat2vec(state0.full()).ravel()
        # in the sso init.
        state0_vec = sso.rho0
        l_vec = state0_vec.shape[0]
        try:
            out_d1 = d1(0., sso.rho0)
        except Exception as e:
            raise RuntimeError("Safety check: d1(0., state0_vec) failed.:\n" +
                               str(e)) from e
        try:
            out_d2 = d2(0., sso.rho0)
        except Exception as e:
            raise RuntimeError("Safety check: d2(0., state0_vec) failed:\n" +
                               str(e)) from e

        msg_d1 = ("d1 must return an 1d numpy array with the same number "
                  "of elements as the initial state as a vector.")
        if not isinstance(out_d1, np.ndarray):
            raise TypeError(msg_d1)
        if (out_d1.ndim != 1
                or out_d1.shape[0] != l_vec or len(out_d1.shape) != 1):
            raise ValueError(msg_d1)

        msg_d2 = ("Safety check: d2 must return a 2d numpy array "
                  "with the shape (len_d2, len(state0_vec) ).")
        if not isinstance(out_d2, np.ndarray):
            raise TypeError(msg_d2)
        if (out_d2.ndim != 2
                or out_d2.shape[1] != l_vec or out_d2.shape[0] != len_d2):
            raise ValueError(msg_d2)
        if out_d1.dtype != np.dtype('complex128') or \
           out_d2.dtype != np.dtype('complex128'):
            raise ValueError("Safety check: d1 and d2 must return " +
                             "complex numpy array.")
        msg_e_ops = ("Safety check: The shape of the e_ops "
                     "does not fit the initial state.")
        for op in sso.e_ops:
            shape_op = op.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise ValueError(msg_e_ops)
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise ValueError(msg_e_ops +
                                     " Expecting e_ops as superoperators.")

    sso.m_ops = []
    sso.cm_ops = []
    if sso.store_measurement:
        if not m_ops:
            raise ValueError("General stochastic needs explicit " +
                             "m_ops to store measurement.")
        sso.m_ops = m_ops
        sso.cm_ops = [QobjEvo(op) for op in sso.m_ops]
        [op.compile() for op in sso.cm_ops]
        if sso.dW_factors is None:
            sso.dW_factors = [1.] * len(sso.m_ops)
        elif len(sso.dW_factors) == 1:
            sso.dW_factors = sso.dW_factors * len(sso.m_ops)
        elif len(sso.dW_factors) != len(sso.m_ops):
            raise ValueError("The number of dW_factors must fit" +
                             " the number of m_ops.")

    if sso.dW_factors is None:
        sso.dW_factors = [1.] * len_d2
    sso.sops = [None] * len_d2
    sso.ce_ops = [QobjEvo(op) for op in sso.e_ops]
    for op in sso.ce_ops:
        op.compile()

    sso.solver_obj = GenericSSolver
    sso.solver_name = "general_stochastic_solver_" + sso.solver

    ssolver = GenericSSolver()
    # ssolver.set_data(sso)
    ssolver.set_solver(sso)

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def _safety_checks(sso):
    l_vec = sso.rho0.shape[0]
    if sso.H.cte.issuper:
        if not sso.me:
            raise ValueError(
                "Given a Liouvillian for a Schrödinger equation problem."
            )
        shape_op = sso.H.cte.shape
        if shape_op[0] != l_vec or shape_op[1] != l_vec:
            raise Exception("The size of the hamiltonian does "
                            "not fit the initial state")
    else:
        shape_op = sso.H.cte.shape
        if sso.me:
            if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                raise Exception("The size of the hamiltonian does "
                                "not fit the initial state")
        else:
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the hamiltonian does "
                                "not fit the initial state")

    for op in sso.sc_ops:
        if op.cte.issuper:
            if not sso.me:
                raise ValueError(
                    "Given a Liouvillian for a Schrödinger equation problem."
                )
            shape_op = op.cte.shape
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the sc_ops does "
                                "not fit the initial state")
        else:
            shape_op = op.cte.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise Exception("The size of the sc_ops does "
                                    "not fit the initial state")
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise Exception("The size of the sc_ops does "
                                    "not fit the initial state")

    for op in sso.c_ops:
        if op.cte.issuper:
            if not sso.me:
                raise ValueError(
                    "Given a Liouvillian for a Schrödinger equation problem."
                )
            shape_op = op.cte.shape
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the c_ops does "
                                "not fit the initial state")
        else:
            shape_op = op.cte.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise Exception("The size of the c_ops does "
                                    "not fit the initial state")
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise Exception("The size of the c_ops does "
                                    "not fit the initial state")

    for op in sso.e_ops:
        shape_op = op.shape
        if sso.me:
            if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                raise Exception("The size of the e_ops does "
                                "not fit the initial state")
        else:
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the e_ops does "
                                "not fit the initial state")

    if sso.m_ops is not None:
        for op in sso.m_ops:
            shape_op = op.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise Exception("The size of the m_ops does "
                                    "not fit the initial state")
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise Exception("The size of the m_ops does "
                                    "not fit the initial state")


def _sesolve_generic(sso, options, progress_bar):
    """
    Internal function. See smesolve.
    """
    res = Result()
    res.times = sso.times
    res.expect = np.zeros((len(sso.e_ops), len(sso.times)), dtype=complex)
    res.ss = np.zeros((len(sso.e_ops), len(sso.times)), dtype=complex)
    res.measurement = []
    res.solver = sso.solver_name
    res.ntraj = sso.ntraj
    res.num_expect = len(sso.e_ops)

    nt = sso.ntraj
    task = _single_trajectory
    map_kwargs = {'progress_bar': sso.progress_bar}
    map_kwargs.update(sso.map_kwargs)
    task_args = (sso,)
    task_kwargs = {}

    results = sso.map_func(task, list(range(sso.ntraj)),
                           task_args, task_kwargs, **map_kwargs)
    noise = []
    for result in results:
        states_list, dW, m, expect = result
        if options.average_states or options.store_states:
            res.states.append(states_list)
        noise.append(dW)
        res.measurement.append(m)
        res.expect += expect
        res.ss += expect * expect
    res.noise = np.stack(noise)

    if sso.store_all_expect:
        paths_expect = []
        for result in results:
            paths_expect.append(result[3])
        res.runs_expect = np.stack(paths_expect)

    # average density matrices (vectorized maybe)
    # ajgpitch 2019-10-25: np.any(res.states) seems to error
    # I guess there may be a potential exception if there are no states?
    # store individual trajectory states
    if options.store_states:
        res.traj_states = res.states
    else:
        res.traj_states = None
    res.avg_states = None
    if options.average_states:
        avg_states_list = []
        for n in range(len(res.times)):
            if res.states[0][n].shape[1] == 1:
                tslot_states = [
                    res.states[mm][n].proj().data
                    for mm in range(nt)
                ]
            else:
                tslot_states = [res.states[mm][n].data for mm in range(nt)]
            if len(tslot_states) > 0:
                state = Qobj(np.sum(tslot_states),
                             dims=[res.states[0][n].dims[0]] * 2).unit()
                avg_states_list.append(state)
        # store average states
        res.states = res.avg_states = avg_states_list

    # average
    res.expect = res.expect / nt

    # standard error
    if nt > 1:
        res.se = (res.ss - nt * (res.expect ** 2)) / (nt * (nt - 1))
    else:
        res.se = None

    # convert complex data to real if hermitian
    res.expect = [np.real(res.expect[n, :])
                  if e.isherm else res.expect[n, :]
                  for n, e in enumerate(sso.e_ops)]

    return res


def _single_trajectory(i, sso):
    # Only one step?
    ssolver = sso.solver_obj()
    ssolver.set_solver(sso)
    result = ssolver.cy_sesolve_single_trajectory(i)
    return result


# The code for ssepdpsolve have been moved to the file pdpsolve.
# The call is still in stochastic for consistance.
def ssepdpsolve(H, psi0, times, c_ops, e_ops, **kwargs):
    """
    A stochastic (piecewse deterministic process) PDP solver for wavefunction
    evolution. For most purposes, use :func:`qutip.mcsolve` instead for quantum
    trajectory simulations.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.

    """
    return main_ssepdpsolve(H, psi0, times, c_ops, e_ops, **kwargs)


# The code for smepdpsolve have been moved to the file pdpsolve.
# The call is still in stochastic for consistance.
def smepdpsolve(H, rho0, times, c_ops, e_ops, **kwargs):
    """
    A stochastic (piecewse deterministic process) PDP solver for density matrix
    evolution.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix.

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`.

    """
    return main_smepdpsolve(H, rho0, times, c_ops, e_ops, **kwargs)
