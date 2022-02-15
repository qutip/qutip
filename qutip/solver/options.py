__all__ = ['SolverOptions', 'SolverOdeOptions']

from ..optionsclass import QutipOptions
import multiprocessing


class SolverOptions(QutipOptions):
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(progress_bar='enhanced', ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts['progress_bar'] = 'enhanced'

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['progress_bar'] = 'enhanced'

    Options
    -------
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.

    store_states : bool {False, True, None}
        Whether or not to store the state vectors or density matrices.
        On `None` the states will be saved if no expectation operators are
        given.

    normalize_output : str {"", "ket", "all"}
        normalize output state to hide ODE numerical errors.
        "all" will normalize both ket and dm.
        On "ket", only 'ket' output are normalized.
        Leave empty for no normalization.

    norm_tol : float {1e-4}
        Tolerance used when finding wavefunction norm in mcsolve.

    norm_t_tol : float {1e-6}
        Tolerance used when finding wavefunction time in mcsolve.

    norm_steps : int {5}
        Max. number of steps used to find wavefunction norm to within norm_tol
        in mcsolve.

    keep_runs_results: bool
        Keep all trajectories results or save only the average.

    map : str  {'parallel', 'serial', 'loky'}
        How to run the trajectories.
        'parallel' use python's multiprocessing.
        'loky' use the pyhon module of the same name (not installed with qutip).

    map_options: dict
        keys:
            'num_cpus': number of cpus to use.
            'timeout': maximum time for all trajectories. (sec)
            'job_timeout': maximum time per trajectory. (sec)
        Only finished trajectories will be returned when timeout is reached.

    mc_corr_eps : float {1e-10}
        Arbitrarily small value for eliminating any divide-by-zero errors in
        correlation calculations when using mcsolve.

    progress_bar : str {'text', 'enhanced', 'tqdm', ''}
        How to present the solver progress.
        True will result in 'text'.
        'tqdm' uses the python module of the same name and raise an error if
        not installed.
        Empty string or False will disable the bar.

    progress_kwargs : dict
        kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
    """
    default = {}
    name = "Solver"
    _ode_options = {}

    def __init__(self, base=None, *, method=None, _strick=True, **options):
        if isinstance(base, dict):
            options.update(base)

        elif type(base) is SolverOptions:
            _strick = False
            opt = {
                key: val
                for key, val in base.options.items()
                if val is not None
            }
            options.update(base.options)

        elif type(base) is self.__class__:
            options.update(base.options)

        self._solver = None
        self.options = self.default.copy()
        self._from_dict(options)
        self.ode = method
        self.ode._from_dict(options)

        solver_keys = set(options) & set(self.default)
        ode_keys = set(options) & set(self.ode.default)
        leftoever = set(options) - solver_keys - ode_keys
        if _strick and leftoever:
            raise KeyError("Unknown option(s): " +
                           f"{set(options) - set(self.default)}")

    def __setitem__(self, key, value):
        self.options[key] = value
        if key == 'method':
            self.ode = value
        if self._solver:
            # Tell solver the options were updated
            self._solver.options = self

    def __str__(self):
        out = self.name + ":\n"
        longest = max(len(key) for key in self.options)
        for key, val in self.options.items():
            if isinstance(val, str):
                out += "{:{width}} : '{}'\n".format(key, val, width=longest)
            else:
                out += "{:{width}} : {}\n".format(key, val, width=longest)
        out += "\n"
        out += str(self.ode)
        return out

    @property
    def ode(self):
        return self._ode

    @ode.setter
    def ode(self, new):
        if isinstance(new, SolverOdeOptions):
            self._ode = new
        else:
            method = new or self.default.get('method', None)
            self._ode = self._ode_options.get(new, SolverOdeOptions)()

        self.options['method'] = self._ode['method']
        self._ode._parent = self
        if self._solver:
            # Tell solver the options were updated
            self._solver.options = self


class SolverOdeOptions(QutipOptions):
    """
    Class of options for the ODE integrator of solvers such as
    :func:`qutip.mesolve` and :func:`qutip.mcsolve`. Options can be
    specified either as arguments to the SolverOptions constructor::

        opts = SolverOptions(method=bdf, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.ode['method'] = 'bdf'

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver.ode['method'] = 'bdf'

    Options
    -------
    method : str {'adams', 'bdf', 'dop853', 'lsoda', ...}
        Integration method.

    atol : float {1e-8}
        Absolute tolerance.

    rtol : float {1e-6}
        Relative tolerance.

    order : int {12}
        Order of integrator (<=12 'adams', <=5 'bdf')

    nsteps : int {2500}
        Max. number of internal steps/call.

    first_step : float {0}
        Size of initial step (0 = automatic).

    min_step : float {0}
        Minimum step size (0 = automatic).

    max_step : float {0}
        Maximum step size (0 = automatic)
    """
    default = {}
    _parent = None  # Instance of SolverOptions that contain this instance.

    def __setitem__(self, key, value):
        self.options[key] = value
        if self._parent:
            # Tell solver the options were updated
            self._parent.ode = self
