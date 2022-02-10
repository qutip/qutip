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
    default = {
        # (turned off for batch unitary propagator mode)
        "progress_bar": "text",
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "progress_kwargs": {"chunk_size":10},
        # store final state?
        "store_final_state": False,
        # store states even if expectation operators are given?
        "store_states": None,
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "normalize_output": "ket",
        # Tolerance for wavefunction norm (mcsolve only)
        "norm_tol": 1e-4,
        # Tolerance for collapse time precision (mcsolve only)
        "norm_t_tol": 1e-6,
        # Max. number of steps taken to find wavefunction norm to within
        # norm_tol (mcsolve only)
        "norm_steps": 5,

        "map": "parallel_map",

        "keep_runs_results": False,

        "mc_corr_eps": 1e-10,

        'num_cpus': multiprocessing.cpu_count(),

        'job_timeout': 1e8,

        'method': 'adams',
    }
    name = "Solver"

    def __init__(self, base=None, *, method=None, _strick=True, **options):
        if isinstance(method, SolverOdeOptions):
            self.ode = method
        else:
            method = method or self.default['method']
            self.ode = SolverOdeOptions()

        if isinstance(base, dict):
            options.update(base)

        elif isinstance(base, QutipOptions):
            options.update(base.options)

        solver_keys = set(options) & set(self.default)
        ode_keys = set(options) & set(self.ode.default)
        leftoever = set(options) - solver_keys - ode_keys
        if _strick and leftoever:
            raise KeyError("Unknown option(s): " +
                           f"{set(options) - set(self.default)}")
        self.ode._from_dict(options)
        self.options = self.default.copy()
        self._from_dict(options)


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
    method : str {'adams', 'bdf', 'dop853', 'lsoda', 'vern7', 'vern9', 'diag'}
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

    tidy: bool {True}
        tidyup Hamiltonian before calculation

    operator_data_type: str {""}
        Data type of the operator to used during the ODE evolution, such as
        'CSR' or 'Dense'. Use an empty string to keep the input state type.

    state_data_type: str {'dense'}
        Name of the data type of the state used during the ODE evolution.
        Use an empty string to keep the input state type. Many integrator can
        only work with `Dense`.

    feedback_normalize: bool
        Normalize the state before passing it to coefficient when using
        feedback.
    """
    default = {
        # Integration method (default = 'adams', for stiff 'bdf')
        "method": 'adams',

        "rhs": '',

        # Absolute tolerance (default = 1e-8)
        "atol": 1e-8,
        # Relative tolerance (default = 1e-6)
        "rtol": 1e-6,
        # Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        "order": 12,
        # Max. number of internal steps/call
        "nsteps": 2500,
        # Size of initial step (0 = determined by solver)
        "first_step": 0,
        # Max step size (0 = determined by solver)
        "max_step": 0,
        # Minimal step size (0 = determined by solver)
        "min_step": 0,
        # tidyup Hamiltonian before calculation (default = True)
        "tidy": True,

        "operator_data_type": "",

        "state_data_type": "dense",
        # Normalize the states received in feedback_args
        "feedback_normalize": True,
    }
    extra_options = set()
