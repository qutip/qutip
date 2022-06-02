__all__ = ['SolverOptions',
           'SolverResultsOptions', 'SolverOdeOptions',
           'McOptions']

from ..optionsclass import optionsclass


@optionsclass("solver")
class SolverOptions:
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
    normalize_output : str {"", "dm", "ket", "all", True}
        Whether to normalize the output state to hide ODE numerical errors.
        The values "all" and True will normalize both ket and dm states.
        The values "ket" and "dm" will only normalize states of the specified
        type. Leave blank for no normalization.

    progress_bar : str {'text', 'enhanced', 'tqdm', ''}
        How to present the solver progress.
        True will result in 'text'.
        'tqdm' uses the python module of the same name and raise an error if
        not installed.
        Empty string or False will disable the bar.

    progress_kwargs : dict
        kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
    """
    options = {
        # Normalize output of solvers
        "normalize_output": "ket",
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size":10},
    }


@optionsclass("ode", SolverOptions)
class SolverOdeOptions:
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

    state_data_type: str {"dense"}
        Name of the data type of the state used during the ODE evolution.
        Use an empty string to keep the input state type. Many integrator can
        only work with `Dense`.

    feedback_normalize: bool
        Normalize the state before passing it to coefficient when using
        feedback.
    """
    options = {
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
        # data type to use for the operator
        "operator_data_type": "",
        # data type to use for the state
        "state_data_type": "dense",
        # Normalize the states received in feedback_args
        "feedback_normalize": True,
    }
    extra_options = set()


@optionsclass("results", SolverOptions)
class SolverResultsOptions:
    """
    Class of options for Results of evolution solvers such as
    :func:`qutip.mesolve` and :func:`qutip.mcsolve`.
    Options can be specified when constructing SolverOptions

        opts = SolverOptions(store_final_state=True, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.results["store_final_state"] = True

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver.result['store_final_state'] = True

    Options
    -------
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.

    store_states : bool {False, True, None}
        Whether or not to store the state vectors or density matrices.
        On `None` the states will be saved if no expectation operators are
        given.
    """
    options = {
        # store final state?
        "store_final_state": False,
        # store states even if expectation operators are given?
        "store_states": None,
    }


@optionsclass("mcsolve", SolverOptions)
class McOptions:
    """
    Class of options specific for :func:`qutip.mcsolve`.
    Options can be specified either as arguments to the constructor of
    SolverOptions::

        opts = SolverOptions(norm_tol=1e-3, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.mcsolve['norm_tol'] = 1e-3

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.options.mcsolve['norm_tol'] = 1e-3

    Options
    -------

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
            'timeout': maximum time (sec) for all trajectories. ``None`` will
                use the maximum timeout value.
            'job_timeout': maximum time (sec) per trajectory. ``None`` will
                use the maximum timeout value.
        Only finished trajectories will be returned when timeout is reached.

    mc_corr_eps : float {1e-10}
        Arbitrarily small value for eliminating any divide-by-zero errors in
        correlation calculations when using mcsolve.
    """
    options = {
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

        "map_options": {
            'num_cpus': None,
            'timeout': None,
            'job_timeout': None,
        },
    }
