__all__ = ['SolverOptions',
           'SolverResultsOptions','SolverRhsOptions','SolverOdeOptions',
           'McOptions']

from ..optionsclass import optionsclass
import multiprocessing

@optionsclass("solver")
class SolverOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------
    """
    options = {
        # (turned off for batch unitary propagator mode)
        "progress_bar": "text",
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "progress_kwargs": {"chunk_size":10},
    }


@optionsclass("ode", SolverOptions)
class SolverOdeOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------
    atol : float {1e-8}
        Absolute tolerance.
    rtol : float {1e-6}
        Relative tolerance.
    method : str {'adams','bdf'}
        Integration method.
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
    options = {
        # Absolute tolerance (default = 1e-8)
        "atol": 1e-8,
        # Relative tolerance (default = 1e-6)
        "rtol": 1e-6,
        # Integration method (default = 'adams', for stiff 'bdf')
        "method": 'adams',
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
        'ifactor': 6.0,
        'dfactor': 0.3,
        'beta': 0.0,
    }


@optionsclass("rhs", SolverOptions)
class SolverRhsOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------
    tidy : bool {True,False}
        Tidyup Hamiltonian and initial state by removing small terms.
    """
    options = {
        # tidyup Hamiltonian before calculation (default = True)
        "tidy": True,
        "Operator_data_type": "input",
        "State_data_type": "dense",
        # Normalize the states received in feedback_args
        "feedback_normalize": True,
        "ahs": False,
        "ahs_options": {
            "ahs_atol": 1e-8,
            "ahs_rtol": 1e-8,
            "ahs_padding": 6,
            "ahs_safety_rtol": 1e-5,
            "ahs_safety_interval": 4,
        }
    }


@optionsclass("results", SolverOptions)
class SolverResultsOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.
    store_states : bool {False, True}
        Whether or not to store the state vectors or density matrices in the
        result class, even if expectation values operators are given. If no
        expectation are provided, then states are stored by default and this
        option has no effect.
    """
    #average_states : bool {False}
    #    Average states values over trajectories in stochastic solvers.
    #average_expect : bool {True}
    #    Average expectation values over trajectories for stochastic solvers.
    options = {
        # store final state?
        "store_final_state": False,
        # store states even if expectation operators are given?
        "store_states": False,
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "normalize_output": "ket",

        # Average expectation values over trajectories (default = True)
        # "average_expect": True,
        # average expectation values
        # "average_states": False,
        # average mcsolver density matricies assuming steady state evolution
        # "steady_state_average": False,
    }


@optionsclass("mcsolve", SolverOptions)
class McOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(norm_tol=1e-3, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts['norm_tol'] = 1e-3

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
    """
    # mc_corr_eps : float {1e-10}
    #     Arbitrarily small value for eliminating any divide-by-zero errors in
    #     correlation calculations when using mcsolve.
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

        "map_options": {
            'num_cpus': multiprocessing.cpu_count(),
            'timeout':1e8,
            'job_timeout':1e8
        }
    }
