__all__ = [
    'correlation_3op',
    'correlation_2op_1t', 'correlation_2op_2t', 'correlation_3op_1t',
    'correlation_3op_2t', 'coherence_function_g1', 'coherence_function_g2'
]

import numpy as np

from ..core import (
    Qobj, QobjEvo, liouvillian, spre, unstack_columns, stack_columns,
    tensor, expect, qeye_like, isket
)
from .mesolve import MESolver
from .mcsolve import MCSolver
from .brmesolve import BRSolver
from .heom.bofin_solvers import HEOMSolver

from .steadystate import steadystate
from ..ui.progressbar import progress_bars
from .parallel import serial_map, _maps


# -----------------------------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------------------------

def _corr_3op_1row(task_data, solver, B, n_tau):
    """
    Compute one row of the 3-operator 2-time correlation matrix.

    Defined at module level so it can be pickled for parallel execution.

    Parameters
    ----------
    task_data : tuple
        ``(rho_modified, taulist_shifted, n_compute)`` where

        - *rho_modified* is ``C(t) @ rho(t) @ A(t)``
        - *taulist_shifted* is ``taulist[:n_compute] + t``
        - *n_compute* is how many tau points to actually evaluate

    solver : :class:`.MESolver` or :class:`.BRSolver`
        Solver with options already configured (``normalize_output=False``,
        ``progress_bar=False``).
    B : :class:`.QobjEvo`
        Operator whose expectation value is measured.
    n_tau : int
        Full length of taulist (for zero-padding skipped entries).

    Returns
    -------
    row : ndarray of complex
        Correlation values, length *n_tau*.
    """
    rho_modified, taulist_shifted, n_compute = task_data
    row = np.zeros(n_tau, dtype=complex)
    if n_compute > 0:
        row[:n_compute] = solver.run(
            rho_modified, taulist_shifted, e_ops=B
        ).expect[0]
    return row


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

# low level correlation


def correlation_2op_1t(H, state0, taulist, c_ops, a_op, b_op,
                       solver="me", reverse=False, args=None,
                       options=None):
    r"""
    Calculate the two-operator one-time correlation function:
    :math:`\left<A(\tau)B(0)\right>`
    along one time axis using the quantum regression theorem and the evolution
    solver indicated by the `solver` parameter.

    Parameters
    ----------

    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        System Hamiltonian, may be time-dependent for solver choice of `me`.
    state0 : :obj:`.Qobj`
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        if ``c_ops`` are provided and the Hamiltonian is constant.
    taulist : array_like
        List of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list of {:obj:`.Qobj`, :obj:`.QobjEvo`}
        List of collapse operators
    a_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator A.
    b_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator B.
    reverse : bool, default: False
        If ``True``, calculate :math:`\left<A(t)B(t+\tau)\right>` instead of
        :math:`\left<A(t+\tau)B(t)\right>`.
    solver : str {'me', 'es'}, default: 'me'
        Choice of solver, ``me`` for master-equation, and ``es`` for
        exponential series. ``es`` is equivalent to `me` with
        ``options={"method": "diag"}``.
    options : dict, optional
        Options for the solver.

    Returns
    -------
    corr_vec : ndarray
        An array of correlation values for the times specified by ``taulist``.

    See Also
    --------
    :func:`correlation_3op` :
        Similar function supporting various solver types.

    References
    ----------
    See, Gardiner, Quantum Noise, Section 5.2.

    """
    solver = _make_solver(H, c_ops, args, options, solver)

    if reverse:
        A_op, B_op, C_op = a_op, b_op, 1
    else:
        A_op, B_op, C_op = 1, a_op, b_op
    if state0 is None:
        state0 = steadystate(H, c_ops)

    return correlation_3op(solver, state0, [0], taulist, A_op, B_op, C_op)[0]


def correlation_2op_2t(H, state0, tlist, taulist, c_ops, a_op, b_op,
                       solver="me", reverse=False, args=None,
                       options=None, *,
                       max_t_plus_tau=None, map='serial', map_kw=None):
    r"""
    Calculate the two-operator two-time correlation function:
    :math:`\left<A(t+\tau)B(t)\right>`
    along two time axes using the quantum regression theorem and the
    evolution solver indicated by the ``solver`` parameter.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        System Hamiltonian, may be time-dependent for solver choice of `me`.
    state0 : :obj:`.Qobj`
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        if ``c_ops`` are provided and the Hamiltonian is constant.
    tlist : array_like
        List of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one ``tlist``
        value is necessary, i.e. when :math:`t \rightarrow \infty`.
        If ``tlist`` is ``None``, ``tlist=[0]`` is assumed.
    taulist : array_like
        List of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list of {:obj:`.Qobj`, :obj:`.QobjEvo`}
        List of collapse operators
    a_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator A.
    b_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator B.
    reverse : bool, default: False
        If ``True``, calculate :math:`\left<A(t)B(t+\tau)\right>` instead of
        :math:`\left<A(t+\tau)B(t)\right>`.
    solver : str {'me', 'es'}, default: 'me'
        Choice of solver, ``me`` for master-equation, and ``es`` for
        exponential series. ``es`` is equivalent to `me` with
        ``options={"method": "diag"}``.
    options : dict, optional
        Options for the solver.
    max_t_plus_tau : float, optional
        If provided, skip computation where ``t + tau > max_t_plus_tau``.
        Skipped entries are filled with ``0``. Default ``None`` means compute
        all entries (equivalent to ``np.inf``).
    map : str, default: ``'serial'``
        How to run the loop over *tlist*. A string is looked up in
        ``qutip.solver.parallel._maps`` (e.g. ``'serial'``,
        ``'parallel'``, ``'loky'``).
    map_kw : dict, optional
        Keyword arguments passed to the map function via its ``map_kw``
        parameter, e.g. ``{'num_cpus': 4}``.

    Returns
    -------
    corr_mat : ndarray
        An 2-dimensional array (matrix) of correlation values for the times
        specified by ``tlist`` (first index) and ``taulist`` (second index).

    See Also
    --------
    :func:`correlation_3op` :
        Similar function supporting various solver types.

    References
    ----------
    See, Gardiner, Quantum Noise, Section 5.2.

    """
    solver = _make_solver(H, c_ops, args, options, solver)
    if tlist is None:
        tlist = [0]
    if state0 is None:
        state0 = steadystate(H, c_ops)

    if reverse:
        A_op, B_op, C_op = a_op, b_op, 1
    else:
        A_op, B_op, C_op = 1, a_op, b_op

    return correlation_3op(solver, state0, tlist, taulist, A_op, B_op, C_op,
                           max_t_plus_tau=max_t_plus_tau,
                           map=map, map_kw=map_kw)


def correlation_3op_1t(H, state0, taulist, c_ops, a_op, b_op, c_op,
                       solver="me", args=None, options=None):
    r"""
    Calculate the three-operator two-time correlation function:
    :math:`\left<A(0)B(\tau)C(0)\right>` along one time axis using the
    quantum regression theorem and the evolution solver indicated by the
    `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\tau<0`.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        System Hamiltonian, may be time-dependent for solver choice of ``me``.
    state0 : :obj:`.Qobj`
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        if ``c_ops`` are provided and the Hamiltonian is constant.
    taulist : array_like
        List of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list of {:obj:`.Qobj`, :obj:`.QobjEvo`}
        List of collapse operators
    a_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator A.
    b_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator B.
    c_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator C.
    solver : str {'me', 'es'}, default: 'me'
        Choice of solver, ``me`` for master-equation, and ``es`` for
        exponential series. ``es`` is equivalent to `me` with
        ``options={"method": "diag"}``.
    options : dict, optional
        Options for the solver.

    Returns
    -------
    corr_vec : array
        An array of correlation values for the times specified by ``taulist``.

    See Also
    --------
    :func:`correlation_3op` :
        Similar function supporting various solver types.

    References
    ----------
    See, Gardiner, Quantum Noise, Section 5.2.

    """
    solver = _make_solver(H, c_ops, args, options, solver)
    if state0 is None:
        state0 = steadystate(H, c_ops)
    return correlation_3op(solver, state0, [0], taulist, a_op, b_op, c_op)[0]


def correlation_3op_2t(H, state0, tlist, taulist, c_ops, a_op, b_op, c_op,
                       solver="me", args=None, options=None, *,
                       max_t_plus_tau=None, map='serial', map_kw=None):
    r"""
    Calculate the three-operator two-time correlation function:
    :math:`\left<A(t)B(t+\tau)C(t)\right>` along two time axes using the
    quantum regression theorem and the evolution solver indicated by the
    `solver` parameter.

    Note: it is not possibly to calculate a physically meaningful correlation
    of this form where :math:`\tau<0`.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        System Hamiltonian, may be time-dependent for solver choice of ``me``.
    state0 : :obj:`.Qobj`
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        if ``c_ops`` are provided and the Hamiltonian is constant.
    tlist : array_like
        List of times for :math:`t`. tlist must be positive and contain the
        element `0`. When taking steady-steady correlations only one tlist
        value is necessary, i.e. when :math:`t \rightarrow \infty`.
        If ``tlist`` is ``None``, ``tlist=[0]`` is assumed.
    taulist : array_like
        List of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list of {:obj:`.Qobj`, :obj:`.QobjEvo`}
        List of collapse operators
    a_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator A.
    b_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator B.
    c_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator C.
    solver : str {'me', 'es'}, default: 'me'
        Choice of solver, ``me`` for master-equation, and ``es`` for
        exponential series. ``es`` is equivalent to `me` with
        ``options={"method": "diag"}``.
    options : dict, optional
        Options for the solver. Only used with ``me`` solver.
    max_t_plus_tau : float, optional
        If provided, skip computation where ``t + tau > max_t_plus_tau``.
        Skipped entries are filled with ``0``. Default ``None`` means compute
        all entries (equivalent to ``np.inf``).
    map : str, default: ``'serial'``
        How to run the loop over *tlist*. A string is looked up in
        ``qutip.solver.parallel._maps`` (e.g. ``'serial'``,
        ``'parallel'``, ``'loky'``).
    map_kw : dict, optional
        Keyword arguments passed to the map function via its ``map_kw``
        parameter, e.g. ``{'num_cpus': 4}``.

    Returns
    -------
    corr_mat : array
        An 2-dimensional array (matrix) of correlation values for the times
        specified by ``tlist`` (first index) and ``taulist`` (second index).

    See Also
    --------
    :func:`correlation_3op` :
        Similar function supporting various solver types.

    References
    ----------

    See, Gardiner, Quantum Noise, Section 5.2.

    """
    solver = _make_solver(H, c_ops, args, options, solver)

    if tlist is None:
        tlist = [0]
    if state0 is None:
        state0 = steadystate(H, c_ops)

    return correlation_3op(solver, state0, tlist, taulist, a_op, b_op, c_op,
                           max_t_plus_tau=max_t_plus_tau,
                           map=map, map_kw=map_kw)


# high level correlation

def coherence_function_g1(
    H, state0, taulist, c_ops, a_op, solver="me", args=None, options=None
):
    r"""
    Calculate the normalized first-order quantum coherence function:

    .. math::

        g^{(1)}(\tau) =
        \frac{\langle A^\dagger(\tau)A(0)\rangle}
        {\sqrt{\langle A^\dagger(\tau)A(\tau)\rangle
                \langle A^\dagger(0)A(0)\rangle}}

    using the quantum regression theorem and the evolution solver indicated by
    the `solver` parameter.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        System Hamiltonian, may be time-dependent for solver choice of ``me``.
    state0 : :obj:`.Qobj`
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        if ``c_ops`` are provided and the Hamiltonian is constant.
    taulist : array_like
        List of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list of {:obj:`.Qobj`, :obj:`.QobjEvo`}
        List of collapse operators
    a_op : :obj:`.Qobj`, :obj:`.QobjEvo`
        Operator A.
    solver : str {'me', 'es'}, default: 'me'
        Choice of solver, ``me`` for master-equation, and ``es`` for
        exponential series. ``es`` is equivalent to `me` with
        ``options={"method": "diag"}``.
    args : dict, optional
        dictionary of parameters for time-dependent Hamiltonians
    options : dict, optional
        Options for the solver.

    Returns
    -------
    g1, G1 : tuple
        The normalized and unnormalized second-order coherence function.

    """
    solver = _make_solver(H, c_ops, args, options, solver)

    # first calculate the photon number
    if state0 is None:
        state0 = steadystate(H, c_ops)
        n = np.array([expect(state0, a_op.dag() * a_op)])
    else:
        n = solver.run(state0, taulist, e_ops=[a_op.dag() * a_op]).expect[0]

    # calculate the correlation function G1 and normalize with n to obtain g1
    G1 = correlation_3op(solver, state0, [0], taulist,
                         None, a_op.dag(), a_op)[0]

    g1 = G1 / np.sqrt(n[0] * np.array(n))[0]
    return g1, G1


def coherence_function_g2(H, state0, taulist, c_ops, a_op, solver="me",
                          args=None, options=None):
    r"""
    Calculate the normalized second-order quantum coherence function:

    .. math::

         g^{(2)}(\tau) =
        \frac{\langle A^\dagger(0)A^\dagger(\tau)A(\tau)A(0)\rangle}
        {\langle A^\dagger(\tau)A(\tau)\rangle
         \langle A^\dagger(0)A(0)\rangle}

    using the quantum regression theorem and the evolution solver indicated by
    the `solver` parameter.

    Parameters
    ----------
    H : :obj:`.Qobj`, :obj:`.QobjEvo`
        System Hamiltonian, may be time-dependent for solver choice of ``me``.
    state0 : :obj:`.Qobj`
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`. If 'state0' is 'None', then the steady state will
        be used as the initial state. The 'steady-state' is only implemented
        if ``c_ops`` are provided and the Hamiltonian is constant.
    taulist : array_like
        List of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    c_ops : list
        List of collapse operators, may be time-dependent for solver choice of
        ``me``.
    a_op : :obj:`.Qobj`
        Operator A.
    args : dict, optional
        Dictionary of arguments to be passed to solver.
    solver : str {'me', 'es'}, default: 'me'
        Choice of solver, ``me`` for master-equation, and ``es`` for
        exponential series. ``es`` is equivalent to ``me`` with
        ``options={"method": "diag"}``.
    options : dict, optional
        Options for the solver.

    Returns
    -------
    g2, G2 : tuple
        The normalized and unnormalized second-order coherence function.

    """
    solver = _make_solver(H, c_ops, args, options, solver)

    # first calculate the photon number
    if state0 is None:
        state0 = steadystate(H, c_ops)
        n = np.array([expect(state0, a_op.dag() * a_op)])
    else:
        n = solver.run(state0, taulist, e_ops=[a_op.dag() * a_op]).expect[0]

    # calculate the correlation function G2 and normalize with n to obtain g2
    G2 = correlation_3op(solver, state0, [0], taulist,
                         a_op.dag(), a_op.dag() * a_op, a_op)[0]

    g2 = G2 / (n[0] * np.array(n))
    return g2, G2


def _make_solver(H, c_ops, args, options, solver):
    H = QobjEvo(H, args=args)
    c_ops = [QobjEvo(c_op, args=args) for c_op in c_ops]
    if solver == "me":
        solver_instance = MESolver(H, c_ops, options=options)
    elif solver == "es":
        options = {"method": "diag"}
        solver_instance = MESolver(H, c_ops, options=options)
    elif solver == "mc":
        raise ValueError("MC solver for correlation has been removed")
    return solver_instance


def correlation_3op(solver, state0, tlist, taulist, A=None, B=None, C=None, *,
                    max_t_plus_tau=None, map='serial', map_kw=None):
    r"""
    Calculate the three-operator two-time correlation function:

        :math:`\left<A(t)B(t+\tau)C(t)\right>`.

    from a open system :class:`.Solver`.

    Note: it is not possible to calculate a physically meaningful correlation
    where :math:`\tau<0`.

    Parameters
    ----------
    solver : :class:`.MESolver`, :class:`.BRSolver`
        Qutip solver for an open system.
    state0 : :obj:`.Qobj`
        Initial state density matrix :math:`\rho(t_0)` or state vector
        :math:`\psi(t_0)`.
    tlist : array_like
        List of times for :math:`t`. tlist must be positive and contain the
        element `0`.
    taulist : array_like
        List of times for :math:`\tau`. taulist must be positive and contain
        the element `0`.
    A, B, C : :class:`.Qobj`, :class:`.QobjEvo`, optional, default=None
        Operators ``A``, ``B``, ``C`` from the equation
        ``<A(t)B(t+\tau)C(t)>`` in the Schrodinger picture. They do not need
        to be all provided. For exemple, if ``A`` is not provided,
        ``<B(t+\tau)C(t)>`` is computed.
    max_t_plus_tau : float, optional
        Maximum value of ``t + tau`` to compute. If provided, entries where
        ``t + tau > max_t_plus_tau`` are skipped and filled with ``0``.
        This is useful for reducing computation when correlations decay
        quickly and long-time behavior is not needed.
        Default ``None`` means compute all entries (equivalent to ``np.inf``).
    map : str, default: ``'serial'``
        How to run the loop over *tlist*. A string is looked up in
        ``qutip.solver.parallel._maps`` (e.g. ``'serial'``,
        ``'parallel'``, ``'loky'``). Use ``'parallel'`` for multi-core
        parallelization to speed up computation.
    map_kw : dict, optional
        Keyword arguments passed to the map function via its ``map_kw``
        parameter, e.g. ``{'num_cpus': 4}``.

    Returns
    -------
    corr_mat : array
        An 2-dimensional array (matrix) of correlation values for the times
        specified by ``tlist`` (first index) and `taulist` (second index). If
        ``tlist`` is ``None``, then a 1-dimensional array of correlation values
        is returned instead.

    Notes
    -----
    **Performance Optimization:**

    This function can be computationally expensive for large `tlist` and
    `taulist`. Two strategies can help reduce computation time:

    1. **Limit computation with ``max_t_plus_tau``:**
       If correlations decay quickly, you can skip computing entries where
       ``t + tau`` is large. For example, if you only need correlations
       up to total time ``T_max``::

           corr = correlation_3op(solver, rho0, tlist, taulist, A, B, C,
                                  max_t_plus_tau=T_max)

    2. **Parallel execution with ``map``:**
       Use multiple CPU cores to parallelize the computation::

           corr = correlation_3op(solver, rho0, tlist, taulist, A, B, C,
                                  map='parallel', map_kw={'num_cpus': 4})

    These options can be combined for maximum performance::

        corr = correlation_3op(solver, rho0, tlist, taulist, A, B, C,
                               max_t_plus_tau=T_max,
                               map='parallel', map_kw={'num_cpus': 4})

    Examples
    --------
    Compute a simple two-time correlation:

    >>> import numpy as np  # doctest: +SKIP
    >>> from qutip import sigmam, sigmap, basis, sigmax, correlation_3op  # doctest: +SKIP
    >>> from qutip.solver import MESolver  # doctest: +SKIP
    >>> H = 0.5 * 2 * np.pi * sigmax()  # doctest: +SKIP
    >>> c_ops = [np.sqrt(0.1) * sigmam()]  # doctest: +SKIP
    >>> solver = MESolver(H, c_ops)  # doctest: +SKIP
    >>> rho0 = basis(2, 0)  # doctest: +SKIP
    >>> tlist = np.linspace(0, 10, 100)  # doctest: +SKIP
    >>> taulist = np.linspace(0, 5, 50)  # doctest: +SKIP
    >>> corr = correlation_3op(solver, rho0, tlist, taulist,  # doctest: +SKIP
    ...                         A=sigmap(), B=sigmam(), C=sigmap())  # doctest: +SKIP

    Compute with performance optimization:

    >>> # Only compute up to t + tau = 12  # doctest: +SKIP
    >>> corr = correlation_3op(solver, rho0, tlist, taulist,  # doctest: +SKIP
    ...                         A=sigmap(), B=sigmam(), C=sigmap(),  # doctest: +SKIP
    ...                         max_t_plus_tau=12.0)  # doctest: +SKIP
    >>>  # doctest: +SKIP
    >>> # Use 4 CPU cores for parallel computation  # doctest: +SKIP
    >>> corr = correlation_3op(solver, rho0, tlist, taulist,  # doctest: +SKIP
    ...                         A=sigmap(), B=sigmam(), C=sigmap(),  # doctest: +SKIP
    ...                         map='parallel', map_kw={'num_cpus': 4})  # doctest: +SKIP
    """
    taulist = np.asarray(taulist)
    if isket(state0):
        state0 = state0.proj()

    A = QobjEvo(qeye_like(state0) if A in [None, 1] else A)
    B = QobjEvo(qeye_like(state0) if B in [None, 1] else B)
    C = QobjEvo(qeye_like(state0) if C in [None, 1] else C)

    map_func = _maps[map]

    if isinstance(solver, (MESolver, BRSolver)):
        out = _correlation_3op_dm(solver, state0, tlist, taulist, A, B, C,
                                  max_t_plus_tau=max_t_plus_tau,
                                  map_func=map_func, map_kw=map_kw)
    elif isinstance(solver, MCSolver):
        raise TypeError("Monte Carlo support for correlation was removed. "
                        "Please, tell us on GitHub issues if you need it!")
    elif isinstance(solver, HEOMSolver):
        raise TypeError("HEOM is not supported by correlation. "
                        "Please, tell us on GitHub issues if you need it!")
    else:
        raise TypeError("Only solvers able to evolve density matrices"
                        " are supported.")

    return out


def _correlation_3op_dm(solver, state0, tlist, taulist, A, B, C,
                        max_t_plus_tau=None, map_func=serial_map,
                        map_kw=None):
    """
    Internal worker for :func:`correlation_3op` using density-matrix solvers.
    """
    n_tau = np.size(taulist)
    old_opt = solver.options.copy()
    try:
        # We don't want to modify the solver
        # TODO: Solver could have a ``with`` or ``copy``.
        solver.options["normalize_output"] = False
        solver.options["progress_bar"] = False

        rho_t = solver.run(state0, tlist).states

        tasks = []
        for t_idx, rho in enumerate(rho_t):
            t = tlist[t_idx]
            rho_modified = C(t) @ rho @ A(t)

            if max_t_plus_tau is not None:
                n_compute = int(np.searchsorted(
                    taulist, max_t_plus_tau - t, side='right'
                ))
                taulist_shifted = taulist[:n_compute] + t
            else:
                n_compute = n_tau
                taulist_shifted = taulist + t

            tasks.append((rho_modified, taulist_shifted, n_compute))

        results = map_func(
            _corr_3op_1row,
            tasks,
            task_args=(solver, B, n_tau),
            progress_bar=old_opt['progress_bar'],
            progress_bar_kwargs=old_opt['progress_kwargs'],
            map_kw=map_kw,
        )

        corr_mat = np.array(results, dtype=complex)

    finally:
        solver.options = old_opt

    return corr_mat