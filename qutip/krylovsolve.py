__all__ = ["krylovsolve"]
"""
This module provides approximations of the time evolution operator
using small dimensional Krylov subspaces.
"""

from scipy.optimize import root_scalar
from math import ceil
import numpy as np
import warnings

from qutip.expect import expect
from qutip.qobj import Qobj
from qutip.solver import Result, Options
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.sparse import eigh


def krylovsolve(
    H: Qobj,
    psi0: Qobj,
    tlist: np.array,
    krylov_dim: int,
    e_ops=None,
    options=None,
    progress_bar: bool = None,
    sparse: bool = False,
):
    """
    Time evolution of state vectors for time independent Hamiltonians.
    Evolve the state vector ("psi0") finding an approximation for the time
    evolution operator of Hamiltonian ("H") by obtaining the projection of
    the time evolution operator on a set of small dimensional Krylov
    subspaces (m << dim(H)).

    The output is either the state vector or the expectation values of
    supplied operators ("e_ops") at arbitrary points at ("tlist").

    **Additional options**

    Additional options to krylovsolve can be set with the following:

    * "store_states": stores states even though expectation values are
      requested via the "e_ops" argument.

    * "store_final_state": store final state even though expectation values are
      requested via the "e_ops" argument.

    Parameters
    ----------
    H : :class:`qutip.Qobj`
        System Hamiltonian.
    psi0 : :class: `qutip.Qobj`
        Initial state vector (ket).
    tlist : None / *list* / *array*
        List of times on which to evolve the initial state. If None, nothing
        happens but the code won't break.
    krylov_dim: int
        Dimension of Krylov approximation subspaces used for the time
        evolution approximation.
    e_ops : None / list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.
    options : Options
        Instance of ODE solver options, as well as krylov parameters.
            atol: controls (approximately) the error desired for the final
                  solution. (Defaults to 1e-8)
            nsteps: maximum number of krylov's internal number of Lanczos
                    iterations. (Defaults to 10000)
    progress_bar : None / BaseProgressBar
         Optional instance of BaseProgressBar, or a subclass thereof, for
         showing the progress of the simulation.
    sparse : bool (default False)
         Use np.array to represent system Hamiltonians. If True, scipy sparse
         arrays are used instead.

    Returns
    -------
    result: :class:`qutip.Result`
        An instance of the class :class:`qutip.Result`, which contains
        either an *array* `result.expect` of expectation values for the times
        `tlist`, or an *array* `result.states` of state vectors corresponding
        to the times `tlist` [if `e_ops` is an empty list].
    """
    # check the physics
    _check_inputs(H, psi0, krylov_dim)

    # check extra inputs
    e_ops, e_ops_dict = _check_e_ops(e_ops)
    pbar = _check_progress_bar(progress_bar)

    # transform inputs type from Qobj to np.ndarray/csr_matrix
    if sparse:
        _H = H.get_data()  # (fast_) csr_matrix
    else:
        _H = H.full().copy()  # np.ndarray
    _psi = psi0.full().copy()
    _psi = _psi / np.linalg.norm(_psi)

    # create internal variable and output containers
    if options is None:
        options = Options(nsteps=10000)
    krylov_results = Result()
    krylov_results.solver = "krylovsolve"

    # handle particular cases of an empty tlist or single element
    n_tlist_steps = len(tlist)
    if n_tlist_steps < 1:
        return krylov_results

    if n_tlist_steps == 1:  # if tlist has only one element, return it
        krylov_results = particular_tlist_or_happy_breakdown(
            tlist, n_tlist_steps, options, psi0, e_ops, krylov_results, pbar
        )  # this will also raise a warning
        return krylov_results

    tf = tlist[-1]
    t0 = tlist[0]

    # optimization step using Lanczos, then reuse it for the first partition
    dim_m = krylov_dim
    krylov_basis, T_m = lanczos_algorithm(
        _H, _psi, krylov_dim=dim_m, sparse=sparse
    )

    # check if a happy breakdown occurred
    if T_m.shape[0] < krylov_dim + 1:
        if T_m.shape[0] == 1:
            # this means that the state does not evolve in time, it lies in a
            # symmetry of H subspace. Thus, theres no work to be done.
            krylov_results = particular_tlist_or_happy_breakdown(
                tlist,
                n_tlist_steps,
                options,
                psi0,
                e_ops,
                krylov_results,
                pbar,
                happy_breakdown=True,
            )
            return krylov_results
        else:
            # no optimization is required, convergence is guaranteed.
            delta_t = tf - t0
            n_timesteps = 1
    else:

        # calculate optimal number of internal timesteps.
        delta_t = _optimize_lanczos_timestep_size(
            T_m, krylov_basis=krylov_basis, tlist=tlist, options=options
        )
        n_timesteps = int(ceil((tf - t0) / delta_t))

        if n_timesteps >= options.nsteps:
            raise Exception(
                f"Optimization requires a number {n_timesteps} of lanczos iterations, "
                f"which exceeds the defined allowed number {options.nsteps}. This can "
                "be increased via the 'Options.nsteps' property."
            )

    partitions = _make_partitions(tlist=tlist, n_timesteps=n_timesteps)

    if progress_bar:
        pbar.start(len(partitions))

    # update parameters regarding e_ops
    krylov_results, expt_callback, options, n_expt_op = _e_ops_outputs(
        krylov_results, e_ops, n_tlist_steps, options
    )

    # parameters for the lazy iteration evolve tlist
    psi_norm = np.linalg.norm(_psi)
    last_t = t0

    for idx, partition in enumerate(partitions):

        evolved_states = _evolve_krylov_tlist(
            H=_H,
            psi0=_psi,
            krylov_dim=dim_m,
            tlist=partition,
            t0=last_t,
            psi_norm=psi_norm,
            krylov_basis=krylov_basis,
            T_m=T_m,
            sparse=sparse,
        )

        if idx == 0:
            krylov_basis = None
            T_m = None
            t_idx = 0

        _psi = evolved_states[-1]
        psi_norm = np.linalg.norm(_psi)
        last_t = partition[-1]

        # apply qobj to each evolved state, remove repeated tail elements
        qobj_evolved_states = [
            Qobj(state, dims=psi0.dims) for state in evolved_states[1:-1]
        ]

        krylov_results = _expectation_values(
            e_ops,
            n_expt_op,
            expt_callback,
            krylov_results,
            qobj_evolved_states,
            partitions,
            idx,
            t_idx,
            options,
        )

        t_idx += len(partition[1:-1])

        pbar.update(idx)

    pbar.finished()

    if e_ops_dict:
        krylov_results.expect = {
            e: krylov_results.expect[n]
            for n, e in enumerate(e_ops_dict.keys())
        }

    return krylov_results


def _expectation_values(
    e_ops,
    n_expt_op,
    expt_callback,
    res,
    evolved_states,
    partitions,
    idx,
    t_idx,
    options,
):

    if options.store_states:
        res.states += evolved_states

    for t, state in zip(
        range(t_idx, t_idx + len(partitions[idx][1:-1])), evolved_states
    ):

        if expt_callback:
            # use callback method
            res.expect.append(e_ops(t, state))

        for m in range(n_expt_op):
            op = e_ops[m]
            if not isinstance(op, Qobj) and callable(op):
                res.expect[m][t] = op(t, state)
                continue

            res.expect[m][t] = expect(op, state)

    if (
        idx == len(partitions) - 1
        and options.store_final_state
        and not options.store_states
    ):
        res.states = [evolved_states[-1]]

    return res


def lanczos_algorithm(
    H,
    psi: np.ndarray,
    krylov_dim: int,
    sparse: bool = False,
):
    """
    Computes a basis of the Krylov subspace for Hamiltonian 'H', a system
    state 'psi' and Krylov dimension 'krylov_dim'. The space is spanned
    by {psi, H psi, H^2 psi, ..., H^(krylov_dim) psi}.
    Parameters
    ------------
    H : np.ndarray or csr_matrix
       System Hamiltonian. If the Hamiltonian is dense, a np.ndarray is
       preferred, whereas if it is sparse, a scipy csr_matrix is optimal.
    psi: np.ndarray
        State used to calculate Krylov subspace.
    krylov_dim: int
        Dimension (krylov_dim + 1) of the spanned Krylov subspace.
    sparse: bool (optional, default False)
        Wether to perform scipy sparse matrix multiplication operations or
        numpy dense matrix multiplications.
    Returns
    ---------
    v: np.ndarray
        Lanczos eigenvector.
    T: np.ndarray
        Tridiagonal decomposition.
    """

    v = np.zeros((krylov_dim + 1, psi.shape[0]), dtype=complex)
    T_m = np.zeros((krylov_dim + 1, krylov_dim + 1), dtype=complex)

    v[0, :] = psi.squeeze()

    w_prime = H.dot(v[0, :])

    alpha = np.vdot(w_prime, v[0, :])

    w = w_prime - alpha * v[0, :]

    T_m[0, 0] = alpha

    for j in range(1, krylov_dim + 1):

        beta = np.linalg.norm(w)

        if beta < 1e-7:

            # Happy breakdown
            v_happy = v[0:j, :]
            T_m_happy = T_m[0:j, 0:j]

            return v_happy, T_m_happy

        v[j, :] = w / beta
        w_prime = H.dot(v[j, :])
        alpha = np.vdot(w_prime, v[j, :])

        w = w_prime - alpha * v[j, :] - beta * v[j - 1, :]

        T_m[j, j] = alpha
        T_m[j, j - 1] = beta
        T_m[j - 1, j] = beta

    return v, T_m


def _evolve(t0: float, krylov_basis: np.ndarray, T_m: np.ndarray):
    """
    Computes the time evolution operator 'U(t - t0) psi0_k', where 'psi0_k'
    is the first basis element of the Krylov subspace, as a function of time.
    Parameters
    ------------
    t0: float
        Initial time for the time evolution.
    krylov_basis: np.ndarray
        Krylov basis projector operator.
    T_m: np.ndarray
        Tridiagonal matrix decomposition of the system given by lanczos
        algorithm.
    Returns
    ---------
    time_evolution: function
        Time evolution given by the Krylov subspace approximation.
    """

    eigenvalues, eigenvectors = eigh(T_m)
    U = np.matmul(krylov_basis.T, eigenvectors)
    e0 = eigenvectors.conj().T[:, 0]

    def time_evolution(t):
        delta_t = t - t0
        aux = np.multiply(np.exp(-1j * delta_t * eigenvalues), e0)
        return np.matmul(U, aux)

    return time_evolution


def _evolve_krylov_tlist(
    H,
    psi0: np.ndarray,
    krylov_dim: int,
    tlist: list,
    t0: float,
    psi_norm: float = None,
    krylov_basis: np.array = None,
    T_m: np.array = None,
    sparse: bool = False,
):
    """
    Computes the Krylov approximation time evolution of dimension 'krylov_dim'
    for Hamiltonian 'H' and initial state 'psi0' for each time in 'tlist'.
    Parameters
    ------------
    H: np.ndarray or csr_matrix
        System Hamiltonian.
    psi0: np.ndarray
        Initial state vector.
    krylov_basis: np.ndarray
        Krylov basis projector operator.
    tlist: list
        List of timesteps for the time evolution.
    t0: float
        Initial time for the time evolution.
    psi_norm: float (optional, default False)
        Norm-2 of psi0.
    krylov_basis: np.ndarray (optional, default None)
        Krylov basis projector operator. If 'krylov_basis' is None, perform
        a lanczos iteration.
    T_m: np.ndarray (optional, default None)
        Tridiagonal matrix decomposition of the system given by lanczos
        algorithm. If 'T_m' is None, perform a lanczos iteration.
    Returns
    ---------
    psi_list: List[np.ndarray]
        List of evolved states at times t in 'tlist'.
    """

    if psi_norm is None:
        psi_norm = np.linalg.norm(psi0)

    if psi_norm != 1:
        psi = psi0 / psi_norm
    else:
        psi = psi0

    if (krylov_basis is None) or (T_m is None):
        krylov_basis, T_m = lanczos_algorithm(
            H=H, psi=psi, krylov_dim=krylov_dim, sparse=sparse
        )

    evolve = _evolve(t0, krylov_basis, T_m)
    psi_list = list(map(evolve, tlist))

    return psi_list


# ----------------------------------------------------------------------
# Auxiliar functions


def _check_inputs(H, psi0, krylov_dim):
    """Check that the inputs 'H' and 'psi0' have the correct structures."""
    if not isinstance(H, Qobj):
        raise TypeError(
            "krylovsolve currently supports Hamiltonian Qobj operators only"
        )

    if not H.isherm:
        raise TypeError("Hamiltonian 'H' must be hermician.")

    if not isinstance(psi0, Qobj):
        raise TypeError("'psi0' must be a Qobj.")

    if not psi0.isket:
        raise TypeError("Initial state must be a ket Qobj.")

    if not ((len(H.shape) == 2) and (H.shape[0] == H.shape[1])):
        raise ValueError("the Hamiltonian must be 2-dimensional square Qobj.")

    if not (psi0.dims[0] == H.dims[0]):
        raise ValueError(
            "'psi0' and the Hamiltonian must share the same dimension."
        )

    if not (H.shape[0] >= krylov_dim):
        raise ValueError(
            "the Hamiltonian dimension must be greater or equal to the maximum"
            " allowed krylov dimension 'krylov_dim'."
        )


def _check_e_ops(e_ops):
    """
    Check instances of e_ops and return the formatted version of e_ops
    and e_ops_dict.
    """
    if e_ops is None:
        e_ops = []
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]
    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None
    return e_ops, e_ops_dict


def _check_progress_bar(progress_bar):
    """
    Check instance of progress_bar and return the object.
    """
    if progress_bar is None:
        pbar = BaseProgressBar()
    if progress_bar is True:
        pbar = TextProgressBar()
    return pbar


def particular_tlist_or_happy_breakdown(
    tlist,
    n_tlist_steps,
    options,
    psi0,
    e_ops,
    res,
    progress_bar,
    happy_breakdown=False,
):
    """Deals with the problem when 'tlist' contains a single element, where
    that same ket is returned and evaluated at 'e_ops', if provided.
    """

    if len(tlist) == 0:
        warnings.warn(
            "Input 'tlist' contains a single element. If 'e_ops' were provided"
            ", return its corresponding expectation values at 'psi0', else "
            "return 'psi0'."
        )

    progress_bar.start(1)

    res, expt_callback, options, n_expt_op = _e_ops_outputs(
        res, e_ops, n_tlist_steps, options
    )

    if options.store_states:
        res.states = [psi0]

    e_0 = None
    if expt_callback:
        # use callback method
        e_0 = e_ops(0, psi0)
        res.expect.append(e_0)

    e_m_0 = []
    for m in range(n_expt_op):
        op = e_ops[m]

        if not isinstance(op, Qobj) and callable(op):
            e_m_0.append(op(0, psi0))
            res.expect[m][0] = e_m_0[m]
            continue

        e_m_0.append(expect(op, psi0))
        res.expect[m][0] = e_m_0[m]

    if happy_breakdown:
        res = _happy_breakdown(
            tlist,
            options,
            res,
            psi0,
            expt_callback,
            e_0,
            n_expt_op,
            e_ops,
            e_m_0,
        )

    if (options.store_final_state) and (not options.store_states):
        res.states = [psi0]

    progress_bar.update(1)
    progress_bar.finished()
    return res


def _happy_breakdown(
    tlist, options, res, psi0, expt_callback, e_0, n_expt_op, e_ops, e_m_0
):
    """
    Dummy evolves the system if a happy breakdown of an eigenstate occurs.
    """
    for i in range(1, len(tlist)):
        if options.store_states:
            res.states.append(psi0)
        if expt_callback:
            res.expect.append(e_0)

        for m in range(n_expt_op):
            op = e_ops[m]
            res.expect[m][i] = e_m_0[m]
    return res


def _optimize_lanczos_timestep_size(T, krylov_basis, tlist, options):
    """
    Solves the equation defined to optimize the number of Lanczos
    iterations to be performed inside Krylov's algorithm.
    """

    f = _lanczos_error_equation_to_optimize_delta_t(
        T,
        krylov_basis=krylov_basis,
        t0=tlist[0],
        tf=tlist[-1],
        target_tolerance=options.atol,
    )

    # To avoid the singularity at t0, we add a small epsilon value
    t_min = (tlist[-1] - tlist[0]) / options.nsteps + tlist[0]
    bracket = [t_min, tlist[-1]]

    if (np.sign(f(bracket[0])) == -1) and (np.sign(f(bracket[-1])) == -1):
        delta_t = tlist[-1] - tlist[0]
        return delta_t

    elif (np.sign(f(bracket[0])) == 1) and (np.sign(f(bracket[-1])) == 1):
        raise ValueError(
            "No solution exists with the given combination of parameters 'krylov_dim', "
            "tolerance = 'options.atol', maximum number allowed of krylov internal "
            "partitions = 'options.nsteps' and 'tlist'. Try reducing the tolerance, or "
            "increasing 'krylov_dim'. If nothing works, then a deeper analysis of the "
            "problem is recommended."
        )

    else:
        sol = root_scalar(f=f, bracket=bracket, method="brentq", xtol=options.atol)
        if sol.converged:
            delta_t = sol.root
            return delta_t
        else:
            raise Exception(
                "Method did not converge, try increasing 'krylov_dim', "
                "taking a lesser final time 'tlist[-1]' or decreasing the "
                "tolerance via Options().atol. "
                "If nothing works, this problem might not be suitable for "
                "Krylov or a deeper analysis might be required."
            )


def _lanczos_error_equation_to_optimize_delta_t(
    T, krylov_basis, t0, tf, target_tolerance
):
    """
    Function to optimize in order to obtain the optimal number of
    Lanczos algorithm iterations, governed by the optimal timestep size between
    Lanczos iteractions.
    """
    eigenvalues1, eigenvectors1 = eigh(T[0:, 0:])
    U1 = np.matmul(krylov_basis[0:, 0:].T, eigenvectors1)
    e01 = eigenvectors1.conj().T[:, 0]

    eigenvalues2, eigenvectors2 = eigh(T[0:-1, 0: T.shape[1] - 1])
    U2 = np.matmul(krylov_basis[0:-1, :].T, eigenvectors2)
    e02 = eigenvectors2.conj().T[:, 0]

    def f(t):
        delta_t = -1j * (t - t0)

        aux1 = np.multiply(np.exp(delta_t * eigenvalues1), e01)
        psi1 = np.matmul(U1, aux1)

        aux2 = np.multiply(np.exp(delta_t * eigenvalues2), e02)
        psi2 = np.matmul(U2, aux2)

        error = np.linalg.norm(psi1 - psi2)

        steps = max(1, (tf - t0) // (t - t0))
        return np.log10(error) + np.log10(steps) - np.log10(target_tolerance)

    return f


def _make_partitions(tlist, n_timesteps):
    """Generates an internal 'partitions' list of np.arrays to iterate Lanczos
    algorithms on each of them, based on 'tlist' and the optimized number of
    iterations 'n_timesteps'.
    """

    _tlist = np.copy(tlist)

    if n_timesteps == 1:
        _tlist = np.insert(_tlist, 0, tlist[0])
        _tlist = np.append(_tlist, tlist[-1])
        partitions = [_tlist]
        return partitions

    n_timesteps += 1
    krylov_tlist = np.linspace(tlist[0], tlist[-1], n_timesteps)
    krylov_partitions = [
        np.array(krylov_tlist[i: i + 2]) for i in range(n_timesteps - 1)
    ]
    partitions = []
    for krylov_partition in krylov_partitions:
        start = krylov_partition[0]
        end = krylov_partition[-1]
        condition = _tlist <= end
        partitions.append([start] + _tlist[condition].tolist() + [end])
        _tlist = _tlist[~condition]

    return partitions


def _e_ops_outputs(krylov_results, e_ops, n_tlist_steps, opt):
    krylov_results.expect = []
    if callable(e_ops):
        n_expt_op = 0
        expt_callback = True
        krylov_results.num_expect = 1
    elif isinstance(e_ops, list):
        n_expt_op = len(e_ops)
        expt_callback = False
        krylov_results.num_expect = n_expt_op
        if n_expt_op == 0:
            # fall back on storing states
            opt.store_states = True
        else:
            for op in e_ops:
                if not isinstance(op, Qobj) and callable(op):
                    krylov_results.expect.append(
                        np.zeros(n_tlist_steps, dtype=complex)
                    )
                    continue
                if op.isherm:
                    krylov_results.expect.append(np.zeros(n_tlist_steps))
                else:
                    krylov_results.expect.append(
                        np.zeros(n_tlist_steps, dtype=complex)
                    )

    else:
        raise TypeError("Expectation parameter must be a list or a function")

    return krylov_results, expt_callback, opt, n_expt_op
