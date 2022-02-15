__all__ = ["krylovsolve", "lanczos_algorithm"]
"""
This module provides approximations of the time evolution operator
using small dimensional Krylov subspaces.
"""

from functools import reduce
from math import ceil
import operator

import numpy as np
from qutip.expect import expect, expect_rho_vec
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
     subspaces (m<<dim(H)).
     The output is either the state vector or the expectation values of
     supplied operators ("e_ops") at arbitrary points at ("tlist").

     **Additional options**
     Additional options to krylovsolve can be set with the following:
     "store_states": stores states even though expectation values are
     requested via the "e_ops" argument.
     "store_final_state": store final state even though expectation values are
     requested via the "e_ops" argument.

    Parameters
    -------------
     H : :class:`qutip.Qobj`
        System Hamiltonian.
     psi0 : :class: `qutip.Qobj`
         initial state vector (ket).
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
         Instance of ODE solver options.
     progress_bar : None / BaseProgressBar
         Optional instance of BaseProgressBar, or a subclass thereof, for
         showing the progress of the simulation.
     sparse : bool (default False)
         Use np.array to represent system Hamiltonians. If True, scipy sparse
         arrays are used instead.
     Returns
     ---------
      result: :class:`qutip.Result`
         An instance of the class :class:`qutip.Result`, which contains
         either an *array* `result.expect` of expectation values for the times
         `tlist`, or an *array* `result.states` of state vectors corresponding
         to the times `tlist` [if `e_ops` is an empty list].
    """

    e_ops, e_ops_dict = check_e_ops(e_ops)
    pbar = check_progress_bar(progress_bar)

    # set the physics
    if not isinstance(H, Qobj):
        raise TypeError(
            "krylovsolve currently supports Hamiltonian Qobj operators only"
        )

    if not (psi0.isket and isinstance(psi0, Qobj)):
        raise Exception("Initial state must be a state vector Qobj.")

    assert (
        len(H.shape) == 2 and H.shape[0] == H.shape[1]
    ), "the Hamiltonian must be 2-dimensional square Qobj."

    assert (
        psi0.shape[0] == H.shape[0]
    ), "The state vector and the Hamiltonian must share the same \
        dimension."

    assert (
        H.shape[0] >= krylov_dim
    ), "the Hamiltonian dimension must be greater or equal to the \
            maximum allowed krylov dimension."

    # transform inputs type from Qobj to np.ndarray/csr_matrix
    if sparse:
        _H = H.get_data()  # (fast_) csr_matrix
    else:
        _H = H.full().copy()  # np.ndarray
    _psi = psi0.full().copy()
    _psi = _psi / np.linalg.norm(_psi)

    # create internal variable and output containers
    if options is None:
        options = Options()

    krylov_results = Result()
    krylov_results.solver = "krylovsolve"

    # optimization step
    dim_m = krylov_dim
    n_tlist_steps = len(tlist)

    # handle particular cases of an empty tlist or single element
    if n_tlist_steps < 1:  # if tlist is empty, return empty results
        _dummy_progress_bar(progress_bar=pbar)
        return krylov_results

    if n_tlist_steps == 1:  # if tlist has only one element, return it
        print(
            "Warning: input 'tlist' contains a single element, assuming \
            initial time is 0 and final time is tlist single element"
        )
        krylov_results = particular_tlist(
            tlist, n_tlist_steps, options, psi0, e_ops, krylov_results, pbar
        )
        return krylov_results

    tf = tlist[-1]
    t0 = tlist[0]

    # this Lanczos iteration it's reused for the first partition
    krylov_basis, T_m = lanczos_algorithm(
        _H, _psi, krylov_dim=dim_m, sparse=sparse
    )

    delta_t = optimizer(
        T_m, krylov_basis=krylov_basis, tlist=tlist, tol=options.atol
    )

    n_timesteps = int(ceil((tf - t0) / delta_t))
    partitions = _make_partitions(tlist=tlist, n_timesteps=n_timesteps)

    if progress_bar:
        pbar.start(len(partitions))

    krylov_results, expt_callback, options, n_expt_op = e_ops_outputs(
        krylov_results, e_ops, n_tlist_steps, options
    )

    # parameters for the lazy iteration evolve tlist
    psi_norm = np.linalg.norm(_psi)
    last_t = t0
    _krylov_basis = krylov_basis.copy()
    _T_m = T_m.copy()

    for idx, partition in enumerate(partitions):

        evolved_states = _evolve_krylov_tlist(
            H=_H,
            psi0=_psi,
            krylov_dim=dim_m,
            tlist=partition,
            t0=last_t,
            psi_norm=psi_norm,
            krylov_basis=_krylov_basis,
            T_m=_T_m,
            sparse=sparse,
        )

        if idx == 0:
            _krylov_basis = None
            _T_m = None
            t_idx = 0

        _psi = evolved_states[-1]
        psi_norm = np.linalg.norm(_psi)
        last_t = partition[-1]

        evolved_states = map(Qobj, evolved_states[1:-1])

        krylov_results = _evolve_states(
            e_ops,
            n_expt_op,
            expt_callback,
            krylov_results,
            evolved_states,
            partitions,
            idx,
            t_idx,
            options,
        )

        t_idx += len(partition[1:-1])

        if progress_bar:
            pbar.update(idx)

    if progress_bar:
        pbar.finished()

    # krylov_results = remove_initial_result(krylov_results, _single_element_list, options, e_ops, n_expt_op)

    if e_ops_dict:
        krylov_results.expect = {
            e: krylov_results.expect[n]
            for n, e in enumerate(e_ops_dict.keys())
        }

    return krylov_results


def _evolve_states(
    e_ops,
    n_expt_op,
    expt_callback,
    krylov_results,
    evolved_states,
    partitions,
    idx,
    t_idx,
    options,
):

    if options.store_states:
        krylov_results.states += evolved_states

    for t, state in zip(
        range(t_idx, t_idx + len(partitions[idx][1:-1])), evolved_states
    ):

        if expt_callback:
            # use callback method
            krylov_results.expect.append(e_ops(t, state))

        for m in range(n_expt_op):
            op = e_ops[m]
            if not isinstance(op, Qobj) and callable(op):

                krylov_results.expect[m][t] = op(t, state)
                continue

            krylov_results.expect[m][t] = expect(op, state)

    if idx == len(partitions) - 1:
        if options.store_final_state:
            krylov_results.states += evolved_states[-1]

    return krylov_results


def dot_mul(A, v, sparse: bool = False):
    """
    Matrix multiplication of square matrix 'A' with vector 'v' for numpy
    'A' an instance of a dense np.ndarray or a scipy sparse array.
    Parameters
    ------------
    A : np.ndarray | csr_matrix
        Square matrix.
    v: np.ndarray
        Vector.
    sparse: bool (optional, default False)
        Wether to perform scipy sparse matrix multiplication operations or
        numpy dense matrix multiplications.
    Returns
    ---------
    Av: np.ndarray
        Resulting matrix multiplication.
    """

    if sparse:  # a is an instance of scr_matrix, v is a np.array
        return A.dot(v)
    else:
        return np.matmul(A, v)


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

    v = np.zeros((krylov_dim + 2, psi.shape[0]), dtype=complex)
    T_m = np.zeros((krylov_dim + 2, krylov_dim + 2), dtype=complex)

    v[0, :] = psi.squeeze()

    w_prime = dot_mul(H, v[0, :], sparse=sparse)

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
        w_prime = dot_mul(H, v[j, :], sparse=sparse)
        alpha = np.vdot(w_prime, v[j, :])

        w = w_prime - alpha * v[j, :] - beta * v[j - 1, :]

        T_m[j, j] = alpha
        T_m[j, j - 1] = beta
        T_m[j - 1, j] = beta

    beta = np.linalg.norm(w)
    v[krylov_dim + 1, :] = w / beta

    T_m[krylov_dim + 1, krylov_dim] = beta

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
        np.array(krylov_tlist[i : i + 2]) for i in range(n_timesteps - 1)
    ]
    partitions = []
    for krylov_partition in krylov_partitions:
        start = krylov_partition[0]
        end = krylov_partition[-1]
        condition = _tlist <= end
        partitions.append([start] + _tlist[condition].tolist() + [end])
        _tlist = _tlist[~condition]

    return partitions


def bound_function(T, krylov_basis, t0, tf):
    """
    Function to optimize in order to obtain the optimal number of
    Lanczos algorithm iterations.
    """
    eigenvalues1, eigenvectors1 = eigh(T[0:, 0:])
    U1 = np.matmul(krylov_basis[0:, 0:].T, eigenvectors1)
    e01 = eigenvectors1.conj().T[:, 0]

    eigenvalues2, eigenvectors2 = eigh(T[0:-1, 0 : T.shape[1] - 1])
    U2 = np.matmul(krylov_basis[0:-1, :].T, eigenvectors2)
    e02 = eigenvectors2.conj().T[:, 0]

    def f(t):
        delta_t = -1j * (t - t0)

        aux1 = np.multiply(np.exp(delta_t * eigenvalues1), e01)
        psi1 = np.matmul(U1, aux1)

        aux2 = np.multiply(np.exp(delta_t * eigenvalues2), e02)
        psi2 = np.matmul(U2, aux2)

        error = np.linalg.norm(psi1 - psi2)

        steps = 1 if t == t0 else max(1, tf // (t - t0))
        return np.log10(error) + np.log10(steps)

    return f


def illinois_algorithm(f, a, b, y, margin=1e-5):
    """
    Bracketed approach of Root-finding with illinois method.
    Parameters
    ----------
    f : callable
        Continuous function.
    a : float
        Lower bound to be searched.
    b : float
        Upper bound to be searched.
    y : float
        Target value.
    margin : float
        Margin of error in absolute term.
    Returns
    -------
    c : float
        Value where f(c) is within the margin of y.
    """

    if margin < 0:
        raise ValueError(
            "tolerance by 'margin' input cannot be null nor negative"
        )

    lower = f(a)
    upper = f(b)

    assert lower <= y, f"y is smaller than the lower bound. {y} < {lower}"

    if upper <= y:
        return b

    stagnant = 0

    while True:
        c = ((a * (upper - y)) - (b * (lower - y))) / (upper - lower)

        y_c = f(c)
        if abs((y_c) - y) < margin:
            # found!
            return c
        elif y < y_c:
            b, upper = c, y_c
            if stagnant == -1:
                # Lower bound is stagnant!
                lower += (y - lower) / 2
            stagnant = -1
        else:
            a, lower = c, y_c
            if stagnant == 1:
                # Upper bound is stagnant!
                upper -= (upper - y) / 2
            stagnant = 1


def optimizer(T, krylov_basis, tlist, tol):
    """
    Solves the equation defined to optimize the number of Lanczos
    iterations to be performed inside Krylov's algorithm.
    """
    f = bound_function(T, krylov_basis=krylov_basis, t0=tlist[0], tf=tlist[-1])
    n_iterations = illinois_algorithm(
        f, a=tlist[0], b=tlist[-1], y=np.log10(tol), margin=0.1
    )
    return n_iterations


def check_e_ops(e_ops):
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


def e_ops_outputs(krylov_results, e_ops, n_tlist_steps, options):
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
            options.store_states = True
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

    return krylov_results, expt_callback, options, n_expt_op


def check_progress_bar(progress_bar):
    """
    Check instance of progress_bar and return the object.
    """
    if progress_bar is None:
        pbar = BaseProgressBar()
    if progress_bar is True:
        pbar = TextProgressBar()
    return pbar


def _dummy_progress_bar(progress_bar):
    """Create a progress bar without jobs."""
    if progress_bar:
        progress_bar.start(1)
        progress_bar.update(1)
        progress_bar.finished()


def particular_tlist(
    tlist, n_tlist_steps, options, psi0, e_ops, res, progress_bar
):

    if progress_bar:
        progress_bar.start(1)

    res, expt_callback, options, n_expt_op = e_ops_outputs(
        res, e_ops, n_tlist_steps, options
    )

    if options.store_states:
        res.states = [psi0]

    if expt_callback:
        # use callback method
        res.expect.append(e_ops(tlist[0], psi0))

    for m in range(n_expt_op):
        op = e_ops[m]
        if not isinstance(op, Qobj) and callable(op):

            res.expect[m][0] = op(tlist[0], psi0)
            continue

        res.expect[m][0] = expect(op, psi0)

    if options.store_final_state:
        res.states = [psi0]

    if progress_bar:
        progress_bar.update(1)
        progress_bar.finished()
    return res
