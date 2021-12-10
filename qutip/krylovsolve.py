"""
This module provides approximations of the time evolution operator
using small dimensional Krylov subspaces.
"""

from functools import reduce
from math import ceil
import operator

import numpy as np
from qutip import mcsolve
from qutip.expect import expect
from qutip.qobj import Qobj
from qutip.solver import Result
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from scipy.linalg import eigh, expm

from qutip import mcsolve


def krylovsolve(
    H: Qobj,
    psi0: Qobj,
    tlist: np.array,
    krylov_dim: int = 30,
    e_ops=None,
    tolerance: float = 1e-7,
    store_states: bool = False,
    store_final_state: bool = False,
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
            List of times on which to evolve the initial state. If None,
            nothing happens but the code won't break.
    krylov_dim: int
            Dimension of Krylov approximation subspaces used for the time
            evolution approximation.
    e_ops : None / list of :class:`qutip.Qobj`
            Single operator or list of operators for which to evaluate
            expectation values.
    if store_states : bool (default False)
            If e_ops is provided, store each state vector corresponding to each
            time in tlist.
    store_final_state : bool (default False)
            If e_ops is provided, store the final state of the evolution.
    progress_bar : None / BaseProgressBar
            Optional instance of BaseProgressBar, or a subclass thereof, for
            showing the progress of the simulation.
    sparse : bool (default False)
            Use np.array to represent system Hamiltonians. If True, scipy
            arrays are used instead.
    tolerance : :float: (default 1e-7)
            Minimum bound value for the final state infidelity with respect to
            the exact state.
    Returns
    ---------
    result: :class:`qutip.Result`
            An instance of the class :class:`qutip.Result`, which contains
            either an *array* `result.expect` of expectation values for times
            `tlist`, or an *array* `result.states` of state vectors at times
            `tlist` [if `e_ops` is an empty list].
    """

    # list of expectation values operators
    if e_ops is None:
        e_ops = []
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]
    if isinstance(e_ops, dict):
        e_ops = [e for e in e_ops.values()]

    # progress bar
    if progress_bar is True:
        pbar = TextProgressBar()
    else:
        pbar = BaseProgressBar()

    # verify that the hamiltonian meets the requirements
    assert isinstance(H, Qobj) | isinstance(
        H, np.ndarray
    ), "the Hamiltonian must be either a Qobj or a np.ndarray."

    assert len(H.shape) == 2, "the Hamiltonian must be 2-dimensional."

    assert (
        H.shape[0] == H.shape[1]
    ), "the Hamiltonian must be a square 2-dimensional."

    assert (
        H.shape[0] >= krylov_dim
    ), "the Hamiltonian dimension must be greater or equal to the \
            maximum allowed krylov dimension."

    if isinstance(H, Qobj):
        if sparse:
            _H = H.get_data()  # -> (fast_) csr_matrix
        else:
            _H = H.full().copy()  # -> np.ndarray
    else:
        _H = H

    # verify that the state vector meets the requirements
    assert isinstance(psi0, Qobj) | isinstance(
        psi0, np.ndarray
    ), "The state vector must be either a Qobj or a np.ndarray."

    assert (
        psi0.shape[0] == _H.shape[0]
    ), "The state vector and the Hamiltonian must share the same \
        dimension."

    # transform state type from Qobj to np.ndarray for faster operations
    if isinstance(psi0, Qobj):
        _psi = psi0.full().copy()
        _psi = _psi / np.linalg.norm(_psi)
    else:
        _psi = psi0.copy()
        _psi = _psi / np.linalg.norm(_psi)

    krylov_results = _krylovsolve(
        H,
        _H,
        psi0,
        _psi,
        tlist,
        krylov_dim,
        e_ops,
        tolerance,
        store_states,
        store_final_state,
        progress_bar,
        pbar,
        sparse,
    )

    return krylov_results


def _krylovsolve(
    H,
    _H,
    psi0,
    _psi,
    tlist,
    krylov_dim,
    e_ops,
    tolerance,
    store_states,
    store_final_state,
    progress_bar,
    pbar,
    sparse,
):
    # optimization step
    dim_m = krylov_dim

    tf = tlist[-1]
    t0 = tlist[0]

    # this Lanczos iteration it's reused for the first partition
    krylov_basis, T_m = lanczos_algorithm(
        _H, _psi, krylov_dim=dim_m, sparse=sparse
    )

    # calculate number of partitions, then make partitions
    n_timesteps = _calculate_dt(
        T_m=T_m,
        krylov_basis=krylov_basis,
        dim_m=dim_m,
        tf=tf,
        t0=t0,
        tlist=tlist,
        tolerance=tolerance,
    )
    partitions = _make_partitions(tlist=tlist, n_timesteps=n_timesteps)

    if progress_bar:
        pbar.start(len(partitions))

    # create output container
    krylov_results = Result()

    # if len(tlist)=1, the initial state (or its expectation value) is returned
    if len(tlist) < 1:
        krylov_results = _dummy_krylov(
            krylov_results=krylov_results,
            psi0=psi0,
            store_states=store_states,
            store_final_state=store_final_state,
            e_ops=e_ops,
        )
        return krylov_results

    # lazy iteration
    psi_norm = np.linalg.norm(_psi)
    evolved_states = _evolve_krylov_tlist(
        H=_H,
        psi0=_psi,
        krylov_dim=dim_m,
        tlist=partitions[0],
        t0=t0,
        psi_norm=psi_norm,
        krylov_basis=krylov_basis,
        T_m=T_m,
        sparse=sparse,
    )
    _psi = evolved_states[-1]
    psi_norm = np.linalg.norm(_psi)
    evolved_states = evolved_states[1:-1]
    evolved_states = [Qobj(state) for state in evolved_states]

    if e_ops:
        for idx, op in enumerate(e_ops):

            op.dims = [
                [reduce(operator.mul, op.dims[0])],
                [reduce(operator.mul, op.dims[1])],
            ]

            krylov_results.expect.append(
                [expect(op, state) for state in evolved_states]
            )
        if store_states:
            krylov_results.states += evolved_states
        if store_final_state:
            if len(partitions) == 1:
                krylov_results.states += evolved_states[-1]
    else:
        krylov_results.states += evolved_states
    last_t = partitions[0][-1]

    if progress_bar:
        pbar.update(0)

    for pdx, partition in enumerate(partitions[1:]):
        pdx += 1
        if progress_bar:
            pbar.update(pdx)

        # for each partition we calculate Lanczos
        evolved_states = _evolve_krylov_tlist(
            H=_H,
            psi0=_psi,
            krylov_dim=dim_m,
            tlist=partition,
            t0=last_t,
            psi_norm=psi_norm,
            sparse=sparse,
        )

        _psi = evolved_states[-1]
        psi_norm = np.linalg.norm(_psi)

        evolved_states = evolved_states[1:-1]
        evolved_states = [Qobj(state) for state in evolved_states]

        if e_ops:
            for idx, op in enumerate(e_ops):
                krylov_results.expect[idx] += [
                    expect(op, state) for state in evolved_states
                ]
            if store_states:
                krylov_results.states += evolved_states
            if store_final_state:
                if pdx == len(partitions) - 1:
                    krylov_results.states += evolved_states[-1]
        else:
            krylov_results.states += evolved_states
        last_t = partition[-1]

    if progress_bar:
        pbar.finished()

    return krylov_results


def _estimate_norm(H: np.ndarray, order: int):
    """
     Estimates the norm-2 of a Hamiltonian using a Lanczos algorithm of
     dimension 'order'.
    Parameters
    -------------
     H : np.ndarray
        System Hamiltonian.
     order: int
         Order of the estimated norm.
     Returns
     ---------
     max_eigenvalue: float
         The maximum eigenvalue resulting from a lanczos algorithm of
         dimension 'order'.
    """

    random_psi = np.random.random(H.shape[0]) + 1j * np.random.random(
        H.shape[0]
    )
    random_psi = random_psi / np.linalg.norm(random_psi)

    _, T_m = lanczos_algorithm(H, psi0=random_psi, krylov_dim=order)
    eigenvalues = eigh(T_m, eigvals_only=True)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    return max_eigenvalue


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

            v_happy = np.zeros([j, psi.shape[0]], dtype=complex)
            T_m_happy = np.zeros([j, j], dtype=complex)

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


def _calculate_dt(T_m, krylov_basis, dim_m, tf, t0, tlist, tolerance):
    if krylov_basis.shape[0] != dim_m + 2:
        deltat = (tf - t0) / 10
    else:
        deltat = optimizer(
            T_m, krylov_basis=krylov_basis, tlist=tlist, tol=tolerance
        )

    return int(ceil((tf - t0) / deltat))  # n_timesteps


def _make_partitions(tlist, n_timesteps):
    if n_timesteps == 1:
        partitions = [np.insert(tlist, 0, tlist[0])]
        return partitions
    n_timesteps += 1
    krylov_tlist = np.linspace(tlist[0], tlist[-1], n_timesteps)
    krylov_partitions = [
        np.array(krylov_tlist[i: i + 2]) for i in range(n_timesteps - 1)
    ]
    partitions = []
    _tlist = np.copy(tlist)
    for krylov_partition in krylov_partitions:
        start = krylov_partition[0]
        end = krylov_partition[-1]
        condition = _tlist <= end
        partitions.append([start] + _tlist[condition].tolist() + [end])
        _tlist = _tlist[~condition]
    return partitions


def _happy_breakdown(T_m, v, beta, w, j):
    v = v[0: j + 1, :]
    v[j + 1, :] = w / beta

    T_m = T_m[0:j, 0:j]

    T_m[j + 2, j + 1] = beta

    return v, T_m


def bound_function(T, krylov_basis, t0, tf):
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
    f = bound_function(T, krylov_basis=krylov_basis, t0=tlist[0], tf=tlist[-1])
    n = illinois_algorithm(
        f, a=tlist[0], b=tlist[-1], y=np.log10(tol), margin=0.1
    )
    return n


def _dummy_krylov(
    krylov_results, psi0, store_states=None, store_final_state=None, e_ops=None
):
    if e_ops:
        for idx, op in enumerate(e_ops):
            krylov_results.expect[idx] += [expect(op, psi0)]
        if store_state or store_final_state:
            krylov_results.states += [psi0]
    else:
        krylov_results.states += [psi0]
    return krylov_results
