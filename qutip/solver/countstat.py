"""
This module contains functions for calculating current and current noise using
the counting statistics formalism.
"""
__all__ = ['countstat_current', 'countstat_current_noise']

import numpy as np
import scipy.sparse as sp

from itertools import product
from ..core import (
    sprepost, spre, qeye, tensor, expect, Qobj,
    operator_to_vector, vector_to_operator, CoreOptions
)
from ..core import data as _data
from .steadystate import pseudo_inverse, steadystate
from ..settings import settings

# Load MKL spsolve if avaiable
if settings.has_mkl:
    from qutip._mkl.spsolve import mkl_spsolve


def countstat_current(L, c_ops=None, rhoss=None, J_ops=None):
    """
    Calculate the current corresponding to a system Liouvillian ``L`` and a
    list of current collapse operators ``c_ops`` or current superoperators
    ``J_ops``.

    Parameters
    ----------

    L : :class:`.Qobj`
        Qobj representing the system Liouvillian.

    c_ops : array / list (optional)
        List of current collapse operators. Required if either ``rhoss`` or
        ``J_ops`` is not given.

    rhoss : :class:`.Qobj` (optional)
        The steadystate density matrix for the given system Liouvillian ``L``
        and collapse operators. If not given, it defaults to
        ``steadystate(L, c_ops)``.

    J_ops : array / list (optional)
        List of current superoperators. If not given, they default to
        ``sprepost(c, c.dag())`` for each ``c`` from ``c_ops``.

    Returns
    --------
    I : array
        The currents ``I`` corresponding to each current collapse operator
        ``J_ops`` (or to each ``c_ops`` if ``J_ops`` was not given).
    """

    if J_ops is None:
        if c_ops is None:
            raise ValueError("c_ops must be given if J_ops is not")
        J_ops = [sprepost(c, c.dag()) for c in c_ops]

    if rhoss is None:
        if c_ops is None:
            raise ValueError("c_ops must be given if rhoss is not")
        rhoss = steadystate(L, c_ops)

    rhoss_vec = _data.column_stack(rhoss.data.copy())

    N = len(J_ops)
    current = np.zeros(N)

    for i, Ji in enumerate(J_ops):
        current[i] = _data.expect_super(Ji.data, rhoss_vec).real

    return current


def _solve(A, V):
    try:
        return _data.solve(A, V)
    except ValueError:
        return _data.solve(A, V, "lstsq")


def _noise_direct(L, wlist, rhoss, J_ops, I_ops):
    J_ops = [op.data for op in J_ops]
    I_ops = [op.data for op in I_ops]
    rhoss_vec = operator_to_vector(rhoss).data

    N_j_ops = len(J_ops)
    current = np.zeros(N_j_ops)
    noise = np.zeros((N_j_ops, N_j_ops, len(wlist)))
    skewness = np.zeros((N_j_ops, len(wlist)))

    tr_op = qeye(L.dims[0][0])
    tr_op_vec = operator_to_vector(tr_op)

    Pop = _data.kron(rhoss_vec, tr_op_vec.data.transpose())
    Iop = _data.identity(np.prod(L.dims[0][0]) ** 2)
    Q = _data.sub(Iop, Pop)

    QJ_ops = [
        _data.matmul(Q, _data.matmul(op, rhoss_vec))
        for op in J_ops
    ]
    QI_ops = [
        _data.matmul(Q, _data.matmul(op, rhoss_vec))
        for op in I_ops
    ]

    QIPI_ops = [
        _data.matmul(
            Q,
            _data.matmul(
                op, _data.matmul(Pop, _data.matmul(op, rhoss_vec))
            )
        )
        for op in I_ops
    ]

    for k, w in enumerate(wlist):
        if w != 0.0:
            L_temp = 1.0j * w * spre(tr_op) + L
        else:
            # At zero frequency some solvers fail for small systems.
            # Adding a small finite frequency of order 1e-15
            # helps prevent the solvers from throwing an exception.
            with CoreOptions(auto_tidyup=False):
                L_temp = 1e-15j * spre(tr_op) + L

        XI_rho = [
            _solve(L_temp.data, op) for op in QI_ops
        ]
        XJ_rho = [
            _solve(L_temp.data, op) for op in QJ_ops
        ]

        RIRI = [
            _solve(
                L_temp.data,
                _data.matmul(Q, _data.matmul(opI, _data.matmul(Q, op)))
            )
            for op, opI in zip(XI_rho, I_ops)
        ]

        RRIPI = [
            _solve(
                L_temp.data,
                _data.matmul(
                    Q, _data.matmul(Q, _solve(L_temp.data, op))
                )
            )
            for op in QIPI_ops
        ]

        for i, j in product(range(N_j_ops), repeat=2):
            if i == j:
                current[i] = _data.expect_super(I_ops[i], rhoss_vec).real
                noise[j, i, k] = _data.expect_super(J_ops[i], rhoss_vec).real
                skewness[i, k] = _data.expect_super(I_ops[i], rhoss_vec).real

                skewness[i, k] -= (
                    3 * (
                        _data.expect_super(
                            _data.matmul(J_ops[i], Q), XI_rho[i]
                        )
                        + _data.expect_super(
                            _data.matmul(I_ops[i], Q), XJ_rho[i]
                        )
                    ).real
                )

                skewness[i, k] -= (
                    6 * (
                        _data.expect_super(
                            _data.matmul(I_ops[i], Q), RRIPI[i]
                        )
                        - _data.expect_super(
                            _data.matmul(I_ops[i], Q), RIRI[i]
                        )
                    ).real
                )

            noise[j, i, k] -= (
                _data.expect_super(
                    _data.matmul(I_ops[j], Q), XI_rho[i]
                )
                + _data.expect_super(
                    _data.matmul(I_ops[i], Q), XI_rho[j]
                )
            ).real

    return current, noise, skewness


def _noise_pseudoinv(L, wlist, rhoss, J_ops, I_ops, sparse, method):
    N_j_ops = len(J_ops)
    current = np.zeros(N_j_ops)
    noise = np.zeros((N_j_ops, N_j_ops, len(wlist)))
    skewness = np.zeros((N_j_ops, len(wlist)))

    rhoss_vec = operator_to_vector(rhoss)

    dtype = type(L.data)

    tr_op = qeye(L.dims[0][0])
    tr_op_vec = operator_to_vector(tr_op)

    P = _data.kron(rhoss_vec.data, tr_op_vec.data.transpose(), dtype=dtype)

    Pop_new = Qobj(P, dims=L.dims)
    for k, w in enumerate(wlist):
        R = pseudo_inverse(L, rhoss=rhoss, w=w, sparse=sparse, method=method)
        for i, j in product(range(N_j_ops), repeat=2):
            if i == j:
                current[i] = I_ops[i](rhoss).tr().real
                noise[i, j, k] = J_ops[i](rhoss).tr().real
                skewness[i, k] = I_ops[i](rhoss).tr().real
                skewness[i, k] -= (
                    3 * (
                        (I_ops[i] @ R @ J_ops[i])(rhoss).tr().real +
                        (J_ops[i] @ R @ I_ops[i])(rhoss).tr().real
                    )
                )

                skewness[i, k] -= (
                    6 * (
                        (I_ops[i] @ R @ (
                            R @ I_ops[i] @ Pop_new - I_ops[i] @ R
                        ) @ I_ops[i])(rhoss).tr().real
                    )
                )
            op = I_ops[i] @ R @ I_ops[j] + I_ops[j] @ R @ I_ops[i]
            noise[i, j, k] -= op(rhoss).tr().real

    return current, noise, skewness


def countstat_current_noise(L, c_ops, wlist=None, rhoss=None, J_ops=None,
                            I_ops=None, sparse=True, method='direct'):
    """
    Compute the cross-current noise spectrum for a list of collapse operators
    `c_ops` corresponding to monitored currents, given the system
    Liouvillian `L`. The current collapse operators `c_ops` should be part
    of the dissipative processes in `L`, but the `c_ops` given here does not
    necessarily need to be all collapse operators contributing to dissipation
    in the Liouvillian. Optionally, the steadystate density matrix `rhoss`,
    the net current operator `I_ops` and the total current operators `J_ops`
    corresponding to the current collapse operators `c_ops` can also be
    specified. If either of `rhoss` or `J_ops` are omitted, they will be
    computed internally. 'wlist' is an optional list of frequencies
    at which to evaluate the noise spectrum.

    Parameters
    ----------

    L : :class:`.Qobj`
        Qobj representing the system Liouvillian.

    c_ops : array / list
        List of current collapse operators.

    rhoss : :class:`.Qobj` (optional)
        The steadystate density matrix corresponding the system Liouvillian
        `L`.

    wlist : array / list (optional)
        List of frequencies at which to evaluate (if none are given, evaluates
        at zero frequency)

    J_ops : array / list (optional)
        List of total current superoperators. Where J_ops is defined as the
        sum of currents across a junction J_ops ≡ I+ + I-. If not provided,
        it will be calculated using the collapse operators.

    I_ops : array / list (optional)
        List of net current superoperators. Where I_ops is defined as the
        difference of currents across a junction I_ops ≡ I+ - I-.
        If not provided, it will be assumed that I- = 0 and
        set I_ops = J_ops.

    sparse : bool [True]
        Flag that indicates whether to use sparse or dense matrix methods when
        computing the pseudo inverse. Default is false, as sparse solvers
        can fail for small systems. For larger systems the sparse solvers
        are recommended.

    method : str, ['direct']
        Method used to compute the noise. The default, 'direct' with
        ``sparse=True``, compute the noise directly. It is the recommended
        method for larger systems. Otherwise, the pseudo inverse is computed
        using the given method. Pseudo inverse supports 'splu' and 'spilu' for
        sparse matrices and 'direct', 'scipy' and 'numpy' methods for
        ``sparse=False``.

    .. note::
        The algoryth is described in page 67 of "Electrons in nanostructures"
        C. Flindt, PhD Thesis, available online:
        https://orbit.dtu.dk/en/publications/electrons-in-nanostructures-coherent-manipulation-and-counting-st

    Returns
    --------
    current : tuple of arrays
        The first cumulant, representing the mean current associated with each
        current superoperator `I_ops`. It quantifies the average charge
        transfer rate across the system.

    noise : tuple of arrays
        The second cumulant, representing the current noise, which measures
        fluctuations around the mean current. This term supports
        cross-correlations between different total current
        superoperators `J_ops`.

    skewness : tuple of arrays
        The third cumulant, representing the asymmetry of current fluctuations.
        It describes whether deviations from the mean current are more
        likely to be positive or negative. Unlike noise, skewness only
        accounts for diagonal (self-correlated) terms and does not
        include cross-correlations between different `J_ops`.
    """
    if rhoss is None:
        rhoss = steadystate(L, c_ops)

    if J_ops is None:
        J_ops = [sprepost(c, c.dag()) for c in c_ops]

    if I_ops is None:
        I_ops = J_ops

    if wlist is None:
        wlist = [0.]

    if sparse and method == 'direct':
        # rhoss_vec = operator_to_vector(rhoss).data
        current, noise, skewness = _noise_direct(L, wlist, rhoss, J_ops, I_ops)
    else:
        current, noise, skewness = _noise_pseudoinv(
            L, wlist, rhoss, J_ops, I_ops, sparse, method)

    return current, noise, skewness
