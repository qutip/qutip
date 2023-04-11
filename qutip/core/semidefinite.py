# -*- coding: utf-8 -*-

"""
This module implements internal-use functions for semidefinite programming.
"""
import collections
import functools

import numpy as np
import scipy.sparse as sp

# Conditionally import CVXPY
try:
    import cvxpy

    __all__ = ["dnorm_problem", "dnorm_sparse_problem"]
except ImportError:
    cvxpy = None
    __all__ = []

from .operators import swap

Complex = collections.namedtuple("Complex", ["re", "im"])


def _complex_var(rows=1, cols=1, name=None):
    return Complex(
        re=cvxpy.Variable((rows, cols), name=(name + "_re") if name else None),
        im=cvxpy.Variable((rows, cols), name=(name + "_im") if name else None),
    )


def _make_constraints(*rhos):
    """
    Create constraints to ensure definied density operators.
    """
    # rhos traces are 1
    constraints = [cvxpy.trace(rho.re) == 1 for rho in rhos]
    # rhos are Hermitian
    for rho in rhos:
        constraints += [rho.re == rho.re.T] + [rho.im == -rho.im.T]
    # Non negative
    constraints += [
        cvxpy.bmat([[rho.re, -rho.im], [rho.im, rho.re]]) >> 0 for rho in rhos
    ]
    return constraints


def _arr_to_complex(A):
    if np.iscomplex(A).any():
        return Complex(re=A.real, im=A.imag)
    return Complex(re=A, im=np.zeros_like(A))


def _kron(A, B):
    if isinstance(A, np.ndarray):
        A = _arr_to_complex(A)
    if isinstance(B, np.ndarray):
        B = _arr_to_complex(B)

    return Complex(
        re=(cvxpy.kron(A.re, B.re) - cvxpy.kron(A.im, B.im)),
        im=(cvxpy.kron(A.im, B.re) + cvxpy.kron(A.re, B.im)),
    )


def _conj(W, A):
    U, V = W.re, W.im
    A, B = A.re, A.im
    return Complex(
        re=(U @ A @ U.T - U @ B @ V.T - V @ A @ V.T - V @ B @ U.T),
        im=(U @ A @ V.T + U @ B @ U.T + V @ A @ U.T - V @ B @ V.T),
    )


@functools.lru_cache
def initialize_constraints_on_dnorm_problem(dim):
    # Start assembling constraints and variables.
    constraints = []

    # Make a complex variable for X.
    X = _complex_var(dim**2, dim**2, "X")

    # Make complex variables for rho0 and rho1.
    rho0 = _complex_var(dim, dim, "rho0")
    rho1 = _complex_var(dim, dim, "rho1")
    constraints += _make_constraints(rho0, rho1)

    # Finally, add the tricky positive semidefinite constraint.
    # Since we're using column-stacking, but Watrous used row-stacking,
    # we need to swap the order in Rho0 and Rho1. This is not straightforward,
    # as CVXPY requires that the constant be the first argument. To solve this,
    # We conjugate by SWAP.
    W = swap(dim, dim).full()
    W = Complex(re=W.real, im=W.imag)
    Rho0 = _conj(W, _kron(np.eye(dim), rho0))
    Rho1 = _conj(W, _kron(np.eye(dim), rho1))

    Y = cvxpy.bmat(
        [
            [Rho0.re, X.re, -Rho0.im, -X.im],
            [X.re.T, Rho1.re, X.im.T, -Rho1.im],
            [Rho0.im, X.im, Rho0.re, X.re],
            [-X.im.T, Rho1.im, X.re.T, Rho1.re],
        ]
    )
    constraints += [Y >> 0]

    return X, constraints


def dnorm_problem(dim):
    """
    Creade the cvxpy ``Problem`` for the dnorm metric using dense arrays
    """
    X, constraints = initialize_constraints_on_dnorm_problem(dim)
    Jr = cvxpy.Parameter((dim**2, dim**2))
    Ji = cvxpy.Parameter((dim**2, dim**2))
    # The objective, however, depends on J.
    objective = cvxpy.Maximize(cvxpy.trace(Jr.T @ X.re + Ji.T @ X.im))
    problem = cvxpy.Problem(objective, constraints)
    return problem, Jr, Ji


def dnorm_sparse_problem(dim, J_dat):
    """
    Creade the cvxpy ``Problem`` for the dnorm metric using sparse arrays
    """
    X, constraints = initialize_constraints_on_dnorm_problem(dim)
    J_val = J_dat.tocoo()

    def adapt_sparse_params(A_val, dim):
        # This detour is needed as pointed out in cvxgrp/cvxpy#1159, as cvxpy
        # can not solve with parameters that aresparse matrices directly.
        # Solutions have to be made through calling cvxpy.reshape on
        # the original sparse matrix.
        side_size = dim**2
        A_nnz = cvxpy.Parameter(A_val.nnz)

        A_data = np.ones(A_nnz.size)
        A_rows = A_val.row * side_size + A_val.col
        A_cols = np.arange(A_nnz.size)
        # We are pushing the data on the location of the nonzero elements
        # to the nonzero rows of A_indexer
        A_Indexer = sp.coo_matrix(
            (A_data, (A_rows, A_cols)), shape=(side_size**2, A_nnz.size)
        )
        # We get finaly the sparse matrix A which we wanted
        A = cvxpy.reshape(A_Indexer @ A_nnz, (side_size, side_size), order="C")
        A_nnz.value = A_val.data
        return A

    Jr_val = J_val.real
    Jr = adapt_sparse_params(Jr_val, dim)

    Ji_val = J_val.imag
    Ji = adapt_sparse_params(Ji_val, dim)

    # The objective, however, depends on J.
    objective = cvxpy.Maximize(cvxpy.trace(Jr.T @ X.re + Ji.T @ X.im))

    problem = cvxpy.Problem(objective, constraints)
    return problem
