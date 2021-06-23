# -*- coding: utf-8 -*-

"""
This module implements internal-use functions for semidefinite programming.
"""

# Python Standard Library
from functools import wraps
from collections import namedtuple

# NumPy/SciPy
import numpy as np
import scipy.sparse as sp
# Conditionally import CVXPY
try:
    import cvxpy
except ImportError:
    cvxpy = None

Complex = namedtuple('Complex', ['re', 'im'])

from qutip.tensor import tensor_swap
from qutip.operators import qeye

from qutip.logging_utils import get_logger
logger = get_logger()

def complex_var(rows=1, cols=1, name=None):
    return Complex(
        re=cvxpy.Variable((rows, cols), name=(name + "_re") if name else None),
        im=cvxpy.Variable((rows, cols), name=(name + "_im") if name else None)
    )


def herm(*Xs):
    return sum([
        [X.re == X.re.T, X.im == -X.im.T]
        for X in Xs
    ], [])


def pos_noherm(*Xs):
    constraints =[
        cvxpy.bmat([
            [X.re, -X.im],
            [X.im, X.re]
        ]) >> 0
        for X in Xs
    ]
    return constraints


def pos(*Xs):
    return pos_noherm(*Xs) + herm(*Xs)


def dens(*rhos):
    return pos(*rhos) + [
        cvxpy.trace(rho.re) == 1
        for rho in rhos
    ]


def _arr_to_complex(A):
    if np.iscomplex(A).any():
        return Complex(re=A.real, im=A.imag)
    else:
        return Complex(re=A, im=np.zeros_like(A))


def kron(A, B):
    if isinstance(A, np.ndarray):
        A = _arr_to_complex(A)
    if isinstance(B, np.ndarray):
        B = _arr_to_complex(B)

    return Complex(
        re=(cvxpy.kron(A.re, B.re) - cvxpy.kron(A.im, B.im)),
        im=(cvxpy.kron(A.im, B.re) + cvxpy.kron(A.re, B.im)),
    )

def conj(W, A):
    U, V = W.re, W.im
    A, B = A.re, A.im
    return Complex(
        re=(U @ A @ U.T - U @ B @ V.T - V @ A @ V.T - V @ B @ U.T),
        im=(U @ A @ V.T + U @ B @ U.T + V @ A @ U.T - V @ B @ V.T)
    )

def bmat(B):
    return Complex(
        re=cvxpy.bmat([[element.re for element in row] for row in B]),
        im=cvxpy.bmat([[element.re for element in row] for row in B]),
    )


def dag(X):
    return Complex(re=X.re.T, im=-X.im.T)


def memoize(fn):
    cache = {}

    @wraps(fn)
    def memoized(*args):
        if args in cache:
            return cache[args]
        else:
            ret = fn(*args)
            cache[args] = ret
            return ret

    memoized.reset_cache = cache.clear
    return memoized


def qudit_swap(dim):
    # We should likely generalize this and include it in qip.gates.
    W = qeye([dim, dim])
    return tensor_swap(W, (0, 1))


@memoize
def initialize_constraints_on_dnorm_problem(dim):
    # Start assembling constraints and variables.
    constraints = []

    # Make a complex variable for X.
    X = complex_var(dim ** 2, dim ** 2, "X")

    # Make complex variables for rho0 and rho1.
    rho0 = complex_var(dim, dim, "rho0")
    rho1 = complex_var(dim, dim, "rho1")
    constraints += dens(rho0, rho1)

    # Finally, add the tricky positive semidefinite constraint.
    # Since we're using column-stacking, but Watrous used row-stacking,
    # we need to swap the order in Rho0 and Rho1. This is not straightforward,
    # as CVXPY requires that the constant be the first argument. To solve this,
    # We conjugate by SWAP.
    W = qudit_swap(dim).full()
    W = Complex(re=W.real, im=W.imag)
    Rho0 = conj(W, kron(np.eye(dim), rho0))
    Rho1 = conj(W, kron(np.eye(dim), rho1))

    Y = cvxpy.bmat([
        [Rho0.re, X.re,      -Rho0.im, -X.im],
        [X.re.T, Rho1.re,    X.im.T, -Rho1.im],

        [Rho0.im, X.im,      Rho0.re, X.re],
        [-X.im.T, Rho1.im,   X.re.T, Rho1.re],
    ])
    constraints += [Y >> 0]

    logger.debug("Using {} constraints.".format(len(constraints)))

    return X, constraints


def dnorm_problem(dim):
    X, constraints = initialize_constraints_on_dnorm_problem(dim)
    Jr = cvxpy.Parameter((dim**2, dim**2))
    Ji = cvxpy.Parameter((dim**2, dim**2))
    # The objective, however, depends on J.
    objective = cvxpy.Maximize(cvxpy.trace(
        Jr.T @ X.re + Ji.T @ X.im
    ))
    problem = cvxpy.Problem(objective, constraints)
    return problem, Jr, Ji


def dnorm_sparse_problem(dim, J_dat):
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
        A_Indexer = sp.coo_matrix((A_data, (A_rows, A_cols)),
                                  shape=(side_size**2, A_nnz.size))
        # We get finaly the sparse matrix A which we wanted
        A = cvxpy.reshape(A_Indexer @ A_nnz, (side_size, side_size), order='C')
        A_nnz.value = A_val.data
        return A

    Jr_val = J_val.real
    Jr = adapt_sparse_params(Jr_val, dim)

    Ji_val = J_val.imag
    Ji = adapt_sparse_params(Ji_val, dim)

    # The objective, however, depends on J.
    objective = cvxpy.Maximize(cvxpy.trace(
        Jr.T @ X.re + Ji.T @ X.im
    ))

    problem = cvxpy.Problem(objective, constraints)
    return problem
