# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module implements internal-use functions for semidefinite programming.
"""

# Python Standard Library
from functools import wraps
from collections import namedtuple

# NumPy/SciPy
import numpy as np

# Conditionally import CVXPY
try:
    import cvxpy
except:
    cvxpy = None

Complex = namedtuple('Complex', ['re', 'im'])

from qutip.qip.gates import swap
from qutip.tensor import tensor_swap
from qutip.operators import qeye

from qutip.logging_utils import get_logger
logger = get_logger()

def complex_var(rows=1, cols=1, name=None):
    return Complex(
        re=cvxpy.Variable(rows, cols, name=(name + "_re") if name else None),
        im=cvxpy.Variable(rows, cols, name=(name + "_im") if name else None)
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
        re=(U * A * U.T - U * B * V.T - V * A * V.T - V * B * U.T),
        im=(U * A * V.T + U * B * U.T + V * A * U.T - V * B * V.T)
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
def dnorm_problem(dim):
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
    W = qudit_swap(dim).data.todense()
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
    
    Jr = cvxpy.Parameter(dim ** 2, dim ** 2)
    Ji = cvxpy.Parameter(dim ** 2, dim ** 2)
    
    # The objective, however, depends on J.
    objective = cvxpy.Maximize(cvxpy.trace(
        Jr.T * X.re + Ji.T * X.im
    ))
    
    problem = cvxpy.Problem(objective, constraints)
    
    return problem, Jr, Ji, X, rho0, rho1

