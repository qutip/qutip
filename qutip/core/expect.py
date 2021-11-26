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

__all__ = ['expect', 'variance']

import numpy as np

from .qobj import Qobj
from . import data as _data


def expect(oper, state):
    """
    Calculate the expectation value for operator(s) and state(s).  The
    expectation of state `k` on operator `A` is defined as `k.dag() @ A @ k`,
    and for density matrix `R` on operator `A` it is `trace(A @ R)`.

    Parameters
    ----------
    oper : qobj/array-like
        A single or a `list` or operators for expectation value.

    state : qobj/array-like
        A single or a `list` of quantum states or density matrices.

    Returns
    -------
    expt : float/complex/array-like
        Expectation value.  ``real`` if `oper` is Hermitian, ``complex``
        otherwise. A (nested) array of expectaction values of state or operator
        are arrays.

    Examples
    --------
    >>> expect(num(4), basis(4, 3)) == 3 # doctest: +NORMALIZE_WHITESPACE
        True

    """
    if isinstance(state, Qobj) and isinstance(oper, Qobj):
        return _single_qobj_expect(oper, state)

    elif isinstance(oper, (list, np.ndarray)):
        if isinstance(state, Qobj):
            dtype = np.complex128
            if all(op.isherm for op in oper) and (state.isket or state.isherm):
                dtype = np.float64
            return np.array([_single_qobj_expect(op, state) for op in oper],
                            dtype=dtype)
        return [expect(op, state) for op in oper]

    elif isinstance(state, (list, np.ndarray)):
        dtype = np.complex128
        if oper.isherm and all(op.isherm or op.isket for op in state):
            dtype = np.float64
        return np.array([_single_qobj_expect(oper, x) for x in state],
                        dtype=dtype)
    raise TypeError('Arguments must be quantum objects')


def _single_qobj_expect(oper, state):
    """
    Private function used by expect to calculate expectation values of Qobjs.
    """
    if not oper.isoper or not (state.isket or state.isoper):
        raise TypeError('invalid operand types')
    if oper.dims[1] != state.dims[0]:
        msg = (
            "incompatible dimensions "
            + str(oper.dims[1]) + " and " + str(state.dims[0])
        )
        raise ValueError(msg)
    out = _data.expect(oper.data, state.data)

    # This ensures that expect can return something that is not a number such
    # as a `tensorflow.Tensor` in qutip-tensorflow.
    return out.real if (oper.isherm
                        and (state.isket or state.isherm)
                        and hasattr(out, "real")
                        ) else out


def variance(oper, state):
    """
    Variance of an operator for the given state vector or density matrix.

    Parameters
    ----------
    oper : qobj
        Operator for expectation value.

    state : qobj/list
        A single or `list` of quantum states or density matrices..

    Returns
    -------
    var : float
        Variance of operator 'oper' for given state.

    """
    return expect(oper**2, state) - expect(oper, state)**2
