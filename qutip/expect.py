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
import scipy.sparse as sp

from qutip.qobj import Qobj, isoper
from qutip.eseries import eseries
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi, cy_spmm_tr,
                                expect_csr_ket)


expect_rho_vec = cy_expect_rho_vec
expect_psi = cy_expect_psi


def expect(oper, state):
    '''Calculates the expectation value for operator(s) and state(s).

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
    >>> expect(num(4), basis(4, 3))
    3

    '''
    if isinstance(state, Qobj) and isinstance(oper, Qobj):
        return _single_qobj_expect(oper, state)

    elif isinstance(oper, Qobj) and isinstance(state, eseries):
        return _single_eseries_expect(oper, state)

    elif isinstance(oper, (list, np.ndarray)):
        if isinstance(state, Qobj):
            if (all([op.isherm for op in oper]) and
                    (state.isket or state.isherm)):
                return np.array([_single_qobj_expect(o, state) for o in oper])
            else:
                return np.array([_single_qobj_expect(o, state) for o in oper],
                                dtype=complex)
        else:
            return [expect(o, state) for o in oper]

    elif isinstance(state, (list, np.ndarray)):
        if oper.isherm and all([(op.isherm or op.type == 'ket')
                                for op in state]):
            return np.array([_single_qobj_expect(oper, x) for x in state])
        else:
            return np.array([_single_qobj_expect(oper, x) for x in state],
                            dtype=complex)
    else:
        raise TypeError('Arguments must be quantum objects or eseries')


def _single_qobj_expect(oper, state):
    """
    Private function used by expect to calculate expectation values of Qobjs.
    """
    if isoper(oper):
        if oper.dims[1] != state.dims[0]:
            raise Exception('Operator and state do not have same tensor ' +
                            'structure: %s and %s' %
                            (oper.dims[1], state.dims[0]))

        if state.type == 'oper':
            # calculates expectation value via TR(op*rho)
            return cy_spmm_tr(oper.data, state.data,
                              oper.isherm and state.isherm)

        elif state.type == 'ket':
            # calculates expectation value via <psi|op|psi>
            return expect_csr_ket(oper.data, state.data,
                                 oper.isherm)
    else:
        raise TypeError('Invalid operand types')


def _single_eseries_expect(oper, state):
    """
    Private function used by expect to calculate expectation values for
    eseries.
    """

    out = eseries()

    if isoper(state.ampl[0]):
        out.rates = state.rates
        out.ampl = np.array([expect(oper, a) for a in state.ampl])

    else:
        out.rates = np.array([])
        out.ampl = np.array([])

        for m in range(len(state.rates)):
            op_m = state.ampl[m].data.conj().T * oper.data

            for n in range(len(state.rates)):
                a = op_m * state.ampl[n].data

                if isinstance(a, sp.spmatrix):
                    a = a.todense()

                out.rates = np.append(out.rates, state.rates[n] -
                                      state.rates[m])
                out.ampl = np.append(out.ampl, a)

    return out


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
    return expect(oper ** 2, state) - expect(oper, state) ** 2
