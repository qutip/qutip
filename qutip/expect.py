# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import numpy as np
import scipy.sparse as sp

from qutip.qobj import Qobj, issuper, isoper
from qutip.eseries import eseries
from qutip.cyQ.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi)


expect_rho_vec = cy_expect_rho_vec
expect_psi = cy_expect_psi


def expect(oper, state):
    '''Calculates the expectation value for operator and state(s).

    Parameters
    ----------
    oper : qobj
        Operator for expectation value.

    state : qobj/list
        A single or `list` of quantum states or density matrices..

    Returns
    -------
    expt : float
        Expectation value.  ``real`` if `oper` is Hermitian, ``complex``
        otherwise.

    Examples
    --------
    >>> expect(num(4), basis(4, 3))
    3

    '''
    if isinstance(state, Qobj) and isinstance(oper, Qobj):
        return _single_qobj_expect(oper, state)

    elif isinstance(oper, Qobj) and isinstance(state, eseries):
        return _single_eseries_expect(oper, state)

    elif isinstance(state, np.ndarray) or isinstance(state, list):
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
        if state.type == 'oper':
            # calculates expectation value via TR(op*rho)
            prod = oper.data * state.data
            tr = prod.diagonal().sum()
            if oper.isherm and state.isherm:
                return float(np.real(tr))
            else:
                return tr

        elif state.type == 'ket':
            # calculates expectation value via <psi|op|psi>
            prod = state.data.conj().T.dot(oper.data * state.data)
            if oper.isherm:
                return float(np.real(prod[0, 0]))
            else:
                return prod[0, 0]
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
