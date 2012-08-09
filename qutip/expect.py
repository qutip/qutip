#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from qutip.eseries import *
from qutip.Qobj import *
from qutip.istests import *
import numpy as np
import scipy.sparse as sp

def expect(oper,state):
    '''Calculates the expectation value for operator and state(s).
    
    Parameters
    ----------
    oper : qobj 
        Operator for expectation value.
        
    state : qobj/list
        A single or `list` of quantum states or density matricies.
    
    Returns
    -------
    expt : float
        Expectation value.  ``real`` if `oper` is Hermitian, ``complex``
        otherwise.
    
    Examples
    --------    
    >>> expect(num(4),basis(4,3))
    3
        
    '''
    if isinstance(state,Qobj) or isinstance(state, eseries):
        return single_expect(oper,state)
    elif isinstance(state,np.ndarray) or isinstance(state,list):
        if oper.isherm and all([(op.isherm or op.type=='ket') for op in state]):
            return array([single_expect(oper,x) for x in state])
        else:
            return array([single_expect(oper,x) for x in state],dtype=complex)


def single_expect(oper,state):
    """
    Private function used by expect
    """
    if isinstance(oper,Qobj) and isinstance(state,Qobj):
        if isoper(oper):
            if state.type=='oper':
                #calculates expectation value via TR(op*rho)
                prod = oper.data*state.data
                tr=sum(prod.diagonal()) #sum of diagonal elements
                if oper.isherm and state.isherm: #if hermitian
                    return float(real(tr))
                else: #not hermitian
                    return tr
            elif state.type=='ket':
                #calculates expectation value via <psi|op|psi>
                #prod = state.data.conj().T * (oper.data * state.data)
                prod = state.data.conj().T.dot(oper.data * state.data)
                if oper.isherm:
                    return float(real(prod[0,0]))
                else:
                    return prod[0,0]
        else:
            raise TypeError('Invalid operand types')
    # eseries
    # 
    elif isinstance(oper,Qobj) and isinstance(state, eseries):
        out = eseries()

        if isoper(state.ampl[0]):

            out.rates = state.rates
            out.ampl = array([expect(oper, a) for a in state.ampl])

        else:

            out.rates = array([])
            out.ampl  = array([])

            for m in range(len(state.rates)):

                op_m = state.ampl[m].data.conj().T * oper.data

                for n in range(len(state.rates)):

                    a = op_m * state.ampl[n].data

                    if isinstance(a, sp.spmatrix):
                        a = a.todense()

                    out.rates = append(out.rates, state.rates[n] - state.rates[m])
                    out.ampl  = append(out.ampl, a)

        return out
    else:# unsupported types
        raise TypeError('Arguments must be quantum objects or eseries')

