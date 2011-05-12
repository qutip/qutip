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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from eseries import *
from Qobj import *
from istests import *
import numpy as np
import scipy.sparse as sp

def expect(oper,state):
    ''' Calculates expectation value of an operator for a given state-vector or density matrix.  
    USAGE:
        - expect(oper,state) returns expactation value corresponding to operator 'oper' and state 'state'
        - expect(oper,rho) returns expactation value corresponding to operator 'oper' and density matrix 'rho'
    OUTPUT:
        returns Float (real number) if operator is Hermitian; otherwise returns Complex number
    
    '''
    if isinstance(oper,Qobj) and isinstance(state,Qobj):
        if isoper(oper):
            if isoper(state):
                #calculates expectation value via TR(op*rho)
                prod = (oper.data*state.data)
                if isinstance(prod, sp.spmatrix):
                    prod = prod.tocsr()
                num=prod.shape[0]
                tr=0.0j
                for j in xrange(num):
                    tr+=prod[j,j]
                if oper.isherm:
                    return real(tr)
                else:
                    return tr
            elif isket(state):
                #calculates expectation value via <psi|op|psi>

                #prod = state.data.conj().T * (oper.data * state.data)
                prod = dot(state.data.conj().T, oper.data * state.data)
                if isinstance(prod, sp.spmatrix):
                    prod = prod.tocsr()

                if oper.isherm:
                    return real(prod[0,0])
                else:
                    return prod[0,0]
        else:
            raise TypeError('Invalid operand types')

    #
    # eseries
    # 
    elif isinstance(oper,Qobj) and isinstance(state, eseries):

        out = eseries()
        out.rates = state.rates
        out.ampl = array([expect(oper, a) for a in state.ampl])

        return out
    else:# unsupported types
        raise TypeError('Arguments must be quantum objects or eseries')


