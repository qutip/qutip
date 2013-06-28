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
from qutip.qobj import *
from qutip.states import *
from qutip.operators import *
from qutip.tensor import tensor
from qutip.gates import *

class Register(Qobj):
    """A class for representing quantum registers.  Subclass
    of the quantum object (Qobj) class.
    """
    def __init__(self,N,state=None):
        if state==None:
            reg=tensor([basis(2) for k in range(N)])
        Qobj.__init__(self, reg.data, reg.dims, reg.shape,
                 reg.type, reg.isherm, fast=False)
        
    def width(self):
        # gives the number of qubits in register.
        return len(self.dims[0])
    
    def __str__(self):
        s = ""
        if self.type == 'oper' or self.type == 'super':
            s += ("Quantum Register: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(self.shape) +
                  ", type = " + self.type +
                  ", isherm = " + str(self.isherm) + "\n")
        else:
            s += ("Quantum Register: " +
                  "dims = " + str(self.dims) +
                  ", shape = " + str(self.shape) +
                  ", type = " + self.type + "\n")
        s += "Register data =\n"
        if all(np.imag(self.data.data) == 0):
            s += str(np.real(self.full()))
        else:
            s += str(self.full())
        return s
    
    def __repr__(self):
        return self.__str__()
    
    ########################################################
    # Gate operations begin here
    ########################################################
    
    def apply_hadamard(self, target):
        #Applies Hadamard gate to target qubits
        target=np.asarray(target)
        _reg_input_check(target,self.width())
        u=basis(2,0)
        d=basis(2,1)
        I=qeye(2)  
        H=1.0 / sqrt(2.0) * (sigmaz()+sigmax())
        if 0 in target:
            op_list=[H]
        else:
            op_list=[I]
        for kk in range(1,self.width()):
            if kk in target:
                op_list+=[H]
            else:
                op_list+=[I]    
        reg_gate=tensor(op_list)  
        #apply reg_gate to register
        self.data=(reg_gate*self).data



########################################################
# End Register class
########################################################

def _reg_input_check(A,width):
    """
    Checks if all elements of input are integers >=0
    and that they are within the register width.
    """
    A=np.asarray(A)
    int_check=np.equal(np.mod(A, 1), 0)
    if np.any(int_check==False):
        raise TypeError('Target and control qubit indices must be integers.')
    if np.any(A<0):
        raise ValueError('Target and control qubit indices must be positive.')
    if np.any(A>width-1):
        raise ValueError('Qubit indices must be within the register width.')

