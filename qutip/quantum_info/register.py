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
        s += ("Quantum Register: " +
            ", width = " + self.width() + "\n")
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
    
    #Single Qubit gates
    #----------------------
    def apply_hadamard(self, target):
        #Applies Hadamard gate to target qubits
        target=np.asarray(target)
        _reg_input_check(target,self.width()) 
        H=1.0 / sqrt(2.0) * (sigmaz()+sigmax())
        reg_gate=_single_op_reg_gate(H,target,self.width())
        self.data=(reg_gate*self).data
        
    def apply_not(self,target):
        #Applies NOT gate (sigmax) to target qubits
        target=np.asarray(target)
        _reg_input_check(target,self.width())
        reg_gate=_single_op_reg_gate(sigmax(),target,self.width())
        self.data=(reg_gate*self).data
    
    def apply_sigmaz(self,target):
        #Applies sigmaz to target qubits
        target=np.asarray(target)
        _reg_input_check(target,self.width())
        reg_gate=_single_op_reg_gate(sigmaz(),target,self.width())
        self.data=(reg_gate*self).data
    
    def apply_sigmay(self,target):
        #Applies sigmaz to target qubits
        target=np.asarray(target)
        _reg_input_check(target,self.width())
        reg_gate=_single_op_reg_gate(sigmay(),target,self.width())
        self.data=(reg_gate*self).data
    
    def apply_sigmax(self,target):
        #Applies sigmax, same as NOT
        self.apply_not(self,target,self.width())
    
    def apply_phasegate(self, target, phase=0):
        #Applies phase gate to target qubits
        target=np.asarray(target)
        _reg_input_check(target,self.width()) 
        P=fock_dm(2,0)+np.exp(1.0j*phase)*fock_dm(2,1)
        reg_gate=_single_op_reg_gate(P,target,self.width())
        self.data=(reg_gate*self).data



########################################################
# End Register class
########################################################

def _reg_input_check(A,width):
    """
    Checks if all elements of input are integers >=0
    and that they are within the register width.
    """
    int_check=np.equal(np.mod(A, 1), 0)
    if np.any(int_check==False):
        raise TypeError('Target and control qubit indices must be integers.')
    if np.any(A<0):
        raise ValueError('Target and control qubit indices must be positive.')
    if np.any(A>width-1):
        raise ValueError('Qubit indices must be within the register width.')


def _single_op_reg_gate(op,target,width):
    """
    Constructs register gate composed of single-qubit operators
    """
    I=qeye(2)  
    H=1.0 / sqrt(2.0) * (sigmaz()+sigmax())
    if 0 in target:
        op_list=[op]
    else:
        op_list=[I]
    for kk in range(1,width):
        if kk in target:
            op_list+=[op]
        else:
            op_list+=[I]    
    return tensor(op_list)
    
















