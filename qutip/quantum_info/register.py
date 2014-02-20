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
import numpy as np
from qutip.qobj import *
from qutip.states import *
from qutip.operators import *
from qutip.tensor import tensor
from qutip.quantum_info.gates import *
from qutip.quantum_info.utils import _reg_str2array

class Register(Qobj):
    """A class for representing quantum registers.  Subclass
    of the quantum object (Qobj) class.
    """
    def __init__(self,N,state=None):
        #construct register intial state
        if state==None:
            reg=tensor([basis(2) for k in range(N)])
        if isinstance(state,str):
            state=_reg_str2array(state,N)
            reg=tensor([basis(2,state[k]) for k in state])
        Qobj.__init__(self, reg.data, reg.dims, reg.shape,
                 reg.type, reg.isherm, fast=False)
        
    def width(self):
        # gives the number of qubits in register.
        return len(self.dims[0])
    
    def __str__(self):
        s = ""
        s += ("Quantum Register: " +
            ", width = " + str(self.width()) + ", type = " + self.type + "\n")
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
        if self.type=='ket':
            self.data=(reg_gate*self).data
        else:
            self.data=(reg_gate*self*reg_gate.dag()).data
            
        
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
    
















