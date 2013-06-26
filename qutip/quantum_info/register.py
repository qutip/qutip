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
from qutip.qobj import *
from qutip.states import basis
from qutip.tensor import tensor


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
        # give complete information on register without print statement in
        # command-line we cant realistically serialize a Qobj into a string,
        # so we simply return the informal __str__ representation instead.)
        return self.__str__()


