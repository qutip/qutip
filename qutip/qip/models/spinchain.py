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
import scipy.sparse as sp
from qutip.qobj import *
from qutip.qip.gates import snot, cphase, swap
from qutip.qip.circuit import QubitCircuit


class SpinChain(CircuitProcessor):
    """
    Representation of the physical implementation of a quantum program/algorithm
    on a spin chain qubit system.
    """
    
    def __init__(self, N, correct_global_phase=True):

        super(SpinChain, self).__init__(N, correct_global_phase)
        
        self.sx_ops = [tensor([sigmax() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        self.sz_ops = [tensor([sigmaz() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]

        self.sxsy_ops = []
        for n in range(N-1):
            x = [identity(2)] * N
            x[n] = x[n+1] = sigmax()
            y = [identity(2)] * N
            y[n] = y[n+1] = sigmay()
            self.sxsy_ops.append(tensor(x) + tensor(y))
        
        
        self.sx_coeff = [0.25 * 2 * pi] * N
        self.sz_coeff = [1.0 * 2 * pi] * N
        self.sxsy_coeff = [0.1 * 2 * pi] * (N - 1)

        
    def get_ops_and_u(self):
        return (self.sx_ops + self.sz_ops + self.sxsy_ops,
                hstack((self.sx_u, self.sz_u, self.sxsy_u)))

    
    def get_ops_labels(self):
        return ([r"$\sigma_x^%d$" % n for n in range(self.N)] + 
                [r"$\sigma_z^%d$" % n for n in range(self.N)] + 
                [r"$\sigma_x^%d\sigma_x^{%d} + \sigma_y^%d\sigma_y^{%d}$"
                 % (n, n, n, n) for n in range(self.N-1)])


    def load_circuit(self, qc):
        
        gates = self.optimize_circuit(qc).gates
        
        self.global_phase = 0
        self.sx_u = np.zeros((len(gates), len(self.sx_ops)))
        self.sz_u = np.zeros((len(gates), len(self.sz_ops)))
        self.sxsy_u = np.zeros((len(gates), len(self.sxsy_ops)))
        self.T_list = []
        
        n = 0
        for gate in gates:
            
            if gate.name == "ISWAP":
                g = self.sxsy_coeff[min(gate.targets)]
                self.sxsy_u[n, min(gate.targets)] = -g
                T = pi / (4 * g)
                self.T_list.append(T)
                n += 1

            elif gate.name == "SQRTISWAP":
                g = self.sxsy_coeff[min(gate.targets)]
                self.sxsy_u[n, min(gate.targets)] = -g
                T = pi / (8 * g)
                self.T_list.append(T)
                n += 1

            elif gate.name == "RZ":
                g = self.sz_coeff[gate.targets[0]]
                self.sz_u[n, gate.targets[0]] = np.sign(gate.arg_value) * g
                T = abs(gate.arg_value) / (2 * g)
                self.T_list.append(T)
                n += 1
                
            elif gate.name == "RX":
                g = self.sx_coeff[gate.targets[0]]
                self.sx_u[n, gate.targets[0]] = np.sign(gate.arg_value) * g
                T = abs(gate.arg_value) / (2 * g)
                self.T_list.append(T)
                n += 1
                
            elif gate.name == "GLOBALPHASE":
                self.global_phase += gate.arg_value
                
            else:
                raise ValueError("Unsupported gate %s" % gate.name)


class LinearSpinChain(SpinChain):
    """
    Representation of the physical implementation of a quantum program/algorithm
    on a spin chain qubit system arranged in a linear formation. It is a 
    sub-class of SpinChain.
    """
    
    def __init__(self, N, correct_global_phase=True):

        super(LinearSpinChain, self).__init__(N, correct_global_phase)


    def optimize_circuit(self, qc):    
        self.qc0 = qc
        qc_temp = SpinChain.adjacent_gates(qc, "linear")
        self.qc1 = qc_temp
        qc = qc_temp.resolve_gates(basis=["ISWAP", "RX", "RZ"])
        self.qc2 = qc
        return qc    


class CircularSpinChain(SpinChain):
    """
    Representation of the physical implementation of a quantum program/algorithm
    on a spin chain qubit system arranged in a circular formation. It is a 
    sub-class of SpinChain.
    """
    
    def __init__(self, N, correct_global_phase=True):

        super(CircularSpinChain, self).__init__(N, correct_global_phase)


    def optimize_circuit(self, qc):    
        self.qc0 = qc
        qc_temp = SpinChain.adjacent_gates(qc, "circular")
        self.qc1 = qc_temp
        qc = qc_temp.resolve_gates(basis=["ISWAP", "RX", "RZ"])
        self.qc2 = qc
        return qc        
