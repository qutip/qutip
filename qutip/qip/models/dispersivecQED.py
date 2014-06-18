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
from qutip.qip.gates import *
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.circuitprocessor import CircuitProcessor


class DispersivecQED(CircuitProcessor):
    """
    Representation of the physical implementation of a quantum program/algorithm
    on a dispersive cavity-QED system.
    """
    
    def __init__(self, N, Nres=5, correct_global_phase=True):
        """
        
        H = .... + g_n * (a^\dagger \sigma_- + a \sigma_+)
        
        """
        super(DispersiveCQED, self).__init__(N, correct_global_phase)

        self.Nres = Nres
        self.w0 = 2 # 2 * pi * 2
        self.Delta = 0.8 #2 * pi * 0.8
        
        self.sx_ops = [tensor([identity(Nres)] + [sigmax() if m == n 
                              else identity(2) for n in range(N)])
                       for m in range(N)]
        self.sz_ops = [tensor([identity(Nres)] + [sigmaz() if m == n 
                              else identity(2) for n in range(N)])
                       for m in range(N)]

        self.a = tensor([destroy(Nres)] + [identity(2) for n in range(N)])

        self.cavityqubit_ops = []
        for n in range(N):
            sm = tensor([identity(Nres)] + [destroy(2) if m == n 
                                            else identity(2) for m in range(N)])
            self.cavityqubit_ops.append(self.a.dag() * sm + self.a * sm.dag())

        self.psi_proj = tensor([basis(Nres, 0)] 
                        + [identity(2) for n in range(N)])

        self.sx_coeff = [1.0] * N
        self.sz_coeff = [1.0] * N
        self.g = [0.001 ] * N #* 2 * pi
 
        
    def get_ops_and_u(self):
        H0 = self.a.dag() * self.a
        return ([H0] + self.sx_ops + self.sz_ops + self.cavityqubit_ops,
                hstack((self.w0 * np.zeros((self.sx_u.shape[0], 1)), 
                self.sx_u, self.sz_u, self.g_u)))

    
    def get_ops_labels(self):
        return ([r"$a^\dagger a$"] + [r"$\sigma_x^%d$" % n 
                                      for n in range(self.N)] + 
                [r"$\sigma_z^%d$" % n for n in range(self.N)] + 
                [r"$\g{%d}$" % (n) for n in range(self.N)])

    
    def optimize_circuit(self, qc):
        self.qc0 = qc
        qc_temp = qc.resolve_gates(basis=["ISWAP", "RX", "RZ"])
        self.qc1 = qc_temp
        qc = self.dispersive_gate_correction(qc_temp)
        self.qc2 = qc

        return qc

    
    def dispersive_gate_correction(self, qc, rwa=True):
        """
        Method to resolve ISWAP and SQRTISWAP gates in a cQED system by adding 
        single qubit gates to get the correct output matrix.
        
        Parameters
        ----------
        qc: Qobj
            The circular spin chain circuit to be resolved
        
        rwa: Boolean
            Specify if RWA is used or not.
            
        Returns
        ----------
        qc_temp: Qobj
            Returns Qobj of resolved gates for the qubit circuit in the desired 
            basis.            
        """  
        qc_temp = QubitCircuit(qc.N, qc.reverse_states)
        
        for gate in qc.gates:       
            qc_temp.gates.append(gate)
            if rwa:
                if gate.name == "SQRTISWAP":
                    qc_temp.gates.append(Gate("RZ", [gate.targets[0]], None,
                                              arg_value=-np.pi/4, 
                                              arg_label=r"-\pi/4"))
                    qc_temp.gates.append(Gate("RZ", [gate.targets[1]], None,
                                              arg_value=-np.pi/4, 
                                              arg_label=r"-\pi/4"))
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=-np.pi/4, 
                                              arg_label=r"-\pi/4"))       
                elif gate.name == "ISWAP":
                    qc_temp.gates.append(Gate("RZ", [gate.targets[0]], None,
                                              arg_value=-np.pi/2, 
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", [gate.targets[1]], None,
                                              arg_value=-np.pi/2, 
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=-np.pi/2, 
                                              arg_label=r"-\pi/2"))
            
        return qc_temp
    
    def eliminate_auxillary_modes(self, U):
        return self.psi_proj.dag() * U * self.psi_proj
    
    def load_circuit(self, qc):
        
        gates = self.optimize_circuit(qc).gates
        
        self.global_phase = 0
        self.sx_u = np.zeros((len(gates), len(self.sx_ops)))
        self.sz_u = np.zeros((len(gates), len(self.sz_ops)))
        self.g_u = np.zeros((len(gates), len(self.cavityqubit_ops)))
        self.T_list = []
        
        n = 0
        for gate in gates:
            
            if gate.name == "ISWAP":
                self.sz_u[n, gate.targets[0]] = self.w0 - self.Delta            
                self.sz_u[n, gate.targets[1]] = self.w0 - self.Delta            
                self.g_u[n, gate.targets[0]] = self.g[gate.targets[0]] 
                self.g_u[n, gate.targets[1]] = self.g[gate.targets[1]]
                T = self.Delta / (4 * self.g[gate.targets[0]] * 
                                  self.g[gate.targets[1]])
                self.T_list.append(T)
                n += 1
                
            elif gate.name == "SQRTISWAP":
                self.sz_u[n, gate.targets[0]] = self.w0 - self.Delta            
                self.sz_u[n, gate.targets[1]] = self.w0 - self.Delta            
                self.g_u[n, gate.targets[0]] = self.g[gate.targets[0]] 
                self.g_u[n, gate.targets[1]] = self.g[gate.targets[1]]
                T = self.Delta / (8 * self.g[gate.targets[0]] * 
                                  self.g[gate.targets[1]])
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

