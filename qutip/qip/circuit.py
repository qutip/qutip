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
import warnings

from qutip.qip.circuit_latex import _latex_write, _latex_pdf, _latex_compile
from qutip.qip.gates import *

class Gate(object):
    
    def __init__(self, name, targets=None, controls=None, arg_value=None,
                 arg_label=None):
        self.name = name
        self.targets = targets
        self.controls = controls

        self.arg_value = arg_value
        self.arg_label = arg_label
        

_gate_name_to_label = {
    'CPHASE': r'{\rm R}',
    'RX': r'R_x',
    'RY': r'R_y',
    'RZ': r'R_z',
    'BERKELEY': r'{\rm BERKELEY}',
    'SWAPalpha': r'{\rm SWAPalpha}',
    'CNOT': r'{\rm CNOT}',
    'CSIGN': r'{\rm Z}',
    'FREDKIN': r'{\rm FREDKIN}',
    'TOFFOLI': r'{\rm TOFFOLI}',
    'SWAP': r'{\rm SWAP}',
    'ISWAP': r'{i}{\rm SWAP}',
    'SQRTISWAP': r'\sqrt{{i}\rm SWAP}',
    'SQRTSWAP': r'\sqrt{\rm SWAP}',
    'SQRTNOT': r'\sqrt{{i}\rm NOT}',
    'SNOT': r'{\rm H}',
    'PHASEGATE': r'{\rm PHASE}',
    'GLOBALPHASE': r'{\rm Ph}',
    }


def _gate_label(name, arg_label):

    if name in _gate_name_to_label:
        gate_label = _gate_name_to_label[name]
    else:
        warnings.warn("Unknown gate %s" % name)
        gate_label = name
    
    if arg_label:
        return r'%s(%s)' % (gate_label, arg_label)
    else:
        return r'%s' % gate_label


class QubitCircuit(object):
    """
    Representation of a quantum program/algorithm. It needs to maintain a list
    of gates (with target and source and time
    """
    def __init__(self, N, reverse_states=True):
        
        # number of qubits in the register
        self.N = N
        self.reverse_states = reverse_state        
        self.gates = []
        self.U_list = []
        
    def add_gate(self, name, targets=None, controls=None, arg_value=None,
                 arg_label=None):
        self.gates.append(Gate(name, targets=targets, controls=controls,
                               arg_value=arg_value, arg_label=arg_label))
 
   
    def resolve_gates(self, basis=["CNOT", "RX", "RY", "RZ"]):
        """
        Unitary matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right in the specified basis.
            
        Parameters
        ----------
        basis: list.
            Basis of the resolved circuit.

        Returns
        -------
        qc_temp: Qobj
            Returns Qobj of resolved gates for the qubit circuit in the desired basis.    
        """  
        qc_temp = QubitCircuit(self.N)
        temp_resolved = []
        qc_temp.gates = []

        basis_1q = []
        basis_2q = None

        basis_1q_valid = ["RX", "RY", "RZ"]
        basis_2q_valid = ["ISWAP", "CSIGN", "CNOT", "SQRTISWAP", "SQRTSWAP"]

        if isinstance(basis, list):
            for gate in basis:
                if gate not in (basis_1q_valid + basis_2q_valid):
                    raise ValueError("%s is not a valid basis gate" % gate)

                if gate in basis_2q_valid:
                    if basis_2q is not None:
                        raise ValueError("At most one two-qubit gate allowed")
                    basis_2q = gate

                else:                    
                    basis_1q.append(gate)

            if len(basis_1q) == 1:
                raise ValueError("Not sufficient single-qubit gates in basis")
            elif len(basis_1q) == 0:
                basis_1q = ["RX", "RY", "RZ"]    

        else:
            basis_1q = ["RX", "RY", "RZ"]
            if basis in basis_2q_valid:
                basis_2q = basis
            else:
                raise ValueError("%s is not a valid two-qubit basis gate" % basis)
        
        
        for gate in self.gates:    
            if gate.name == "CPHASE":
                raise NotImplementedError("Cannot be resolved by the program")
            elif gate.name == "RX":
                temp_resolved.append(Gate(gate.name, gate.targets, gate.controls,
                                          gate.arg_value, gate.arg_label))
            elif gate.name == "RY":
                temp_resolved.append(Gate(gate.name, gate.targets, gate.controls,
                                          gate.arg_value, gate.arg_label))
            elif gate.name == "RZ":
                temp_resolved.append(Gate(gate.name, gate.targets, gate.controls,
                                          gate.arg_value, gate.arg_label))
            elif gate.name == "CNOT":
                temp_resolved.append(Gate(gate.name, gate.targets, gate.controls,
                                          gate.arg_value, gate.arg_label))
            elif gate.name == "CSIGN" and not basis_2q is "CSIGN":
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("CNOT", gate.targets, gate.controls))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
            elif gate.name == "BERKELEY":
                raise NotImplementedError("Cannot be resolved by the program")
            elif gate.name == "SWAPalpha":
                raise NotImplementedError("Cannot be resolved by the program")
            elif gate.name == "FREDKIN":
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.controls]))
                temp_resolved.append(Gate("RZ", gate.controls, None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", [gate.targets[0]], None,
                                          arg_value=-np.pi/8, arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], gate.controls))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", [gate.targets[1]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RZ", [gate.targets[0]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], gate.controls))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=-np.pi/8, arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], gate.controls))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=-np.pi/8, arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", [gate.targets[1]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
            elif gate.name == "TOFFOLI":
                temp_resolved.append(Gate("CNOT", [gate.controls[1]], [gate.controls[0]]))
                temp_resolved.append(Gate("RZ", [gate.controls[0]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", [gate.controls[1]], None,
                                          arg_value=-np.pi/8, arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.controls[1]], [gate.controls[0]]))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RZ", [gate.controls[1]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets, [gate.controls[0]]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=-np.pi/8, arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets, [gate.controls[1]]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets, [gate.controls[0]]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=-np.pi/8, arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets, [gate.controls[1]]))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
            elif gate.name == "SWAP":
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
            elif gate.name == "ISWAP" and not basis_2q is "ISWAP":
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("RZ", [gate.targets[0]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", [gate.targets[0]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("RY", [gate.targets[0]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
            elif gate.name == "SQRTISWAP" and not basis_2q is "SQRTISWAP":
                raise NotImplementedError("Cannot be resolved by the program")
            elif gate.name == "SQRTSWAP" and not basis_2q is "SQRTSWAP":
                raise NotImplementedError("Cannot be resolved by the program")
            elif gate.name == "SQRTNOT":
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                                   arg_value=np.pi/4, arg_label=r"\pi/4"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
            elif gate.name == "SNOT":
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
            elif gate.name == "PHASEGATE":
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          gate.arg_value, gate.arg_label))
            elif gate.name == "GLOBALPHASE":
                temp_resolved.append(Gate(gate.name, gate.targets, gate.controls,
                                          gate.arg_value, gate.arg_label))
            else:
                temp_resolved.append(gate)


        if basis_2q == "CSIGN":
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))        
                    qc_temp.gates.append(Gate("CSIGN", gate.targets, gate.controls))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        elif basis_2q == "ISWAP":
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=np.pi/4, arg_label=r"\pi/4"))
                    qc_temp.gates.append(Gate("ISWAP", [gate.controls[0], gate.targets[0]], None))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RY", gate.controls, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("ISWAP", [gate.controls[0], gate.targets[0]], None))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        elif basis_2q == "SQRTISWAP":
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("RY", gate.controls, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RX", gate.controls, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("SQRTISWAP", [gate.controls[0], gate.targets[0]], None))
                    qc_temp.gates.append(Gate("RX", gate.controls, None,
                                              arg_value=np.pi, arg_label=r"\pi"))
                    qc_temp.gates.append(Gate("SQRTISWAP", [gate.controls[0], gate.targets[0]], None))
                    qc_temp.gates.append(Gate("RY", gate.controls, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=np.pi/4, arg_label=r"\pi/4"))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=np.pi, arg_label=r"\pi"))
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=3*np.pi/2, arg_label=r"3\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        elif basis_2q == "SQRTSWAP":
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("SQRTSWAP", [gate.controls[0], gate.targets[0]], None))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=np.pi, arg_label=r"\pi"))
                    qc_temp.gates.append(Gate("SQRTSWAP", [gate.controls[0], gate.targets[0]], None))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        else:
            qc_temp.gates = temp_resolved              


        if len(basis_1q) == 2:
            temp_resolved = qc_temp.gates
            qc_temp.gates = []
            for gate in temp_resolved:            
                if gate.name == "RX" and "RX" not in basis_1q:
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                elif gate.name == "RY" and "RY" not in basis_1q:
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))
                elif gate.name == "RZ" and "RZ" not in basis_1q:
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=-np.pi/2, arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=np.pi/2, arg_label=r"\pi/2"))            
                else:
                    qc_temp.gates.append(gate)            

        return qc_temp
                

    def unitary_matrix(self):
        """
        Unitary matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right.
        
        Returns
        -------
        self.U_list: list
            Returns list of unitary matrices for the qubit circuit.

        """          
        self.U_list = []

        for gate in self.gates:    
            if gate.name == "CPHASE":
                self.U_list.append(cphase(gate.arg_value, self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "RX":
                self.U_list.append(rx(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RY":
                self.U_list.append(ry(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RZ":
                self.U_list.append(rz(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "CNOT":
                self.U_list.append(cnot(self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "CSIGN":
                self.U_list.append(csign(self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "BERKELEY":
                self.U_list.append(berkeley(self.N, gate.targets))
            elif gate.name == "SWAPalpha":
                self.U_list.append(swapalpha(gate.arg_value, self.N, gate.targets))
            elif gate.name == "FREDKIN":
                self.U_list.append(fredkin(self.N, gate.controls, gate.targets[0]))
            elif gate.name == "TOFFOLI":
                self.U_list.append(toffoli(self.N, gate.controls, gate.targets[0]))
            elif gate.name == "SWAP":
                self.U_list.append(swap(self.N, gate.targets))
            elif gate.name == "ISWAP":
                self.U_list.append(iswap(self.N, gate.targets))
            elif gate.name == "SQRTISWAP":
                self.U_list.append(sqrtiswap(self.N, gate.targets))
            elif gate.name == "SQRTSWAP":
                self.U_list.append(sqrtswap(self.N, gate.targets))
            elif gate.name == "SQRTNOT":
                self.U_list.append(sqrtnot(self.N, gate.targets[0]))
            elif gate.name == "SNOT":
                self.U_list.append(snot(self.N, gate.targets[0]))
            elif gate.name == "PHASEGATE":
                self.U_list.append(phasegate(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "GLOBALPHASE":
                self.U_list.append(globalphase(gate.arg_value, self.N))

        return self.U_list


    def latex_code(self):
        rows = []

        gates = self.gates

        for gate in gates:
            col = []
            for n in range(self.N):
                if gate.targets and n in gate.targets:
                    
                    if len(gate.targets) > 1:
                        if (self.reverse_states and n == max(gate.targets)) or (not self.reverse_states and n == min(gate.targets)):
                            col.append(r" \multigate{%d}{%s} " %
                                       (len(gate.targets) - 1,
                                        _gate_label(gate.name, gate.arg_label)))
                        else:
                            col.append(r" \ghost{%s} " %
                                       (_gate_label(gate.name, gate.arg_label)))
                    
                    elif gate.name == "CNOT":
                            col.append(r" \targ ")
                    elif gate.name == "SWAP":
                        col.append(r" \qswap ")
                    else:
                        col.append(r" \gate{%s} " %
                                   _gate_label(gate.name, gate.arg_label))
                        
                elif gate.controls and n in gate.controls:
                    m = (gate.targets[0] - n) * (-1 if self.reverse_states else 1)
                    if gate.name == "SWAP":
                        col.append(r" \qswap \ctrl{%d} " % m)
                    else:
                        col.append(r" \ctrl{%d} " % m)
                
                elif (not gate.controls and not gate.targets):
                    # global gate
                    if (self.reverse_states and n == self.N - 1) or (not self.reverse_states and n == 0):
                        col.append(r" \multigate{%d}{%s} " %
                                   (self.N - 1,
                                    _gate_label(gate.name, gate.arg_label)))
                    else:
                        col.append(r" \ghost{%s} " %
                                   (_gate_label(gate.name, gate.arg_label)))

                else:
                    col.append(r" \qw ")

            col.append(r" \qw ")
            rows.append(col)

        code = ""
        n_iter = reversed(range(self.N)) if self.reverse_states else range(self.N)
        for n in n_iter:
            for m in range(len(gates)):
                code += r" & %s" % rows[m][n]
            code += r" & \qw \\ " + "\n"
        
        return code
        
    def _repr_png_(self):
        return _latex_compile(self.latex_code(), format="png")

    def _repr_svg_(self):
        return _latex_compile(self.latex_code(), format="svg")

    @property
    def png(self):
        from IPython.display import Image
        return Image(self._repr_png_(), embed=True)

    @property
    def svg(self):
        from IPython.display import SVG
        return SVG(self._repr_svg_())
    
    def qasm(self):
        
        code = "# qasm code generated by QuTiP\n\n"
        
        for n in range(self.N):
            code += "\tqubit\tq%d\n" % n
        
        code += "\n"
        
        for gate in self.gates:
            code += "\t%s\t" % gate.name
            qtargets = ["q%d" % t for t in gate.targets] if gate.targets else []
            qcontrols = ["q%d" % c for c in gate.controls] if gate.controls else []
            code += ",".join(qtargets + qcontrols)
            code += "\n"
            
        return code


class Circuit():
    """A class for representing quantum circuits.  Circuits may be defined for
    any combination of oscillator, qubit, and qudit components.
    
    """
    def __init__(self,num,name=None):
        # check for circuit name
        if name==None:
            name='QuTiP_Circuit'
        self.name=name
        
        self.record={'name': name, "elements" : num, 'element_order': ['']*num,'oscillator_components': {},
            'qubit_components': {}, 'qudit_components' : {}}


    def __str__(self):
        s = ""

        s += "Quantum Circuit: "
        s += "Number of elements : "+str(self.record['elements'])+"\n"
        s += "Num. of oscillators : "+str(len(self.record['oscillator_components']))+ ", "
        s += "Num. of qubits : "+str(len(self.record['qubit_components']))+ ", "
        s += "Num. of qudits : "+str(len(self.record['qudit_components']))+"\n"
        return s
    
    def __repr__(self):
        return self.__str__()
    
    def add_oscillator(self,N,element_order=None,label=None):
        """
        Adds an oscillator to the quantum circuit.
        
        Parameters
        ----------
        N : int
            Number of Fock states in oscillator Hilbert space
        element_order : int {optional}
            The element_order of the oscillator component in the tensor product structure
        label : str {optional}
            Custom label for the oscillator component
        
        """
        
        #add osc to list of osc components with label
        if label==None:
            num_of_osc=len(self.record['oscillator_components'])
            label='osc'+str(num_of_osc)
        self.record['oscillator_components'][label]=N
        
        #add osc to circuit in given element_order
        if element_order==None:
            ind=np.where(np.array(self.record['element_order'])=='')[0]
            if len(ind)==0:
                print('Circuit has no empty components.')
            else:
                element_order=ind[0]
        if element_order>self.record['elements']-1:
            raise ValueError('Oscillator element_order is higher than the total number of components.')
        self.record['element_order'][element_order]=label
        
    
    def add_qubit(self,element_order=None,label=None):
         """
         Adds a qubit to the quantum circuit.
        
         Parameters
         ----------
         element_order : int {optional}
             The element_order of the qubit component in the tensor product structure
         label : str {optional}
             Custom label for the qubit component
        
         """
         #add qubit to list of qubit components with label
         if label==None:
             num_of_qubit=len(self.record['qubit_components'])
             label='q'+str(num_of_qubit)
         self.record['qubit_components'][label]=2
         #add qubit to circuit in given element_order
         if element_order==None:
             ind=np.where(np.array(self.record['element_order'])=='')[0]
         if len(ind)==0:
             print('Circuit has no empty components.')
         else:
             element_order=ind[0]
         if element_order>self.record['elements']-1:
             raise ValueError('Qubit element_order is higher than the total number of components.')
         self.record['element_order'][element_order]=label
    
    def add_qudit(self,N,element_order=None,label=None):
         """
         Adds a qudit to the quantum circuit.
        
         Parameters
         ----------
         N : int
             Number of Fock states in qudit Hilbert space
         element_order : int {optional}
             The element_order of the qudit component in the tensor product structure
         label : str {optional}
             Custom label for the qudit component
        
         """
         #add qubit to list of qubit components with label
         if label==None:
             num_of_qubit=len(self.record['qudit_components'])
             label='q'+str(num_of_qubit)
         self.record['qudit_components'][label]=N
         #add qubit to circuit in given element_order
         if element_order==None:
             ind=np.where(np.array(self.record['element_order'])=='')[0]
         if len(ind)==0:
             print('Circuit has no empty components.')
         else:
             element_order=ind[0]
         if element_order>self.record['elements']-1:
             raise ValueError('Qudit element_order is higher than the total number of components.')
         self.record['element_order'][element_order]=label
    
                
    
    def output_latex(self):
        """
        Takes circuit description and outputs a latex file
        for the corresponding circuit diagram.
        """
        _latex_write(self.name)
    
    
    def output_pdf(self):
        """
        Takes circuit description and outputs both a latex and PDF file
        for the corresponding circuit diagram.
        """
        _latex_pdf(self.name)
        

