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
        self.gates = []
        self.gates_resolved = []
        self.U_list = []
        self.U_list_resolved = []
        self.reverse_states = reverse_states

    def add_gate(self, name, targets=None, controls=None, arg_value=None,
                 arg_label=None):
        self.gates.append(Gate(name, targets=targets, controls=controls,
                               arg_value=arg_value, arg_label=arg_label))
 
   
    def qubit_circuit_resolved(self, basis="CNOT"):
        """
        Unitary matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right in the specified basis.
            
        Parameters
        ----------
        basis: string.
            Basis of the resolved circuit.
            
        """  
        temp_resolved = []
        self.gates_resolved = []
        
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
            elif gate.name == "CSIGN" and not basis is "CSIGN":
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("CNOT", gate.targets, gate.controls))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
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
                                          arg_value=-np.pi/8, arg_label=r"\-pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], gate.controls))
                temp_resolved.append(Gate("RY", [gate.targets[1]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", [gate.targets[1]], None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RZ", [gate.targets[0]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], gate.controls))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=-np.pi/8, arg_label=r"\-pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], gate.controls))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=-np.pi/8, arg_label=r"\-pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("RY", [gate.targets[1]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", [gate.targets[1]], None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
            elif gate.name == "TOFFOLI":
                temp_resolved.append(Gate("CNOT", [gate.controls[1]], [gate.controls[0]]))
                temp_resolved.append(Gate("RZ", [gate.controls[0]], None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", [gate.controls[1]], None,
                                          arg_value=-np.pi/8, arg_label=r"\-pi/8"))
                temp_resolved.append(Gate("CNOT", [gate.controls[1]], [gate.controls[0]]))
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
                                          arg_value=-np.pi/8, arg_label=r"\-pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets, [gate.controls[1]]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi/8, arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets, [gate.controls[0]]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=-np.pi/8, arg_label=r"\-pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets, [gate.controls[1]]))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
            elif gate.name == "SWAP":
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
            elif gate.name == "ISWAP" and not basis is "ISWAP":
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[1]], [gate.targets[0]]))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("RY", [gate.targets[0]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RZ", [gate.targets[1]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RZ", [gate.targets[0]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", [gate.targets[0]], None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("CNOT", [gate.targets[0]], [gate.targets[1]]))
                temp_resolved.append(Gate("RY", [gate.targets[0]], None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", [gate.targets[0]], None,
                                          arg_value=np.pi, arg_label=r"\pi"))
            elif gate.name == "SQRTISWAP" and not basis is "SQRTISWAP":
                raise NotImplementedError("Cannot be resolved by the program")
            elif gate.name == "SQRTSWAP" and not basis is "SQRTSWAP":
                raise NotImplementedError("Cannot be resolved by the program")
            elif gate.name == "SQRTNOT":
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
            elif gate.name == "SNOT":
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi/2, arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
            elif gate.name == "PHASEGATE":
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          gate.arg_value, gate.arg_label))
            
        if basis == "CSIGN":
            for gate in temp_resolved:
                if temp_resolved.name == "CNOT":
                    self.gates_resolved.append(Gate("RY", gate.targets, None,
                                                    arg_value=-np.pi/2, arg_label=r"\-pi/2"))        
                    self.gates_resolved.append(Gate("CSIGN", gate.targets, gate.controls))
                    self.gates_resolved.append(Gate("RY", gate.targets, None,
                                                   arg_value=np.pi/2, arg_label=r"\pi/2"))
                else:
                    self.gates_resolved.append(gate)
        elif basis == "ISWAP":
            for gate in temp_resolved:
                if temp_resolved.name == "CNOT":
                    self.gates_resolved.append(Gate("ISWAP", [gate.controls[0], gate.targets[0]], None))
                    self.gates_resolved.append(Gate("RZ", gate.targets, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                    self.gates_resolved.append(Gate("RY", gate.controls, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                    self.gates_resolved.append(Gate("RZ", gate.controls, None,
                                                   arg_value=np.pi/2, arg_label=r"\pi/2"))
                    self.gates_resolved.append(Gate("ISWAP", [gate.controls[0], gate.targets[0]], None))
                    self.gates_resolved.append(Gate("RY", gate.targets, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                    self.gates_resolved.append(Gate("RZ", gate.targets, None,
                                                   arg_value=np.pi/2, arg_label=r"\pi/2"))
                else:
                    self.gates_resolved.append(gate)
        elif basis == "SQRTISWAP":
            for gate in temp_resolved:
                if temp_resolved.name == "CNOT":
                    self.gates_resolved.append(Gate("RY", gate.controls, None,
                                                   arg_value=np.pi/2, arg_label=r"\pi/2"))
                    self.gates_resolved.append(Gate("SQRTISWAP", [gate.controls[0], gate.targets[0]], None))
                    self.gates_resolved.append(Gate("RX", gate.controls, None,
                                                   arg_value=np.pi, arg_label=r"\pi"))
                    self.gates_resolved.append(Gate("SQRTISWAP", [gate.controls[0], gate.targets[0]], None))
                    self.gates_resolved.append(Gate("RX", gate.targets, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                    self.gates_resolved.append(Gate("RX", gate.controls, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                    self.gates_resolved.append(Gate("RY", gate.controls, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                else:
                    self.gates_resolved.append(gate)
        elif basis == "SQRTSWAP":
            for gate in temp_resolved:
                if temp_resolved.name == "CNOT":
                    self.gates_resolved.append(Gate("RY", gate.targets, None,
                                                   arg_value=np.pi/2, arg_label=r"\pi/2"))
                    self.gates_resolved.append(Gate("SQRTSWAP", [gate.controls[0], gate.targets[0]], None))
                    self.gates_resolved.append(Gate("RZ", gate.controls, None,
                                                   arg_value=np.pi, arg_label=r"\pi"))
                    self.gates_resolved.append(Gate("SQRTSWAP", [gate.controls[0], gate.targets[0]], None))
                    self.gates_resolved.append(Gate("RZ", gate.targets, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                    self.gates_resolved.append(Gate("RY", gate.targets, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                    self.gates_resolved.append(Gate("RZ", gate.controls, None,
                                                   arg_value=-np.pi/2, arg_label=r"\-pi/2"))
                else:
                    self.gates_resolved.append(gate)
        else:
            self.gates_resolved = temp_resolved              
                
                
    def unitary_matrix(self, resolved=False):
        """
        Unitary matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right.
        
        Parameters
        ----------
        resolved: Boolean.
            Checks if matrix is for resolved circuit or not.
        
        Returns
        -------
        unitary_list: list
            Returns list of unitary matrices for the qubit circuit.

        """          
        if resolved:
            gatelist = self.gates_resolved
            unitary_list = self.U_list_resolved = []
        else:
            gatelist = self.gates
            unitary_list = self.U_list = []

        for gate in gatelist:    
            if gate.name == "CPHASE":
                unitary_list.append(cphase(gate.arg_value, self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "RX":
                unitary_list.append(rx(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RY":
                unitary_list.append(ry(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RZ":
                unitary_list.append(rz(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "CNOT":
                unitary_list.append(cnot(self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "CSIGN":
                unitary_list.append(csign(self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "BERKELEY":
                unitary_list.append(berkeley(self.N, gate.targets))
            elif gate.name == "SWAPalpha":
                unitary_list.append(swapalpha(gate.arg_value, self.N, gate.targets))
            elif gate.name == "FREDKIN":
                unitary_list.append(fredkin(self.N, gate.controls, gate.targets[0]))
            elif gate.name == "TOFFOLI":
                unitary_list.append(toffoli(self.N, gate.controls, gate.targets[0]))
            elif gate.name == "SWAP":
                unitary_list.append(swap(self.N, gate.targets))
            elif gate.name == "ISWAP":
                unitary_list.append(iswap(self.N, gate.targets))
            elif gate.name == "SQRTISWAP":
                unitary_list.append(sqrtiswap(self.N, gate.targets))
            elif gate.name == "SQRTSWAP":
                unitary_list.append(sqrtswap(self.N, gate.targets))
            elif gate.name == "SQRTNOT":
                unitary_list.append(sqrtnot(self.N, gate.targets[0]))
            elif gate.name == "SNOT":
                unitary_list.append(snot(self.N, gate.targets[0]))
            elif gate.name == "PHASEGATE":
                unitary_list.append(phasegate(gate.arg_value, self.N, gate.targets[0]))

        return unitary_list


    def latex_code(self):
        rows = []

        if len(self.gates_resolved) > 0:
            gates = self.gates_resolved
        else:
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
        

