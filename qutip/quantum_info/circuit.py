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
from qutip.quantum_info.circuit_latex import _latex_write, _latex_pdf

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
        








