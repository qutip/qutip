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

import sys
sys.path.append('..')
from qutip import *

import unittest

class TestBasisTransformations(unittest.TestCase):

    """
    A test class for the QuTiP function for performing basis transformations
    """

    def setUp(self):
        """
        setup
        """

    def testTransformation1(self):
        """
        Transform a two-level hamiltonian to its eigebasis and back.
        """

        H1 = rand() * sigmax() + rand() * sigmay() + rand() * sigmaz()
        ekets, evals = H1.eigenstates()
        Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
        H2 = Heb.transform(ekets, True) # back to original basis
        
        #print "H1 =\n", H1
        #print "H2 =\n", H2        
        
        self.assertTrue((H1 - H2).norm() < 1e-6)


    def testTransformation2(self):
        """
        Transform a 10-level real hamiltonian to its eigebasis and back.
        """
        N = 10
        H1 = Qobj((0.5-rand(N,N)))
        H1 = H1 + H1.dag()
        ekets, evals = H1.eigenstates()
        Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
        H2 = Heb.transform(ekets, True) # back to original basis
        self.assertTrue((H1 - H2).norm() < 1e-6)
        

    def testTransformation3(self):
        """
        Transform a 10-level hamiltonian to its eigebasis and back.
        """
        N = 10
        H1 = Qobj((0.5-rand(N,N)) + 1j*(0.5-rand(N,N)))
        H1 = H1 + H1.dag()
        ekets, evals = H1.eigenstates()
        Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
        H2 = Heb.transform(ekets, True) # back to original basis
        self.assertTrue((H1 - H2).norm() < 1e-6)    
        
            
    def testTransformation4(self):
        """
        Transform a 10-level imaginary hamiltonian to its eigebasis and back.
        """
        N = 10
        H1 = Qobj(1j*(0.5-rand(N,N)))
        H1 = H1 + H1.dag()
        ekets, evals = H1.eigenstates()
        Heb = H1.transform(ekets)       # eigenbasis (should be diagonal)
        H2 = Heb.transform(ekets, True) # back to original basis
        self.assertTrue((H1 - H2).norm() < 1e-6)    
     

if __name__ == '__main__':

    unittest.main()
