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
from numpy import allclose
import unittest

class TestRand(unittest.TestCase):

    """
    A test class for the built-in QuTiP operators
    """

    def setUp(self):
        """
        setup
        """
    def testRandUnitary(self):
        U=array([rand_unitary(5) for k in range(5)])
        for k in range(5):
            self.assertEqual(U[k]*U[k].dag()==qeye(5), True)
    
    def testRandherm(self):
        H=array([rand_herm(5) for k in range(5)])
        for k in range(5):
            self.assertEqual(H[k].isherm==True, True)
            
    def testRanddm(self):
        R=array([rand_dm(5) for k in range(5)])
        for k in range(5):
            self.assertEqual(sum(R[k].tr())-1.0<1e-15, True)
    
    def testRandket(self):
        P=array([rand_ket(5) for k in range(5)])
        for k in range(5):
            self.assertEqual(P[k].type=='ket', True)
        
if __name__ == '__main__':
    unittest.main()
