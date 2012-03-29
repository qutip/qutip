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
from qutip import *

import unittest

class TestEigenstates(unittest.TestCase):

    """
    A test class for the QuTiP function for calculating eigenstates/values
    """

    def setUp(self):
        """
        setup
        """

    def testHamiltonian1(self):
        """
        Test diagonalization of a random two-level system
        """

        H = rand() * sigmax() + rand() * sigmay() + rand() * sigmaz()

        evals, ekets = H.eigenstates()

        for n in range(len(evals)):
            # assert that max(H * ket - e * ket) is small
            self.assertTrue(amax(abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10)

    def testHamiltonian2(self):
        """
        Test diagonalization of a composite system
        """

        H1 = rand() * sigmax() + rand() * sigmay() + rand() * sigmaz()
        H2 = rand() * sigmax() + rand() * sigmay() + rand() * sigmaz()

        H = tensor(H1, H2)

        evals, ekets = H.eigenstates()

        for n in range(len(evals)):
            # assert that max(H * ket - e * ket) is small
            self.assertTrue(amax(abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10)

    def testHamiltonian3(self):
        """
        Test diagonalization of another composite system
        """
        N1 = 10
        N2 = 2
        
        a1 = tensor(destroy(N1), qeye(N2))
        a2 = tensor(qeye(N1), destroy(N2))
    
        H = rand() * a1.dag() * a1 + rand() * a2.dag() * a2 + rand() * (a1 + a1.dag()) * (a2 + a2.dag())

        evals, ekets = H.eigenstates()

        for n in range(len(evals)):
            # assert that max(H * ket - e * ket) is small
            self.assertTrue(amax(abs((H * ekets[n] - evals[n] * ekets[n]).full())) < 1e-10)

     

if __name__ == '__main__':

    unittest.main()
