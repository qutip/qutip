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


from qutip import *

import unittest

class TestStates(unittest.TestCase):

    """
    A test class for the QuTiP functions for generating quantum states
    """

    def setUp(self):
        """
        setup
        """

    def testCoherentDensityMatrix(self):
        
        N = 10

        rho = coherent_dm(N, 1)

        # make sure rho has trace close to 1.0
        self.assertEqual(abs(rho.tr() - 1.0) < 1e-12, True)

    def testThermalDensityMatrix(self):
        
        N = 40

        rho = thermal_dm(N, 1)

        # make sure rho has trace close to 1.0
        self.assertEqual(abs(rho.tr() - 1.0) < 1e-12, True)


    def testFockDensityMatrix(self):
        N = 10
        for i in range(N):
            rho = fock_dm(N, i)
            # make sure rho has trace close to 1.0
            self.assertEqual(abs(rho.tr() - 1.0) < 1e-12, True)
            self.assertEqual(rho.data[i,i], 1.0)        

if __name__ == '__main__':

    unittest.main()
