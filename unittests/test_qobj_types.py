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

class TestQobjTypes(unittest.TestCase):

    """
    A test class for the QuTiP functions for generating quantum states
    """

    def setUp(self):
        """
        setup
        """

    def testKetType(self):
    
        psi = basis(2,1)
        
        self.assertEqual(isket(psi),   True)
        self.assertEqual(isbra(psi),   False)
        self.assertEqual(isoper(psi),  False)
        self.assertEqual(issuper(psi), False)


    def testBraType(self):
    
        psi = basis(2,1).dag()
        
        self.assertEqual(isket(psi),   False)
        self.assertEqual(isbra(psi),   True)
        self.assertEqual(isoper(psi),  False)
        self.assertEqual(issuper(psi), False)


    def testOperType(self):

        psi = basis(2,1)
        rho = psi * psi.dag()

        self.assertEqual(isket(rho),   False)
        self.assertEqual(isbra(rho),   False)
        self.assertEqual(isoper(rho),  True)
        self.assertEqual(issuper(rho), False)

    def testSuperType(self):

        psi = basis(2,1)
        rho = psi * psi.dag()

        sop = spre(rho)

        self.assertEqual(isket(sop),   False)
        self.assertEqual(isbra(sop),   False)
        self.assertEqual(isoper(sop),  False)
        self.assertEqual(issuper(sop), True)

        sop = spost(rho)

        self.assertEqual(isket(sop),   False)
        self.assertEqual(isbra(sop),   False)
        self.assertEqual(isoper(sop),  False)
        self.assertEqual(issuper(sop), True)

if __name__ == '__main__':

    unittest.main()
