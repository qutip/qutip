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



from scipy import *
from qutip import *

import unittest

class TestQobj(unittest.TestCase):

    """
    A test class for QuTiP's  core quantum object class
    """

    def setUp(self):
        """
        setup
        """

    def testQobjAddition(self):
        
        data1 = array([[1,2], [3,4]])
        data2 = array([[5,6], [7,8]])       

        data3 = data1 + data2

        q1 = Qobj(data1)
        q2 = Qobj(data2)
        q3 = Qobj(data3)

        q4 = q1 + q2

        # check elementwise addition/subtraction
        self.assertEqual(q3, q4)

        # check that addition is commutative
        self.assertEqual(q1+q2, q2+q1)


    def testQobjMultiplication(self):
        
        data1 = array([[1,2], [3,4]])
        data2 = array([[5,6], [7,8]])       

        data3 = dot(data1, data2)

        q1 = Qobj(data1)
        q2 = Qobj(data2)
        q3 = Qobj(data3)

        q4 = q1 * q2

        self.assertEqual(q3, q4)

if __name__ == '__main__':

    unittest.main()
