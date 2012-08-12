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
from numpy import allclose
import unittest

class TestOperators(unittest.TestCase):

    """
    A test class for the built-in QuTiP operators
    """

    def setUp(self):
        """
        setup
        """

    def test_Jmat_12(self):
    
        spinhalf = jmat(1/2.)
        
        paulix=array([[ 0.0+0.j,  0.5+0.j],[ 0.5+0.j,  0.0+0.j]])
        pauliy=array([[ 0.+0.j ,  0.+0.5j],[ 0.-0.5j,  0.+0.j ]])
        pauliz=array([[ 0.5+0.j,  0.0+0.j],[ 0.0+0.j, -0.5+0.j]])
        sigmap=array([[ 0.+0.j,  1.+0.j],[ 0.+0.j,  0.+0.j]])
        sigmam=array([[ 0.+0.j,  0.+0.j],[ 1.+0.j,  0.+0.j]])
        
        self.assertEqual(allclose(spinhalf[0].full(),paulix),True)
        self.assertEqual(allclose(spinhalf[1].full(),pauliy),True)
        self.assertEqual(allclose(spinhalf[2].full(),pauliz),True)
        self.assertEqual(allclose(jmat(1/2.,'+').full(),sigmap),True)
        self.assertEqual(allclose(jmat(1/2.,'-').full(),sigmam),True)
        
    def test_Jamt_32(self):
        
        spin32=jmat(3/2.)
        
        paulix32=array([[ 0.0000000+0.j,  0.8660254+0.j,  0.0000000+0.j,  0.0000000+0.j],
               [ 0.8660254+0.j,  0.0000000+0.j,  1.0000000+0.j,  0.0000000+0.j],
               [ 0.0000000+0.j,  1.0000000+0.j,  0.0000000+0.j,  0.8660254+0.j],
               [ 0.0000000+0.j,  0.0000000+0.j,  0.8660254+0.j,  0.0000000+0.j]])
        
        pauliy32=array([[ 0.+0.j       ,  0.+0.8660254j,  0.+0.j       ,  0.+0.j       ],
               [ 0.-0.8660254j,  0.+0.j       ,  0.+1.j       ,  0.+0.j       ],
               [ 0.+0.j       ,  0.-1.j       ,  0.+0.j       ,  0.+0.8660254j],
               [ 0.+0.j       ,  0.+0.j       ,  0.-0.8660254j,  0.+0.j       ]])
        
        pauliz32=array([[ 1.5+0.j,  0.0+0.j,  0.0+0.j,  0.0+0.j],
               [ 0.0+0.j,  0.5+0.j,  0.0+0.j,  0.0+0.j],
               [ 0.0+0.j,  0.0+0.j, -0.5+0.j,  0.0+0.j],
               [ 0.0+0.j,  0.0+0.j,  0.0+0.j, -1.5+0.j]])
        
        self.assertEqual(allclose(spin32[0].full(),paulix32),True)
        self.assertEqual(allclose(spin32[1].full(),pauliy32),True)
        self.assertEqual(allclose(spin32[2].full(),pauliz32),True)
    
    def test_Jmat_42(self):
        spin42 = jmat(4/2.,'+')
        self.assertEqual(spin42.dims==[[5],[5]],True)
    
    def test_Jamt_52(self):
        
        spin52 = jmat(5/2.,'+')
        self.assertEqual(spin52.shape==[6,6],True)
        
    def test_Destroy(self):
        b4=basis(5,4)
        d5=destroy(5)
        test1=d5*b4
        self.assertEqual(allclose(test1.full(),2.0*basis(5,3).full()), True)
        d3=destroy(3)
        matrix3=array([[ 0.00000000+0.j,  1.00000000+0.j,  0.00000000+0.j],
               [ 0.00000000+0.j,  0.00000000+0.j,  1.41421356+0.j],
               [ 0.00000000+0.j,  0.00000000+0.j,  0.00000000+0.j]])
        
        self.assertEqual(allclose(matrix3,d3.full()),True)
    
    def test_Create(self):
        b3=basis(5,3)
        c5=create(5)
        test1=c5*b3
        self.assertEqual(allclose(test1.full(),2.0*basis(5,4).full()), True)
        c3=create(3)
        matrix3=array([[ 0.00000000+0.j,  0.00000000+0.j,  0.00000000+0.j],
               [ 1.00000000+0.j,  0.00000000+0.j,  0.00000000+0.j],
               [ 0.00000000+0.j,  1.41421356+0.j,  0.00000000+0.j]])
        
        self.assertEqual(allclose(matrix3,c3.full()),True)
        
        
    def test_Qeye(self):
        eye3=qeye(5)
        self.assertEqual(allclose(eye3.full(),eye(5,dtype=complex)),True)
        
    def test_Num(self):
        n5=num(5)
        self.assertEqual(allclose(n5.full(),diag([0+0j,1+0j,2+0j,3+0j,4+0j])),True)
    
    def test_Squeez(self):
        sq=squeez(4,0.1+0.1j)
        sqmatrix=array([[ 0.99500417+0.j        ,  0.00000000+0.j        ,
                 0.07059289-0.07059289j,  0.00000000+0.j        ],
               [ 0.00000000+0.j        ,  0.98503746+0.j        ,
                 0.00000000+0.j        ,  0.12186303-0.12186303j],
               [-0.07059289-0.07059289j,  0.00000000+0.j        ,
                 0.99500417+0.j        ,  0.00000000+0.j        ],
               [ 0.00000000+0.j        , -0.12186303-0.12186303j,
                 0.00000000+0.j        ,  0.98503746+0.j        ]])
                 
        self.assertEqual(allclose(sq.full(),sqmatrix),True)
        
    def test_Displace(self):
        dp=displace(4,0.25)
        dpmatrix=array([[ 0.96923323+0.j, -0.24230859+0.j,  0.04282883+0.j, -0.00626025+0.j],
               [ 0.24230859+0.j,  0.90866411+0.j, -0.33183303+0.j,  0.07418172+0.j],
               [ 0.04282883+0.j,  0.33183303+0.j,  0.84809499+0.j, -0.41083747+0.j],
               [ 0.00626025+0.j,  0.07418172+0.j,  0.41083747+0.j,  0.90866411+0.j]])
        
        
        self.assertEqual(allclose(dp.full(),dpmatrix),True)
        
        
        

if __name__ == '__main__':
    unittest.main()
