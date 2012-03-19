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
from qutip.sparse import *
from numpy import allclose
import unittest

class TestRand(unittest.TestCase):

    """
    A test class for the built-in Qobj sparse eigensolver.
    in sparse module.
    """

    def setUp(self):
        """
        setup
        """
    def testSparseHermValsVecs(self):
        """
        A collection of tests for the sparse eigensolver routine
        returning eigenvalues and eigenvectors for a Hermitian 
        operator.
        """
        
        #check using number operator
        N=num(10)
        spvals,spvecs=N.eigenstates(sparse=True)
        for k in range(10):
            #check that eigvals are in proper order
            self.assertTrue(abs(spvals[k]-k)<=1e-13)
            #check that eigenvectors are right and in right order
            self.assertTrue(abs(expect(N,spvecs[k])-spvals[k])<5e-14)
        
        #check ouput of only a few eigenvals/vecs
        spvals,spvecs=N.eigenstates(sparse=True,eigvals=7)
        self.assertTrue(len(spvals)==7)
        self.assertTrue(spvals[0]<=spvals[-1])
        for k in range(7):
            self.assertTrue(abs(spvals[k]-k)<1e-14)
        
        spvals,spvecs=N.eigenstates(sparse=True,sort='high',eigvals=5)
        self.assertTrue(len(spvals)==5)
        self.assertTrue(spvals[0]>=spvals[-1])
        vals=arange(9,4,-1)
        for k in range(5):
            #check that eigvals are ordered from high to low
            self.assertTrue(abs(spvals[k]-vals[k])<5e-14)
            self.assertTrue(abs(expect(N,spvecs[k])-vals[k])<1e-14)
        #check using random Hermitian
        H=rand_herm(10)
        spvals,spvecs=H.eigenstates(sparse=True)
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]<=spvals[-1])
        #check that spvals equal expect vals
        for k in range(10):
            self.assertTrue(abs(expect(H,spvecs[k])-spvals[k])<5e-14)
            #check that ouput is real for Hermitian operator
            self.assertEqual(isreal(spvals[k]),True)
    
    def testSparseValsVecs(self):
        """
        A collection of tests for the sparse eigensolver routine
        returning eigenvalues and eigenvectors for a non-Hermitian 
        operator.
        """
        U=rand_unitary(10)
        spvals,spvecs=U.eigenstates(sparse=True)
        self.assertTrue(real(spvals[0])<=real(spvals[-1]))
        for k in range(10):
            #check that eigenvectors are right and in right order
            self.assertTrue(abs(expect(U,spvecs[k])-spvals[k])<5e-14)
            self.assertEqual(iscomplex(spvals[k]),True)
        
        #check sorting
        spvals,spvecs=U.eigenstates(sparse=True,sort='high')
        self.assertTrue(real(spvals[0])>=real(spvals[-1]))
        
        #check for N-1 eigenvals
        U=rand_unitary(10)
        spvals,spvecs=U.eigenstates(sparse=True,eigvals=9)
        self.assertEqual(len(spvals),9)
    
    def testSparseValsOnly(self):
        """
        A collection of tests for the sparse eigensolver routine
        returning eigenvalues ONLY for a non-Hermitian operator.
        """
        H=rand_herm(10)
        spvals=H.eigenenergies(sparse=True)
        self.assertEqual(len(spvals),10)
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]<=spvals[-1])
        #check that spvals equal expect vals
        for k in range(10):
            #check that ouput is real for Hermitian operator
            self.assertEqual(isreal(spvals[k]),True)
        spvals=H.eigenenergies(sparse=True,sort='high')
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]>=spvals[-1])
        spvals=H.eigenenergies(sparse=True,sort='high',eigvals=4)
        self.assertEqual(len(spvals),4)
        
        U=rand_unitary(10)
        spvals=U.eigenenergies(sparse=True)
        self.assertEqual(len(spvals),10)
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]<=spvals[-1])
        #check that spvals equal expect vals
        for k in range(10):
            #check that ouput is real for Hermitian operator
            self.assertEqual(iscomplex(spvals[k]),True)
        spvals=U.eigenenergies(sparse=True,sort='high')
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]>=spvals[-1])
        spvals=U.eigenenergies(sparse=True,sort='high',eigvals=4)
        self.assertEqual(len(spvals),4)
    
    def testDenseHermValsVecs(self):
        """
        A collection of tests for the dense eigensolver routine
        returning eigenvalues ONLY for a Hermitian operator.
        """
        #check using number operator
        N=num(10)
        spvals,spvecs=N.eigenstates(sparse=False)
        for k in range(10):
            #check that eigvals are in proper order
            self.assertTrue(abs(spvals[k]-k)<1e-14)
            #check that eigenvectors are right and in right order
            self.assertTrue(abs(expect(N,spvecs[k])-spvals[k])<5e-14)
        
        #check ouput of only a few eigenvals/vecs
        spvals,spvecs=N.eigenstates(sparse=False,eigvals=7)
        self.assertTrue(len(spvals)==7)
        self.assertTrue(spvals[0]<=spvals[-1])
        for k in range(7):
            self.assertTrue(abs(spvals[k]-k)<1e-14)
        
        spvals,spvecs=N.eigenstates(sparse=False,sort='high',eigvals=5)
        self.assertTrue(len(spvals)==5)
        self.assertTrue(spvals[0]>=spvals[-1])
        vals=arange(9,4,-1)
        for k in range(5):
            #check that eigvals are ordered from high to low
            self.assertTrue(abs(spvals[k]-vals[k])<5e-14)
            self.assertTrue(abs(expect(N,spvecs[k])-vals[k])<5e-14)
        #check using random Hermitian
        H=rand_herm(10)
        spvals,spvecs=H.eigenstates(sparse=False)
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]<=spvals[-1])
        #check that spvals equal expect vals
        for k in range(10):
            self.assertTrue(abs(expect(H,spvecs[k])-spvals[k])<5e-14)
            #check that ouput is real for Hermitian operator
            self.assertEqual(isreal(spvals[k]),True)
    
    def testDenseValsVecs(self):
        """
        A collection of tests for the dense eigensolver routine
        returning eigenvalues and eigenvectors for a non-Hermitian 
        operator.
        """
        U=rand_unitary(10)
        spvals,spvecs=U.eigenstates(sparse=False)
        self.assertTrue(real(spvals[0])<=real(spvals[-1]))
        for k in range(10):
            #check that eigenvectors are right and in right order
            self.assertTrue(abs(expect(U,spvecs[k])-spvals[k])<1e-14)
            self.assertEqual(iscomplex(spvals[k]),True)

        #check sorting
        spvals,spvecs=U.eigenstates(sparse=False,sort='high')
        self.assertTrue(real(spvals[0])>=real(spvals[-1]))

        #check for N-1 eigenvals
        U=rand_unitary(10)
        spvals,spvecs=U.eigenstates(sparse=False,eigvals=9)
        self.assertEqual(len(spvals),9) 
    
    def testDenseValsOnly(self):
        """
        A collection of tests for the dense eigensolver routine
        returning eigenvalues ONLY for a non-Hermitian operator.
        """
        H=rand_herm(10)
        spvals=H.eigenenergies(sparse=False)
        self.assertEqual(len(spvals),10)
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]<=spvals[-1])
        #check that spvals equal expect vals
        for k in range(10):
            #check that ouput is real for Hermitian operator
            self.assertEqual(isreal(spvals[k]),True)
        spvals=H.eigenenergies(sparse=False,sort='high')
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]>=spvals[-1])
        spvals=H.eigenenergies(sparse=False,sort='high',eigvals=4)
        self.assertEqual(len(spvals),4)

        U=rand_unitary(10)
        spvals=U.eigenenergies(sparse=False)
        self.assertEqual(len(spvals),10)
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]<=spvals[-1])
        #check that spvals equal expect vals
        for k in range(10):
            #check that ouput is real for Hermitian operator
            self.assertEqual(iscomplex(spvals[k]),True)
        spvals=U.eigenenergies(sparse=False,sort='high')
        #check that sorting is lowest eigval first
        self.assertTrue(spvals[0]>=spvals[-1])
        spvals=U.eigenenergies(sparse=False,sort='high',eigvals=4)
        self.assertEqual(len(spvals),4)
        
        
if __name__ == '__main__':
    unittest.main()
