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

class TestEntropy(unittest.TestCase):

    """
    A test class for the built-in entropy functions
    """

    def setUp(self):
        """
        setup
        """
    
    def testEntropyVN(self):
        #verify that entropy_vn gives correct binary entropy 
        a=linspace(0,1,20)
        for k in xrange(len(a)):
            # a*|0><0|
            x=a[k]*ket2dm(basis(2,0))
            # (1-a)*|1><1|
            y=(1-a[k])*ket2dm(basis(2,1))
            rho=x+y
            # Von-Neumann entropy (base 2) of rho
            out=entropy_vn(rho,2)
            if k==0 or k==19:
                self.assertTrue(out==0)
            else:
                self.assertTrue(out==-a[k]*log2(a[k])-(1-a[k])*log2((1-a[k])))
        #test entropy_vn = 0 for pure state
        psi=rand_ket(10)
        self.assertTrue(abs(entropy_vn(psi))<=1e-13)
    
    def testEntropyLinear(self):
        #test entropy_vn = 0 for pure state
        psi=rand_ket(10)
        self.assertTrue(abs(entropy_linear(psi))<=1e-13)
        
        #test linear entropy always less than or equal to VN entropy
        rhos=[rand_dm(6) for k in xrange(10)]
        for k in rhos:
            self.assertTrue(entropy_linear(k)<=entropy_vn(k))
    
    def testEntropyConcurrence(self):
        #check concurrence = 1 for maximal entangled (Bell) state
        bell=ket2dm((tensor(basis(2),basis(2))+tensor(basis(2,1),basis(2,1))).unit())
        self.assertTrue(abs(concurrence(bell)-1.0)<1e-15)
        
        #check all concurrence values >=0
        rhos=[rand_dm(4,dims=[[2,2],[2,2]]) for k in xrange(10)]
        for k in rhos:
            self.assertTrue(concurrence(k)>=0)
    
    def testEntropyMutual(self):
        #verify mutual information = S(A)+S(B) for pure state
        rhos=[rand_dm(25,dims=[[5,5],[5,5]],pure=True) for k in xrange(10)]
        for r in rhos:
            self.assertTrue(abs(entropy_mutual(r,[0],[1])-(entropy_vn(ptrace(r,0))+entropy_vn(ptrace(r,1))))<1e-13)
        #check component selection
        rhos=[rand_dm(8,dims=[[2,2,2],[2,2,2]],pure=True) for k in xrange(10)]
        for r in rhos:
            self.assertTrue(abs(entropy_mutual(r,[0,2],[1])-(entropy_vn(ptrace(r,[0,2]))+entropy_vn(ptrace(r,1))))<1e-13)

    def testEntropyConditional(self):
        #test S(A,B|C,D)<=S(A|C)+S(B|D)
        rhos=[rand_dm(16,dims=[[2,2,2,2],[2,2,2,2]],pure=True) for k in xrange(20)]
        for ABCD in rhos:
            AC=ptrace(ABCD,[0,2])
            BD=ptrace(ABCD,[1,3])
            self.assertTrue(entropy_conditional(ABCD,[2,3])<=(entropy_conditional(AC,1)+entropy_conditional(BD,1)))
        #test S(A|B,C)<=S(A|B)
        rhos=[rand_dm(8,dims=[[2,2,2],[2,2,2]],pure=True) for k in xrange(20)]
        for ABC in rhos:
            AB=ptrace(ABC,[0,1])
            self.assertTrue(entropy_conditional(ABC,[1,2])<=entropy_conditional(AB,1))
if __name__ == '__main__':
    unittest.main()
