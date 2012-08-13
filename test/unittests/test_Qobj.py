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

import numpy as np
from scipy import *
from qutip import *
import scipy.sparse as sp
import unittest

class TestQobj(unittest.TestCase):

    """
    A test class for QuTiP's  core quantum object class
    """

    def setUp(self):
        """
        setup
        """
    #-------- test the Qobj properties ------------#
    def testQobjData(self):
        N=10
        data1=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
        q1=Qobj(data1)
        #check if data is a csr_matrix if originally array
        self.assertTrue(sp.isspmatrix_csr(q1.data))
        #check if dense ouput is equal to original data
        self.assertTrue(all(q1.data.todense()-matrix(data1))==0)
        
        data2=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
        data2=sp.csr_matrix(data2)
        q2=Qobj(data2)
        #check if data is a csr_matrix if originally csr_matrix
        self.assertTrue(sp.isspmatrix_csr(q2.data))
        
        data3=1
        q3=Qobj(data3)
        #check if data is a csr_matrix if originally int
        self.assertTrue(sp.isspmatrix_csr(q3.data))
        
        data4=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
        data4=matrix(data4)
        q4=Qobj(data4)
        #check if data is a csr_matrix if originally csr_matrix
        self.assertTrue(sp.isspmatrix_csr(q4.data))
        self.assertTrue(all(q4.data.todense()-matrix(data4))==0)
    
    
    def testQobjType(self):
        
        N=int(ceil(10.0*np.random.random()))+5
        
        ket_data=np.random.random((N,1))
        ket_qobj=Qobj(ket_data)
        self.assertTrue(ket_qobj.type=='ket')
        
        bra_data=np.random.random((1,N))
        bra_qobj=Qobj(bra_data)
        self.assertTrue(bra_qobj.type=='bra')
        
        oper_data=np.random.random((N,N))
        oper_qobj=Qobj(oper_data)
        self.assertTrue(oper_qobj.type=='oper')
        
        N=9
        super_data=np.random.random((N,N))
        super_qobj=Qobj(super_data,dims=[[[3]],[[3]]])
        self.assertTrue(super_qobj.type=='super')
        
    def testQobjHerm(self): 
        N=10
        data=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
        q1=Qobj(data)
        self.assertFalse(q1.isherm==True)
        
        data=data+data.conj().T
        q2=Qobj(data)
        self.assertTrue(q2.isherm==True)
    
    def testQobjDimsShape(self): 
        N=10
        data=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
        
        q1=Qobj(data)
        self.assertEqual(q1.dims,[[10],[10]])
        self.assertEqual(q1.shape,[10,10])
        
        data=np.random.random((N,1))+1j*np.random.random((N,1))-(0.5+0.5j)
        
        q1=Qobj(data)
        self.assertEqual(q1.dims,[[10],[1]])
        self.assertEqual(q1.shape,[10,1])
        
        N=4
        
        data=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
        
        q1=Qobj(data,dims=[[2,2],[2,2]])
        self.assertEqual(q1.dims,[[2,2],[2,2]])
        self.assertEqual(q1.shape,[4,4])
    
    #-------- test the Qobj math ------------#
    def testQobjAddition(self):
        
        data1 = array([[1,2], [3,4]])
        data2 = array([[5,6], [7,8]])       

        data3 = data1 + data2

        q1 = Qobj(data1)
        q2 = Qobj(data2)
        q3 = Qobj(data3)

        q4 = q1 + q2
        
        self.assertTrue(q4.type==ischeck(q4))
        self.assertTrue(q4.isherm==isherm(q4))
        
        # check elementwise addition/subtraction
        self.assertEqual(q3, q4)

        # check that addition is commutative
        self.assertEqual(q1+q2, q2+q1)
        
        data=np.random.random((5,5))
        q=Qobj(data)
        
        x1=q+5
        x2=5+q
        
        data=data+5
        self.assertTrue(all(x1.data.todense()-(matrix(data)))==0)
        self.assertTrue(all(x2.data.todense()-(matrix(data)))==0)
        
        data=np.random.random((5,5))
        q=Qobj(data)
        x3=q+data
        x4=data+q
        
        data=2.0*data
        self.assertTrue(all(x3.data.todense()-(matrix(data)))==0)
        self.assertTrue(all(x4.data.todense()-(matrix(data)))==0)
        
    
    def testQobjSubtraction(self):
        data1=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        q1=Qobj(data1)
        
        data2=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        q2=Qobj(data2)
        
        q3=q1-q2
        data3=data1-data2
        
        
        
        self.assertTrue(all(q3.data.todense()-matrix(data3))==0)
        
        q4=q2-q1
        data4=data2-data1
        
        self.assertTrue(all(q4.data.todense()-matrix(data4))==0)
        
    def testQobjMultiplication(self):
        
        data1 = array([[1,2], [3,4]])
        data2 = array([[5,6], [7,8]])       

        data3 = dot(data1, data2)

        q1 = Qobj(data1)
        q2 = Qobj(data2)
        q3 = Qobj(data3)

        q4 = q1 * q2

        self.assertEqual(q3, q4)
        
    
    def testQobjDivision(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        q=Qobj(data)
        randN=10*np.random.random()
        q=q/randN
        self.assertTrue(all(q.data.todense()-matrix(data)/randN)==0)
    
    
    def testQobjPower(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        q=Qobj(data)
        
        q2=q**2
        self.assertTrue(all(q2.data.todense()-matrix(data)**2)==0)
        
        q3=q**3
        self.assertTrue(all(q3.data.todense()-matrix(data)**3)==0)
    
    
    
    def testQobjNeg(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        q=Qobj(data)
        x=-q
        self.assertTrue(all(x.data.todense()+matrix(data))==0)
    
    
    def testQobjEquals(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        q1=Qobj(data)
        q2=Qobj(data)
        self.assertTrue(q1==q2)
        
        q1=Qobj(data)
        q2=Qobj(-data)
        self.assertTrue(q1!=q2)
    
    def testQobjGetItem(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        q=Qobj(data)
        self.assertTrue(q[0,0]==data[0,0])
        self.assertTrue(q[-1,2]==data[-1,2])
    
    def testCheckMulType(self):
        psi=basis(5)
        dm=psi*psi.dag()
        self.assertTrue(dm.type=='oper')
        self.assertTrue(dm.isherm==True)
        
        nrm=psi.dag()*psi
        self.assertTrue(dm.type=='oper')
        self.assertTrue(dm.isherm==True)
        
        H1=rand_herm(3)
        H2=rand_herm(3)
        out=H1*H2
        self.assertTrue(out.type=='oper')
        self.assertTrue(out.isherm==isherm(out))
        
        U=rand_unitary(5)
        out=U.dag()*U
        self.assertTrue(out.type=='oper')
        self.assertTrue(out.isherm==True)
        
        N=num(5)
        
        out=N*N
        self.assertTrue(out.type=='oper')
        self.assertTrue(out.isherm==True)
        
        
    #-------- test the Qobj methods ------------#
    
    def testQobjConjugate(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        A=Qobj(data)
        B=A.conj()
        self.assertTrue(all(B.data.todense()-matrix(data.conj()))==0)
    
    def testQobjDagger(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        A=Qobj(data)
        B=A.dag()
        self.assertTrue(all(B.data.todense()-matrix(data.conj().T))==0)
    
    def testQobjDiagonals(self):
        data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
        A=Qobj(data)
        b=A.diag()
        self.assertTrue(all(b-diag(data))==0)
    
    def testQobjEigenEnergies(self):
        data=eye(5)
        A=Qobj(data)
        b=A.eigenenergies()
        self.assertTrue(all(b-ones(5))==0)
        
        data=diag(arange(10))
        A=Qobj(data)
        b=A.eigenenergies()
        self.assertTrue(all(b-arange(10))==0)
        
        
        data=diag(arange(10))
        A=5*Qobj(data)
        b=A.eigenenergies()
        self.assertTrue(all(b-5*arange(10))==0)
    
    
    def testQobjEigenStates(self):
        data=eye(5)
        A=Qobj(data)
        b,c=A.eigenstates()
        self.assertTrue(all(b-ones(5))==0)
        
        kets=array([basis(5,k) for k in range(5)])
        
        for k in range(5):
            self.assertTrue(c[k]==kets[k])
        
    
    def testQobjExpm(self):
        data=np.random.random((15,15))+1j*np.random.random((15,15))-(0.5+0.5j)
        A=Qobj(data)
        B=A.expm()
        self.assertTrue(all(B.data.todense()-matrix(scipy.linalg.expm(data,13)))<1e-15)
    
    def testQobjFull(self):
        data=np.random.random((15,15))+1j*np.random.random((15,15))-(0.5+0.5j)
        A=Qobj(data)
        b=A.full()
        self.assertTrue(all(b-data)==0)
    

    
    
    
    
    
    

if __name__ == '__main__':

    unittest.main()
