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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################



from scipy import *
from qutip import *
import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy.testing import assert_equal, assert_


#-------- test_ the Qobj properties ------------#
def test_QobjData():
    "Qobj data"
    N=10
    data1=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
    q1=Qobj(data1)
    #check if data is a csr_matrix if originally array
    assert_equal(sp.isspmatrix_csr(q1.data),True)
    #check if dense ouput is equal to original data
    assert_equal(all(q1.data.todense()-matrix(data1)),0)
    
    data2=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
    data2=sp.csr_matrix(data2)
    q2=Qobj(data2)
    #check if data is a csr_matrix if originally csr_matrix
    assert_equal(sp.isspmatrix_csr(q2.data),True)
    
    data3=1
    q3=Qobj(data3)
    #check if data is a csr_matrix if originally int
    assert_equal(sp.isspmatrix_csr(q3.data),True)
    
    data4=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
    data4=matrix(data4)
    q4=Qobj(data4)
    #check if data is a csr_matrix if originally csr_matrix
    assert_equal(sp.isspmatrix_csr(q4.data),True)
    assert_equal(all(q4.data.todense()-matrix(data4)),0)


def test_QobjType():
    "Qobj type"
    N=int(ceil(10.0*np.random.random()))+5
    
    ket_data=np.random.random((N,1))
    ket_qobj=Qobj(ket_data)
    assert_equal(ket_qobj.type,'ket')
    
    bra_data=np.random.random((1,N))
    bra_qobj=Qobj(bra_data)
    assert_equal(bra_qobj.type,'bra')
    
    oper_data=np.random.random((N,N))
    oper_qobj=Qobj(oper_data)
    assert_equal(oper_qobj.type,'oper')
    
    N=9
    super_data=np.random.random((N,N))
    super_qobj=Qobj(super_data,dims=[[[3]],[[3]]])
    assert_equal(super_qobj.type,'super')
    
def test_QobjHerm():
    "Qobj Hermicity" 
    N=10
    data=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
    q1=Qobj(data)
    assert_equal(q1.isherm,False)
    
    data=data+data.conj().T
    q2=Qobj(data)
    assert_equal(q2.isherm,True)

def test_QobjDimsShape():
    "Qobj shape" 
    N=10
    data=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
    
    q1=Qobj(data)
    assert_equal(q1.dims,[[10],[10]])
    assert_equal(q1.shape,[10,10])
    
    data=np.random.random((N,1))+1j*np.random.random((N,1))-(0.5+0.5j)
    
    q1=Qobj(data)
    assert_equal(q1.dims,[[10],[1]])
    assert_equal(q1.shape,[10,1])
    
    N=4
    
    data=np.random.random((N,N))+1j*np.random.random((N,N))-(0.5+0.5j)
    
    q1=Qobj(data,dims=[[2,2],[2,2]])
    assert_equal(q1.dims,[[2,2],[2,2]])
    assert_equal(q1.shape,[4,4])

#-------- test_ the Qobj math ------------#
def test_QobjAddition():
    "Qobj addition"
    data1 = array([[1,2], [3,4]])
    data2 = array([[5,6], [7,8]])       

    data3 = data1 + data2

    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)

    q4 = q1 + q2
    
    assert_equal(q4.type,ischeck(q4))
    assert_equal(q4.isherm,isherm(q4))
    
    # check elementwise addition/subtraction
    assert_equal(q3, q4)

    # check that addition is commutative
    assert_equal(q1+q2, q2+q1)
    
    data=np.random.random((5,5))
    q=Qobj(data)
    
    x1=q+5
    x2=5+q
    
    data=data+5
    assert_equal(all(x1.data.todense()-(matrix(data))),0)
    assert_equal(all(x2.data.todense()-(matrix(data))),0)
    
    data=np.random.random((5,5))
    q=Qobj(data)
    x3=q+data
    x4=data+q
    
    data=2.0*data
    assert_equal(all(x3.data.todense()-(matrix(data))),0)
    assert_equal(all(x4.data.todense()-(matrix(data))),0)
    

def test_QobjSubtraction():
    "Qobj subtraction"
    data1=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    q1=Qobj(data1)
    
    data2=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    q2=Qobj(data2)
    
    q3=q1-q2
    data3=data1-data2
    
    
    
    assert_equal(all(q3.data.todense()-matrix(data3)),0)
    
    q4=q2-q1
    data4=data2-data1
    
    assert_equal(all(q4.data.todense()-matrix(data4)),0)
    
def test_QobjMultiplication():
    "Qobj multiplication"
    data1 = array([[1,2], [3,4]])
    data2 = array([[5,6], [7,8]])       

    data3 = dot(data1, data2)

    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)

    q4 = q1 * q2

    assert_equal(q3, q4)
    

def test_QobjDivision():
    "Qobj division"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    q=Qobj(data)
    randN=10*np.random.random()
    q=q/randN
    assert_equal(all(q.data.todense()-matrix(data)/randN),0)


def test_QobjPower():
    "Qobj power"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    q=Qobj(data)
    
    q2=q**2
    assert_equal(all(q2.data.todense()-matrix(data)**2),0)
    
    q3=q**3
    assert_equal(all(q3.data.todense()-matrix(data)**3),0)



def test_QobjNeg():
    "Qobj negation"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    q=Qobj(data)
    x=-q
    assert_equal(all(x.data.todense()+matrix(data)),0)


def test_QobjEquals():
    "Qobj equals"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    q1=Qobj(data)
    q2=Qobj(data)
    assert_equal(q1,q2)
    
    q1=Qobj(data)
    q2=Qobj(-data)
    assert_equal(q1!=q2,True)

def test_QobjGetItem():
    "Qobj getitem"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    q=Qobj(data)
    assert_equal(q[0,0],data[0,0])
    assert_equal(q[-1,2],data[-1,2])

def test_CheckMulType():
    "Qobj multiplication type"
    psi=basis(5)
    dm=psi*psi.dag()
    assert_equal(dm.type,'oper')
    assert_equal(dm.isherm,True)
    
    nrm=psi.dag()*psi
    assert_equal(dm.type,'oper')
    assert_equal(dm.isherm,True)
    
    H1=rand_herm(3)
    H2=rand_herm(3)
    out=H1*H2
    assert_equal(out.type,'oper')
    assert_equal(out.isherm,isherm(out))
    
    U=rand_unitary(5)
    out=U.dag()*U
    assert_equal(out.type,'oper')
    assert_equal(out.isherm,True)
    
    N=num(5)
    
    out=N*N
    assert_equal(out.type,'oper')
    assert_equal(out.isherm,True)
    
    
#-------- test_ the Qobj methods ------------#

def test_QobjConjugate():
    "Qobj conjugate"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    A=Qobj(data)
    B=A.conj()
    assert_equal(all(B.data.todense()-matrix(data.conj())),0)

def test_QobjDagger():
    "Qobj adjoint (dagger)"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    A=Qobj(data)
    B=A.dag()
    assert_equal(all(B.data.todense()-matrix(data.conj().T)),0)

def test_QobjDiagonals():
    "Qobj diagonals"
    data=np.random.random((5,5))+1j*np.random.random((5,5))-(0.5+0.5j)
    A=Qobj(data)
    b=A.diag()
    assert_equal(all(b-diag(data)),0)

def test_QobjEigenEnergies():
    "Qobj eigenenergies"
    data=eye(5)
    A=Qobj(data)
    b=A.eigenenergies()
    assert_equal(all(b-ones(5)),0)
    
    data=diag(arange(10))
    A=Qobj(data)
    b=A.eigenenergies()
    assert_equal(all(b-arange(10)),0)
    
    
    data=diag(arange(10))
    A=5*Qobj(data)
    b=A.eigenenergies()
    assert_equal(all(b-5*arange(10)),0)


def test_QobjEigenStates():
    "Qobj eigenstates"
    data=eye(5)
    A=Qobj(data)
    b,c=A.eigenstates()
    assert_equal(all(b-ones(5)),0)
    
    kets=array([basis(5,k) for k in range(5)])
    
    for k in range(5):
        assert_equal(c[k],kets[k])
    

def test_QobjExpm():
    "Qobj expm"
    data=np.random.random((15,15))+1j*np.random.random((15,15))-(0.5+0.5j)
    A=Qobj(data)
    B=A.expm()
    assert_equal(all(B.data.todense()-matrix(la.expm(data)))<1e-15,True)

def test_QobjFull():
    "Qobj full"
    data=np.random.random((15,15))+1j*np.random.random((15,15))-(0.5+0.5j)
    A=Qobj(data)
    b=A.full()
    assert_equal(all(b-data),0)
    
# --- Test types

def test_KetType():
    "Qobj ket type"

    psi = basis(2,1)
    
    assert_(isket(psi))
    assert_(not isbra(psi))
    assert_(not isoper(psi))
    assert_(not issuper(psi))

    psi = tensor(basis(2,1), basis(2,0))

    assert_(isket(psi))
    assert_(not isbra(psi))
    assert_(not isoper(psi))
    assert_(not issuper(psi))


def test_BraType():
    "Qobj bra type"

    psi = basis(2,1).dag()
    
    assert_equal(isket(psi),   False)
    assert_equal(isbra(psi),   True)
    assert_equal(isoper(psi),  False)
    assert_equal(issuper(psi), False)

    psi = tensor(basis(2,1).dag(), basis(2,0).dag())

    assert_equal(isket(psi),   False)
    assert_equal(isbra(psi),   True)
    assert_equal(isoper(psi),  False)
    assert_equal(issuper(psi), False)


def test_OperType():
    "Qobj operator type"

    psi = basis(2,1)
    rho = psi * psi.dag()

    assert_equal(isket(rho),   False)
    assert_equal(isbra(rho),   False)
    assert_equal(isoper(rho),  True)
    assert_equal(issuper(rho), False)

def test_SuperType():
    "Qobj superoperator type"

    psi = basis(2,1)
    rho = psi * psi.dag()

    sop = spre(rho)

    assert_equal(isket(sop),   False)
    assert_equal(isbra(sop),   False)
    assert_equal(isoper(sop),  False)
    assert_equal(issuper(sop), True)

    sop = spost(rho)

    assert_equal(isket(sop),   False)
    assert_equal(isbra(sop),   False)
    assert_equal(isoper(sop),  False)
    assert_equal(issuper(sop), True)

if __name__ == "__main__":
    run_module_suite()

