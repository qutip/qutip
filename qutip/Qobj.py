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
import scipy.sparse as sp
import scipy.linalg as la
from istests import *

##
# @package Qobj
# Quantum Object class. Describes the composition of a quantum system and
# implements common operations.
#


class Qobj():
    """
    @brief Quantum object class.
    requires: scipy, scipy.sparse.csr_matrix, scipy.linalg
    """
    ################## Define Qobj class #################
    __array_priority__=100 #sets Qobj priority above numpy arrays
    def __init__(self,inpt=array([[0]]),dims=[[],[]],shape=[]):
        """
        Qobj constructor. Optionally takes the dimension array and/or
        shape array as arguments.

        @param array    Data for vector/matrix representation of the quantum object
        @param dims
        @param shape
        """
        if isinstance(inpt,Qobj):#if input is already Qobj then return identical copy
            self.data=sp.csr_matrix(inpt.data) #make sure matrix is sparse (safety check)
            if not any(dims):
                self.dims=inpt.dims
            else:
                self.dims=dims
            if not any(shape):
                self.shape=inpt.shape
            else:
                self.shape=shape
        else:#if input is NOT Qobj
            #if input is int, float, or complex then convert to array
            if isinstance(inpt,(int,float,complex)):
                inpt=array([[inpt]])
            #case where input is array or sparse
            if (isinstance(inpt,ndarray)) or (isinstance(inpt,sp.csr_matrix)):
                self.data=sp.csr_matrix(inpt) #data stored as space array
                if not any(dims):
                    self.dims=[[inpt.shape[0]],[inpt.shape[1]]] #list of object dimensions
                else:
                    self.dims=dims
                if not any(shape):
                    self.shape=[inpt.shape[0],inpt.shape[1]] # list of matrix dimensions
                else:
                    self.shape=shape
            elif isinstance(inpt,list):# case where input is not array or sparse, i.e. a list
                if len(array(inpt).shape)==1:#if list has only one dimension (i.e [5,4])
                    inpt=array([inpt])
                else:#if list has two dimensions (i.e [[5,4]])
                    inpt=array(inpt)
                    self.data=sp.csr_matrix(inpt)
                    if not any(dims):
                        self.dims=[[inpt.shape[0]],[inpt.shape[1]]]
                    else:
                        self.dims=dims
                    if not any(shape):
                        self.shape=[inpt.shape[0],inpt.shape[1]]
                    else:
                        self.shape=shape
    
    ##### Definition of PLUS with Qobj on LEFT (ex. Qobj+4) #########                
    def __add__(self, other): #defines left addition for Qobj class
        if classcheck(other)=='eseries':
            return other.__radd__(self)
        other=Qobj(other)
        if prod(other.shape)==1 and prod(self.shape)!=1: #case for scalar quantum object
            dat=array(other.full())[0][0]
            out=Qobj()
            out.data=self.data+dat*sp.csr_matrix(ones(self.shape))
            out.dims=self.dims
            out.shape=self.shape
            return Qobj(out)
        elif prod(self.shape)==1 and prod(other.shape)!=1:#case for scalar quantum object
            dat=array(self.full())[0][0]
            out=Qobj()
            out.data=dat*sp.csr_matrix(ones(other.shape))+other.data
            out.dims=other.dims
            out.shape=other.shape
            return Qobj(out)
        elif self.dims!=other.dims:
            raise TypeError('Incompatible quantum object dimensions')
        elif self.shape!=other.shape:
            raise TypeError('Matrix shapes do not match')
        else:#case for matching quantum objects
            out=Qobj()
            out.data=self.data+other.data
            out.dims=self.dims
            out.shape=self.shape
            return Qobj(out)

    ##### Definition of PLUS with Qobj on RIGHT (ex. 4+Qobj) ###############
    def __radd__(self,other): #defines left addition for Qobj class
        return self+other

    ##### Definition of SUBTRACTION with Qobj on LEFT (ex. Qobj-4) #########
    def __sub__(self,other):
        return self+(-other)

    ##### Definition of SUBTRACTION with Qobj on RIGHT (ex. 4-Qobj) #########
    def __rsub__(self,other):
        return (-self)+other
    
    ##### Definition of Multiplication with Qobj on left (ex. Qobj*5) #########
    def __mul__(self,other):
        if isinstance(other,Qobj): #if both are quantum objects
            if prod(other.shape)==1 and prod(self.shape)!=1:#case for scalar quantum object
                dat=array(other.data.todense())[0][0]
                out=Qobj()
                out.data=dat * (sp.csr_matrix(ones(self.shape))*self.data)
                out.dims=self.dims
                out.shape=self.shape
                return Qobj(out)
            elif prod(self.shape)==1 and prod(other.shape)!=1:#take care of right mul as well for scalar Qobjs
                dat=array(self.data.todense())[0][0]
                out=Qobj()
                out.data=dat*sp.csr_matrix(ones(other.shape))*other.data
                out.dims=other.dims
                out.shape=other.shape
                return Qobj(out)
            elif self.shape[1]==other.shape[0] and self.dims[1]==other.dims[0]:
                out=Qobj()
                out.data=self.data*other.data
                out.dims  = [self.dims[0],  other.dims[1]]
                out.shape = [self.shape[0], other.shape[1]]
                return Qobj(out)
            else:
                raise TypeError("Incompatible Qobj shapes")

        if isinstance(other, list): # if other is a list, do element-wise multiplication
            prod_list = []
            for item in other:
                prod_list.append(self * item)
            return prod_list

        if classcheck(other)=='eseries':
            return other.__rmul__(self)

        if isinstance(other,(int,float,complex)): #if other is int,float,complex
            out=Qobj()
            out.data=self.data*other
            out.dims=self.dims
            out.shape=self.shape
            return Qobj(out)
        else:
            raise TypeError("Incompatible object for multiplication")
    
    ##### Definition of Multiplication with Qobj on right (ex. 5*Qobj) #########
    def __rmul__(self,other):
        if isinstance(other,Qobj): #if both are quantum objects
            if prod(other.shape)==1 and prod(self.shape)!=1:#case for scalar quantum object
                dat=array(other.data.todense())[0][0]
                out=Qobj()
                out.data=dat * (sp.csr_matrix(ones(self.shape))*self.data)
                out.dims=self.dims
                out.shape=self.shape
                return Qobj(out)
            elif prod(self.shape)==1 and prod(other.shape)!=1:#take care of right mul as well for scalar Qobjs
                dat=array(self.data.todense())[0][0]
                out=Qobj()
                out.data=dat*sp.csr_matrix(ones(other.shape))*other.data
                out.dims=other.dims
                out.shape=other.shape
                return Qobj(out)
            elif self.shape[1]==other.shape[0] and self.dims[1]==other.dims[0]:
                out=Qobj()
                out.data=other.data * self.data
                out.dims=self.dims
                out.shape=[self.shape[0],other.shape[1]]
                return Qobj(out)
            else:
                raise TypeError("Incompatible Qobj shapes")

        if isinstance(other, list): # if other is a list, do element-wise multiplication
            prod_list = []
            for item in other:
                prod_list.append(item * self)
            return prod_list

        if classcheck(other)=='eseries':
            return other.__mul__(self)

        if isinstance(other,(int,float,complex)): #if other is int,float,complex
            out=Qobj()
            out.data=other*self.data
            out.dims=self.dims
            out.shape=self.shape
            return Qobj(out)
        else:
            raise TypeError("Incompatible object for multiplication")

    ##### Definition of Division with Qobj on left (ex. Qobj / sqrt(2)) #########
    def __div__(self,other):
        if isinstance(other,Qobj): #if both are quantum objects
            raise TypeError("Incompatible Qobj shapes [division with Qobj not implemented]")
        #if isinstance(other,Qobj): #if both are quantum objects
        #    raise TypeError("Incompatible Qobj shapes [division with Qobj not implemented]")
        if isinstance(other,(int,float,complex)): #if other is int,float,complex
            out=Qobj()
            out.data=self.data/other
            out.dims=self.dims
            out.shape=self.shape
            return Qobj(out)
        else:
            raise TypeError("Incompatible object for division")
    def __neg__(self):
        out=Qobj()
        out.data=-self.data
        out.dims=self.dims
        out.shape=self.shape
        return Qobj(out)
    def __getitem__(self,ind):
        return self.data[ind]

    def __eq__(self, other):
        if isinstance(other,Qobj) and self.dims == other.dims and \
           self.shape == other.shape and abs(la.norm((self.data-other.data).todense())) < 1e-12:
            return True
        else:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        #return "Quantum object: ", dimensions = " + str(self.shape) + "\n" + str(self.data)
        print "Quantum object: " + "dims = " + str(self.dims) + ", shape = " + str(self.shape)
        print "Qobj data = "
        print self.full()
        return ""
    
    #####---functions acting on quantum objects---######################
    def dag(self):
        #returns Adjont (dagger) of operator
        out=Qobj()
        out.data=self.data.T.conj()
        out.dims=[self.dims[1],self.dims[0]]
        out.shape=[self.shape[1],self.shape[0]]
        return Qobj(out)
    def norm(self):
        #returns norm of quantum object
        return la.norm(self.full())
    def tr(self):
        #returns trace of quantum object
        return sum(diag(self.full()))   
    def full(self):
        #returns dense matrix form of quantum object data
        if isinstance(self.data, ndarray):
            return self.data
        return array(self.data.todense())
    def expm(self):
        if self.dims[0][0]==self.dims[1][0]:
            return sp_expm(self)
        else:
            raise TypeError('Invalid operand for matrix exponential')
    def sqrtm(self):
        if self.dims[0][0]==self.dims[1][0]:
            evals,evecs=la.eig(self.full())
            return Qobj(dot(evecs,dot(diag(sqrt(evals)),la.inv(evecs))),dims=self.dims,shape=self.shape)
        else:
	        raise TypeError('Invalid operand for matrix square root')

##############################################################################
#
#
# functions acting on Qobj class
#
#
def dag(inQobj):
	if not isinstance(inQobj,Qobj): #checks for Qobj
		raise TypeError("Input is not a quantum object")
	outQobj=Qobj()
	outQobj.data=inQobj.data.T.conj()
	outQobj.dims=[inQobj.dims[1],inQobj.dims[0]]
	outQobj.shape=[inQobj.shape[1],inQobj.shape[0]]
	return Qobj(outQobj)

def trans(A):
    out=Qobj()
    out.data=A.data.T
    out.dims=[A.dims[1],A.dims[0]]
    out.shape=[A.shape[1],A.shape[0]]
    return Qobj(out)


def isherm(ops):
    if isinstance(ops,Qobj):
        ops=array([ops])
    ops=array(ops)
    out=zeros(len(ops))
    for k in range(len(ops)):
        if (ops[k].dag()-ops[k]).norm()/ops[k].norm()>1e-12:#any non-zero elements
            out[k]=0
        else:
            out[k]=1
    return out

##############################################################################
#      
#
# some functions for increased compatibility with quantum optics toolbox:
#
#

def dims(obj):
    if isinstance(obj,Qobj):
        return Qobj.dims
    else:
        raise TypeError("Incompatible object for dims (not a Qobj)")


def shape(inpt):
    from scipy import shape as shp
    if isinstance(inpt,Qobj):
        return Qobj.shape
    else:
        return shp(inpt)


##############################################################################
#      
# functions for storing and loading Qobj instances to files 
#
import pickle

def qobj_save(qobj, filename):

    f = open(filename, 'wb')

    pickle.dump(qobj, f)

    f.close()


def qobj_load(filename):

    f = open(filename, 'rb')

    qobj = pickle.load(f)

    f.close()

    return qobj


##############################################################################
#
#
# check for class type (ESERIES,FSERIES)
#
#
#
def classcheck(inpt):
    '''
    Checks for ESERIES and FSERIES class types
    '''
    from eseries import eseries
    if isinstance(inpt,eseries):
        return 'eseries'
    else:
        pass


########################################################################
def sp_expm(qo):
    #############################
    def pade(m):
        n=shape(A)[0]
        c=padecoeff(m)
        if m!=13:
            apows= [[] for jj in xrange(int(ceil((m+1)/2)))]
            apows[0]=sp.eye(n,n).tocsr()
            apows[1]=A*A
            for jj in xrange(2,int(ceil((m+1)/2))):
                apows[jj]=apows[jj-1]*apows[1]
            U=sp.lil_matrix(zeros((n,n))).tocsr(); V=sp.lil_matrix(zeros((n,n))).tocsr()
            for jj in xrange(m,0,-2):
                U=U+c[jj]*apows[jj/2]
            U=A*U
            for jj in xrange(m-1,-1,-2):
                V=V+c[jj]*apows[(jj+1)/2]
            F=la.solve((-U+V).todense(),(U+V).todense())
            return sp.lil_matrix(F).tocsr()
        elif m==13:
            A2=A*A
            A4=A2*A2
            A6=A2*A4
            U = A*(A6*(c[13]*A6+c[11]*A4+c[9]*A2)+c[7]*A6+c[5]*A4+c[3]*A2+c[1]*sp.eye(n,n).tocsr())
            V = A6*(c[12]*A6 + c[10]*A4 + c[8]*A2)+ c[6]*A6 + c[4]*A4 + c[2]*A2 + c[0]*sp.eye(n,n).tocsr()
            F=la.solve((-U+V).todense(),(U+V).todense()) 
            return sp.lil_matrix(F).tocsr()
    #################################
    A=qo.data #extract Qobj data (sparse matrix)
    m_vals=array([3,5,7,9,13])
    theta=array([0.01495585217958292,0.2539398330063230,0.9504178996162932,2.097847961257068,5.371920351148152],dtype=float)
    normA=la.norm(A.todense(),1)
    if normA<=theta[-1]:
        for ii in xrange(len(m_vals)):
            if normA<=theta[ii]:
                F=pade(m_vals[ii])
                break
    else:
        t,s=frexp(normA/theta[-1])
        s=s-(t==0.5)
        A=A/2.0**s
        F=pade(m_vals[-1])
        for i in xrange(s):
            F=F*F
    return Qobj(F,dims=qo.dims,shape=qo.shape)


def padecoeff(m):
    if m==3:
        return array([120, 60, 12, 1])
    elif m==5:
        return array([30240, 15120, 3360, 420, 30, 1])
    elif m==7:
        return array([17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1])
    elif m==9:
        return array([17643225600, 8821612800, 2075673600, 302702400, 30270240,2162160, 110880, 3960, 90, 1])
    elif m==13:
        return array([64764752532480000, 32382376266240000, 7771770303897600,1187353796428800, 129060195264000, 10559470521600,670442572800, 33522128640, 1323241920,40840800, 960960, 16380, 182, 1])









