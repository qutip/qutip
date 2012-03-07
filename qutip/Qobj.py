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
import scipy.sparse as sp
import scipy.linalg as la
from qutip.istests import *
import types
from scipy import finfo

class Qobj():
    """
    A class for representing quantum objects (**Qobj**), such as quantum operators
    and states. The **Qobj** class implements math operations +,-,* between **Qobj**
    isntances (and / by a C-number).

    The Qobj constructor optionally takes the dimension array and/or
    shape array as arguments.

    Arguments:
        
        inpt (*array*): Data for vector/matrix representation of the quantum object.
    
        dims (*list*):  Dimensions of object used for tensor products.
   
        shape (*list*): Shape of underlying data structure (matrix shape).
       
    Returns a quantum object **Qobj** instance for the given input data.
    """
    ################## Define Qobj class #################
    __array_priority__=100 #sets Qobj priority above numpy arrays
    def __init__(self,inpt=array([[0]]),dims=[[],[]],shape=[],fast=False):
        """
        Qobj constructor.
        """
        if fast=='ket':#fast Qobj construction for use in mcsolve
            self.data=sp.csr_matrix(inpt,dtype=complex)
            self.dims=dims
            self.shape=shape
            self.isherm=False
            self.type='ket'
            pass
        elif isinstance(inpt,Qobj):#if input is already Qobj then return identical copy
            ##Quantum object data
            self.data=sp.csr_matrix(inpt.data,dtype=complex) #make sure matrix is sparse (safety check)
            if not any(dims):
                ## Dimensions of quantum object used for keeping track of tensor components
                self.dims=inpt.dims
            else:
                self.dims=dims
            if not any(shape):
                ##Shape of undelying quantum obejct data matrix
                self.shape=inpt.shape
            else:
                self.shape=shape
        else:#if input is NOT Qobj
            #if input is int, float, or complex then convert to array
            if isinstance(inpt,(int,float,complex)):
                inpt=array([[inpt]])
            #case where input is array or sparse
            if (isinstance(inpt,ndarray)) or (isinstance(inpt,sp.csr_matrix)):
                self.data=sp.csr_matrix(inpt,dtype=complex) #data stored as space array
                if not any(dims):
                    self.dims=[[int(inpt.shape[0])],[int(inpt.shape[1])]] #list of object dimensions
                else:
                    self.dims=dims
                if not any(shape):
                    self.shape=[int(inpt.shape[0]),int(inpt.shape[1])] # list of matrix dimensions
                else:
                    self.shape=shape
            elif isinstance(inpt,list):# case where input is not array or sparse, i.e. a list
                if len(array(inpt).shape)==1:#if list has only one dimension (i.e [5,4])
                    inpt=array([inpt])
                else:#if list has two dimensions (i.e [[5,4]])
                    inpt=array(inpt)
                self.data=sp.csr_matrix(inpt,dtype=complex)
                if not any(dims):
                    self.dims=[[int(inpt.shape[0])],[int(inpt.shape[1])]]
                else:
                    self.dims=dims
                if not any(shape):
                    self.shape=[int(inpt.shape[0]),int(inpt.shape[1])]
                else:
                    self.shape=shape
        ##Signifies if quantum object corresponds to Hermitian operator
        self.isherm=isherm(self)
        ##Signifies if quantum object corresponds to a ket, bra, operator, or super-operator
        self.type=ischeck(self)
    
    ##### Definition of PLUS with Qobj on LEFT (ex. Qobj+4) #########                
    def __add__(self, other): #defines left addition for Qobj class
        if classcheck(other)=='eseries':
            return other.__radd__(self)
        other=Qobj(other)
        if prod(other.shape)==1 and prod(self.shape)!=1: #case for scalar quantum object
            dat=array(other.full())[0][0]
            if dat!=0:
                out=Qobj()
                out.data=self.data+dat*sp.csr_matrix(ones(self.shape))
                out.dims=self.dims
                out.shape=self.shape
                return Qobj(out)
            else: #if other qobj is zero object
                return self
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

    #- Definition of PLUS with Qobj on RIGHT (ex. 4+Qobj) ###############
    def __radd__(self,other): #defines left addition for Qobj class
        return self+other

    #- Definition of SUBTRACTION with Qobj on LEFT (ex. Qobj-4) #########
    def __sub__(self,other):
        return self+(-other)

    #- Definition of SUBTRACTION with Qobj on RIGHT (ex. 4-Qobj) #########
    def __rsub__(self,other):
        return (-self)+other
    
    #-Definition of Multiplication with Qobj on left (ex. Qobj*5) #########
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
    
    #- Definition of Multiplication with Qobj on right (ex. 5*Qobj) #########
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

    #- Definition of Division with Qobj on left (ex. Qobj / sqrt(2)) #########
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
    
    def __pow__(self,n):#calculates powers of Qobj
        try:
            data=self.data**n
            return Qobj(data,dims=self.dims,shape=self.shape)
        except:
            raise ValueError('Invalid choice of exponent.')
            
    def __str__(self):
        s = ""
        if self.type=='oper' or self.type=='super':
            s += "Quantum object: " + "dims = " + str(self.dims) + ", shape = " + str(self.shape)+", type = "+self.type+", isHerm = "+str(self.isherm)+"\n"
        else:
            s += "Quantum object: " + "dims = " + str(self.dims) + ", shape = " + str(self.shape)+", type = "+self.type+"\n"
        s += "Qobj data =\n"
        s += str(self.full())
        return s
        
    def __repr__(self):#give complete information on Qobj without print statement in commandline
        # we cant realistically serialize a Qobj into a string, so we simply
        # return the informal __str__ representation instead.)
        return self.__str__()

    #---functions acting on quantum objects---######################
    def dag(self):
        """
        Returns the adjont operator (dagger) of a given quantum object.
        """
        out=Qobj()
        out.data=self.data.T.conj()
        out.dims=[self.dims[1],self.dims[0]]
        out.shape=[self.shape[1],self.shape[0]]
        return Qobj(out)
    def conj(self):
        """
        Returns the conjugate operator of a given quantum object.
        """
        out=Qobj()
        out.data=self.data.conj()
        out.dims=[self.dims[1],self.dims[0]]
        out.shape=[self.shape[1],self.shape[0]]
        return Qobj(out)
    def norm(self):
        """
        Returns norm of a quantum object. Norm is L2-norm for kets and 
        trace-norm for operators.
        """
        if self.type=='oper' or self.type=='super':
            return float(real((self.dag()*self).sqrtm().tr()))
        else:
            return la.norm(self.full(),2)
    def tr(self):
        """
        Returns the trace of a quantum object
        """
        if self.isherm==True:
            return real(sum(self.data.diagonal()))
        else:
            return sum(self.data.diagonal())
    def full(self):
        """
        Returns a dense array from quantum object data
        """
        return array(self.data.todense())
    def diag(self):
        """
        Returns diagonal elements of object
        """
        out=self.data.diagonal()
        if any(imag(out)!=0):
            return out
        else:
            return real(out)
    def expm(self):
        """
        Returns a quantum object corresponding to the matrix exponential of 
        a given square operator.
        """
        if self.dims[0][0]==self.dims[1][0]:
            return sp_expm(self)
        else:
            raise TypeError('Invalid operand for matrix exponential')
    def sqrtm(self):
        """
        Returns the operator corresponding
        to the sqrt of a given square operator.
        """
        if self.dims[0][0]==self.dims[1][0]:
            evals,evecs=la.eig(self.full())
            return Qobj(dot(evecs,dot(diag(sqrt(evals)),la.inv(evecs))),dims=self.dims,shape=self.shape)
        else:
            raise TypeError('Invalid operand for matrix square root')
    def unit(self):
        """
        Returns the operator normalized to unity.
        """
        return self/self.norm()
    def tidyup(self,Atol=1e-12):
        """
        Removes small elements from a Qobj

        Args:

            op (Qobj): input quantum object
            Atol (float): absolute tolerance

        Returns:

            Qobj with small elements removed
        """
        mx=max(abs(self.data.data))
        data=abs(self.data.data)
        outdata=self.data.copy()
        outdata.data[data<(Atol*mx+finfo(float).eps)]=0
        outdata.eliminate_zeros()
        return Qobj(outdata,dims=self.dims,shape=self.shape)


    #
    # basis transformation
    #
    def transform(self, inpt, inverse=False):
        """
        Perform a basis transformation. inpt can be a matrix defining the
        transformation or a list of kets that defines the new basis.

        .. note:: work in progress
        """
    
        if isinstance(inpt, list):
            if len(inpt) != self.shape[0] and len(inpt) != self.shape[1]:
                raise TypeError('Invalid size of ket list for basis transformation')
            S = matrix([inpt[n].full()[:,0] for n in xrange(len(inpt))]).H
        elif isinstance(inpt,ndarray):
            S = matrix(inpt)
        else:
            raise TypeError('Invalid operand for basis transformation')

        # normalize S just in case the supplied basis states aren't normalized
        #S = S/la.norm(S)

        out=Qobj()
        out.dims=[self.dims[1],self.dims[0]]
        out.shape=[self.shape[1],self.shape[0]]
        out.isherm=self.isherm
        out.type=self.type

        # transform data
        if inverse:
            if isket(self):
                out.data = S.H * self.data
            elif isbra(self):
                out.data = self.data * S
            else:
                out.data = S * self.data * S.H
        else:
            if isket(self):
                out.data = S * self.data
            elif isbra(self):
                out.data = self.data * S.H
            else:
                out.data = S.H * self.data * S

        # force sparse
        out.data = sp.csr_matrix(out.data,dtype=complex)
        
        return out

    #
    # calculate the matrix element between self and a bra and a ket
    #
    def matrix_element(self, bra, ket):
        """
        Calculate the matrix element for the Qobj sandwiched between bra and ket.
        """
        
        if isoper(self):
            if isbra(bra) and isket(ket):
                return (bra.data * self.data * ket.data)[0,0]

            if isket(bra) and isket(ket):
                return (bra.data.T * self.data * ket.data)[0,0]

        raise TypeError("Can only calculate matrix elements for operators and between ket and bra Qobj")
           

    #
    # Find the eigenstates and eigenenergies (defined for operators and
    # superoperators)
    # 
    def eigenstates(self):
        """
        Find the eigenstates and eigenenergies (defined for operators and superoperators)
        """
        if isket(self) or isbra(self):
            raise TypeError("Can only diagonalize operators and superoperators")

        evals, evecs = la.eig(self.full())
    
        zipped = zip(evals, range(len(evals)))
        zipped.sort()
        vals, perm = zip(*zipped)

        evals_sorted = array([evals[perm[i]] for i in xrange(len(perm))])
        new_dims  = [self.dims[0], [1] * len(self.dims[0])]
        new_shape = [self.shape[0], 1]
        ekets_sorted = [Qobj(matrix(evecs[:,perm[i]]/la.norm(evecs[perm[i]])).T, dims=new_dims, shape=new_shape) for i in xrange(len(perm))]

        return ekets_sorted, evals_sorted

    #
    # Find only the eigenenergies (defined for operators and superoperators)
    # 
    def eigenenergies(self):

        if isket(self) or isbra(self):
            raise TypeError("Can only diagonalize operators and superoperators")

        evals, evecs = la.eig(self.full())

        return evals


#-------------------------------------------------------------------------------
# This functions evaluates a time-dependent quantum object on the list-string
# and list-function formats that are used by the time-dependent solvers.
# Although not used directly in by those solvers, it can for test purposes be
# conventient to be able to evaluate the expressions passed to the solver for
# arbitrary value of time. This function provides this functionality.
#
#
def qobj_list_evaluate(qobj_list, t, args):
    """
    Evaluate a time-dependent qobj in list format. For example,
    
        qobj_list = [H0, [H1, func_t]]
        
    is evaluated to 
    
        Qobj(t) = H0 + H1 * func_t(t, args)
        
    and

        qobj_list = [H0, [H1, sin(w * t)]]
        
    is evaluated to 
    
        Qobj(t) = H0 + H1 * sin(args['w'] * t)  
    
    Returns:    The Qobj that represents the value of qobj_list at time t.    
    
    """
    q_sum = 0
    if isinstance(qobj_list, Qobj):
        q_sum = qobj_list
    elif isinstance(qobj_list, list):
        for q in qobj_list:
            if isinstance(q, Qobj):
                q_sum += q
            elif isinstance(q, list) and len(q) == 2 and isinstance(q[0], Qobj):
                if isinstance(q[1], types.FunctionType):
                    q_sum += q[0] * q[1](t, args)
                elif isinstance(q[1], str):
                    args['t'] = t
                    q_sum += q[0] * float(eval(q[1], globals(), args))
                else:
                    raise TypeError('Unrecongized format for specification of time-dependent Qobj')
            else:
                raise TypeError('Unrecongized format for specification of time-dependent Qobj')
    else:
        raise TypeError('Unrecongized format for specification of time-dependent Qobj')
        
    return q_sum

#-############################################################################
#
#
# functions acting on Qobj class
#
#
def dag(inQobj):
    """
    Returns the adjont operator (dagger) of a given quantum object.
    
    Argument inQobj *Qobj* input quantum object
    
    Returns *Qobj* adjoint of input operator
    """
    if not isinstance(inQobj,Qobj): #checks for Qobj
        raise TypeError("Input is not a quantum object")
    outQobj=Qobj()
    outQobj.data=inQobj.data.T.conj()
    outQobj.dims=[inQobj.dims[1],inQobj.dims[0]]
    outQobj.shape=[inQobj.shape[1],inQobj.shape[0]]
    return Qobj(outQobj)

def trans(A):
    """
    Returns the transposed operator of the given input quantum object.
    
    Argument A *Qobj* input quantum object
    
    Returns *Qobj* transpose of input operator
    """
    out=Qobj()
    out.data=A.data.T
    out.dims=[A.dims[1],A.dims[0]]
    out.shape=[A.shape[1],A.shape[0]]
    return Qobj(out)


#-############################################################################
#      
#
# some functions for increased compatibility with quantum optics toolbox:
#
#

def dims(obj):
    """
    Returns the dims attribute of a quantum object
    
    Argument qobj input quantum object
    
    Return list list of object dims
    
    Note: using 'Qobj.dims' is recommended
    """
    if isinstance(obj,Qobj):
        return Qobj.dims
    else:
        raise TypeError("Incompatible object for dims (not a Qobj)")


def shape(inpt):
    """
    Returns the shape attribute of a quantum object
    Argument qobj input quantum object
    
    Returns list list of object shape
    
    Note: using 'Qobj.shape' is recommended
    """
    from scipy import shape as shp
    if isinstance(inpt,Qobj):
        return Qobj.shape
    else:
        return shp(inpt)


#-############################################################################
#      
# functions for storing and loading Qobj instances to files 
#
import pickle

def qobj_save(qobj, filename):
    """
    Saves the given qobj to file 'filename'
    Argument qobj input operator
    Argument filename string for output file name 
    
    Returns file returns qobj as file in current directory
    """
    f = open(filename, 'wb')
    pickle.dump(qobj, f,protocol=2)
    f.close()

def qobj_load(filename):
    """
    Loads a quantum object saved using qobj_save
    Argument filename filename of request qobject
    
    Returns Qobj returns quantum object
    """
    f = open(filename, 'rb')
    qobj = pickle.load(f)
    f.close()
    return qobj

#-############################################################################
#
#
# check for class type (ESERIES,FSERIES)
#
#
#
def classcheck(inpt):
    """
    Checks for ESERIES class types
    """
    from qutip.eseries import eseries
    if isinstance(inpt,eseries):
        return 'eseries'
    else:
        pass


#-######################################################################
def sp_expm(qo):
    """
    Sparse matrix exponential of a Qobj instance
    """
    #-###########################
    def pade(m):
        n=shape(A)[0]
        c=padecoeff(m)
        if m!=13:
            apows= [[] for jj in xrange(int(ceil((m+1)/2)))]
            apows[0]=sp.eye(n,n).tocsr()
            apows[1]=A*A
            for jj in xrange(2,int(ceil((m+1)/2))):
                apows[jj]=apows[jj-1]*apows[1]
            U=sp.lil_matrix((n,n)).tocsr(); V=sp.lil_matrix((n,n)).tocsr()
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
    #-###############################
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
    """
    Private function returning coefficients for Pade apprximaion
    """
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



        





