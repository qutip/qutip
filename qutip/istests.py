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
from scipy import any,prod,allclose,shape
import scipy.linalg as la
from numpy import where

"""
A collection of tests used to determine the type of quantum objects.
"""

def isket(Q):
    """
    Determines if given quantum object is a ket-vector.
	
	Parameters
	----------
	Q : qobj
	    Quantum object
	
	Returns
	------- 
	isket : bool
	    True if qobj is ket-vector, False otherwise.
	
	Examples
	--------	    
    >>> psi=basis(5,2)
    >>> isket(psi)
    True
	    
	"""
    result = isinstance(Q.dims[0],list)
    if result:
        result = result and prod(Q.dims[1])==1
    return result

#***************************
def isbra(Q):
	"""Determines if given quantum object is a bra-vector.
	
	Parameters
	----------
	Q : qobj
	    Quantum object
	
	Returns
	-------
	isbra : bool
	    True if Qobj is bra-vector, False otherwise.
	
	Examples
	--------	    
    >>> psi=basis(5,2)
    >>> isket(psi)
    False
	
	"""
	result = isinstance(Q.dims[1],list)
	if result:
		result = result and (prod(Q.dims[0])==1)
	return result


#***************************
def isoper(Q):
	"""Determines if given quantum object is a operator.
	
	Parameters
	----------
	Q : qobj
	    Quantum object
	
	Returns
	-------
	isoper : bool
	    True if Qobj is operator, False otherwise.
	
	Examples
	--------	    
    >>> a=destroy(5)
    >>> isoper(a)
    True
	
	"""
	return isinstance(Q.dims[0],list) and isinstance(Q.dims[0][0], int) and (Q.dims[0]==Q.dims[1])
	

#***************************
def issuper(Q):
	"""Determines if given quantum object is a super-operator.
	
	Parameters
	----------
	Q : qobj
	    Quantum object
	
	Returns
	------- 
	issuper  : bool
	    True if Qobj is superoperator, False otherwise.
	
	"""
	result = isinstance(Q.dims[0],list) and isinstance(Q.dims[0][0],list)
	if result:
	    result = (Q.dims[0]==Q.dims[1]) & (Q.dims[0][0]==Q.dims[1][0])
	return result


#**************************
def isequal(A,B,tol=1e-15):
    """Determines if two qobj objects are equal to within given tolerance.
    
    Parameters
    ----------    
    A : qobj 
        Qobj one
    B : qobj 
        Qobj two
    tol : float
        Tolerence for equality to be valid
    
    Returns
    -------
    isequal : bool
        True if qobjs are equal, False otherwise.
    
    """
    if A.dims!=B.dims:
        return False
    else:
        Adat=A.data
        Bdat=B.data
        elems=(Adat-Bdat).data
        if any(abs(elems)>tol):
            return False
        else:
            return True

#**************************
def ischeck(Q):
    if isoper(Q):
        return 'oper'
    elif isket(Q):
        return 'ket'
    elif isbra(Q):
        return 'bra'
    elif issuper(Q):
        return 'super'
    else:
        raise TypeError('Quantum object has undetermined type.')


#**************************
def isherm(Q):
    """Determines if given operator is Hermitian.
    
    Parameters
	----------
	Q : qobj
	    Quantum object
    
    Returns
    ------- 
    isherm : bool
        True if operator is Hermitian, False otherwise.
    
    Examples
    --------    
    >>> a=destroy(4)
    >>> isherm(a)
    False
    
    """
    if Q.dims[0]!=Q.dims[1]:
        return False
    else:
        dat=Q.data
        elems=(dat.transpose().conj()-dat).data
        if any(abs(elems)>1e-15):
            return False
        else:
            return True
