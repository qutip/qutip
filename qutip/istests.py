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
from scipy import any,prod,allclose,shape
import scipy.linalg as la
from numpy import where

"""
Set of tests used to determine type of quantum objects
"""

def isket(Q):
    """
    Determines if given quantum object is a ket-vector
	
	Parameter Q *Qobj* quantum object
	
	Return *bool* True or False
	
	Example::
	    >>> psi=basis(5,2)
	    >>> print isket(psi)
	    True
	    
	"""
    result = isinstance(Q.dims[0],list)
    if result:
        result = result and prod(Q.dims[1])==1
    return result

#***************************
def isbra(Q):
	"""
	Determines if given quantum object is a bra-vector
	
	Parameter Q *Qobj* quantum object
	
	Returns *bool* True or False
	
	Example::
	    
	    >>> psi=basis(5,2)
	    >>> print isket(psi)
	    False
	

	"""
	result = isinstance(Q.dims[1],list)
	if result:
		result = result and (prod(Q.dims[0])==1)
	return result


#***************************
def isoper(Q):
	"""
	Determines if given quantum object is a operator
	
	Parameter Q *Qobj* quantum object
	
	Returns *bool* True or False
	
	Example::
	    
	    >>> a=destroy(4)
	    >>> isoper(a)
	    True
	
	"""
	return isinstance(Q.dims[0],list) and isinstance(Q.dims[0][0], int) and (Q.dims[0]==Q.dims[1])
	

#***************************
def issuper(Q):
	"""
	Determines if given quantum object is a super-operator
	
	Parameter Q *Qobj* quantum object
	
	Returns *bool* True or False
	"""
	result = isinstance(Q.dims[0],list) and isinstance(Q.dims[0][0],list)
	if result:
	    result = (Q.dims[0]==Q.dims[1]) & (Q.dims[0][0]==Q.dims[1][0])
	return result


#**************************
def isequal(A,B,rtol=1e-10,atol=1e-12):
    """
    Determines if two array objects are equal to within tolerances
    
    Parameter A *array* array one
    Parameter B *array* array two
    Parameter rtol *float* relative tolerence
    Parameter atol *float* absolute tolerence
    
    Returns *bool* True or False
    """
    if shape(A)!=shape(B):
        raise TypeError('Inputs do not have same shape.')
    else:
        x=allclose(A,B,rtol,atol)
        y=allclose(B,A,rtol,atol)
        if x and y:
            return True
        elif x or y:
            print 'isequal result is not symmetric with respect to inputs.\n See numpy.allclose documentation'
            return True
        else:
            return False

#**************************
def ischeck(Q):
    if isket(Q):
        return 'ket'
    elif isbra(Q):
        return 'bra'
    elif isoper(Q):
        return 'oper'
    elif issuper(Q):
        return 'super'
    else:
        raise TypeError('Quantum object has undetermined type.')


#**************************
def isherm(oper):
    """
    Determines whether a given operator is Hermitian
    
    Parameter oper *Qobj* input quantum object
    
    Returns *bool* True if operator is Hermitian, False otherwise
    
    Example::
        
        >>> a=destroy(4)
        >>> isherm(a)
        False
    
    """
    if oper.dims[0]!=oper.dims[1]:
        return False
    else:
        dat=oper.data
        elems=(dat.transpose().conj()-dat).data
        if any(abs(elems)>1e-12):
            return False
        else:
            return True
