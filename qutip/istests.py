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
##@package istests
#Set of tests used to determine type of quantum objects
#

def isket(Q):
    """
    @brief Determines if given quantum object is a ket-vector
	@param Q quantum object
	@return bool True or False
	"""
    result = isinstance(Q.dims[0],list)
    if result:
        result = result and prod(Q.dims[1])==1
    return result

#***************************
def isbra(Q):
	"""
	@brief Determines if given quantum object is a bra-vector
	@param Qobj quantum object
	@return bool True or False
	"""
	result = isinstance(Q.dims[1],list)
	if result:
		result = result and (prod(Q.dims[0])==1)
	return result


#***************************
def isoper(Q):
	"""
	@brief Determines if given quantum object is a operator
	@param Qobj quantum object
	@return bool True or False
	"""
	return isinstance(Q.dims[0],list) and isinstance(Q.dims[0][0], int) and (Q.dims[0]==Q.dims[1])
	

#***************************
def issuper(Q):
	"""
	@brief Determines if given quantum object is a super-operator
	@param Qobj quantum object
	@return bool True or False
	"""
	result = isinstance(Q.dims[0],list) and isinstance(Q.dims[0][0],list)
	if result:
	    result = (Q.dims[0]==Q.dims[1]) & (Q.dims[0][0]==Q.dims[1][0])
	return result


#**************************
def isequal(A,B,rtol=1e-10,atol=1e-14):
    """Determines if two array objects are equal to within tolerances
    @brief Determines if two array objects are equal to within tolerances
    @return bool True or False
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
    @param qobj input quantum object
    @return bool returns True if operator is Hermitian, False otherwise
    """
    if oper.dims[0]!=oper.dims[1]:
        return False
    else:
        data=oper.data.todense()
        if la.norm(data,2)==0:
            if any(data>1e-14):
                raise ValueError('Norm=0 but nonzero data in array') 
        return allclose(data.T.conj(),data,rtol=1e-8, atol=1e-10)
