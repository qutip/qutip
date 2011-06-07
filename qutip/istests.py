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

def isket(L):
    """
	Determines if given quantum object is a ket-vector
	@param Qobj quantum object
	@return bool True or False
	"""
    result = isinstance(L.dims[0],list)
    if result:
        result = result and (prod(L.dims[0])==1 or prod(L.dims[1])==1)
    return result

#***************************
def isbra(L):
	"""
	Determines if given quantum object is a bra-vector
	@param Qobj quantum object
	@return bool True or False
	"""
	result = isinstance(L.dims[0],list)
	if result:
		result = result and (prod(L.dims[0])==1)
	return result


#***************************
def isoper(*args):
	"""
	Determines if given quantum object is a operator
	@param Qobj quantum object
	@return bool True or False
	"""
	if len(args)==1:
		L=args[0]
		return isinstance(L.dims[0],list) & (L.dims[0]==L.dims[1])
	elif args[1]=='rect':
		return isinstance(L.dims[0],list)
	else:
		raise TypeError('Unknown option. Only "rect" option is valid')
	


#***************************
def issuper(L,*args):
	"""
	Determines if given quantum object is a super-operator
	@param Qobj quantum object
	@return bool True or False
	"""
	result = isinstance(L.dims[0],list) & (len(L.dims[0])>1)
	if not any(args):
		if result:
			result = (L.dims[0]==L.dims[1]) & (L.dims[0][0]==L.dims[1][0])
	elif args[0]!='rect':
		raise TypeError('Unknown option. Only valid option is rect')
	return result


#**************************
def isequal(A,B,rtol=1e-8,atol=1e-12):
    """
    Determines if two array objects are equal to within tolerances
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

#******************************




