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
from Qobj import *
from states import basis
from tensor import *


def qstate(string):
	"""
	Creates a tensor product for a set of qubits in either 
	the 'up' |0> or 'down' |1> state.
    
        Parameter *string* containing 'u' or 'd' for each qubit (ex. 'ududd')

        Returns *Qobj* Tensor product corresponding to input string.
	"""
	n=len(string)
	if n!=(string.count('u')+string.count('d')):
		raise TypeError('String input to QSTATE must consist of "u" and "d" elements only')
	else:
		up=basis(2,1)
		dn=basis(2,0)
	lst=[]
	for k in xrange(n):
		if string[k]=='u':
			lst.append(up)
		else:
			lst.append(dn)
	return tensor(lst)
