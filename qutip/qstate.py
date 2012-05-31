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
from qutip.Qobj import *
from qutip.states import basis
from qutip.tensor import *


def qstate(string):
	"""Creates a tensor product for a set of qubits in either 
	the 'up' :math:`|0>` or 'down' :math:`|1>` state.
    
    Parameters
    ----------
    string : str 
        String containing 'u' or 'd' for each qubit (ex. 'ududd')

    Returns
    ------- 
    qstate : qobj
        Qobj for tensor product corresponding to input string.
    
    Examples
    --------
    >>> qstate('udu')
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = [8, 1], type = ket
    Qobj data = 
    [[ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 1.]
     [ 0.]
     [ 0.]]
    
	"""
	n=len(string)
	if n!=(string.count('u')+string.count('d')):
		raise TypeError('String input to QSTATE must consist of "u" and "d" elements only')
	else:
		up=basis(2,1)
		dn=basis(2,0)
	lst=[]
	for k in range(n):
		if string[k]=='u':
			lst.append(up)
		else:
			lst.append(dn)
	return tensor(lst)
