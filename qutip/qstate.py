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
from qobj import *
from basis import *
from tensor import *
import time

def qstate(string):
	n=len(string)
	if n!=(string.count('u')+string.count('d')):
		raise TypeError('String input to QSTATE must consist of "u" and "d" elements only')
	else:
		up=basis(2,1)
		dn=basis(2,0)
	lst=range(0,n)
	for k in range(n):
		if string[k]=='u':
			lst[k]=up
		else:
			lst[k]=dn
	return tensor(lst)

