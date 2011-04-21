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

def selct(sel,dims):
	sel=asarray(sel)#make sure sel is array
	dims=asarray(dims)#make sure dims is array
	rlst=dims.take(sel)
	rprod=prod(rlst)
	ilist=ones((rprod,len(dims)));
	counter=arange(rprod)
	for k in range(len(sel)):
		ilist[:,sel[k]]=remainder(fix(counter/prod(dims[sel[k+1:]])),dims[sel[k]])+1
	return ilist