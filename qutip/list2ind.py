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

def list2ind(ilist,dims):
	ilist=asarray(ilist)
	dims=asarray(dims)
	irev=fliplr(ilist)-1
	fact=append(array([1]),(cumprod(flipud(dims)[:-1])))
	fact=fact.reshape(len(fact),1)
	return sort(dot(irev,fact)+1,0)
	
	
	