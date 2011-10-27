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
from multiprocessing import Pool
import os

def parfor(func,frange):
	"""
	Parallel execution of a for-loop over function 'func()' 
	    for a single variable 'frange'.
	
	Returns *array* with length equal to number of input parameters 
	"""
	pool=Pool(processes=int(os.environ['NUM_THREADS']))
	par_return=list(pool.map(func,frange))
	if isinstance(par_return[0],tuple):
	    par_return=[elem for elem in par_return]
	    num_elems=len(par_return[0])
	    return [array([elem[ii] for elem in par_return]) for ii in xrange(num_elems)]
	else:
	    return list(par_return)

