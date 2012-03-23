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
from multiprocessing import Pool
import os,sys
import qutip.settings as qset

def parfor(func,frange):
	"""Executes a single-variable function in parallel.
	
	Parallel execution of a for-loop over function `func` 
	for a single variable `frange`.
	
	Parameters
	----------
	func: function_type
	    A single-variable function.
	frange: array_type
	    An ``array`` of values to be passed on to `func`.
	
	Returns
	------- 
	ans : list
	    A ``list`` with length equal to number of input parameters
	    containting the output from `func`.  In general, the ordering
	    of the output variables will not be in the same order as `frange`.
	     
	"""
	
	pool=Pool(processes=qset.num_cpus)
	par_return=list(pool.map(func,frange))
	if isinstance(par_return[0],tuple):
	    par_return=[elem for elem in par_return]
	    num_elems=len(par_return[0])
	    return [array([elem[ii] for elem in par_return]) for ii in xrange(num_elems)]
	else:
	    return list(par_return)

