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
from scipy import array
from multiprocessing import Pool
import os,sys,signal
import qutip.settings as qset

def _task_wrapper(args):
    try:
        return args[0](args[1])
    except KeyboardInterrupt:
        os.kill(args[2], signal.SIGINT)
        os.exit(1)
        
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
    
    .. note::
    
        Multiple values can be passed into the parfor function using Pythons
        builtin 'zip' command, or using multidimensional `lists` or `arrays`.
         
    """
    
    pool=Pool(processes=qset.num_cpus)
    try:
        par_return=list(pool.map(_task_wrapper,[(func,f,os.getpid()) for f in frange]))
        if isinstance(par_return[0],tuple):
            par_return=[elem for elem in par_return]
            num_elems=len(par_return[0])
            return [array([elem[ii] for elem in par_return]) for ii in range(num_elems)]
        else:
            return list(par_return)
    except KeyboardInterrupt:
        pool.terminate()

