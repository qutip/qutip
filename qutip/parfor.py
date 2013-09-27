# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from scipy import array
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import sys
import signal
import qutip.settings as qset


def _task_wrapper(args):
    try:
        return args[0](*args[1])
    except KeyboardInterrupt:
        os.kill(args[2], signal.SIGINT)
        sys.exit(1)


def _task_wrapper_with_args(args, user_args):
    try:
        return args[0](*args[1], **user_args)
    except KeyboardInterrupt:
        os.kill(args[2], signal.SIGINT)
        sys.exit(1)


def parfor(func, *args, **kwargs):
    """Executes a multi-variable function in parallel on the local machine.

    Parallel execution of a for-loop over function `func` for multiple input
    arguments and keyword arguments.

    Parameters
    ----------
    func : function_type
        A function to run in parallel on the local machine. The function 'func' 
        accepts a series of arguments that are passed to the function as variables. 
        In general, the function can have multiple input variables, and these 
        arguments must be passed in the same order as they are defined in
        the function definition.  In addition, the user can pass multiple keyword
        arguments to the function.

    The following keyword argument is reserved:
    
    num_cpus : int
        Number of CPU's to use.  Default uses maximum number of CPU's. 
        Performance degrades if num_cpus is larger than the physical CPU 
        count of your machine.


    Returns
    -------
    result : list
        A ``list`` with length equal to number of input parameters
        containing the output from `func`.

    """
    kw = _default_parfor_settings()
    for keys in kwargs.keys():
        if keys in kw.keys():
            kw[keys]=kwargs[keys]
            del kwargs[keys]
    if len(kwargs) != 0:
        task_func = partial(_task_wrapper_with_args, user_args=kwargs)
    else:
        task_func = _task_wrapper
    
    if kw['num_cpus'] > cpu_count():
        print("Requested number of CPUs (%s) " % cpus +
              "is larger than physical number (%s)." % cpu_count())
        print("Reduce 'num_cpus' for greater performance.")

    pool = Pool(processes=kw['num_cpus'])
    var = [[args[j][i] for j in range(len(args))] for i in range(len(args[0]))]
    try:
        map_args = ((func, v, os.getpid()) for v in var)
        par_return = list(pool.map(task_func, map_args))

        if isinstance(par_return[0], tuple):
            par_return = [elem for elem in par_return]
            num_elems = len(par_return[0])
            return [array([elem[ii] for elem in par_return])
                    for ii in range(num_elems)]
        else:
            return list(par_return)

    except KeyboardInterrupt:
        pool.terminate()

def _default_parfor_settings():
    settings = {'num_cpus' : qset.num_cpus}
    return settings

