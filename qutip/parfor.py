# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

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
        accepts a series of arguments that are passed to the function as
        variables. In general, the function can have multiple input variables,
        and these arguments must be passed in the same order as they are
        defined in the function definition.  In addition, the user can pass
        multiple keyword arguments to the function.

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
    if 'num_cpus' in kwargs.keys():
        kw['num_cpus'] = kwargs['num_cpus']
        del kwargs['num_cpus']
    if len(kwargs) != 0:
        task_func = partial(_task_wrapper_with_args, user_args=kwargs)
    else:
        task_func = _task_wrapper

    if kw['num_cpus'] > qset.num_cpus:
        print("Requested number of CPUs (%s) " % kw['num_cpus'] +
              "is larger than physical number (%s)." % qset.num_cpus)
        print("Reduce 'num_cpus' for greater performance.")

    pool = Pool(processes=kw['num_cpus'])
    args = [list(arg) for arg in args]
    var = [[args[j][i] for j in range(len(args))]
           for i in range(len(list(args[0])))]
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
    settings = {'num_cpus': qset.num_cpus}
    return settings
