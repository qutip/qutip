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
import cloud
import numpy as np


def picloud(func, *args, **kwargs):
    """
    Runs the given function in parallel over the PiCloud cluster.

    Parameters
    ----------
    func : function
        Function to run in parallel.

    In addition to the function 'func' to be run in parallel, the picloud
    function accepts a series of arguments that are passed to the function
    as variables. In general, the function can have multiple input variables,
    and these arguments must be passed in the same order as they are defined in
    the function definition.

    Furthermore, several keyword arguments may be given that set the settings
    for the PiCloud cluster:

    _type - Type of core used in picloud: 'c1', 'c2', 'f2' (default), 'm1',
            's1'
    _cores - Number of cores used: 1 (default)
    _env - Custom environment for computation. Set to current version of qutip.
    _label - Provide a label for the current computation.

    For more information see the PiCloud website: http://www.picloud.com/

    """
    kw = _default_cloud_settings()
    for keys in kwargs.keys():
        if keys not in kw.keys():
            raise Exception(str(keys) + ' is not a valid kwarg.')
        else:
            kw[keys] = kwargs[keys]
    job_ids = cloud.map(func, *args, **kw)
    results = cloud.result(job_ids)
    if isinstance(results[0], tuple):
        par_return = [elem for elem in results]
        num_elems = len(results[0])
        return [np.array([elem[ii] for elem in results])
                for ii in range(num_elems)]
    else:
        return list(results)


def _default_cloud_settings():
    settings = {'_type': 'f2', '_cores': 1, '_env': '/pnation/qutip_2_2',
                '_label': 'qutip job'}
    return settings
