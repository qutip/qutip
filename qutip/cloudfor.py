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
import cloud
import numpy as np

def cloudfor(func, *args, **kwargs):
    """
    Runs the given function in parallel over the PiCloud cluster
    
    Parameters
    ----------
    func : function
        Function to run in parallel.
        
    In addition to the function 'func' to be run in parallel, the cloudfor
    function accepts a series of arguments that are passed to the function
    as variables. In general, the function can have multiple input variables, 
    and these arguemnts must be passed in the same order as they are defined in
    the function definition.
    
    Furthermore, several keyword arguments may be given that set the settings
    for the PiCloud cluster.
    
    type - Type of core used in picloud: 'c1', 'c2', 'f2' (default), 'm1', 's1'
    cores - Number of cores used: 4 (default)
    env - Custom environment for computation.  Set to current version of qutip.
    label - Provide a label for the current computation.
    
    """
    kw = _default_cloud_settings()
    for keys in kwargs.keys():
        if not keys in kw.keys():
            raise Exception(str(keys)+' is not a valid kwarg.')
        else:
            kw[keys]=kwargs[keys]
    job_ids = cloud.map(func, *args, _type=kw['type'], _cores=kw['cores'], 
                        _env=kw['env'], _label=kw['label'])
    results = cloud.result(job_ids)
    if isinstance(results[0], tuple):
        par_return = [elem for elem in results]
        num_elems = len(results[0])
        return [np.array([elem[ii] for elem in results])
                for ii in range(num_elems)]
    else:
        return list(results)


def _default_cloud_settings():
    settings = {'type':'f2','cores':4,'env':'/pnation/qutip_2_2',
                'label': 'qutip job'}
    return settings
