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
"""
Module for the creation of composite quantum
objects via the tensor product.

"""
from numpy import ndarray, array
import scipy.sparse as sp
from qutip.qobj import Qobj, _typecheck
import qutip.settings


def tensor(*args):
    """Calculates the tensor product of input operators.

    Parameters
    ----------
    args : array_like
        ``list`` or ``array`` of quantum objects for tensor product.

    Returns
    --------
    obj : qobj
        A composite quantum object.

    Examples
    --------
    >>> tensor([sigmax(), sigmax()])
    Quantum object: dims = [[2, 2], [2, 2]], \
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
     [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
     [ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]]

    """
    if not args:
        raise TypeError("Requires at least one input argument")
    num_args = len(args)
    step = 0
    isherm = True
    for n in range(num_args):
        if isinstance(args[n], Qobj):
            qos = args[n]
            if step == 0:
                dat = qos.data
                dim = qos.dims
                shp = qos.shape
                isherm = isherm and qos.isherm
                step = 1
            else:
                dat = sp.kron(dat, qos.data, format='csr')
                isherm = isherm and qos.isherm
                dim = [dim[0] + qos.dims[0],
                       dim[1] + qos.dims[1]]  # append dimensions of Qobjs
                shp = [dat.shape[0], dat.shape[1]]  # new shape of matrix

        elif isinstance(args[n], (list, ndarray)):
            qos = args[n]
            items = len(qos)  # number of inputs
            if not all([isinstance(k, Qobj) for k in qos]):
                # raise error if one of the inputs is not a quantum object
                raise TypeError("One of inputs is not a quantum object")
            if items == 1:  # if only one Qobj, do nothing
                if step == 0:
                    dat = qos[0].data
                    dim = qos[0].dims
                    shp = qos[0].shape
                    isherm = isherm and qos[0].isherm
                    step = 1
                else:
                    dat = sp.kron(dat, qos[0].data, format='csr')
                    isherm = isherm and qos[0].isherm
                    dim = [dim[0] + qos[0].dims[0],
                           dim[1] + qos[0].dims[1]]  # append dimensions of qos
                    shp = [dat.shape[0], dat.shape[1]]  # new shape of matrix
            elif items != 1:
                if step == 0:
                    dat = qos[0].data
                    dim = qos[0].dims
                    shp = qos[0].shape
                    step = 1
                    isherm = isherm and qos[0].isherm
                for k in range(items - 1):  # cycle over all items
                    dat = sp.kron(dat, qos[k + 1].data, format='csr')
                    isherm = isherm and qos[k + 1].isherm
                    dim = [dim[0] + qos[k + 1].dims[0],
                           dim[1] + qos[k + 1].dims[1]]
                    shp = [dat.shape[0], dat.shape[1]]  # new shape of matrix
    out = Qobj()
    out.data = dat
    out.dims = dim
    out.shape = shp
    out.type = _typecheck(out)
    out.isherm = isherm
    if qutip.settings.auto_tidyup:
        return out.tidyup()  # returns tidy Qobj
    else:
        return out
