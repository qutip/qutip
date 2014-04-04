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
from qutip.qobj import Qobj
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
    
    if len(args) == 1 and isinstance(args[0], (list, ndarray)):
        # this is the case when tensor is called on the form: 
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    elif len(args) == 1 and isinstance(args[0], Qobj):
        # tensor is called with a single Qobj as an argument, do nothing
        return args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args        

    if not all([isinstance(q, Qobj) for q in qlist]):
        # raise error if one of the inputs is not a quantum object
        raise TypeError("One of inputs is not a quantum object")

    step = 0
    isherm = True
    for n, qos in enumerate(qlist):
        if n == 0:
            dat = q.data
            dim = q.dims
        else:
            dat = sp.kron(dat, q.data, format='csr')
            dim = [dim[0] + q.dims[0], dim[1] + q.dims[1]]

        isherm = isherm and q.isherm

    out = Qobj()
    out.data = dat
    out.dims = dim
    out.isherm = isherm

    return out.tidyup() if qutip.settings.auto_tidyup else out
