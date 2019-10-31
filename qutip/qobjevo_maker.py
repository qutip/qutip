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
"""Factory fonction to create a QobjEvo or QobjEvoFunc from a valid
time dependent Qobj definition.
"""
__all__ = ['qobjevo_maker']
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo, EvoElement
from qutip.qobjevofunc import QobjEvoFunc


class _StateAsArgs:
    # old with state (f(t, psi, args)) to new (args["state"] = psi)
    def __init__(self, func):
        self.original_func = func

    def __call__(self, t, args={}):
        return self.original_func(t, args["_state_vec"], args)


class _NoArgs:
    def __init__(self, func):
        self.original_func = func

    def __call__(self, t, args={}):
        return self.original_func(t)


def qobjevo_maker(Q_object=None, args={}, tlist=None, copy=True,
                  rhs_with_state=False, no_args=False):
    """Create a QobjEvo or QobjEvoFunc from a valid definition.
    Valid format are:
    list format:

        Q_object = [H_0, [H_1, c_1], [H_2, c_2], ...]

        with the c_i as
            - callable: c_i(t, args)
            - string : "sin(t)"
            - array : np.sin(tlist)

    callable:

        Q_object(t, args) -> Qobj

    Parameters
    ----------
    Q_object : list or callable or Qobj
        time dependent Quantum object

    args : dict
        list of argument to pass to function contributing to the QobjEvo

    tlist : np.array
        times at which the array format coefficients are sampled

    copy : bool
        Whether make a copy of the args or the QobjEvo if Q_object is already a
        QobjEvo.

    rhs_with_state : bool
        Whether the function are defined using the old rhs_with_state format:
        c_i(t, state, args)

    no_args : bool
        Whether the function are defined without states:
        c_i(t)

    Returns
    -------
    L : QobjEvo or QobjEvoFunc
        The time-dependent Qobj.

    """
    if isinstance(Q_object, QobjEvo):
        obj = Q_object.copy() if copy else Q_object
    elif isinstance(Q_object, (list, Qobj)):
        obj = QobjEvo(Q_object, args, tlist, copy)
        if rhs_with_state:
            _with_state(obj)
        if no_args:
            _noargs(obj)
    elif callable(Q_object):
        if no_args:
            Q_object = _NoArgs(Q_object)
        elif rhs_with_state:
            Q_object = _StateAsArgs(Q_object)
            args["_state_vec=vec"] = None
        obj = QobjEvoFunc(Q_object, args, tlist, copy)
    return obj


def _with_state(obj):
    add_vec = False
    for op in obj.ops:
        if op.type == "func":
            nfunc = _StateAsArgs(obj.coeff)
            op = EvoElement((op.qobj, nfunc, nfunc, "func"))
            add_vec = True
    if add_vec:
        obj.dynamics_args += [("_state_vec", "vec", None)]


def _noargs(obj):
    for op in obj.ops:
        if op.type == "func":
            nfunc = _NoArgs(obj.coeff)
            op = EvoElement((op.qobj, nfunc, nfunc, "func"))
