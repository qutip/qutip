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
import numpy as np
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo, EvoElement
from qutip.qobjevofunc import QobjEvoFunc
import inspect


class StateArgs:
    """Object to indicate to use the state in args outside solver.
    args[key] = StateArgs(type, op)
    """
    def __init__(self, type="Qobj", op=None):
        self.dyn_args = (type, op)

    def __call__(self):
        return self.dyn_args


class _StateAsArgs:
    # old with state (f(t, psi, args)) to new (args["state"] = psi)
    def __init__(self, func):
        self.original_func = func

    def __call__(self, t, args={}):
        return self.original_func(t, args["state_vec"], args)


class _NoArgs:
    def __init__(self, func):
        self.original_func = func

    def __call__(self, t, args={}):
        return self.original_func(t)


class _KwArgs:
    def __init__(self, func):
        self.original_func = func

    def __call__(self, t, args={}):
        return self.original_func(t, **args)


def _set_signature(func):
    if hasattr(func, "__code__"):
        code = func.__code__
    else:
        code = func.__call__.__code__

    has_kwargs = bool(inspect.getargs(code).varkw)
    # is_method = inspect.ismethod(func) or inspect.ismethod(func.__call__)
    num_args = len(inspect.getargs(code).args)

    if has_kwargs:
        # func(t, **kwargs)
        return _KwArgs(func)
    elif num_args == 1:
        # func(t) of func(self, t)
        return _NoArgs(func)
    elif num_args == 2:
        # func(t, args) of func(self, t, args)
        return func
    elif num_args == 3:
        # func(t, state, args) of func(self, t, state, args)
        return _StateAsArgs(func)
    else:
        raise Exception("Could not reconise function signature")


def qobjevo_maker(Q_object=None, args={}, tlist=None, copy=True,
                  e_ops=[], state=None):
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

    Valid function signature are:
        - f(t)
        - f(t, args)
        - f(t, state, args)  *to be deprecated
        - f(t, **kwargs)

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

    state : Qobj
        default state if rhs_with_state.

    e_ops : list of Qobj
        operators for expect dynamics args

    Returns
    -------
    L : QobjEvo or QobjEvoFunc
        The time-dependent Qobj.

    """
    if isinstance(Q_object, QobjEvo):
        obj = Q_object.copy() if copy else Q_object
    elif isinstance(Q_object, (list, Qobj)):
        obj = QobjEvo(Q_object, args, tlist, copy)
        if _all_sig_check(obj):
            args["state_vec"] = None
        obj.solver_set_args(args, state, e_ops)
    elif callable(Q_object):
        Q_object = _set_signature(Q_object)
        if isinstance(Q_object, _StateAsArgs):
            args["state_vec"] = state
        obj = QobjEvoFunc(Q_object, args, tlist, copy)
        obj.solver_set_args(args, state, e_ops)
    else:
        raise NotImplementedError(type(Q_object))
    return obj


def _all_sig_check(obj):
    new_ops = []
    state_args = False
    for op in obj.ops:
        if op.type == "func":
            fixed_sig = _set_signature(op.coeff)
            new_ops.append(EvoElement(op.qobj, fixed_sig, fixed_sig, "func"))
            if isinstance(fixed_sig, _StateAsArgs):
                state_args = True
        else:
            new_ops.append(op)
    obj.ops = new_ops
    return state_args
