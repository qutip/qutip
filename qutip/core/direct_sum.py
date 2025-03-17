from .qobj import Qobj
from . import data as _data

from numbers import Number

import numpy as np

__all__ = ['direct_sum']


def _is_like_qobj(qobj):
    return isinstance(qobj, Number) or isinstance(qobj, Qobj)

def _qobj_data(qobj):
    return qobj.data if isinstance(qobj, Qobj) else np.array([[qobj]])

def _is_like_ket(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'ket']
        ))

def _is_like_bra(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'bra']
        ))

def _is_like_oper(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'ket', 'bra']
        ))

def _is_like_operator_ket(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'operator-ket']
        ))

def _is_like_operator_bra(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'operator-bra']
        ))

def _is_like_super(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'super',
                              'operator-ket', 'operator-bra']
        ))

def direct_sum(qobjs: list[Qobj | float] | list[list[Qobj | float]]) -> Qobj:
    if len(qobjs) == 0:
        raise ValueError("No Qobjs provided for direct sum.")

    linear = _is_like_qobj(qobjs[0])

    if linear and all(_is_like_ket(qobj) for qobj in qobjs):
        out_data = _data.concat_data([[_qobj_data(qobj)] for qobj in qobjs])
        return Qobj(out_data, copy=False)

    raise NotImplementedError
