__all__ = ['expect', 'variance']

import numpy as np
from typing import overload, Sequence

from .qobj import Qobj
from . import data as _data
from ..settings import settings


@overload
def expect(oper: Qobj, state: Qobj) -> complex: ...

@overload
def expect(
    oper: Qobj,
    state: Qobj | Sequence[Qobj],
) -> np.typing.NDArray[complex]: ...

@overload
def expect(
    oper: Qobj | Sequence[Qobj],
    state: Qobj,
) -> list[complex]: ...

@overload
def expect(
    oper: Qobj | Sequence[Qobj],
    state: Qobj | Sequence[Qobj]
) -> list[np.typing.NDArray[complex]]: ...

def expect(oper, state):
    """
    Calculate the expectation value for operator(s) and state(s).  The
    expectation of state ``k`` on operator ``A`` is defined as
    ``k.dag() @ A @ k``, and for density matrix ``R`` on operator ``A`` it is
    ``trace(A @ R)``.

    Parameters
    ----------
    oper : qobj / list of Qobj
        A single or a `list` of operators for expectation value.

    state : qobj / list of Qobj
        A single or a `list` of quantum states or density matrices.

    Returns
    -------
    expt : float / complex / list / array
        Expectation value(s).  ``real`` if ``oper`` is Hermitian, ``complex``
        otherwise. If multiple ``oper`` are passed, a list of array.
        A (nested) array of expectaction values if ``state`` or
        ``oper`` are arrays.

    Examples
    --------
    >>> expect(num(4), basis(4, 3)) == 3 # doctest: +NORMALIZE_WHITESPACE
        True

    """
    if isinstance(state, Qobj) and isinstance(oper, Qobj):
        return _single_qobj_expect(oper, state)

    elif isinstance(oper, Sequence):
        return [expect(op, state) for op in oper]

    elif isinstance(state, Sequence):
        dtype = np.complex128
        if oper.isherm and all(op.isherm or op.isket for op in state):
            dtype = np.float64
        return np.array([_single_qobj_expect(oper, x) for x in state],
                        dtype=dtype)
    raise TypeError('Arguments must be quantum objects')


def _single_qobj_expect(oper, state):
    """
    Private function used by expect to calculate expectation values of Qobjs.
    """
    if not oper.isoper or not (state.isket or state.isoper):
        raise TypeError('invalid operand types')
    if oper._dims[1] != state._dims[0]:
        msg = (
            "incompatible dimensions "
            + str(oper.dims[1]) + " and " + str(state.dims[0])
        )
        raise ValueError(msg)
    out = _data.expect(oper.data, state.data)

    # This ensures that expect can return something that is not a number such
    # as a `tensorflow.Tensor` in qutip-tensorflow.
    if (
        settings.core["auto_real_casting"]
        and oper.isherm
        and (state.isket or state.isherm)
    ):
        out = out.real
    return out


@overload
def variance(oper: Qobj, state: Qobj) -> complex: ...

@overload
def variance(oper: Qobj, state: list[Qobj]) -> np.typing.NDArray[complex]: ...

def variance(oper, state):
    """
    Variance of an operator for the given state vector or density matrix.

    Parameters
    ----------
    oper : Qobj
        Operator for expectation value.

    state : Qobj / list of Qobj
        A single or ``list`` of quantum states or density matrices..

    Returns
    -------
    var : float
        Variance of operator 'oper' for given state.

    """
    return expect(oper**2, state) - expect(oper, state)**2
