__all__ = ['expect', 'variance']

import numpy as np
from typing import overload, Sequence
import itertools

from .qobj import Qobj
from .properties import isoper
from . import data as _data
from ..settings import settings
from .cy.coefficient import Coefficient
from .coefficient import coefficient
from .cy.qobjevo import QobjEvo


@overload
def expect(oper: Qobj, state: Qobj) -> complex: ...

@overload
def expect(oper: Qobj | QobjEvo, state: Qobj | QobjEvo) -> Coefficient: ...

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

    if (
        isinstance(state, (Qobj, QobjEvo))
        and isinstance(oper, (Qobj, QobjEvo))
    ):
        return _single_qobjevo_expect(oper, state)
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


def _single_qobjevo_expect(oper, state):
    oper = QobjEvo(oper)
    state = QobjEvo(state)
    if not isoper(state):
        state = state @ state.dag()

    op_list = oper.to_list()
    state_list = state.to_list()

    out_coeff = coefficient(0.)

    for op, rho in itertools.product(op_list, state_list):
        if isinstance(op, Qobj):
            op = [op, coefficient(1.)]
        if isinstance(rho, Qobj):
            rho = [rho, coefficient(1.)]

        if isinstance(op[0], Qobj) and isinstance(rho[0], Qobj):
            out_coeff = out_coeff + coefficient(
                _single_qobj_expect(op[0], rho[0])
            ) * op[1] * rho[1]

        # One of the QobjEvo is in the function format:
        # QobjEvo(lambda t, **kw: Qobj(...)
        elif isinstance(rho[0], Qobj):

            class _QevoOperExpect:
                def __init__(self, func, args, state):
                    self.oper = QobjEvo(func, args=args)
                    self.state = state.copy()

                def __call__(self, t, **args):
                    return expect(self.oper(t, **args), self.state)

            _qevo_oper_expect = _QevoOperExpect(op[0], op[1], rho[0])

            out_coeff = out_coeff + coefficient(_qevo_oper_expect) * rho[1]

        elif isinstance(op[0], Qobj):

            class _QevoStateExpect:
                def __init__(self, oper, func, args):
                    self.oper = oper.copy()
                    self.state = QobjEvo(func, args=args)

                def __call__(self, t, **args):
                    return expect(self.oper, self.state(t, **args))

            _qevo_state_expect = _QevoStateExpect(op[0], rho[0], rho[1])

            out_coeff = out_coeff + coefficient(_qevo_state_expect) * op[1]

        else:

            class _QevoBothExpect:
                def __init__(self, oper, state):
                    self.oper = QobjEvo(oper[0], args=oper[1])
                    self.state = QobjEvo(state[0], args=state[1])

                def __call__(self, t, **args):
                    return expect(self.oper(t, **args), self.state(t, **args))

            _qevo_both_expect = _QevoBothExpect(op, rho)

            out_coeff = out_coeff + coefficient(_qevo_both_expect)

    return out_coeff


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
