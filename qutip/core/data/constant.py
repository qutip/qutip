# This module exists to supply a couple of very standard constant matrices
# which are used in the data layer, and within `Qobj` itself.  Other matrices
# (e.g. `create`) should not be here, but should be defined within the
# higher-level components of QuTiP instead.

from . import csr, dense
from .csr import CSR
from .dense import Dense
from .base import Data
from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

__all__ = ['zeros', 'identity', 'zeros_like', 'identity_like']

zeros = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('rows', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('cols', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='zeros',
    module=__name__,
    inputs=(),
    out=True,
)
zeros.__doc__ =\
    """
    Create matrix representation of 0 with the given dimensions.

    Depending on the selected output type, this may or may not actually
    contained explicit values; sparse matrices will typically contain nothing
    (which is their representation of 0), and dense matrices will still be
    filled.

    Parameters
    ----------
    rows, cols : int
        The number of rows and columns in the output matrix.
    """
zeros.add_specialisations([
    (CSR, csr.zeros),
    (Dense, dense.zeros),
], _defer=True)

identity = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('dimension',
                           _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('scale', _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           default=1),
    ]),
    name='identity',
    module=__name__,
    inputs=(),
    out=True,
)
identity.__doc__ =\
    """
    Create a square identity matrix of the given dimension.  Optionally, the
    `scale` can be given, where all the diagonal elements will be that instead
    of 1.

    Parameters
    ----------
    dimension : int
        The dimension of the square output identity matrix.
    scale : complex, optional
        The element which should be placed on the diagonal.
    """
identity.add_specialisations([
    (CSR, csr.identity),
    (Dense, dense.identity),
], _defer=True)


del _Dispatcher, _inspect


def zeros_like(data, /):
    """
    Create an zeros matrix of the same type and shape.
    """
    if type(data) is Dense:
        return dense.zeros(*data.shape, fortran=data.fortran)
    return zeros[type(data)](*data.shape)


def identity_like(data, /):
    """
    Create an identity matrix of the same type and shape.
    """
    if not data.shape[0] == data.shape[1]:
        raise ValueError("Can't create and identity like a non square matrix.")
    if type(data) is Dense:
        return dense.identity(data.shape[0], fortran=data.fortran)
    return identity[type(data)](data.shape[0])
