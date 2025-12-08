# This module exists to supply a couple of very standard constant matrices
# which are used in the data layer, and within `Qobj` itself.  Other matrices
# (e.g. `create`) should not be here, but should be defined within the
# higher-level components of QuTiP instead.

from . import csr, dense, dia
from .csr import CSR
from .dia import Dia
from .dense import Dense
from .base import Data
from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

__all__ = ['zeros', 'identity', 'zeros_like', 'identity_like',
           'zeros_like_dense', 'identity_like_dense',
           'zeros_like_data', 'identity_like_data']

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
    (Dia, dia.zeros),
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
    (Dia, dia.identity),
    (Dense, dense.identity),
], _defer=True)


def zeros_like_data(data, /):
    """
    Create an zeros matrix of the same type and shape.
    """
    return zeros[type(data)](*data.shape)


def zeros_like_dense(data, /):
    """
    Create an zeros matrix of the same type and shape.
    """
    return dense.zeros(*data.shape, fortran=data.fortran)


def identity_like_data(data, /):
    """
    Create an identity matrix of the same type and shape.
    """
    if not data.shape[0] == data.shape[1]:
        raise ValueError(
            "Can't create an identity matrix like a non square matrix."
        )
    return identity[type(data)](data.shape[0])


def identity_like_dense(data, /):
    """
    Create an identity matrix of the same type and shape.
    """
    if not data.shape[0] == data.shape[1]:
        raise ValueError(
            "Can't create an identity matrix like a non square matrix."
        )
    return dense.identity(data.shape[0], fortran=data.fortran)


identity_like = _Dispatcher(
    identity_like_data, name='identity_like',
    module=__name__, inputs=("data",), out=False,
)
identity_like.add_specialisations([
    (Data, identity_like_data),
    (Dense, identity_like_dense),
], _defer=True)


zeros_like = _Dispatcher(
    zeros_like_data, name='zeros_like',
    module=__name__, inputs=("data",), out=False,
)
zeros_like.add_specialisations([
    (Data, zeros_like_data),
    (Dense, zeros_like_dense),
], _defer=True)


del _Dispatcher, _inspect
