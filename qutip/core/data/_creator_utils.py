"""
Define functions to use as Data creator for `create` in `convert.py`.
"""
import numpy as np
from .csr import CSR
from .base import Data
from .dense import Dense
from ._scipy_sparse import is_csr, is_dia


__all__ = [
    'data_copy',
    'is_data',
    'is_nparray',
    'is_csr',
    'is_dia',
    'issparse'
]


def is_data(arg):
    return isinstance(arg, Data)


def is_nparray(arg):
    return isinstance(arg, np.ndarray)


def true(arg):
    return True


def data_copy(arg, shape, copy=True):
    if shape is not None and shape != arg.shape:
        raise ValueError("".join([
            "shapes do not match: ", str(shape), " and ", str(arg.shape),
        ]))
    return arg.copy() if copy else arg
