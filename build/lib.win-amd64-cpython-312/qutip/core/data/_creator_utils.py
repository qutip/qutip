"""
Define functions to use as Data creator for `create` in `convert.py`.
"""

from scipy.sparse import isspmatrix_csr, issparse, isspmatrix_dia
import numpy as np
from .csr import CSR
from .base import Data
from .dense import Dense

__all__ = [
    'data_copy',
    'is_data',
    'is_nparray',
    'isspmatrix_csr',
    'isspmatrix_dia',
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
