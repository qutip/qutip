""" Hypothesis strategies for QuTiP. """

import numpy

from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from qutip.core import data as _data


def qobj_dtypes():
    """
    A strategy for Qobj data-layer dtypes.
    """
    dtype_list = sorted(_data.to.dtypes, key=lambda t: t.__name__)
    return st.sampled_from(dtype_list)


def qobj_shapes():
    """
    A strategy for Qobj data-layer shapes.
    """
    return npst.array_shapes(max_dims=2)


def qobj_np(shape=qobj_shapes()):
    """
    A strategy for returning Qobj compatible numpy arrays.
    """
    return npst.arrays(shape=shape, dtype=numpy.complex128)


@st.composite
def qobj_datas(draw, shape=qobj_shapes(), dtype=qobj_dtypes()):
    """
    A strategy for returning Qobj data-layer instances.

    Parameters
    ----------
    shape : strategy
        A strategy to produce the array shapes. Defaults to `qobj_shapes()`.
    dtype : strategy
        A strategy to produce the array QuTiP dtypes. Defaults to
        `qobj_dtypes()`.
    """
    # TODO: In future it might be good to have flags like unitary,
    #       hermitian, ket, dm, oper, etc to restrict the kinds of
    #       objects produced.
    dtype = draw(dtype)
    data = draw(qobj_np(shape=shape))
    return _data.to(dtype, _data.create(data))
