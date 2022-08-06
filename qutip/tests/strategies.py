""" Hypothesis strategies for QuTiP. """

import warnings

import numpy

from hypothesis import strategies as st, note as _note
from hypothesis.extra import numpy as npst

from qutip import Qobj
from qutip.core import data as _data


def assert_allclose(a, b, atol=1e-15, rtol=1e-15):
    """ Call numpy.testing.assert_allclose, but ignore it's warnings generated
        when comparing NaNs equal.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "invalid value encountered in multiply", RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore", "overflow encountered in absolute", RuntimeWarning,
        )
        numpy.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def note(**kw):
    """ Generate hypothesis .note() calls for the supplied keyword arguments
        apply .to_array() or .full() as necessary.
    """
    for key, value in kw.items():
        if isinstance(value, Qobj):
            key = f"{key}.full()"
            value = (type(value.data).__name__, value.full())
        elif isinstance(value, _data.Data):
            key = f"{key}.to_array()"
            value = (type(value).__name__, value.to_array())
        _note(f"{key}: {value!r}")


def qobj_dtypes():
    """
    A strategy for Qobj data-layer dtypes.
    """
    return st.sampled_from([_data.Dense, _data.CSR])


def qobj_shapes():
    """
    A strategy for Qobj data-layer shapes.
    """
    return npst.array_shapes(max_dims=2)


@st.composite
def qobj_np(draw, shape=qobj_shapes()):
    """
    A strategy for returning Qobj compatible numpy arrays.
    """
    np_array = draw(npst.arrays(shape=shape, dtype=numpy.complex128))
    if len(np_array.shape) == 1:
        np_array = numpy.atleast_2d(np_array).transpose()
    else:
        np_array = numpy.atleast_2d(np_array)
    return np_array


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
