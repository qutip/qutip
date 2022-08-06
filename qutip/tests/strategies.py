""" Hypothesis strategies for QuTiP. """

import contextlib
import warnings

import numpy as np

from hypothesis import strategies as st, note as _note
from hypothesis.extra import numpy as npst

from qutip import Qobj
from qutip.core import data as _data


def _convert_inf_to_nan(a):
    """ Convert infinite values (i.e. where np.isinf) to np.nan. """
    if np.issubdtype(a.dtype, complex):
        a = a.copy()
        a[np.isinf(a)] = complex(np.nan, np.nan)
    elif np.issubdtype(a.dtype, float):
        a = a.copy()
        a[np.isnf(a)] = np.nan
    return a


@contextlib.contextmanager
def ignore_arithmetic_warnings():
    """ Ignore numpy's arithmetic warnings when encountering nans and infs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "invalid value encountered in", RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore", "overflow encountered in", RuntimeWarning,
        )
        yield


def assert_allclose(
    a, b, atol=1e-15, rtol=1e-15, treat_inf_as_nan=False,
):
    """ Call numpy.testing.assert_allclose, but ignore it's warnings generated
        when comparing NaNs equal.

        Parameters
        ----------
        a : numpy.array
            The first array to compare.
        b : numpy.array
            The second array to compare.
        atol : float
            The absolute tolerance to use in the comparison. Default 1e-15.
        rtol : float
            The relative tolerance to use in the comparion. Default 1e-15.
        treat_inf_as_nan: bool
            Parts of BLAS were defined before IEEE 754 was released in 1985.
            Thus BLAS sometimes returns (nan+infj) for (0+infj). This
            option allows this break from the specification by setting all
            infinite elements (i.e. where np.isinf) to nan.
    """
    if treat_inf_as_nan:
        a = _convert_inf_to_nan(a)
        b = _convert_inf_to_nan(b)
    with ignore_arithmetic_warnings():
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


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
    np_array = draw(npst.arrays(shape=shape, dtype=np.complex128))
    if len(np_array.shape) == 1:
        np_array = np.atleast_2d(np_array).transpose()
    else:
        np_array = np.atleast_2d(np_array)
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
