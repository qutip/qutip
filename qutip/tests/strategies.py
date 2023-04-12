""" Hypothesis strategies for QuTiP. """

import functools
import contextlib
import warnings

import numpy as np
import pytest

from hypothesis import strategies as st, note as _note
from hypothesis.extra import numpy as npst

from qutip import Qobj
from qutip.core import data as _data


class MatrixShapesStrategy(st.SearchStrategy):
    """ A strategy for producing compatible matrix shapes.

        Parameters
        ----------
        shapes : list of shape specifications
            Each shape specification should be a tuple of dimension labels,
            e.g. ("n", "k"). Dimensions with the same labels will be
            given the same length in the resulting shapes. Fixed dimensions
            may be specified with an integer constant, e.g. ("n", 3).
        min_side : int
            The minimum length of any one dimension. Default 0.
        max_side : int
            The maximum length of any one dimension. Default 64.

        Examples
        --------
        MatrixShapesStrategy([("n", "k"), ("k", "m")])
        MatrixShapesStrategy([("j", 3), (3, "j")])
    """
    def __init__(
        self,
        shapes,
        min_side=1,
        max_side=64,
    ):
        super().__init__()
        self.side_strat = st.integers(min_side, max_side)
        self.shapes = tuple(shapes)
        self.min_side = min_side
        self.max_side = max_side

    def do_draw(self, data):
        dims = {}
        shapes = []
        for shape in self.shapes:
            shapes.append([])
            for name in shape:
                if isinstance(name, int):
                    shapes[-1].append(name)
                    continue
                if name not in dims:
                    dim = name
                    dims[dim] = data.draw(self.side_strat)
                shapes[-1].append(dims[name])

        shapes = tuple(tuple(s) for s in shapes)
        return shapes


def qobj_dtypes():
    """
    A strategy returning Qobj data-layer dtypes.
    """
    return st.sampled_from([_data.Dense, _data.CSR])


def qobj_shapes():
    """
    A strategy returning Qobj data-layer shapes.
    """
    return npst.array_shapes(max_dims=2)


@st.composite
def qobj_np(draw, shape=qobj_shapes(), elements=None):
    """
    A strategy returning Qobj compatible numpy arrays.
    """
    np_array = draw(npst.arrays(
        shape=shape, elements=elements, dtype=np.complex128,
    ))
    if len(np_array.shape) == 1:
        np_array = np.atleast_2d(np_array).transpose()
    else:
        np_array = np.atleast_2d(np_array)
    return np_array


@st.composite
def qobj_datas(draw, shape=qobj_shapes(), dtype=qobj_dtypes(), elements=None):
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
    # In future it might be good to have flags like unitary,
    # hermitian, ket, dm, oper, etc to restrict the kinds of
    # objects produced.
    dtype = draw(dtype)
    data = draw(qobj_np(shape=shape, elements=elements))
    return _data.to(dtype, _data.create(data))


def qobj_datas_matmul(*args, **kw):
    """
    A strategy for returning Qobj data-layer instances with suitable
    for use in matrix multiplication. Matrix multiplication is particularly
    prone imprecise results because it involves multiplication and
    multiplication, addition and subtraction of floating point numbers.

    We start with 53 bits of precision (double precision). Then we must
    divide by 2 (multiplying two matrices) and that gives the maximum
    range in matrix entries that can be accurately supported (approximately
    26 bits or 1e7). We select a range matrix entry values from 1e-3 to 1e3.

    Matrix entries that are identically zero are also allowed.
    """
    elements = st.one_of(
        st.just(0j),
        st.complex_numbers(
            allow_nan=False, allow_infinity=False, allow_subnormal=False,
            min_magnitude=1e-3, max_magnitude=1e3,
        )
    )
    return qobj_datas(*args, elements=elements, **kw)


def qobj_shared_shapes(shapes):
    """
    Return a tuple of strategies that each return one shape from
    a shared MatrixShapesStrategy strategy defined by the given shapes.

    Examples
    --------
    >>> shape_a, shape_b = qobj_shared_shapes([("n", "k"), ("k", "m")])

    In the example above shape_a would return the shapes generated for
    ("n", "k") and shape_b the shapes generated for ("k", "m").
    """
    shapes_st = MatrixShapesStrategy(shapes)
    shapes_st = st.shared(shapes_st)
    import operator
    return tuple(
        st.builds(operator.itemgetter(i), shapes_st)
        for i in range(len(shapes))
    )


def note(**kw):
    """ Generate hypothesis .note() calls for the supplied keyword arguments
        apply .to_array() or .full() as necessary.

        Hypothesis notes are output when a test case fails.

        See https://hypothesis.readthedocs.io/en/latest/details.html#hypothesis.note
        for details.
    """
    for key, value in kw.items():
        if isinstance(value, Qobj):
            key = f"{key} [Qobj, {type(value.data).__name__}]"
            value = value.full()
        elif isinstance(value, _data.Data):
            key = f"{key} [Data, {type(value).__name__}]"
            value = value.to_array()
        _note(f"{key}: {value!r}")


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


def raises_when(exc_type, msg, when, format_args=None):
    """ A decorator that assert that a test raises a given condition when
        a condition is fulfilled.

        Parameters
        ----------
        exc_type : type
            The type of exception that will be raised. E.g. ``ValueError``.
        msg : str
            The message the exception will raise. The message will be
            formatted by calling ``.format`` on it with the test
            parameters and so may include formatting such as ``{x}``
            where ``x`` is the name of a test parameter.
        when : callable
            A function of the test arguments that is true when the
            exception will be raised and false otherwise.
        format_args : callable
            A function of the test arguments that returns a set
            of keyword arguments for formatting the exception
            ``msg``. By default the function arguments are passed
            to ``.format`` directly.
    """
    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kw):
            if when(*args, **kw):
                with pytest.raises(exc_type) as err:
                    f(*args, **kw)
                if format_args:
                    args = []
                    kw = format_args(*args, **kw)
                assert str(err.value) == msg.format(*args, **kw)
            else:
                f(*args, **kw)

        return wrapper

    return decorator
