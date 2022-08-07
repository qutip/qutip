""" Hypothesis strategies for QuTiP. """

import contextlib
import warnings

import numpy as np

from hypothesis import strategies as st, note as _note
from hypothesis.extra import numpy as npst

from qutip import Qobj
from qutip.core import data as _data


class MatrixShapesStrategy(st.SearchStrategy):
    """ A strategy for producing compatible matrix shapes.

        Parameters
        ----------
        shapes : list of shape specifications
            Each shape specification show be a tuple of dimension labels,
            e.g. ("n", "k"). Dimensions with the same labels will be
            given the same length in the resulting shapes. Fixed dimensions
            may be specified with an integer constant, e.g. ("n", 3).
        min_side : int
            The minimum length of any one dimension. Default 0.
        max_side : int
            The maximum length of any one dimension. Default 64.

        Examples
        --------
        MatrixShapeStrategy([("n", "k"), ("k", "m")])
        MatrixShapeStrategy([("j", 3), (3, "j")])
    """
    def __init__(
        self,
        shapes,
        min_side=1,
        max_side=64,
    ):
        super().__init__()
        self.side_strat = st.integers(min_side, max_side)
        self.shapes = shapes
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
def qobj_np(draw, shape=qobj_shapes()):
    """
    A strategy returning Qobj compatible numpy arrays.
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


@st.composite
def qobj_shaped_datas(
    draw,
    shapes=MatrixShapesStrategy(shapes=(("n", "k"), ("k", "m"))),
    dtype=qobj_dtypes(),
):
    """
    A strategy for returning a list of Qobj data-layer instances.
    """
    shapes = draw(shapes)
    datas = [
        draw(qobj_datas(shape=shape, dtype=dtype))
        for shape in shapes
    ]
    return datas


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
