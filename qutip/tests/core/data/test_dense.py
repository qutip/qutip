import numpy as np
import pytest

from qutip.core import data
from qutip.core.data import dense

from . import conftest


# Set up some fixtures for automatic parametrisation.

@pytest.fixture(params=[
    pytest.param((1, 5), id='ket'),
    pytest.param((5, 1), id='bra'),
    pytest.param((5, 5), id='square'),
    pytest.param((2, 4), id='wide'),
    pytest.param((4, 2), id='tall'),
])
def shape(request): return request.param


@pytest.fixture(params=[True, False], ids=['Fortran', 'C'])
def fortran(request): return request.param


def _valid_numpy():
    # Arbitrary valid numpy array.
    return conftest.random_numpy_dense((5, 5), False)


@pytest.fixture(scope='function')
def numpy_dense(shape, fortran):
    return conftest.random_numpy_dense(shape, fortran)


@pytest.fixture(scope='function')
def data_dense(shape, fortran):
    return conftest.random_dense(shape, fortran)


class TestClassMethods:
    def test_init_from_ndarray(self, numpy_dense):
        test = data.Dense(numpy_dense)
        assert test.shape == numpy_dense.shape
        assert np.all(test.as_ndarray() == numpy_dense)

    @pytest.mark.parametrize('dtype', ['complex128',
                                       'float64',
                                       'int32', 'int64',
                                       'uint32'])
    def test_init_from_ndarray_other_dtype(self, shape, dtype):
        numpy_dense = np.random.rand(*shape).astype(dtype, casting='unsafe')
        test = data.Dense(numpy_dense)
        assert test.shape == shape
        assert test.as_ndarray().dtype == np.complex128
        assert np.all(test.as_ndarray() == numpy_dense)

    @pytest.mark.parametrize(['arg', 'kwargs', 'error'], [
        pytest.param(_valid_numpy(), {'shape': ()}, ValueError,
                     id="numpy-shape 0 tuple"),
        pytest.param(_valid_numpy(), {'shape': (1,)}, ValueError,
                     id="numpy-shape 1 tuple"),
        pytest.param(_valid_numpy(), {'shape': (None, None)}, ValueError,
                     id="numpy-shape None tuple"),
        pytest.param(_valid_numpy(), {'shape': [2, 2]}, ValueError,
                     id="numpy-shape list"),
        pytest.param(_valid_numpy(), {'shape': (1, 2, 3)}, ValueError,
                     id="numpy-shape 3 tuple"),
        pytest.param(_valid_numpy(), {'shape': (-1, 1)}, ValueError,
                     id="numpy-negative shape"),
        pytest.param(_valid_numpy(), {'shape': (-4, -4)}, ValueError,
                     id="numpy-both negative shape"),
        pytest.param(_valid_numpy(), {'shape': (1213, 1217)}, ValueError,
                     id="numpy-different shape"),
    ])
    def test_init_from_wrong_input(self, arg, kwargs, error):
        """
        Test that the __init__ method raises a suitable error when passed
        incorrectly formatted inputs.

        This test also serves as a *partial* check that Dense safely handles
        deallocation in the presence of exceptions in its __init__ method.  If
        the tests segfault, it's quite likely that the memory management isn't
        being done correctly in the hand-off us setting our data buffers up and
        marking the numpy actually owns the data.
        """
        with pytest.raises(error):
            data.Dense(arg, **kwargs)

    def test_copy_returns_a_correct_copy(self, data_dense):
        """
        Test that the copy() method produces an actual copy, and that the
        result represents the same matrix.
        """
        original = data_dense
        copy = data_dense.copy()
        assert original is not copy
        assert np.all(original.as_ndarray() == copy.as_ndarray())

    def test_as_ndarray_returns_a_view(self, data_dense):
        """
        Test that modifying the views in the result of as_ndarray() also
        modifies the underlying data structures.  This is important for
        allowing data modification from within Python-space.
        """
        unmodified_copy = data_dense.copy()
        data_dense.as_ndarray()[0, 0] += 1
        modified_copy = data_dense.copy()
        assert np.any(data_dense.as_ndarray() != unmodified_copy.as_ndarray())
        assert np.all(data_dense.as_ndarray() == modified_copy.as_ndarray())

    def test_as_ndarray_caches_result(self, data_dense):
        """
        Test that the as_ndarray() method always returns the same view, even if
        called multiple times.
        """
        assert data_dense.as_ndarray() is data_dense.as_ndarray()

    def test_as_ndarray_of_dense_from_ndarray_is_different(self, numpy_dense):
        """
        Test that we produce a new ndarray, regardless of how we have
        initialised the type.
        """
        assert data.Dense(numpy_dense).as_ndarray() is not numpy_dense

    def test_as_ndarray_of_copy_is_different(self, data_dense):
        """
        Test that as_ndarray() does not return the same array or a view to the
        same data, if it's not the same input matrix.  We don't want two Dense
        matrices to be linked.
        """
        original = data_dense.as_ndarray()
        copy = data_dense.copy().as_ndarray()
        assert original is not copy
        assert not np.may_share_memory(original, copy)

    def test_as_ndarray_is_correct_result(self, numpy_dense):
        """
        Test that as_ndarray is actually giving the matrix we expect for a given
        input.
        """
        data_dense = data.Dense(numpy_dense)
        nd_view = data_dense.as_ndarray()
        assert isinstance(data_dense.as_ndarray(), np.ndarray)
        assert nd_view.ndim == 2
        assert nd_view.shape == numpy_dense.shape
        assert nd_view.strides == numpy_dense.strides
        assert np.all(nd_view == numpy_dense)
        assert nd_view.flags.c_contiguous == numpy_dense.flags.c_contiguous
        assert nd_view.flags.f_contiguous == numpy_dense.flags.f_contiguous

    def test_to_array_is_correct_result(self, data_dense):
        test_array = data_dense.to_array()
        nd_view = data_dense.as_ndarray()
        assert isinstance(test_array, np.ndarray)
        assert test_array.ndim == 2
        assert test_array.shape == nd_view.shape
        assert test_array.strides == nd_view.strides
        assert test_array.flags.c_contiguous == nd_view.flags.c_contiguous
        assert test_array.flags.f_contiguous == nd_view.flags.f_contiguous
        # It's not enough to be accurate within a tolerance here - there's no
        # mathematics, so they should be _identical_.
        assert np.all(test_array == nd_view)

    @pytest.mark.parametrize('new_fortran', [
        pytest.param(-1, id='swap'),
        pytest.param(False, id='C'),
        pytest.param(True, id='Fortran'),
    ])
    def test_reorder(self, data_dense, new_fortran):
        reordered = data_dense.reorder(new_fortran)
        assert isinstance(reordered, data.Dense)
        assert reordered.shape == data_dense.shape
        orig = data_dense.to_array()
        test = reordered.to_array()
        if new_fortran == -1:
            orig.flags.f_contiguous == test.flags.c_contiguous
            orig.flags.c_contiguous == test.flags.f_contiguous
        elif new_fortran is True:
            assert test.flags.f_contiguous
        else:
            assert test.flags.c_contiguous
        assert np.all(orig == test)

class TestFactoryMethods:
    def test_empty(self, shape):
        base = dense.empty(shape[0], shape[1])
        nd = base.as_ndarray()
        assert isinstance(base, data.Dense)
        assert base.shape == shape
        assert nd.shape == shape

    def test_zeros(self, shape):
        base = dense.zeros(shape[0], shape[1])
        nd = base.as_ndarray()
        assert isinstance(base, data.Dense)
        assert base.shape == shape
        assert nd.shape == shape
        assert np.count_nonzero(nd) == 0

    @pytest.mark.parametrize('dimension', [1, 5, 100])
    @pytest.mark.parametrize(
        'scale',
        [None, 2, -0.1, 1.5, 1.5+1j],
        ids=['none', 'int', 'negative', 'float', 'complex']
    )
    def test_identity(self, dimension, scale):
        # scale=None is testing that the default value returns the identity.
        base = (dense.identity(dimension) if scale is None
                else dense.identity(dimension, scale))
        nd = base.as_ndarray()
        numpy_test = np.eye(dimension, dtype=np.complex128)
        if scale is not None:
            numpy_test *= scale
        assert isinstance(base, data.Dense)
        assert base.shape == (dimension, dimension)
        assert np.count_nonzero(nd - numpy_test) == 0
