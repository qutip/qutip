import numpy as np
import scipy.sparse
import pytest

from qutip.core import data
from qutip.core.data import csc

from . import conftest

# We only choose a small subset of dtypes to test so it isn't crazy.
_dtype_complex = ['complex128']
_dtype_float = ['float64']
_dtype_int = ['int32', 'int64']
_dtype_uint = ['uint32']


# Set up some fixtures for automatic parametrisation.

@pytest.fixture(params=[
    pytest.param((1, 5), id='bra'),
    pytest.param((5, 1), id='ket'),
    pytest.param((5, 5), id='square'),
    pytest.param((2, 4), id='wide'),
    pytest.param((4, 2), id='tall'),
])
def shape(request): return request.param


@pytest.fixture(params=[0.001, 1], ids=['sparse', 'dense'])
def density(request): return request.param


@pytest.fixture(params=[True, False], ids=['sorted', 'unsorted'])
def sorted_(request): return request.param


def _valid_scipy():
    """Arbitrary valid scipy CSC"""
    return conftest.random_scipy_csc((10, 10), 0.5, True)


def _valid_arg():
    """
    Arbitrary valid 3-tuple which is a valid `arg` parameter for __init__.
    """
    sci = _valid_scipy()
    return (sci.data, sci.indices, sci.indptr)


@pytest.fixture(scope='function')
def scipy_csc(shape, density, sorted_):
    return conftest.random_scipy_csc(shape, density, sorted_)


@pytest.fixture(scope='function')
def data_csc(shape, density, sorted_):
    return conftest.random_csc(shape, density, sorted_)



class TestClassMethods:
    def test_init_from_tuple(self, scipy_csc):
        """
        Test that __init__ does not throw when passed a 3-tuple.  Also tests
        the as_scipy() method succeeds.
        """
        arg = (scipy_csc.data, scipy_csc.indices, scipy_csc.indptr)
        out = data.CSC(arg, shape=scipy_csc.shape)
        assert out.shape == scipy_csc.shape
        assert (out.as_scipy() - scipy_csc).nnz == 0

    @pytest.mark.parametrize('d_type', (
        _dtype_complex + _dtype_float + _dtype_int + _dtype_uint
    ))
    @pytest.mark.parametrize('c_type', _dtype_int + _dtype_uint)
    @pytest.mark.parametrize('r_type', _dtype_int + _dtype_uint)
    def test_init_from_tuple_allowed_dtypes(self, d_type, c_type, r_type):
        """
        Test that initialisation can use a variety of dtypes and converts into
        the correct type.
        """
        sci = _valid_scipy()
        data_nz = np.random.randn(sci.nnz).astype(d_type, casting='unsafe')
        col_index = sci.indices.astype(c_type, casting='unsafe')
        row_index = sci.indptr.astype(r_type, casting='unsafe')
        scipy_csc = scipy.sparse.csc_matrix((data_nz, col_index, row_index),
                                            shape=sci.shape)
        out = data.CSC((data_nz, col_index, row_index), shape=sci.shape)
        out_scipy = out.as_scipy()
        assert out.shape == scipy_csc.shape
        assert out_scipy.data.dtype == np.complex128
        assert (out_scipy - scipy_csc).nnz == 0

    def test_init_from_scipy(self, scipy_csc):
        """Test that __init__ can accept a scipy CSC matrix."""
        out = data.CSC(scipy_csc)
        assert out.shape == scipy_csc.shape
        assert (out.as_scipy() - scipy_csc).nnz == 0

    @pytest.mark.parametrize(['arg', 'kwargs', 'error'], [
        pytest.param((), {}, ValueError, id="arg 0 tuple"),
        pytest.param((None,), {}, ValueError, id="arg 1 tuple"),
        pytest.param((None,)*2, {}, ValueError, id="arg 2 tuple"),
        pytest.param((None,)*3, {}, TypeError, id="arg None tuple"),
        pytest.param((None,)*4, {}, ValueError, id="arg 4 tuple"),
        pytest.param(_valid_scipy(), {'shape': ()}, ValueError,
                     id="scipy-shape 0 tuple"),
        pytest.param(_valid_scipy(), {'shape': (1,)}, ValueError,
                     id="scipy-shape 1 tuple"),
        pytest.param(_valid_scipy(), {'shape': (None, None)}, ValueError,
                     id="scipy-shape None tuple"),
        pytest.param(_valid_scipy(), {'shape': [2, 2]}, ValueError,
                     id="scipy-shape list"),
        pytest.param(_valid_scipy(), {'shape': (1, 2, 3)}, ValueError,
                     id="scipy-shape 3 tuple"),
        pytest.param(_valid_arg(), {'shape': ()}, ValueError,
                     id="arg-shape 0 tuple"),
        pytest.param(_valid_arg(), {'shape': (1,)}, ValueError,
                     id="arg-shape 1 tuple"),
        pytest.param(_valid_arg(), {'shape': (None, None)}, ValueError,
                     id="arg-shape None tuple"),
        pytest.param(_valid_arg(), {'shape': [2, 2]}, TypeError,
                     id="arg-shape list"),
        pytest.param(_valid_arg(), {'shape': (1, 2, 3)}, ValueError,
                     id="arg-shape 3 tuple"),
        pytest.param(_valid_arg(), {'shape': (-1, -1)}, ValueError,
                     id="arg-negative shape"),
    ])
    def test_init_from_wrong_input(self, arg, kwargs, error):
        """
        Test that the __init__ method raises a suitable error when passed
        incorrectly formatted inputs.

        This test also serves as a *partial* check that CSC safely handles
        deallocation in the presence of exceptions in its __init__ method.  If
        the tests segfault, it's quite likely that the memory management isn't
        being done correctly in the hand-off us setting our data buffers up and
        marking the numpy actually owns the data.
        """
        with pytest.raises(error):
            data.CSC(arg, **kwargs)

    def test_copy_returns_a_correct_copy(self, data_csc):
        """
        Test that the copy() method produces an actual copy, and that the
        result represents the same matrix.
        """
        original = data_csc
        copy = data_csc.copy()
        assert original is not copy
        assert (original.as_scipy() - copy.as_scipy()).nnz == 0

    def test_as_scipy_returns_a_view(self, data_csc):
        """
        Test that modifying the views in the result of as_scipy() also modifies
        the underlying data structures.  This is important for allowing minor
        data modifications from within Python-space.
        """
        unmodified_copy = data_csc.copy()
        data_csc.as_scipy().data[0] += 1
        modified_copy = data_csc.copy()
        assert (data_csc.as_scipy() - unmodified_copy.as_scipy()).nnz != 0
        assert (data_csc.as_scipy() - modified_copy.as_scipy()).nnz == 0

    def test_as_scipy_caches_result(self, data_csc):
        """
        Test that the as_scipy() method always returns the same view, even if
        called multiple times.
        """
        assert data_csc.as_scipy() is data_csc.as_scipy()

    def test_as_scipy_of_csc_from_scipy_is_different(self, scipy_csc):
        """
        Test that we produce a new scipy matrix, regardless of how we have
        initialised the type.
        """
        assert data.CSC(scipy_csc).as_scipy() is not scipy_csc

    def test_as_scipy_of_copy_is_different(self, data_csc):
        """
        Test that as_scipy() does not return the same array, or the same views
        if it's not the same input matrix.  We don't want two CSC matrices to
        be linked.
        """
        original = data_csc.as_scipy()
        copy = data_csc.copy().as_scipy()
        assert original is not copy
        assert not np.may_share_memory(original.data, copy.data)
        assert not np.may_share_memory(original.indices, copy.indices)
        assert not np.may_share_memory(original.indptr, copy.indptr)

    def test_as_scipy_is_correct_result(self, scipy_csc):
        """
        Test that as_scipy is actually giving the matrix we expect for a given
        input.
        """
        data_csc = data.CSC(scipy_csc)
        assert isinstance(data_csc.as_scipy(), scipy.sparse.csc_matrix)
        assert (data_csc.as_scipy() - scipy_csc).nnz == 0

    def test_as_scipy_of_uninitialised_is_empty(self, shape, density):
        nnz = int(shape[0] * shape[1] * density) or 1
        base = csc.empty(shape[0], shape[1], nnz)
        sci = base.as_scipy()
        assert sci.nnz == 0
        assert len(sci.data) == 0
        assert len(sci.indices) == 0

    def test_to_array_is_correct_result(self, data_csc):
        test_array = data_csc.to_array()
        assert isinstance(test_array, np.ndarray)
        # It's not enough to be accurate within a tolerance here - there's no
        # mathematics, so they should be _identical_.
        assert np.all(test_array == data_csc.as_scipy().toarray())

    def test_sorted_indices(self, data_csc):
        # Some matrices _cannot_ be unsorted (e.g. if they have only one entry
        # per row), so we add in this additional assertion message just to help
        # out.
        message = (
            "Sort on {}sorted indices failed."
            .format("" if data_csc.as_scipy().has_sorted_indices else "un")
        )
        # We test on a copy because scipy attempts to cache
        # `has_sorted_indices`, but since it's a view, it has no idea what
        # we've done to the indices behind the scenes and typically would not
        # notice the change.  The copy will return a difference scipy matrix,
        # so the cache will not be built.
        copy = data_csc.copy()
        copy.sort_indices()
        assert copy.as_scipy().has_sorted_indices, message


class TestFactoryMethods:
    def test_empty(self, shape, density):
        nnz = int(shape[0] * shape[1] * density) or 1
        base = csc.empty(shape[0], shape[1], nnz)
        sci = base.as_scipy(full=True)
        assert isinstance(base, data.CSC)
        assert isinstance(sci, scipy.sparse.csc_matrix)
        assert base.shape == shape
        assert sci.data.shape == (nnz,)
        assert sci.indices.shape == (nnz,)
        assert sci.indptr.shape == (shape[1] + 1,)

    def test_zeros(self, shape):
        base = csc.zeros(shape[0], shape[1])
        sci = base.as_scipy()
        assert isinstance(base, data.CSC)
        assert base.shape == shape
        assert sci.nnz == 0
        assert sci.indptr.shape == (shape[1] + 1,)

    @pytest.mark.parametrize('dimension', [1, 5, 100])
    @pytest.mark.parametrize(
        'scale',
        [None, 2, -0.1, 1.5, 1.5+1j],
        ids=['none', 'int', 'negative', 'float', 'complex']
    )
    def test_identity(self, dimension, scale):
        # scale=None is testing that the default value returns the identity.
        base = (csc.identity(dimension) if scale is None
                else csc.identity(dimension, scale))
        sci = base.as_scipy()
        scipy_test = scipy.sparse.eye(dimension,
                                      dtype=np.complex128, format='csc')
        if scale is not None:
            scipy_test *= scale
        assert isinstance(base, data.CSC)
        assert base.shape == (dimension, dimension)
        assert sci.nnz == dimension
        assert (sci - scipy_test).nnz == 0
