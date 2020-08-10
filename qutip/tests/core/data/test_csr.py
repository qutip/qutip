import numpy as np
import scipy.sparse
import pytest

from qutip.core import data
from qutip.core.data import csr

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
    """Arbitrary valid scipy CSR"""
    return conftest.random_scipy_csr((10, 10), 0.5, True)


def _valid_arg():
    """
    Arbitrary valid 3-tuple which is a valid `arg` parameter for __init__.
    """
    sci = _valid_scipy()
    return (sci.data, sci.indices, sci.indptr)


@pytest.fixture(scope='function')
def scipy_csr(shape, density, sorted_):
    return conftest.random_scipy_csr(shape, density, sorted_)


@pytest.fixture(scope='function')
def data_csr(shape, density, sorted_):
    return conftest.random_csr(shape, density, sorted_)


class TestClassMethods:
    def test_init_from_tuple(self, scipy_csr):
        """
        Test that __init__ does not throw when passed a 3-tuple.  Also tests
        the as_scipy() method succeeds.
        """
        arg = (scipy_csr.data, scipy_csr.indices, scipy_csr.indptr)
        out = data.CSR(arg, shape=scipy_csr.shape)
        assert out.shape == scipy_csr.shape
        assert (out.as_scipy() - scipy_csr).nnz == 0

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
        scipy_csr = scipy.sparse.csr_matrix((data_nz, col_index, row_index),
                                            shape=sci.shape)
        out = data.CSR((data_nz, col_index, row_index), shape=sci.shape)
        out_scipy = out.as_scipy()
        assert out.shape == scipy_csr.shape
        assert out_scipy.data.dtype == np.complex128
        assert (out_scipy - scipy_csr).nnz == 0

    def test_init_from_scipy(self, scipy_csr):
        """Test that __init__ can accept a scipy CSR matrix."""
        out = data.CSR(scipy_csr)
        assert out.shape == scipy_csr.shape
        assert (out.as_scipy() - scipy_csr).nnz == 0

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

        This test also serves as a *partial* check that CSR safely handles
        deallocation in the presence of exceptions in its __init__ method.  If
        the tests segfault, it's quite likely that the memory management isn't
        being done correctly in the hand-off us setting our data buffers up and
        marking the numpy actually owns the data.
        """
        with pytest.raises(error):
            data.CSR(arg, **kwargs)

    def test_copy_returns_a_correct_copy(self, data_csr):
        """
        Test that the copy() method produces an actual copy, and that the
        result represents the same matrix.
        """
        original = data_csr
        copy = data_csr.copy()
        assert original is not copy
        assert (original.as_scipy() - copy.as_scipy()).nnz == 0

    def test_as_scipy_returns_a_view(self, data_csr):
        """
        Test that modifying the views in the result of as_scipy() also modifies
        the underlying data structures.  This is important for allowing minor
        data modifications from within Python-space.
        """
        unmodified_copy = data_csr.copy()
        data_csr.as_scipy().data[0] += 1
        modified_copy = data_csr.copy()
        assert (data_csr.as_scipy() - unmodified_copy.as_scipy()).nnz != 0
        assert (data_csr.as_scipy() - modified_copy.as_scipy()).nnz == 0

    def test_as_scipy_caches_result(self, data_csr):
        """
        Test that the as_scipy() method always returns the same view, even if
        called multiple times.
        """
        assert data_csr.as_scipy() is data_csr.as_scipy()

    def test_as_scipy_of_csr_from_scipy_is_different(self, scipy_csr):
        """
        Test that we produce a new scipy matrix, regardless of how we have
        initialised the type.
        """
        assert data.CSR(scipy_csr).as_scipy() is not scipy_csr

    def test_as_scipy_of_copy_is_different(self, data_csr):
        """
        Test that as_scipy() does not return the same array, or the same views
        if it's not the same input matrix.  We don't want two CSR matrices to
        be linked.
        """
        original = data_csr.as_scipy()
        copy = data_csr.copy().as_scipy()
        assert original is not copy
        assert not np.may_share_memory(original.data, copy.data)
        assert not np.may_share_memory(original.indices, copy.indices)
        assert not np.may_share_memory(original.indptr, copy.indptr)

    def test_as_scipy_is_correct_result(self, scipy_csr):
        """
        Test that as_scipy is actually giving the matrix we expect for a given
        input.
        """
        data_csr = data.CSR(scipy_csr)
        assert isinstance(data_csr.as_scipy(), scipy.sparse.csr_matrix)
        assert (data_csr.as_scipy() - scipy_csr).nnz == 0

    def test_as_scipy_of_uninitialised_is_empty(self, shape, density):
        nnz = int(shape[0] * shape[1] * density) or 1
        base = csr.empty(shape[0], shape[1], nnz)
        sci = base.as_scipy()
        assert sci.nnz == 0
        assert len(sci.data) == 0
        assert len(sci.indices) == 0

    def test_to_array_is_correct_result(self, data_csr):
        test_array = data_csr.to_array()
        assert isinstance(test_array, np.ndarray)
        # It's not enough to be accurate within a tolerance here - there's no
        # mathematics, so they should be _identical_.
        assert np.all(test_array == data_csr.as_scipy().toarray())

    def test_sorted_indices(self, data_csr):
        # Some matrices _cannot_ be unsorted (e.g. if they have only one entry
        # per row), so we add in this additional assertion message just to help
        # out.
        message = (
            "Sort on {}sorted indices failed."
            .format("" if data_csr.as_scipy().has_sorted_indices else "un")
        )
        # We test on a copy because scipy attempts to cache
        # `has_sorted_indices`, but since it's a view, it has no idea what
        # we've done to the indices behind the scenes and typically would not
        # notice the change.  The copy will return a difference scipy matrix,
        # so the cache will not be built.
        copy = data_csr.copy()
        copy.sort_indices()
        assert copy.as_scipy().has_sorted_indices, message


class TestFactoryMethods:
    def test_empty(self, shape, density):
        nnz = int(shape[0] * shape[1] * density) or 1
        base = csr.empty(shape[0], shape[1], nnz)
        sci = base.as_scipy(full=True)
        assert isinstance(base, data.CSR)
        assert isinstance(sci, scipy.sparse.csr_matrix)
        assert base.shape == shape
        assert sci.data.shape == (nnz,)
        assert sci.indices.shape == (nnz,)
        assert sci.indptr.shape == (shape[0] + 1,)

    def test_zeros(self, shape):
        base = csr.zeros(shape[0], shape[1])
        sci = base.as_scipy()
        assert isinstance(base, data.CSR)
        assert base.shape == shape
        assert sci.nnz == 0
        assert sci.indptr.shape == (shape[0] + 1,)

    @pytest.mark.parametrize('dimension', [1, 5, 100])
    @pytest.mark.parametrize(
        'scale',
        [None, 2, -0.1, 1.5, 1.5+1j],
        ids=['none', 'int', 'negative', 'float', 'complex']
    )
    def test_identity(self, dimension, scale):
        # scale=None is testing that the default value returns the identity.
        base = (csr.identity(dimension) if scale is None
                else csr.identity(dimension, scale))
        sci = base.as_scipy()
        scipy_test = scipy.sparse.eye(dimension,
                                      dtype=np.complex128, format='csr')
        if scale is not None:
            scipy_test *= scale
        assert isinstance(base, data.CSR)
        assert base.shape == (dimension, dimension)
        assert sci.nnz == dimension
        assert (sci - scipy_test).nnz == 0
