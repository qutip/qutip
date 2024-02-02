import numpy as np
import scipy.sparse
import pytest

from qutip.core import data
from qutip.core.data import coo
from qutip import qeye, CoreOptions

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
    """Arbitrary valid scipy COO"""
    return conftest.random_scipy_coo((10, 10), 0.5)


def _valid_arg():
    """
    Arbitrary valid 2-tuple which is a valid `arg` parameter for __init__.
    """
    sci = _valid_scipy()
    return (sci.data, (sci.col, sci.row))


@pytest.fixture(scope='function')
def scipy_coo(shape, density, sorted_):
    return conftest.random_scipy_coo(shape, density)


@pytest.fixture(scope='function')
def data_coo(shape, density):
    return conftest.random_coo(shape, density)


class TestClassMethods:
    def test_init_from_tuple(self, scipy_coo):
        """
        Test that __init__ does not throw when passed a 3-tuple.  Also tests
        the as_scipy() method succeeds.
        """
        arg = (scipy_coo.data, (scipy_coo.row, scipy_coo.col))
        out = data.COO(arg, shape=scipy_coo.shape)
        assert out.shape == scipy_coo.shape
        assert (out.as_scipy() - scipy_coo).nnz == 0

    @pytest.mark.parametrize('d_type', (
        _dtype_complex + _dtype_float + _dtype_int + _dtype_uint
    ))
    @pytest.mark.parametrize('c_type', _dtype_int + _dtype_uint)
    def test_init_from_tuple_allowed_dtypes(self, d_type, c_type):
        """
        Test that initialisation can use a variety of dtypes and converts into
        the correct type.
        """
        sci = _valid_scipy()
        data_nz = np.random.randn(sci.nnz).astype(d_type, casting='unsafe')
        col_index = sci.col.astype(c_type, casting='unsafe')
        row_index = sci.row.astype(c_type, casting='unsafe')
        scipy_coo = scipy.sparse.coo_matrix((data_nz, (row_index, col_index)),
                                            shape=sci.shape)
        out = data.COO((data_nz, (row_index, col_index)), shape=sci.shape)
        out_scipy = out.as_scipy()
        assert out.shape == scipy_coo.shape
        assert out_scipy.data.dtype == np.complex128
        assert (out_scipy - scipy_coo).nnz == 0

    def test_init_from_scipy(self, scipy_coo):
        """Test that __init__ can accept a scipy COO matrix."""
        out = data.COO(scipy_coo)
        assert out.shape == scipy_coo.shape
        assert (out.as_scipy() - scipy_coo).nnz == 0

    @pytest.mark.parametrize(['arg', 'kwargs', 'error'], [
        pytest.param((), {}, ValueError, id="arg 0 tuple"),
        pytest.param((None,), {}, ValueError, id="arg 1 tuple"),
        pytest.param((None,)*2, {}, ValueError, id="arg 2 tuple"),
        pytest.param((None,)*3, {}, ValueError, id="arg None tuple"),
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

        This test also serves as a *partial* check that COO safely handles
        deallocation in the presence of exceptions in its __init__ method.  If
        the tests segfault, it's quite likely that the memory management isn't
        being done correctly in the hand-off us setting our data buffers up and
        marking the numpy actually owns the data.
        """
        with pytest.raises(error):
            data.COO(arg, **kwargs)

    def test_copy_returns_a_correct_copy(self, data_coo):
        """
        Test that the copy() method produces an actual copy, and that the
        result represents the same matrix.
        """
        original = data_coo
        copy = data_coo.copy()
        assert original is not copy
        assert (original.as_scipy() - copy.as_scipy()).nnz == 0

    def test_as_scipy_returns_a_view(self, data_coo):
        """
        Test that modifying the views in the result of as_scipy() also modifies
        the underlying data structures.  This is important for allowing minor
        data modifications from within Python-space.
        """
        unmodified_copy = data_coo.copy()
        data_coo.as_scipy().data[0] += 1
        modified_copy = data_coo.copy()
        assert (data_coo.as_scipy() - unmodified_copy.as_scipy()).nnz != 0
        assert (data_coo.as_scipy() - modified_copy.as_scipy()).nnz == 0

    def test_as_scipy_caches_result(self, data_coo):
        """
        Test that the as_scipy() method always returns the same view, even if
        called multiple times.
        """
        assert data_coo.as_scipy() is data_coo.as_scipy()

    def test_as_scipy_of_coo_from_scipy_is_different(self, scipy_coo):
        """
        Test that we produce a new scipy matrix, regardless of how we have
        initialised the type.
        """
        assert data.COO(scipy_coo).as_scipy() is not scipy_coo

    def test_as_scipy_of_copy_is_different(self, data_coo):
        """
        Test that as_scipy() does not return the same array, or the same views
        if it's not the same input matrix.  We don't want two COO matrices to
        be linked.
        """
        original = data_coo.as_scipy()
        copy = data_coo.copy().as_scipy()
        assert original is not copy
        assert not np.may_share_memory(original.data, copy.data)
        assert not np.may_share_memory(original.row, copy.row)
        assert not np.may_share_memory(original.col, copy.col)

    def test_as_scipy_is_correct_result(self, scipy_coo):
        """
        Test that as_scipy is actually giving the matrix we expect for a given
        input.
        """
        data_coo = data.COO(scipy_coo)
        assert isinstance(data_coo.as_scipy(), scipy.sparse.coo_matrix)
        assert (data_coo.as_scipy() - scipy_coo).nnz == 0

    def test_as_scipy_of_uninitialised_is_empty(self, shape, density):
        nnz = int(shape[0] * shape[1] * density) or 1
        base = coo.empty(shape[0], shape[1], nnz)
        sci = base.as_scipy()
        assert sci.nnz == 0
        assert len(sci.data) == 0
        assert len(sci.row) == 0
        assert len(sci.col) == 0

    def test_to_array_is_correct_result(self, data_coo):
        test_array = data_coo.to_array()
        assert isinstance(test_array, np.ndarray)
        # It's not enough to be accurate within a tolerance here - there's no
        # mathematics, so they should be _identical_.
        np.testing.assert_array_equal(test_array, data_coo.as_scipy().toarray())


class TestFactoryMethods:
    def test_empty(self, shape, density):
        nnz = int(shape[0] * shape[1] * density) or 1
        base = coo.empty(shape[0], shape[1], nnz)
        assert isinstance(base, data.COO)
        assert base.shape == shape

    def test_zeros(self, shape):
        base = coo.zeros(shape[0], shape[1])
        sci = base.as_scipy()
        assert isinstance(base, data.COO)
        assert base.shape == shape
        assert sci.row.shape == (0,)
        assert sci.col.shape == (0,)

    @pytest.mark.parametrize('dimension', [1, 5, 100])
    @pytest.mark.parametrize(
        'scale',
        [None, 2, -0.1, 1.5, 1.5+1j],
        ids=['none', 'int', 'negative', 'float', 'complex']
    )
    def test_identity(self, dimension, scale):
        # scale=None is testing that the default value returns the identity.
        base = (coo.identity(dimension) if scale is None
                else coo.identity(dimension, scale))
        sci = base.as_scipy()
        scipy_test = scipy.sparse.eye(dimension,
                                      dtype=np.complex128, format='coo')
        if scale is not None:
            scipy_test *= scale
        assert isinstance(base, data.COO)
        assert base.shape == (dimension, dimension)
        assert sci.nnz == dimension
        assert (sci - scipy_test).nnz == 0


    @pytest.mark.parametrize(['shape', 'position', 'value'], [
        pytest.param((1, 1), (0, 0), None, id='minimal'),
        pytest.param((10, 10), (5, 5), 1.j, id='on diagonal'),
        pytest.param((10, 10), (1, 5), 1., id='upper'),
        pytest.param((10, 10), (5, 1), 2., id='lower'),
        pytest.param((10, 1), (5, 0), None, id='column'),
        pytest.param((1, 10), (0, 5), -5j, id='row'),
        pytest.param((10, 2), (5, 1), 1+2j, id='tall'),
        pytest.param((2, 10), (1, 5), 10, id='wide'),
    ])
    def test_one_element(self, shape, position, value):
        test = np.zeros(shape, dtype=np.complex128)
        if value is None:
            base = data.one_element_coo(shape, position)
            test[position] = 1.0+0.0j
        else:
            base = data.one_element_coo(shape, position, value)
            test[position] = value
        assert isinstance(base, data.COO)
        assert base.shape == shape
        np.testing.assert_allclose(base.to_array(), test, atol=1e-10)

    @pytest.mark.parametrize(['shape', 'position', 'value'], [
        pytest.param((0, 0), (0, 0), None, id='zero shape'),
        pytest.param((10, -2), (5, 0), 1.j, id='neg shape'),
        pytest.param((10, 10), (10, 5), 1., id='outside'),
        pytest.param((10, 10), (5, -1), 2., id='outside neg'),
    ])
    def test_one_element_error(self, shape, position, value):
        with pytest.raises(ValueError) as exc:
            base = data.one_element_coo(shape, position, value)
        assert str(exc.value).startswith("Position of the elements"
                                         " out of bound: ")



def test_tidyup():
    small = qeye(1) * 1e-5
    with CoreOptions(auto_tidyup_atol=1e-3):
        assert (small + small).tr() == 0
    with CoreOptions(auto_tidyup_atol=1e-3, auto_tidyup=False):
        assert (small + small).tr() == 2e-5
