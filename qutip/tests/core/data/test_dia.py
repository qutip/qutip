import numpy as np
import scipy.sparse
import pytest

from qutip.core import data, qeye, CoreOptions
from qutip.core.data import dia, Dense, Dia

from . import conftest

# We only choose a small subset of dtypes to test so it isn't crazy.
_dtype_complex = ['complex128']
_dtype_float = ['float64']
_dtype_int = ['int32', 'int64']
_dtype_uint = ['uint32']


# Set up some fixtures for automatic parametrisation.

@pytest.fixture(params=[
    pytest.param((1, 5), id='ket'),
    pytest.param((5, 1), id='bra'),
    pytest.param((5, 5), id='square'),
    pytest.param((2, 4), id='wide'),
    pytest.param((4, 2), id='tall'),
])
def shape(request): return request.param

@pytest.fixture(params=[
    pytest.param(0, id='empty'),
    pytest.param(1, id='full'),
    pytest.param(0.2, id='sparse'),
])
def density(request): return request.param


@pytest.fixture(scope='function')
def scipy_dia(shape, density):
    return conftest.random_scipy_dia(shape, density)


def _valid_scipy():
    """Arbitrary valid scipy Dia"""
    return conftest.random_scipy_dia((10, 10), 0.5)


def _valid_arg():
    """
    Arbitrary valid 3-tuple which is a valid `arg` parameter for __init__.
    """
    sci = _valid_scipy()
    return (sci.data, sci.offsets)


@pytest.fixture(scope='function')
def data_diag(shape, density):
    return conftest.random_diag(shape, density)


class TestClassMethods:
    def test_init_from_scipy(self, scipy_dia):
        """Test that __init__ can accept a scipy dia matrix."""
        out = dia.Dia(scipy_dia)
        assert out.shape == scipy_dia.shape
        assert (out.as_scipy() - scipy_dia).nnz == 0

    def test_init_from_tuple(self, scipy_dia):
        """
        Test that __init__ does not throw when passed a 3-tuple.  Also tests
        the as_scipy() method succeeds.
        """
        arg = (scipy_dia.data, scipy_dia.offsets)
        out = dia.Dia(arg, shape=scipy_dia.shape)
        assert out.shape == scipy_dia.shape
        assert (out.as_scipy() - scipy_dia).nnz == 0

    @pytest.mark.parametrize('d_type', (
        _dtype_complex + _dtype_float + _dtype_int + _dtype_uint
    ))
    @pytest.mark.parametrize('o_type', _dtype_int + _dtype_uint)
    def test_init_from_tuple_allowed_dtypes(self, d_type, o_type):
        """
        Test that initialisation can use a variety of dtypes and converts into
        the correct type.
        """
        sci = _valid_scipy()
        data = sci.data.real.astype(d_type, casting='unsafe')
        offsets = sci.offsets.astype(o_type, casting='unsafe')
        scipy_dia = scipy.sparse.dia_matrix((data, offsets), shape=sci.shape)
        out = dia.Dia((data, offsets), shape=sci.shape)
        out_scipy = out.as_scipy()
        assert out.shape == scipy_dia.shape
        assert out_scipy.data.dtype == np.complex128
        assert (out_scipy - scipy_dia).nnz == 0

    @pytest.mark.parametrize(['arg', 'kwargs', 'error'], [
        pytest.param((), {}, ValueError, id="arg 0 tuple"),
        pytest.param((None,), {}, ValueError, id="arg 1 tuple"),
        pytest.param((None,)*2, {}, TypeError, id="arg None tuple"),
        pytest.param((None,)*3, {}, ValueError, id="arg 3 tuple"),
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

        This test also serves as a *partial* check that Dia safely handles
        deallocation in the presence of exceptions in its __init__ method.  If
        the tests segfault, it's quite likely that the memory management isn't
        being done correctly in the hand-off us setting our data buffers up and
        marking the numpy actually owns the data.
        """
        with pytest.raises(error):
            dia.Dia(arg, **kwargs)

    def test_copy_returns_a_correct_copy(self, data_diag):
        """
        Test that the copy() method produces an actual copy, and that the
        result represents the same matrix.
        """
        original = data_diag
        copy = data_diag.copy()
        assert original is not copy
        assert np.all(original.to_array() == copy.to_array())

    def test_as_scipy_returns_a_view(self, data_diag):
        """
        Test that modifying the views in the result of as_scipy() also modifies
        the underlying data structures.  This is important for allowing minor
        data modifications from within Python-space.
        """
        unmodified_copy = data_diag.copy()
        data_diag.as_scipy().data += 1
        modified_copy = data_diag.copy()
        assert np.any(data_diag.to_array() != unmodified_copy.to_array())
        assert np.all(data_diag.to_array() == modified_copy.to_array())

    def test_as_scipy_caches_result(self, data_diag):
        """
        Test that the as_scipy() method always returns the same view, even if
        called multiple times.
        """
        assert data_diag.as_scipy() is data_diag.as_scipy()

    def test_as_scipy_of_dia_from_scipy_is_different(self, scipy_dia):
        """
        Test that we produce a new scipy matrix, regardless of how we have
        initialised the type.
        """
        assert dia.Dia(scipy_dia).as_scipy() is not scipy_dia

    def test_as_scipy_of_copy_is_different(self, data_diag):
        """
        Test that as_scipy() does not return the same array, or the same views
        if it's not the same input matrix.  We don't want two Dia matrices to
        be linked.
        """
        original = data_diag.as_scipy()
        copy = data_diag.copy().as_scipy()
        assert original is not copy
        assert not np.may_share_memory(original.data, copy.data)
        assert not np.may_share_memory(original.offsets, copy.offsets)

    def test_as_scipy_is_correct_result(self, scipy_dia):
        """
        Test that as_scipy is actually giving the matrix we expect for a given
        input.
        """
        data_diag = dia.Dia(scipy_dia)
        assert isinstance(data_diag.as_scipy(), scipy.sparse.dia_matrix)
        assert (data_diag.as_scipy() - scipy_dia).nnz == 0

    def test_as_scipy_of_uninitialised_is_empty(self, shape):
        ndiag = 0
        base = dia.empty(shape[0], shape[1], ndiag)
        sci = base.as_scipy()
        assert len(sci.data) == 0
        assert len(sci.offsets) == 0

    def test_to_array_is_correct_result(self, data_diag):
        test_array = data_diag.to_array()
        assert isinstance(test_array, np.ndarray)
        # It's not enough to be accurate within a tolerance here - there's no
        # mathematics, so they should be _identical_.
        assert np.all(test_array == data_diag.as_scipy().toarray())


class TestFactoryMethods:
    def test_empty(self, shape, density):
        ndiag = int(shape[0] * shape[1] * density) or 1
        base = dia.empty(shape[0], shape[1], ndiag)
        sci = base.as_scipy(full=True)
        assert isinstance(base, dia.Dia)
        assert isinstance(sci, scipy.sparse.dia_matrix)
        assert base.shape == shape
        assert sci.data.shape == (ndiag, shape[1])
        assert sci.offsets.shape == (ndiag,)

    def test_zeros(self, shape):
        base = dia.zeros(shape[0], shape[1])
        sci = base.as_scipy()
        assert isinstance(base, dia.Dia)
        assert base.shape == shape
        assert sci.nnz == 0
        assert np.all(base.to_array() == 0)

    @pytest.mark.parametrize('dimension', [1, 5, 100])
    @pytest.mark.parametrize(
        'scale',
        [None, 2, -0.1, 1.5, 1.5+1j],
        ids=['none', 'int', 'negative', 'float', 'complex']
    )
    def test_identity(self, dimension, scale):
        # scale=None is testing that the default value returns the identity.
        base = (dia.identity(dimension) if scale is None
                else dia.identity(dimension, scale))
        sci = base.as_scipy()
        scipy_test = scipy.sparse.eye(dimension,
                                      dtype=np.complex128, format='dia')
        if scale is not None:
            scipy_test *= scale
        assert isinstance(base, dia.Dia)
        assert base.shape == (dimension, dimension)
        assert sci.nnz == dimension
        assert (sci - scipy_test).nnz == 0


    @pytest.mark.parametrize(['diagonals', 'offsets', 'shape'], [
        pytest.param([2j, 3, 5, 9], None, None, id='main diagonal'),
        pytest.param([1], None, None, id='1x1'),
        pytest.param([[0.2j, 0.3]], None, None, id='main diagonal list'),
        pytest.param([0.2j, 0.3], 2, None, id='superdiagonal'),
        pytest.param([0.2j, 0.3], -2, None, id='subdiagonal'),
        pytest.param([[0.2, 0.3, 0.4], [0.1, 0.9]], [-2, 3], None,
                     id='two diagonals'),
        pytest.param([1, 2, 3], 0, (3, 5), id='main wide'),
        pytest.param([1, 2, 3], 0, (5, 3), id='main tall'),
        pytest.param([[1, 2, 3], [4, 5]], [-1, -2], (4, 8), id='two wide sub'),
        pytest.param([[1, 2, 3, 4], [4, 5, 4j, 1j]], [1, 2], (4, 8),
                     id='two wide super'),
        pytest.param([[1, 2, 3], [4, 5]], [1, 2], (8, 4), id='two tall super'),
        pytest.param([[1, 2, 3, 4], [4, 5, 4j, 1j]], [-1, -2], (8, 4),
                     id='two tall sub'),
        pytest.param([[1, 2, 3], [4, 5, 6], [1, 2]], [1, -1, -2], (4, 4),
                     id='out of order'),
        pytest.param([[1, 2, 3], [4, 5, 6], [1, 2]], [1, 1, -2], (4, 4),
                     id='sum duplicates'),
    ])
    def test_diags(self, diagonals, offsets, shape):
        base = dia.diags(diagonals, offsets, shape)
        # Build numpy version test.
        if not isinstance(diagonals[0], list):
            diagonals = [diagonals]
        offsets = np.atleast_1d(offsets if offsets is not None else [0])
        if shape is None:
            size = len(diagonals[0]) + abs(offsets[0])
            shape = (size, size)
        test = np.zeros(shape, dtype=np.complex128)
        for diagonal, offset in zip(diagonals, offsets):
            test[np.where(np.eye(*shape, k=offset) == 1)] += diagonal
        assert isinstance(base, dia.Dia)
        assert base.shape == shape
        np.testing.assert_allclose(base.to_array(), test, rtol=1e-10)


    @pytest.mark.parametrize(['shape', 'position', 'value'], [
        pytest.param((1, 1), (0, 0), None, id='minimal'),
        pytest.param((10, 10), (5, 5), 1.j, id='on diagonal'),
        pytest.param((10, 10), (1, 5), 1., id='upper'),
        pytest.param((10, 10), (5, 1), 2., id='lower'),
        pytest.param((10, 1), (5, 0), None, id='column'),
        pytest.param((1, 10), (0, 5), -5.j, id='row'),
        pytest.param((10, 2), (5, 1), 1+2j, id='tall'),
        pytest.param((2, 10), (1, 5), 10, id='wide'),
    ])
    def test_one_element(self, shape, position, value):
        test = np.zeros(shape, dtype=np.complex128)
        if value is None:
            base = data.one_element_dia(shape, position)
            test[position] = 1.0+0.0j
        else:
            base = data.one_element_dia(shape, position, value)
            test[position] = value
        assert isinstance(base, data.Dia)
        assert base.shape == shape
        assert np.allclose(base.to_array(), test, atol=1e-10)

    @pytest.mark.parametrize(['shape', 'position', 'value'], [
        pytest.param((0, 0), (0, 0), None, id='zero shape'),
        pytest.param((10, -2), (5, 0), 1.j, id='neg shape'),
        pytest.param((10, 10), (10, 5), 1., id='outside'),
        pytest.param((10, 10), (5, -1), 2., id='outside neg'),
    ])
    def test_one_element_error(self, shape, position, value):
        with pytest.raises(ValueError) as exc:
            base = data.one_element_dia(shape, position, value)
        assert str(exc.value).startswith("Position of the elements"
                                         " out of bound: ")


def test_tidyup(data_diag):
    before = data_diag.to_array()
    sp_before = data_diag.as_scipy().toarray()
    largest = max(np.abs(before.real).max(), np.abs(before.imag).max())
    min_r = np.abs(before.real[np.abs(before.real) > 0]).min()
    min_i = np.abs(before.imag[np.abs(before.imag) > 0]).min()
    smallest = min(min_r, min_i)
    print(largest, smallest)
    if largest == smallest:
        return
    tol = (largest + smallest) / 2
    tidy = data.tidyup_dia(data_diag, tol, False)
    # Inplace=False, does not modify the original
    np.testing.assert_array_equal(data_diag.to_array(), before)
    np.testing.assert_array_equal(data_diag.as_scipy().toarray(), sp_before)
    # Is tidyup
    assert not np.allclose(tidy.to_array(), before)
    assert not np.allclose(tidy.as_scipy().toarray(), sp_before)

    data.tidyup_dia(data_diag, tol, True)
    assert not np.allclose(data_diag.to_array(), before)
    assert not np.allclose(data_diag.as_scipy().toarray(), sp_before)


def test_autotidyup():
    small = (qeye(1) * 1e-5).to(Dia)
    with CoreOptions(auto_tidyup_atol=1e-3, default_dtype=Dia):
        assert (small + small).tr() == 0
    with CoreOptions(
        auto_tidyup_atol=1e-3, auto_tidyup=False, default_dtype=Dia
    ):
        assert (small + small).tr() == 2e-5
