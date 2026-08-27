import itertools
import numpy as np
import pytest
import scipy
import warnings

from qutip.core import data
from qutip.core.data import Data, Dense, CSR, Dia
from qutip.core.data.dense import OrderEfficiencyWarning

from qutip.testing import random_data
from qutip.testing import mixin

# The ParameterSet is actually a pretty hidden type, so it's easiest to access
# it like this.
_ParameterSet = type(pytest.param())

# First set up a bunch of allowable shapes, for different types of functions so
# we don't have to respecify a whole lot of things on repeat.


def shapes_unary(dim=100):
    """Base shapes to test for unary functions."""
    # Be sure to test a full spectrum bra-type, ket-type and square and
    # non-square operators.  Keep the dimension sensible, particularly for
    # things like kron, since these shapes are reused to build the shapes for
    # higher-order functions too.
    return [
        pytest.param((1, 1), id="scalar"),
        pytest.param((1, dim), id="bra"),
        pytest.param((dim, 1), id="ket"),
        pytest.param((dim, dim), id="square"),
        pytest.param((2, dim), id="nonsquare"),
    ]


def shapes_binary_identical(dim=100):
    """
    Allowed shapes for binary operators that need the two matrices to be the
    same shape, e.g. addition.
    """
    return [(x, x) for x in shapes_unary(dim)]


def shapes_binary_bad_identical(dim=100):
    """
    Disallowed shapes for binary operators that need the two matrices to be the
    same shape, e.g. addition.
    """
    return [
        (x, y)
        for x, y in itertools.product(shapes_unary(dim), repeat=2)
        if x.values[0] != y.values[0]
    ]


def shapes_binary_unrestricted(dim=100):
    """
    Allowed shapes for binary operators which can take any shapes, e.g. the
    Kronecker product.
    """
    return list(itertools.product(shapes_unary(dim), repeat=2))


def shapes_binary_bad_unrestricted(dim=100):
    """
    Disallowed shapes for binary operators which can take any shapes, e.g. the
    Kronecker product.  There aren't actually any of these, but we keep it just
    for consistency.
    """
    return []


def shapes_binary_matmul(dim=100):
    """
    Allowed shapes for "matmul"-like operators that require that the "inner"
    two indices are equal, i.e. the columns on the left equal the rows on the
    right.
    """
    return [
        (x, y)
        for x, y in itertools.product(shapes_unary(dim), repeat=2)
        if x.values[0][1] == y.values[0][0]
    ]


def shapes_binary_bad_matmul(dim=100):
    """
    Disallowed shapes for "matmul"-like operators that require that the "inner"
    two indices are equal, i.e. the columns on the left equal the rows on the
    right.
    """
    return [
        (x, y)
        for x, y in itertools.product(shapes_unary(dim), repeat=2)
        if x.values[0][1] != y.values[0][0]
    ]


def shapes_ternary_matmul_inplace(dim=100):
    """
    Shapes for in-place matmul: (left, right, out) where out.shape ==
    (left.rows, right.cols) and left.cols == right.rows.
    """
    shapes = []
    for left, right in shapes_binary_matmul(dim):
        left_shape = left.values[0]
        right_shape = right.values[0]
        out_shape = (left_shape[0], right_shape[1])
        out = pytest.param(out_shape, id=f"{out_shape[0]}x{out_shape[1]}")
        shapes.append((left, right, out))
    return shapes


def shapes_square(dim=100):
    """Allowed shapes for operations that require square matrices. Examples of
    these operations are trace, pow, expm and the trace norm."""
    return [
        (pytest.param((1, 1), id="1"),),
        (pytest.param((dim, dim), id=str(dim)),),
    ]


def shapes_not_square(dim=100):
    """Disallowed shapes for operations that require square matrices. Examples
    of these operations are trace, pow, expm and the trace norm."""
    return [
        (x,) for x in shapes_unary(dim) if x.values[0][0] != x.values[0][1]
    ]


# Set up the special cases for each type of matrix that will be tested.  These
# should be kept low, because mathematical operations will test a Cartesian
# product of all the cases of the same order as the operation, which can get
# very large very fast.  The operations should each complete in a small amount
# of time, so having 10000+ tests in this file still ought to take less than 2
# minutes, but it's easy to accidentally add orders of magnitude on.
#
# There is a layer of indirection---the cases are returned as 0-ary generator
# closures---for two reasons:
#   1. we don't have to store huge amounts of data at test collection time, but
#      the matrices are only generated, and subsequently freed, within in each
#      individual test.
#   2. each test can be repeated, and new random matrices will be generated for
#      each repeat, rather than re-using the same set.  This is somewhat
#      "defeating" pytest fixtures, but here we're not worried about re-usable
#      inputs, we just want the managed parametrisation.

def cases_csr(shape):
    """
    Return a list of generators of the different special cases for CSR
    matrices of a given shape.
    """
    def factory(density, sort):
        return lambda gen: random_generator.random_csr(shape, density, sort, gen)

    def zero_factory():
        return lambda _: data.csr.zeros(shape[0], shape[1])
    return [
        pytest.param(factory(0.001, True), id="sparse"),
        pytest.param(factory(0.8, True), id="filled,sorted"),
        pytest.param(factory(0.8, False), id="filled,unsorted"),
        pytest.param(zero_factory(), id="zero"),
    ]


def cases_dense(shape):
    """
    Return a list of generators of the different special cases for Dense
    matrices of a given shape.
    """
    def factory(fortran):
        return lambda gen: random_generator.random_dense(shape, fortran, gen)
    return [
        pytest.param(factory(False), id="C"),
        pytest.param(factory(True), id="Fortran"),
    ]


def cases_diag(shape):
    """
    Return a list of generators of the different special cases for Dense
    matrices of a given shape.
    """
    def factory(density, sort=False):
        return lambda gen: random_generator.random_diag(shape, density, sort, gen)

    def zero_factory():
        return lambda _: data.dia.zeros(shape[0], shape[1])

    return [
        pytest.param(factory(0.001), id="sparse"),
        pytest.param(factory(0.8, True), id="filled,sorted"),
        pytest.param(factory(0.8, False), id="filled,unsorted"),
        pytest.param(zero_factory(), id="zero"),
    ]


# Factory methods for generating the cases, mapping type to the function.
# _ALL_CASES is for getting all the special cases to test, _RANDOM is for
# getting just a single case from each.
mixin.CORRECT_CASES = {
    CSR: cases_csr,
    Dia: cases_diag,
    Dense: cases_dense,
}
mixin.WRONG_CASES = {
    CSR: lambda shape: [lambda gen: random_generator.random_csr(shape, 0.5, True, gen)],
    Dense: lambda shape: [lambda: random_generator.random_dense(shape, False, gen)],
    Dia: lambda shape: [lambda: random_generator.random_diag(shape, 0.5, gen=gen)],
}


UnaryOpMixin = mixin.UnaryOpMixin
UnaryScalarOpMixin = mixin.UnaryScalarOpMixin
BinaryOpMixin = mixin.BinaryOpMixin
ScaledBinaryOpMixin = mixin.ScaledBinaryOpMixin
TernaryOpMixin = mixin.TernaryOpMixin


# And now finally we get into the meat of the actual mathematical tests.

class TestAdd(ScaledBinaryOpMixin):
    def op_numpy(self, left, right, scale=1):
        return np.add(left, scale * right)

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()
    specialisations = [
        pytest.param(data.add_csr, CSR, CSR, CSR),
        pytest.param(data.add_dense, Dense, Dense, Dense),
        pytest.param(data.add_dia, Dia, Dia, Dia),
        pytest.param(data.iadd_dense, Dense, Dense, Dense),
        pytest.param(data.iadd_dense_data_dense, Dense, Dia, Dense),
        pytest.param(data.iadd_data, CSR, Dia, Data),
    ]


class TestAdjoint(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.conj(matrix.T)

    specialisations = [
        pytest.param(data.adjoint_csr, CSR, CSR),
        pytest.param(data.adjoint_dense, Dense, Dense),
        pytest.param(data.adjoint_dia, Dia, Dia),
    ]


class TestConj(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.conj(matrix)

    specialisations = [
        pytest.param(data.conj_csr, CSR, CSR),
        pytest.param(data.conj_dense, Dense, Dense),
        pytest.param(data.conj_dia, Dia, Dia),
    ]


class TestInner(BinaryOpMixin):
    # The inner product is a bit more specialist, since it has to handle inputs
    # in a 1D space specially.  In order to keep things simple, we just
    # generate those test cases completely separately from the standard
    # `mathematically_correct`.

    def op_numpy(self, left, right, scalar_is_ket=False):
        if left.shape[1] == 1:
            if left.shape[0] != 1 or scalar_is_ket:
                left = np.conj(left.T)
        return (left @ right)[0, 0]

    # These shapes are a little more non-standard.
    _dim = 100
    _scalar = pytest.param((1, 1), id="scalar")
    _bra = pytest.param((1, _dim), id="bra")
    _ket = pytest.param((_dim, 1), id="ket")
    _op = pytest.param((_dim, _dim), id="square")
    shapes = [
        (_bra, _ket),
        (_ket, _ket),
    ]
    bad_shapes = [
        (_bra, _bra),
        (_ket, _bra),
        (_op, _ket),
        (_op, _bra),
        (_bra, _op),
        (_ket, _op),
    ]

    specialisations = [
        pytest.param(data.inner_csr, CSR, CSR, complex),
        pytest.param(data.inner_dia, Dia, Dia, complex),
        pytest.param(data.inner_dense, Dense, Dense, complex),
        pytest.param(data.inner_data, Dense, Dense, complex),
        pytest.param(data.inner_data, CSR, CSR, complex),
    ]

    def generate_scalar_is_ket(self, metafunc):
        # For 1D subspaces, the special cases don't really matter since there's
        # only really one type of matrix available, so this is parametrised
        # with only case for each input argument.
        parameters = (
            ['op']
            + [x for x in metafunc.fixturenames
               if x.startswith("data_")]
            + ['out_type']
        )
        cases = []
        for p_op in self.specialisations:
            op, *types, out_type = p_op.values
            args = (op, types, [(self._scalar, self._scalar)], out_type)
            cases.extend(cases_type_shape_product(_RANDOM, *args))
        metafunc.parametrize(parameters, cases)
        metafunc.parametrize('scalar_is_ket',
                             [True, False],
                             ids=["ket", "bra"])

    def test_scalar_is_ket(self, op, data_l, data_r, out_type, scalar_is_ket):
        left, right = data_l(), data_r()
        expected = self.op_numpy(left.to_array(), right.to_array(),
                                 scalar_is_ket)
        test = op(left, right, scalar_is_ket)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)


class TestInnerOp(TernaryOpMixin):
    # This is very very similar to TestInner.
    def op_numpy(self, left, mid, right, scalar_is_ket=False):
        if left.shape[1] == 1:
            if left.shape[0] != 1 or scalar_is_ket:
                left = np.conj(left.T)
        return (left @ mid @ right)[0, 0]

    _dim = 100
    _scalar = pytest.param((1, 1), id="scalar")
    _bra = pytest.param((1, _dim), id="bra")
    _ket = pytest.param((_dim, 1), id="ket")
    _op = pytest.param((_dim, _dim), id="square")
    shapes = [
        (_bra, _op, _ket),
        (_ket, _op, _ket),
    ]
    bad_shapes = [
        (_bra, _op, _bra),
        (_ket, _op, _bra),
        (_op, _op, _ket),
        (_op, _op, _bra),
        (_bra, _op, _op),
        (_ket, _op, _op),
        (_bra, _bra, _ket),
        (_ket, _bra, _ket),
        (_bra, _ket, _ket),
        (_ket, _ket, _ket),
    ]

    specialisations = [
        pytest.param(data.inner_op_csr, CSR, CSR, CSR, complex),
        pytest.param(data.inner_op_dia, Dia, Dia, Dia, complex),
        pytest.param(data.inner_op_dense, Dense, Dense, Dense, complex),
        pytest.param(data.inner_op_data, Dense, CSR, Dense, complex),
    ]

    def generate_scalar_is_ket(self, metafunc):
        parameters = (
            ['op']
            + [x for x in metafunc.fixturenames
               if x.startswith("data_")]
            + ['out_type']
        )
        cases = []
        for p_op in self.specialisations:
            op, *types, out_type = p_op.values
            args = (op, types, [(self._scalar,) * 3], out_type)
            cases.extend(cases_type_shape_product(_RANDOM, *args))
        metafunc.parametrize(parameters, cases)
        metafunc.parametrize('scalar_is_ket',
                             [True, False], ids=["ket", "bra"])

    def test_scalar_is_ket(self, op, data_l, data_m, data_r, out_type,
                           scalar_is_ket):
        left, mid, right = data_l(), data_m(), data_r()
        expected = self.op_numpy(left.to_array(),
                                 mid.to_array(),
                                 right.to_array(),
                                 scalar_is_ket)
        test = op(left, mid, right, scalar_is_ket)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol,
                                       rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)


class TestKron(BinaryOpMixin):
    def op_numpy(self, left, right):
        return np.kron(left, right)

    # Keep the dimension low because kron can get very expensive.
    shapes = shapes_binary_unrestricted(dim=5)
    bad_shapes = shapes_binary_bad_unrestricted(dim=5)
    specialisations = [
        pytest.param(data.kron_csr, CSR, CSR, CSR),
        pytest.param(data.kron_dense, Dense, Dense, Dense),
        pytest.param(data.kron_dia, Dia, Dia, Dia),
        pytest.param(data.kron_dense_csr_csr, Dense, CSR, CSR),
        pytest.param(data.kron_csr_dense_csr, CSR, Dense, CSR),
        pytest.param(data.kron_dense_dia_dia, Dense, Dia, Dia),
        pytest.param(data.kron_dia_dense_dia, Dia, Dense, Dia),
    ]


class TestKronT(BinaryOpMixin):
    def op_numpy(self, left, right):
        return np.kron(left.T, right)

    # Keep the dimension low because kron can get very expensive.
    shapes = shapes_binary_unrestricted(dim=5)
    bad_shapes = shapes_binary_bad_unrestricted(dim=5)
    specialisations = [
        pytest.param(data.kron_transpose_data, CSR, CSR, Data),
        pytest.param(data.kron_transpose_dense, Dense, Dense, Dense),
    ]


class TestMatmul(ScaledBinaryOpMixin):
    def op_numpy(self, left, right, scale=1):
        return scale * np.matmul(left, right)

    shapes = shapes_binary_matmul()
    bad_shapes = shapes_binary_bad_matmul()
    specialisations = [
        pytest.param(data.matmul_csr, CSR, CSR, CSR),
        pytest.param(data.matmul_csr_dense_dense, CSR, Dense, Dense),
        pytest.param(data.matmul_dense, Dense, Dense, Dense),
        pytest.param(data.matmul_dia, Dia, Dia, Dia),
        pytest.param(data.matmul_dia_dense_dense, Dia, Dense, Dense),
        pytest.param(data.matmul_dense_dia_dense, Dense, Dia, Dense),
    ]


class TestMatmulDag(ScaledBinaryOpMixin):
    def op_numpy(self, left, right, scale=1):
        return scale * np.matmul(left, right)

    shapes = shapes_binary_matmul()
    bad_shapes = shapes_binary_bad_matmul()

    # Wrapper functions that apply adjoint to right operand
    @staticmethod
    def matmul_dag_data(left, right, scale=1):
        return data.matmul_dag_data(left, right.adjoint(), scale)

    @staticmethod
    def matmul_dag_dense_csr_dense(left, right, scale=1):
        return data.matmul_dag_dense_csr_dense(left, right.adjoint(), scale)

    @staticmethod
    def matmul_dag_dense_dia_dense(left, right, scale=1):
        return data.matmul_dag_dense_dia_dense(left, right.adjoint(), scale)

    @staticmethod
    def matmul_dag_dense(left, right, scale=1):
        return data.matmul_dag_dense(left, right.adjoint(), scale)

    specialisations = [
        pytest.param(matmul_dag_data, CSR, CSR, CSR),
        pytest.param(matmul_dag_dense_csr_dense, Dense, CSR, Dense),
        pytest.param(matmul_dag_dense_dia_dense, Dense, Dia, Dense),
        pytest.param(matmul_dag_dense, Dense, Dense, Dense),
    ]


class InPlaceMatmulMixin(mixin._GenericOpMixin):
    """
    Mix-in for in-place matmul operations: op(left, right, scale, out) -> out.
    Treats the operation as ternary with (left, right, out) as the three data
    arguments, where out is a pre-allocated Dense buffer.
    """

    @pytest.mark.parametrize('scale', [1, 0.5, 0.5j],
                             ids=['scale[1]', 'scale[real]', 'scale[complex]'])
    @pytest.mark.parametrize('out_order', ['C', 'F'], ids=['out[C]', 'out[F]'])
    def test_mathematically_correct(self, op, data_l, data_r, data_out,
                                    out_type, scale, out_order):
        """
        Test in-place matmul accumulates correctly into pre-allocated buffer.
        """
        left, right = data_l(), data_r()
        out_shape = data_out().shape
        expected_product = scale * np.matmul(left.to_array(), right.to_array())

        # Create output buffer with non-zero initial values
        initial = np.ones(out_shape, dtype=complex)
        if out_order == 'F':
            initial = np.asfortranarray(initial)
        out = Dense(initial.copy(), copy=False)
        expected = initial + expected_product

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OrderEfficiencyWarning)
            result = op(left, right, scale, out)

        assert result is out
        assert isinstance(result, out_type)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result.to_array(), expected,
                                   atol=self.atol, rtol=self.rtol)


class TestMatmulInPlace(InPlaceMatmulMixin):
    """Test in-place matmul operations that support out parameter."""

    shapes = shapes_ternary_matmul_inplace()
    bad_shapes = []
    specialisations = [
        pytest.param(data.matmul_csr_dense_dense, CSR, Dense, Dense, Dense),
        pytest.param(data.matmul_dense, Dense, Dense, Dense, Dense),
        pytest.param(data.matmul_dia_dense_dense, Dia, Dense, Dense, Dense),
        pytest.param(data.matmul_dense_dia_dense, Dense, Dia, Dense, Dense),
    ]


class TestMatmulDagInPlace(InPlaceMatmulMixin):
    """Test in-place matmul_dag operations that support out parameter."""

    shapes = shapes_ternary_matmul_inplace()
    bad_shapes = []

    @staticmethod
    def matmul_dag_dense_csr_dense(left, right, scale=1, out=None):
        return data.matmul_dag_dense_csr_dense(left, right.adjoint(), scale,
                                               out)

    @staticmethod
    def matmul_dag_dense_dia_dense(left, right, scale=1, out=None):
        return data.matmul_dag_dense_dia_dense(left, right.adjoint(), scale,
                                               out)

    @staticmethod
    def matmul_dag_dense(left, right, scale=1, out=None):
        return data.matmul_dag_dense(left, right.adjoint(), scale, out)

    specialisations = [
        pytest.param(matmul_dag_dense_csr_dense, Dense, CSR, Dense, Dense),
        pytest.param(matmul_dag_dense_dia_dense, Dense, Dia, Dense, Dense),
        pytest.param(matmul_dag_dense, Dense, Dense, Dense, Dense),
    ]


class TestMultiply(BinaryOpMixin):
    def op_numpy(self, left, right):
        return left * right

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()
    specialisations = [
        pytest.param(data.multiply_csr, CSR, CSR, CSR),
        pytest.param(data.multiply_dense, Dense, Dense, Dense),
        pytest.param(data.multiply_dia, Dia, Dia, Dia),
    ]


class TestMatmul_Outer(BinaryOpMixin):
    def op_numpy(self, left, right):
        return np.matmul(left, right)

    shapes = shapes_binary_matmul()
    bad_shapes = shapes_binary_bad_matmul()
    from qutip.core.data.matmul import (
        matmul_outer_csr_dense_sparse,
        matmul_outer_dia_dense_sparse,
        matmul_outer_dense_Data,
    )
    specialisations = [
        pytest.param(matmul_outer_csr_dense_sparse, CSR, Dense, CSR),
        pytest.param(matmul_outer_csr_dense_sparse, Dense, CSR, CSR),
        pytest.param(matmul_outer_dia_dense_sparse, Dia, Dense, Dia),
        pytest.param(matmul_outer_dia_dense_sparse, Dense, Dia, Dia),
        pytest.param(matmul_outer_dense_Data, Dense, Dense, Data),
    ]


class TestMul(UnaryScalarOpMixin):
    def op_numpy(self, matrix, scalar):
        return scalar * matrix

    specialisations = [
        pytest.param(data.mul_csr, CSR, CSR),
        pytest.param(data.mul_dense, Dense, Dense),
        pytest.param(data.mul_dia, Dia, Dia),
    ]


class TestNeg(UnaryOpMixin):
    def op_numpy(self, matrix):
        return -matrix

    specialisations = [
        pytest.param(data.neg_csr, CSR, CSR),
        pytest.param(data.neg_dense, Dense, Dense),
        pytest.param(data.neg_dia, Dia, Dia),
    ]


class TestSub(BinaryOpMixin):
    def op_numpy(self, left, right):
        return left - right

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()
    specialisations = [
        pytest.param(data.sub_csr, CSR, CSR, CSR),
        pytest.param(data.sub_dense, Dense, Dense, Dense),
        pytest.param(data.sub_dia, Dia, Dia, Dia),
    ]


class TestTrace(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.sum(np.diag(matrix))

    shapes = shapes_square()
    bad_shapes = shapes_not_square()
    specialisations = [
        pytest.param(data.trace_csr, CSR, complex),
        pytest.param(data.trace_dense, Dense, complex),
        pytest.param(data.trace_dia, Dia, complex),
    ]


class TestTrace_oper_ket(UnaryOpMixin):
    def op_numpy(self, matrix):
        N = int(matrix.shape[0] ** 0.5)
        return np.sum(np.diag(matrix.reshape((N, N))))

    shapes = [
        (pytest.param((100, 1), id="oper-ket"),),
    ]
    bad_shapes = [
        (pytest.param((1, 100), id="bra"),),
        (pytest.param((99, 1), id="ket"),),
        (pytest.param((99, 99), id="ket"),),
        (pytest.param((2, 99), id="nonsquare"),),
    ]
    specialisations = [
        pytest.param(data.trace_oper_ket_csr, CSR, complex),
        pytest.param(data.trace_oper_ket_dense, Dense, complex),
        pytest.param(data.trace_oper_ket_dia, Dia, complex),
        pytest.param(data.trace_oper_ket_data, CSR, complex),
        pytest.param(data.trace_oper_ket_data, Dense, complex),
    ]


class TestPow(UnaryOpMixin):
    def op_numpy(self, matrix, n):
        return np.linalg.matrix_power(matrix, n)

    shapes = shapes_square()
    bad_shapes = shapes_not_square()
    specialisations = [
        pytest.param(data.pow_csr, CSR, CSR),
        pytest.param(data.pow_dense, Dense, Dense),
        pytest.param(data.pow_dia, Dia, Dia),
    ]

    @pytest.mark.parametrize("n", [0, 1, 10], ids=["n_0", "n_1", "n_10"])
    def test_mathematically_correct(self, op, data_m, out_type, n):
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), n)
        test = op(matrix, n)
        assert isinstance(test, out_type)
        assert test.shape == expected.shape
        np.testing.assert_allclose(test.to_array(), expected, atol=self.atol,
                                   rtol=self.rtol)

    # Pow actually does have bad shape, so we put that in too.
    def test_incorrect_shape_raises(self, op, data_m):
        """
        Test that the operation produces a suitable error if the shape is not a
        square matrix.
        """
        with pytest.raises(ValueError):
            op(data_m(), 10)


# Scipy complain went creating full dia matrix.
@pytest.mark.filterwarnings("ignore:Constructing a DIA matrix")
class TestExpm(UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.expm(matrix)

    shapes = shapes_square()
    bad_shapes = shapes_not_square()
    specialisations = [
        pytest.param(data.expm_csr, CSR, CSR),
        pytest.param(data.expm_csr_dense, CSR, Dense),
        pytest.param(data.expm_dense, Dense, Dense),
        pytest.param(data.expm_dia, Dia, Dia),
    ]


class TestLogm(UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.logm(matrix)

    shapes = shapes_square()
    bad_shapes = shapes_not_square()
    specialisations = [
        pytest.param(data.logm_dense, Dense, Dense),
    ]


class TestSqrtm(UnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.sqrtm(matrix)

    shapes = shapes_square()
    bad_shapes = shapes_not_square()
    specialisations = [
        pytest.param(data.sqrtm_dense, Dense, Dense),
    ]


class TestTranspose(UnaryOpMixin):
    def op_numpy(self, matrix):
        return matrix.T

    specialisations = [
        pytest.param(data.transpose_csr, CSR, CSR),
        pytest.param(data.transpose_dense, Dense, Dense),
        pytest.param(data.transpose_dia, Dia, Dia),
    ]


class TestProject(UnaryOpMixin):
    def op_numpy(self, matrix):
        if matrix.shape[0] == 1:
            return np.outer(np.conj(matrix), matrix)
        else:
            return np.outer(matrix, np.conj(matrix))

    shapes = [
        (pytest.param((1, 1), id="scalar"),),
        (pytest.param((1, 100), id="bra"),),
        (pytest.param((100, 1), id="ket"),),
    ]
    bad_shapes = [
        (pytest.param((10, 10), id="square"),),
        (pytest.param((2, 10), id="nonsquare"),),
    ]

    specialisations = [
        pytest.param(data.project_csr, CSR, CSR),
        pytest.param(data.project_dia, Dia, Dia),
        pytest.param(data.project_dense_data, Dense, Data),
    ]


def _inv_dense(matrix):
    # Add a diagonal so `matrix` is not singular
    diag = data.diag([2.] * matrix.shape[0], shape=matrix.shape, dtype='dense')
    return data.inv_dense(data.to(Dense, data.add(matrix, diag)))


def _inv_csr(matrix):
    # Add a diagonal so `matrix` is not singular
    diag = data.diag([2.] * matrix.shape[0], shape=matrix.shape, dtype='csr')
    return data.inv_csr(data.to(CSR, data.add(matrix, diag)))


class TestInv(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.linalg.inv(matrix + np.eye(matrix.shape[0]) * 2.)

    shapes = [
        (pytest.param((1, 1), id="scalar"),),
        (pytest.param((10, 10), id="square"),),
    ]
    bad_shapes = [
        (pytest.param((2, 10), id="nonsquare"),),
        (pytest.param((1, 100), id="bra"),),
        (pytest.param((100, 1), id="ket"),),
    ]

    specialisations = [
        pytest.param(_inv_csr, CSR, CSR),
        pytest.param(_inv_dense, Dense, Dense),
    ]


class TestZeros_like(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.zeros_like(matrix)

    specialisations = [
        pytest.param(data.zeros_like_data, CSR, CSR),
        pytest.param(data.zeros_like_dense, Dense, Dense),
    ]


class TestIdentity_like(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.eye(matrix.shape[0])

    shapes = shapes_square()
    bad_shapes = shapes_not_square()

    specialisations = [
        pytest.param(data.identity_like_data, CSR, CSR),
        pytest.param(data.identity_like_dense, Dense, Dense),
    ]


class TestWRMN_error(BinaryOpMixin):
    def op_numpy(self, left, right, atol, rtol):
        return np.linalg.norm(
            np.abs(left)
            / (atol + rtol * np.abs(right))
        ) / left.size**0.5

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()
    specialisations = [
        pytest.param(data.ode.wrmn_error_csr, CSR, CSR, float),
        pytest.param(data.ode.wrmn_error_dense, Dense, Dense, float),
        pytest.param(data.ode.wrmn_error_dia, Dia, Dia, float),
    ]

    # `wrmn_error` has additional scalar parameters: the tolerances.
    @pytest.mark.parametrize('atol', [1e-7, 0.5],
                             ids=['atol[small]', 'atol[large]'])
    @pytest.mark.parametrize('rtol', [0, 1e-10, 0.5],
                             ids=['rtol[0]', 'rtol[small]', 'rtol[large]'])
    def test_mathematically_correct(self, op, data_l, data_r, out_type,
                                    atol, rtol):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations.
        """
        left, right = data_l(), data_r()
        expected = self.op_numpy(left.to_array(), right.to_array(), atol, rtol)
        test = op(left, right, atol, rtol)

        assert isinstance(test, out_type)
        np.testing.assert_allclose(
            test, expected, atol=self.atol, rtol=self.rtol
        )

    def test_incorrect_shape_raises(self, op, data_l, data_r):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        with pytest.raises(ValueError):
            op(data_l(), data_r(), 1e-5, 1e-5)
