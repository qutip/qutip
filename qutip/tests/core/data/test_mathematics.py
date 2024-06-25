import itertools
import numpy as np
import scipy
import pytest

from qutip.core import data
from qutip.core.data import Data, Dense, CSR, Dia

from . import conftest

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
        return lambda: conftest.random_csr(shape, density, sort)

    def zero_factory():
        return lambda: data.csr.zeros(shape[0], shape[1])
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
        return lambda: conftest.random_dense(shape, fortran)
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
        return lambda: conftest.random_diag(shape, density, sort)

    def zero_factory():
        return lambda: data.dia.zeros(shape[0], shape[1])

    return [
        pytest.param(factory(0.001), id="sparse"),
        pytest.param(factory(0.8, True), id="filled,sorted"),
        pytest.param(factory(0.8, False), id="filled,unsorted"),
        pytest.param(zero_factory(), id="zero"),
    ]


# Factory methods for generating the cases, mapping type to the function.
# _ALL_CASES is for getting all the special cases to test, _RANDOM is for
# getting just a single case from each.
_ALL_CASES = {
    CSR: cases_csr,
    Dia: cases_diag,
    Dense: cases_dense,
}
_RANDOM = {
    CSR: lambda shape: [lambda: conftest.random_csr(shape, 0.5, True)],
    Dense: lambda shape: [lambda: conftest.random_dense(shape, False)],
    Dia: lambda shape: [lambda: conftest.random_diag(shape, 0.5)],
}


def cases_type_shape_product(cases_lookup, op, types, shapes, out_type=None):
    """
    Return a list of `pytest.ParameterSet` which is a flat list of all the
    special cases that should be tested for this operator specialisation `op`,
    which takes in several types `types`, where the arguments have shapes taken
    from the iterable of iterables `shapes`.  If `out_type` is specified, it
    will be added into the output parameter set and its name added to the id,
    but is otherwise not used.

    Parameters
    ----------
    cases_lookup: Map[Type, (shape: 2-tuple) -> list]
        Mapping to get the special case generator from.  This is one of
        _ALL_CASES or _RANDOM (or similar).  The elements of the list returned
        from the case generator should all be closures of the form `() -> Data`
        (e.g. `lambda: data.csr.identity(10)`), or a `pytest.ParameterSet`
        containing exactly one value, which is the same closure type.

    op: Function
        The specialisation of a mathematical operation that is being tested.
        This is actually generally unused - it's just added in to the output
        ParameterSet at the end so that it will get correctly parametrised.

    types: Tuple[Type]
        A tuple of data.Data subclass types (not instances).  This defines the
        inputs to this particular specialisation under test.  There should be
        as many types as there are Data arguments to `op`.

    shapes: Iterable[Tuple[(int, int) | ParameterSet[(int, int)]]]
        An iterable containing several sets of shapes to parameterise over.
        Each element of the iterable should be s tuple of shapes, or
        `pytest.ParameterSet` instances each containing a single shape.  A
        shape is a 2-tuple of integers.  There should be as many elements of
        this inner tuple as there are `types`, since each individual `shape`
        goes with a `type`.

    out_type: Type
        The output type for this specialisation.  Not used other than its name
        being added to the id, and it being added at the end of the
        specialisation (if it is given), similarly to `op`.

    Returns
    -------
    List[ParameterSet]
        A list of individual test cases for parametrisation.  Each ParameterSet
        will be
            [op, *data_inputs, ?out_type]
        where `data_inputs` is of the same length as the input parameter
        `types` and represents the Data arguments to the specialisation `op`.
        Each element of `data_inputs` is a generator function which takes no
        arguments and returns a data.Data subclass of the correct type and
        shape.  `out_type` is present in the output only if it were given as a
        parameter itself.
    """
    def case(type_, shape_case, generator_case):
        """
        Build a case parameter for _one_ generator function which will return
        a given type and shape.
        """
        id_ = type_.__name__
        inner = ""
        for extra in [shape_case, generator_case]:
            if hasattr(extra, 'id') and extra.id:
                inner += ("," if inner else "") + extra.id
        if inner:
            id_ += "[" + inner + "]"
        func = (
            generator_case.values[0]
            if isinstance(generator_case, _ParameterSet)
            else generator_case
        )
        return pytest.param(func, id=id_)

    cases = []
    for shapes_ in shapes:
        # Convert the list of types into a list of lists of the special cases
        # needed for each type.
        matrix_cases = [
            [case(type_, shape_case, type_case)
             for type_case in cases_lookup[type_](shape_case.values[0])]
            for type_, shape_case in zip(types, shapes_)
        ]
        # Now Cartesian product all the special cases together to make the full
        # set of parametrisations.
        for matrices in itertools.product(*matrix_cases):
            id_ = ",".join(m.id for m in matrices)
            args = [m for p_m in matrices for m in p_m.values]
            if out_type is not None:
                id_ += "->" + out_type.__name__
                args += [out_type]
            cases.append(pytest.param(op, *args, id=id_))
    return cases


# Now we start to actually build up all the test cases.  Since all the
# mathematical operations are really pretty similar, and most just need the
# basic testing functionality, we do this with a series of mix-in classes which
# provide various components of the testing and test-generation infrastructure.
#
# In each, we use the idiom that "test_<x>" is a test function which pytest
# will collect for us, and "generate_<x>" a method which will be called by
# `pytest_generate_tests` in order to generate all the parametrisations for the
# given test.

class _GenericOpMixin:
    """
    Abstract base mix-in which sets up the test generation for the two basic
    test operations, and puts in the very generic test generator.  This does
    not actually define the tests themselves, because subclasses need to define
    them so the method arguments can be introspected to parametrise over the
    correct number of arguments.

    The tests `mathematically_correct` and `incorrect_shape_raises` will
    parametrise Data types over method arguments which have names `data_*`.

    The class arguments defined here are effectively parametrising the
    mathematical tests.

    Attributes
    ----------
    op_numpy: *args -> np.ndarray
        Function which takes the same arguments as the mathematical operation,
        but with all data arguments numpy arrays, and returns the expected
        result.

    atol: float
        The absolute tolerance to use when comparing the test value with the
        expected value.  If the output is a Data type, the tolerance is
        per-element of the output.

    rtol: float
        The relative tolerance to use when comparing the test value with the
        expected value.  If the output is a Data type, the tolerance is
        per-element of the output.

    shapes: list of (list of shapes)
        A list of the sets of shapes which should be used for the tests of
        mathematical correctness.  Each element of the list is a set of shapes,
        each one corresponding to one of the arguments of the operation.

    bad_shapes: list of (list of shapes)
        Similar to `shapes`, but these should be shapes which are invalid for
        the given mathematical operation.

    specialisations: list of (function, Type, Type, [Type, ...])
        The specialisations of each mathematical function, and the types that
        it takes in and returns.  For example, the function
            add(CSR, Dense) -> Other
        would be specified as `(add, CSR, Dense, Other)`.
    """
    def op_numpy(self, *args): raise NotImplementedError
    # With dimensions of around 100, we have to account for floating-point
    # addition not being associative; the maths on full numpy arrays will often
    # produce slightly different results to sparse algebra, since the order of
    # multiplications and additions will be different.
    atol = 1e-10
    rtol = 1e-7  # Same default as numpy
    shapes = []
    bad_shapes = []
    specialisations = []

    def generate_mathematically_correct(self, metafunc):
        parameters = (
            ['op']
            + [x for x in metafunc.fixturenames
               if x.startswith("data_")]
            + ['out_type']
        )
        cases = []
        for p_op in self.specialisations:
            op, *types, out_type = p_op.values
            args = (op, types, self.shapes, out_type)
            cases.extend(cases_type_shape_product(_ALL_CASES, *args))
        metafunc.parametrize(parameters, cases)

    def generate_incorrect_shape_raises(self, metafunc):
        parameters = (
            ['op']
            + [x for x in metafunc.fixturenames
               if x.startswith("data_")]
        )
        if not self.bad_shapes:
            reason = "".join([
                "no shapes are 'incorrect' for ",
                metafunc.cls.__name__,
                "::",
                metafunc.function.__name__,
            ])
            false_case = pytest.param(*([None]*len(parameters)),
                                      marks=pytest.mark.skip(reason),
                                      id="no test")
            metafunc.parametrize(parameters, [false_case])
            return
        cases = []
        for p_op in self.specialisations:
            op, *types, _ = p_op.values
            args = (op, types, self.bad_shapes)
            cases.extend(cases_type_shape_product(_RANDOM, *args))
        metafunc.parametrize(parameters, cases)

    def pytest_generate_tests(self, metafunc):
        # For every test function "test_xyz", we use the test generator
        # "generate_xyz" if it exists.  This allows derived classes to add
        # their own tests and generators without overiding this method, cutting
        # down on boilerplate, but also that derived classes _may_ override the
        # generation of tests defined in a base class, say if they have
        # additional special arguments that need parametrising over.
        generator_name = (
            "generate_"
            + metafunc.function.__name__.replace("test_", "")
        )
        try:
            generator = getattr(self, generator_name)
        except AttributeError:
            return
        generator(metafunc)


class UnaryOpMixin(_GenericOpMixin):
    """
    Mix-in for unary mathematical operations on Data instances (e.g. unary
    negation).
    """
    shapes = [(x,) for x in shapes_unary()]
    bad_shapes = []

    def test_mathematically_correct(self, op, data_m, out_type):
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array())
        test = op(matrix)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        elif out_type is list:
            for test_, expected_ in zip(test, expected):
                assert test_.shape == expected_.shape
                np.testing.assert_allclose(test_.to_array(), expected_,
                                           atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)

    def test_incorrect_shape_raises(self, op, data_m):
        """
        Test that the operation produces a suitable error if the shape of the
        given operand is not compatible with the operation. Useful for
        operations that require square matrices (trace, pow, ...).
        """
        with pytest.raises(ValueError):
            op(data_m())


class UnaryScalarOpMixin(_GenericOpMixin):
    """
    Mix-in for unary mathematical operations on Data instances, but that also
    take in a numeric scalar (e.g. scalar multiplication).  Only generates
    the test `mathematically_correct`, since there can't be a shape mismatch
    when there's only one Data argument.
    """
    shapes = [(x,) for x in shapes_unary()]

    @pytest.mark.parametrize('scalar', [
        pytest.param(0, id='zero'),
        pytest.param(4.5, id='real'),
        pytest.param(3j, id='complex'),
    ])
    def test_mathematically_correct(self, op, data_m, scalar, out_type):
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), scalar)
        test = op(matrix, scalar)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol,
                                       rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)


class BinaryOpMixin(_GenericOpMixin):
    """
    Mix-in for binary mathematical operations on Data instances (e.g. binary
    addition).
    """
    def test_mathematically_correct(self, op, data_l, data_r, out_type):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations.
        """
        left, right = data_l(), data_r()
        expected = self.op_numpy(left.to_array(), right.to_array())
        test = op(left, right)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)

    def test_incorrect_shape_raises(self, op, data_l, data_r):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        with pytest.raises(ValueError):
            op(data_l(), data_r())


class TernaryOpMixin(_GenericOpMixin):
    """
    Mix-in for ternary mathematical operations on Data instances (e.g. inner
    product with an operator in the middle).  This is pretty rare.
    """
    def test_mathematically_correct(self, op,
                                    data_l, data_m, data_r,
                                    out_type):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations.
        """
        left, mid, right = data_l(), data_m(), data_r()
        expected = self.op_numpy(left.to_array(),
                                 mid.to_array(),
                                 right.to_array())
        test = op(left, mid, right)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)

    def test_incorrect_shape_raises(self, op, data_l, data_m, data_r):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        with pytest.raises(ValueError):
            op(data_l(), data_m(), data_r())


# And now finally we get into the meat of the actual mathematical tests.

class TestAdd(BinaryOpMixin):
    def op_numpy(self, left, right, scale):
        return np.add(left, scale * right)

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()
    specialisations = [
        pytest.param(data.add_csr, CSR, CSR, CSR),
        pytest.param(data.add_dense, Dense, Dense, Dense),
        pytest.param(data.add_dia, Dia, Dia, Dia),
    ]

    # `add` has an additional scalar parameter, because the operation is
    # actually more like `A + c*B`.  We just parametrise that scalar
    # separately.
    @pytest.mark.parametrize('scale', [None, 0.2, 0.5j],
                             ids=['unscaled', 'scale[real]', 'scale[complex]'])
    def test_mathematically_correct(self, op, data_l, data_r, out_type, scale):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations.
        """
        left, right = data_l(), data_r()
        if scale is not None:
            expected = self.op_numpy(left.to_array(), right.to_array(), scale)
            test = op(left, right, scale)
        else:
            expected = self.op_numpy(left.to_array(), right.to_array(), 1)
            test = op(left, right)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)


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
    ]


class TestKronT(BinaryOpMixin):
    def op_numpy(self, left, right):
        return np.kron(left.T, right)

    # Keep the dimension low because kron can get very expensive.
    shapes = shapes_binary_unrestricted(dim=5)
    bad_shapes = shapes_binary_bad_unrestricted(dim=5)
    specialisations = [
        pytest.param(data.kron_transpose_data, CSR, CSR, CSR),
        pytest.param(data.kron_transpose_dense, Dense, Dense, Dense),
    ]


class TestMatmul(BinaryOpMixin):
    def op_numpy(self, left, right):
        return np.matmul(left, right)

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
        pytest.param(data.project_dense, Dense, Dense),
    ]


def _inv_dense(matrix):
    # Add a diagonal so `matrix` is not singular
    return data.inv_dense(
        data.add(
            matrix,
            data.diag([1.1]*matrix.shape[0], shape=matrix.shape, dtype='dense')
        )
    )


def _inv_csr(matrix):
    # Add a diagonal so `matrix` is not singular
    return data.inv_csr(
        data.add(
            matrix,
            data.diag([1.1]*matrix.shape[0], shape=matrix.shape, dtype='csr')
        )
    )


class TestInv(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.linalg.inv(matrix + np.eye(matrix.shape[0]) * 1.1)

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
