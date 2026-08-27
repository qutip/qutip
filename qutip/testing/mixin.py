import itertools
import numpy as np
import pytest
import scipy
import warnings


# The ParameterSet is actually a pretty hidden type, so it's easiest to access
# it like this.
_ParameterSet = type(pytest.param())

# First set up a bunch of allowable shapes, for different types of functions so
# we don't have to respecify a whole lot of things on repeat.

DIM = 100


def shapes_unary(dim=DIM):
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


def shapes_binary_identical(dim=DIM):
    """
    Allowed shapes for binary operators that need the two matrices to be the
    same shape, e.g. addition.
    """
    return [(x, x) for x in shapes_unary(dim)]


def shapes_binary_bad_identical(dim=DIM):
    """
    Disallowed shapes for binary operators that need the two matrices to be the
    same shape, e.g. addition.
    """
    return [
        (x, y)
        for x, y in itertools.product(shapes_unary(dim), repeat=2)
        if x.values[0] != y.values[0]
    ]


def shapes_binary_unrestricted(dim=DIM):
    """
    Allowed shapes for binary operators which can take any shapes, e.g. the
    Kronecker product.
    """
    return list(itertools.product(shapes_unary(dim), repeat=2))


def shapes_binary_bad_unrestricted(dim=DIM):
    """
    Disallowed shapes for binary operators which can take any shapes, e.g. the
    Kronecker product.  There aren't actually any of these, but we keep it just
    for consistency.
    """
    return []


def shapes_binary_matmul(dim=DIM):
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


def shapes_binary_bad_matmul(dim=DIM):
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


def shapes_ternary_matmul_inplace(dim=DIM):
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


def shapes_square(dim=DIM):
    """Allowed shapes for operations that require square matrices. Examples of
    these operations are trace, pow, expm and the trace norm."""
    return [
        (pytest.param((1, 1), id="1"),),
        (pytest.param((dim, dim), id=str(dim)),),
    ]


def shapes_not_square(dim=DIM):
    """Disallowed shapes for operations that require square matrices. Examples
    of these operations are trace, pow, expm and the trace norm."""
    return [
        (x,) for x in shapes_unary(dim) if x.values[0][0] != x.values[0][1]
    ]


# Data generators for the test
# For a type, map to a function of the shape that return a list of cases for
# that type.
# Case are functions taking taking an optional RNG
# {
#    CRS: lambda shape: [
#            lambda _: csr.zeros(*shape),
#            lambda gen: random_csr(*shape, gen)
#         ],
# }

# Cases for mathematically correct
CORRECT_CASES = {}

# Cases for wrong shape
WRONG_CASES = {}


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
    cases_lookup: Map[Type, Callable[(shape: 2-tuple), list]]
        Mapping to get the special case generator from.  The elements of the
        list returned from the case generator should all be closures of the
        form `Callable[(), Data]` (e.g. `lambda: data.csr.identity(10)`), or a
        `pytest.ParameterSet` containing exactly one value, which is the same
        closure type.

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
            op_name = getattr(op, '__name__', str(op))
            id_ = op_name + ":" + ",".join(m.id for m in matrices)
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
            cases.extend(cases_type_shape_product(CORRECT_CASES, *args))

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
            cases.extend(cases_type_shape_product(WRONG_CASES, *args))
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

    def test_mathematically_correct(
        self, op, data_m, out_type, random_generator
    ):
        matrix = data_m(random_generator)
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

    def test_incorrect_shape_raises(self, op, data_m, random_generator):
        """
        Test that the operation produces a suitable error if the shape of the
        given operand is not compatible with the operation. Useful for
        operations that require square matrices (trace, pow, ...).
        """
        with pytest.raises(ValueError):
            op(data_m(random_generator))


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
    def test_mathematically_correct(
        self, op, data_m, scalar, out_type, random_generator
    ):
        matrix = data_m(random_generator)
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
    def test_mathematically_correct(
        self, op, data_l, data_r, out_type, random_generator
    ):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations.
        """
        left, right = data_l(random_generator), data_r(random_generator)
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

    def test_incorrect_shape_raises(
        self, op, data_l, data_r, random_generator
    ):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        with pytest.raises(ValueError):
            op(data_l(random_generator), data_r(random_generator))


class ScaledBinaryOpMixin(BinaryOpMixin):
    """
    Mix-in for binary mathematical operations that support a scale parameter.

    Subclasses should define op_numpy(left, right, scale=1) that returns
    the expected result with scaling applied.
    """

    @pytest.mark.parametrize('scale', [None, 0.2, 0.5j],
                             ids=['unscaled', 'scale[real]',
                                  'scale[complex]'])
    def test_mathematically_correct(
        self, op, data_l, data_r, out_type, scale, random_generator
    ):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations, including with scaling.
        """
        left, right = data_l(random_generator), data_r(random_generator)
        if scale is not None:
            expected = self.op_numpy(left.to_array(), right.to_array(), scale)
            test = op(left, right, scale)
        else:
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


class TernaryOpMixin(_GenericOpMixin):
    """
    Mix-in for ternary mathematical operations on Data instances (e.g. inner
    product with an operator in the middle).  This is pretty rare.
    """
    def test_mathematically_correct(self, op,
                                    data_l, data_m, data_r,
                                    out_type, random_generator):
        """
        Test that the ternary operation is mathematically correct for all the
        known type specialisations.
        """
        rng = random_generator
        left, mid, right = data_l(rng), data_m(rng), data_r(rng)
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

    def test_incorrect_shape_raises(
        self, op, data_l, data_m, data_r, random_generator
    ):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        rng = random_generator
        with pytest.raises(ValueError):
            op(data_l(rng), data_m(rng), data_r(rng))
