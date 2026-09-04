import itertools
import numpy as np
import pytest
import scipy
import warnings

from qutip.core.data import Data

# The ParameterSet is actually a pretty hidden type, so it's easiest to access
# it like this.
_ParameterSet = type(pytest.param())
NoParam = object()


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
# Case are functions taking taking an RNG
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


def cases_type_shape_product(
    cases_lookup,
    op,
    types,
    shapes,
    out_type=None
):
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


def build_extra_cases(extra_param):
    """
    Create the combination of all extra_param entrys.

    extra_param = {
        "scale": [0, 1],
        "alpha": [1, pytest.param(1j, id="complex")],
    }
    with create the 4 cases for each values of scale and alpha.
    If the entrise are pytest.param, id and mask will be kept.
    """
    cases = [({}, [], ())]
    for param_name, param_values in extra_param.items():
        new_cases = []
        for case, value in itertools.product(cases, param_values):
            if isinstance(value, _ParameterSet):
                if len(value.values) != 1:
                    raise ValueError
                id = value.id
                val = value.values[0]
                mark = value.marks
            else:
                id = None
                val = value
                mark = ()

            if val is NoParam:
                id = f"No {param_name}"
                val = {}
            else:
                id = id or f"{param_name}={val}"
                val = {param_name: val}

            new_cases.append((
                {**val, **case[0]},
                case[1] + [id],
                case[2] + mark
            ))
        cases = new_cases

    return [
        pytest.param(
            kwargs,
            id="-".join(ids),
            marks=marks
        )
        for kwargs, ids, marks in cases
    ]

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
    correct_cases = None
    wrong_cases = None
    extra_param = {}
    wrong_extra_param = {}

    def generate_mathematically_correct(self, metafunc):
        parameters = (
            ['op']
            + [x for x in metafunc.fixturenames
               if x.startswith("data_")]
            + ['out_type']
        )

        cases = []
        cases_map = self.correct_cases or CORRECT_CASES
        for p_op in self.specialisations:
            op, *types, out_type = p_op.values
            args = (op, types, self.shapes, out_type)
            cases.extend(cases_type_shape_product(cases_map, *args))

        metafunc.parametrize(parameters, cases)

        if self.extra_param:
            cases = build_extra_cases(self.extra_param)
            metafunc.parametrize(["kw"], cases)
        else:
            metafunc.parametrize(["kw"], [pytest.param({}, id="")])

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
        cases_map = self.wrong_cases or WRONG_CASES
        for p_op in self.specialisations:
            op, *types, _ = p_op.values
            args = (op, types, self.bad_shapes)
            cases.extend(cases_type_shape_product(cases_map, *args))
        metafunc.parametrize(parameters, cases)

        if self.wrong_extra_param:
            cases = build_extra_cases(self.wrong_extra_param)
            metafunc.parametrize(["kw"], cases)
        else:
            metafunc.parametrize(["kw"], [pytest.param({}, id="")])

    def generate_exception(self, metafunc):
        """
        Generate good shape and wrong_cases parametrization without extra
        keyword.
        To test exception from bad extra input.
        """
        parameters = (
            ['op']
            + [x for x in metafunc.fixturenames
               if x.startswith("data_")]
        )

        cases = []
        cases_map = self.wrong_cases or WRONG_CASES
        for p_op in self.specialisations:
            op, *types, out_type = p_op.values
            args = (op, types, self.shapes)
            cases.extend(cases_type_shape_product(cases_map, *args))

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

    def check_result(self, test, expected, out_type):
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)


class UnaryOpMixin(_GenericOpMixin):
    """
    Mix-in for unary mathematical operations on Data instances (e.g. unary
    negation).
    """
    shapes = [(x,) for x in shapes_unary()]
    bad_shapes = []

    def test_mathematically_correct(
        self, op, data_m, out_type, random_generator, kw,
    ):
        matrix = data_m(random_generator)
        expected = self.op_numpy(matrix.to_array(), **kw)
        test = op(matrix, **kw)

        self.check_result(test, expected, out_type)

    def test_incorrect_shape_raises(self, op, data_m, random_generator, kw):
        """
        Test that the operation produces a suitable error if the shape of the
        given operand is not compatible with the operation. Useful for
        operations that require square matrices (trace, pow, ...).
        """
        with pytest.raises(ValueError):
            op(data_m(random_generator), **kw)


class SquareUnaryOpMixin(UnaryOpMixin):
    shapes = shapes_square()
    bad_shapes = shapes_not_square()


class BinaryOpMixin(_GenericOpMixin):
    """
    Mix-in for binary mathematical operations on Data instances (e.g. binary
    addition).
    """
    def test_mathematically_correct(
        self, op, data_l, data_r, out_type, random_generator, kw,
    ):
        """
        Test that the binary operation is mathematically correct for all the
        known type specialisations.
        """
        left, right = data_l(random_generator), data_r(random_generator)
        expected = self.op_numpy(left.to_array(), right.to_array(), **kw)
        test = op(left, right, **kw)

        self.check_result(test, expected, out_type)

    def test_incorrect_shape_raises(
        self, op, data_l, data_r, random_generator, kw,
    ):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        with pytest.raises(ValueError):
            op(data_l(random_generator), data_r(random_generator), **kw)


class TernaryOpMixin(_GenericOpMixin):
    """
    Mix-in for ternary mathematical operations on Data instances (e.g. inner
    product with an operator in the middle).  This is pretty rare.
    """
    def test_mathematically_correct(
        self, op,
        data_l, data_m, data_r,
        out_type, random_generator,
        kw,
    ):
        """
        Test that the ternary operation is mathematically correct for all the
        known type specialisations.
        """
        rng = random_generator
        left, mid, right = data_l(rng), data_m(rng), data_r(rng)
        expected = self.op_numpy(
            left.to_array(),
            mid.to_array(),
            right.to_array(),
            **kw,
        )
        test = op(left, mid, right, **kw)

        self.check_result(test, expected, out_type)

    def test_incorrect_shape_raises(
        self, op, data_l, data_m, data_r, random_generator, kw,
    ):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        rng = random_generator
        with pytest.raises(ValueError):
            op(data_l(rng), data_m(rng), data_r(rng), **kw)


#=============================================================================#
#                           Unitary specialisation                            #
#=============================================================================#

class TestAdjoint(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.conj(matrix.T)


class TestConj(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.conj(matrix)


class TestTranspose(UnaryOpMixin):
    def op_numpy(self, matrix):
        return matrix.T


class TestNeg(UnaryOpMixin):
    def op_numpy(self, matrix):
        return -matrix


class TestZeros_like(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.zeros_like(matrix)


class TestIdentity_like(SquareUnaryOpMixin):
    def op_numpy(self, matrix):
        return np.eye(matrix.shape[0])


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


class TestSqrtm(SquareUnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.sqrtm(matrix)


class TestLogm(SquareUnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.logm(matrix)


@pytest.mark.filterwarnings("ignore:Constructing a DIA matrix")
class TestExpm(SquareUnaryOpMixin):
    def op_numpy(self, matrix):
        return scipy.linalg.expm(matrix)


class TestPow(SquareUnaryOpMixin):
    def op_numpy(self, matrix, n):
        return np.linalg.matrix_power(matrix, n)

    extra_param = {"n": [0, 1, 10]}
    wrong_extra_param = {"n": [10]}


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


class TestTrace(SquareUnaryOpMixin):
    def op_numpy(self, matrix):
        return np.sum(np.diag(matrix))


class TestMul(UnaryOpMixin):
    def op_numpy(self, matrix, value):
        return value * matrix

    extra_param = {"value": [
        pytest.param(0, id='zero'),
        pytest.param(4.5, id='real'),
        pytest.param(3j, id='complex'),
    ]}
    wrong_extra_param = {"value": [10]}


class TestInv(UnaryOpMixin):
    def op_numpy(self, matrix):
        return np.linalg.inv(matrix)

    shapes = [
        (pytest.param((1, 1), id="scalar"),),
        (pytest.param((10, 10), id="square"),),
    ]
    bad_shapes = [
        (pytest.param((2, 10), id="nonsquare"),),
        (pytest.param((1, 100), id="bra"),),
        (pytest.param((100, 1), id="ket"),),
    ]

    # Usual creator do not ensure that the array can be inverted
    # Specialized creator are needed.
    # NotImplementedError will raise an error if not overwritten
    correct_cases = NotImplementedError
    wrong_cases = NotImplementedError


class TestPtrace(UnaryOpMixin):
    def op_numpy(self, matrix, dims, sel):
        sel.sort()
        ndims = len(dims)
        dkeep = [dims[x] for x in sel]
        qtrace = list(set(range(ndims)) - set(sel))
        dtrace = [dims[x] for x in qtrace]

        matrix = matrix.reshape(dims + dims)
        matrix = matrix.transpose(
            qtrace + [ndims + i for i in qtrace] + sel + [ndims + i for i in sel]
        )
        matrix = matrix.reshape(
            [
                np.prod(dtrace, dtype=int),
                np.prod(dtrace, dtype=int),
                np.prod(dkeep, dtype=int),
                np.prod(dkeep, dtype=int),
            ]
        )
        return np.trace(matrix)

    # Custom shapes to have also custom dims and sel arguments.
    # These values should not be changed.
    dims = [2] * 7
    shapes = [(pytest.param((np.prod(dims), np.prod(dims))),)]
    bad_shapes = shapes_not_square(np.prod(dims))

    extra_param = {
        "dims": [dims],
        "sel": [
            pytest.param([0], id="keep_one"),
            pytest.param([0, 3, 6], id="keep_multiple_sorted"),
            pytest.param([0, 6, 3], id="keep_multiple_unsorted"),
            pytest.param(list(range(7)), id="trace_none"),
            pytest.param([], id="trace_all"),
        ]
    }
    wrong_extra_param = {"dims": [dims], "sel": [[0, 1]]}

    # `out_type` is included but not used so that
    # `generate_mathematically_correct` can be re-used.
    @pytest.mark.parametrize(
        "dims",
        [[2], [0], [-2, -2] + [2] * 5, [1.2, 2.2, 3.3]],
        ids=[
            "dims_different_to_shape",
            "dims_0",
            "dims_prod_is_shape_but_negative",
            "dims_is_not_int",
        ],
    )
    def test_incorrect_dims_raises(self, op, data_m, random_generator, dims):
        with pytest.raises(ValueError):
            op(data_m(random_generator), dims, sel=[0, 1])

    def generate_incorrect_dims_raises(self, metafunc):
        self.generate_exception(metafunc)

    @pytest.mark.parametrize(
        "sel",
        [[2, 10], [-1, 2]],
        ids=[
            "sel_value_larger_than_dims",
            "sel_value_negative",
        ],
    )
    def test_incorrect_sel_raises(self, op, data_m, random_generator, sel):
        with pytest.raises(IndexError):
            op(data_m(random_generator), dims=self.dims, sel=sel)

    def generate_incorrect_sel_raises(self, metafunc):
        self.generate_exception(metafunc)


#=============================================================================#
#                            Binary specialisation                            #
#=============================================================================#

class TestAdd(BinaryOpMixin):
    def op_numpy(self, left, right, scale=1):
        return np.add(left, scale * right)

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()

    extra_param = {"scale": [
        pytest.param(0.2, id="scale[real]"),
        pytest.param(1.2 + 0.5j, id="scale[complex]"),
        pytest.param(NoParam, id="unscaled"),
    ]}


class TestSub(BinaryOpMixin):
    def op_numpy(self, left, right):
        return left - right

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()


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
        cases_map = self.correct_cases or CORRECT_CASES
        for p_op in self.specialisations:
            op, *types, out_type = p_op.values
            args = (op, types, [(self._scalar, self._scalar)], out_type)
            cases.extend(cases_type_shape_product(cases_map, *args))
        metafunc.parametrize(parameters, cases)
        metafunc.parametrize('scalar_is_ket',
                             [True, False],
                             ids=["ket", "bra"])

    def test_scalar_is_ket(
        self, op, data_l, data_r, out_type, scalar_is_ket, random_generator
    ):
        left, right = data_l(random_generator), data_r(random_generator)
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


class TestKron(BinaryOpMixin):
    def op_numpy(self, left, right):
        return np.kron(left, right)

    # Keep the dimension low because kron can get very expensive.
    shapes = shapes_binary_unrestricted(dim=5)
    bad_shapes = shapes_binary_bad_unrestricted(dim=5)


class TestKronT(BinaryOpMixin):
    def op_numpy(self, left, right):
        return np.kron(left.T, right)

    # Keep the dimension low because kron can get very expensive.
    shapes = shapes_binary_unrestricted(dim=5)
    bad_shapes = shapes_binary_bad_unrestricted(dim=5)


class TestMatmul(BinaryOpMixin):
    def op_numpy(self, left, right, scale=1):
        return scale * np.matmul(left, right)

    shapes = shapes_binary_matmul()
    bad_shapes = shapes_binary_bad_matmul()

    extra_param = {"scale": [
        pytest.param(NoParam),
        pytest.param(0.2, id="scale[real]"),
        pytest.param(0.5j, id="scale[complex]"),
    ]}


class TestMatmulDag(BinaryOpMixin):
    def op_numpy(self, left, right, scale=1):
        return scale * np.matmul(left, right.T.conj())

    shapes = [
        (x, y)
        for x, y in itertools.product(shapes_unary(50), repeat=2)
        if x.values[0][1] == y.values[0][1]
    ]
    bad_shapes = [
        (x, y)
        for x, y in itertools.product(shapes_unary(50), repeat=2)
        if x.values[0][1] != y.values[0][1]
    ]

    extra_param = {"scale": [
        pytest.param(NoParam),
        pytest.param(0.2, id="scale[real]"),
        pytest.param(0.5j, id="scale[complex]"),
    ]}


class TestMultiply(BinaryOpMixin):
    def op_numpy(self, left, right):
        return left * right

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()


class TestMatmul_Outer(BinaryOpMixin):
    def op_numpy(self, left, right, scale=1):
        return np.matmul(left, right)

    shapes = [
        (pytest.param((100, 1), id="ket"), pytest.param((1, 100), id="bra")),
        (pytest.param((100, 1), id="ket"), pytest.param((1, 1), id="scalar")),
        (pytest.param((100, 2), id="kets"), pytest.param((2, 100), id="bras")),
    ]
    bad_shapes = [
        (pytest.param((10, 1), id="ket"), pytest.param((10, 1), id="ket")),
        (pytest.param((10, 10), id="square"), pytest.param((1, 10), id="bra")),
    ]

    extra_param = {"scale": [
        pytest.param(NoParam),
        pytest.param(0.2, id="scale[real]"),
        pytest.param(0.5j, id="scale[complex]"),
    ]}


class TestExpect(BinaryOpMixin):
    def op_numpy(self, op, state):
        is_ket = state.shape[1] == 1
        if is_ket:
            return np.conj(state.T) @ op @ state
        else:
            return np.trace(op @ state)

    _dim = 100
    _ket = pytest.param((_dim, 1), id="ket")
    _dm = pytest.param((_dim, _dim), id="dm")
    _op = pytest.param((_dim, _dim), id="op")
    _bra = pytest.param((1, _dim), id="bra")
    _nonsquare = pytest.param((2, _dim), id="nonsquare")
    _not_op = [_bra, _ket, _nonsquare]

    shapes = [
        (_op, _ket),
        (_op, _dm),
    ]
    bad_shapes = list(itertools.product(_not_op, [_ket, _dm]))  # Bad op
    bad_shapes += [
        (_op, _nonsquare),
        (_op, _bra),
    ]  # Bad ket/dm


class TestExpectSuper(BinaryOpMixin):
    def op_numpy(self, op, state):
        n = np.sqrt(state.shape[0]).astype(int)
        out_shape = (n, n)
        return np.trace(np.reshape(op@state, out_shape))

    _dim = 100
    _super_ket = pytest.param((_dim, 1), id="super_ket")
    _super_op = pytest.param((_dim, _dim), id="super_op")
    _bra = pytest.param((1, _dim), id="row_stacked")
    _nonsquare = pytest.param((2, _dim), id="nonsquare")
    _not_super_ket = [_super_op, _bra, _nonsquare]
    _not_super_op = [_super_ket, _bra, _nonsquare]

    shapes = [(_super_op, _super_ket), ]
    bad_shapes = list(itertools.product(_not_super_op, [_super_ket]))  # Bad super op
    bad_shapes += list(itertools.product([_super_op], _not_super_ket))  # Bad super ket


class TestWRMN_error(BinaryOpMixin):
    def op_numpy(self, left, right, atol, rtol):
        return np.linalg.norm(
            np.abs(left)
            / (atol + rtol * np.abs(right))
        ) / left.size**0.5

    shapes = shapes_binary_identical()
    bad_shapes = shapes_binary_bad_identical()

    extra_param = {
        "atol": [1e-7, 0.5],
        "rtol": [0, 1e-10, 0.5],
    }
    wrong_extra_param = {
        "atol": [0.5],
        "rtol": [0.5],
    }

#=============================================================================#
#                           Ternary specialisation                            #
#=============================================================================#

class TestMatmulInPlace(TernaryOpMixin):
    """Test in-place matmul operations that support out parameter."""

    def op_numpy(self, left, right, out, scale=1):
        return scale * np.matmul(left, right) + out

    def test_mathematically_correct(
        self, op,
        data_l, data_r, data_out,
        out_type, random_generator,
        kw,
    ):
        left, right = data_l(random_generator), data_r(random_generator)
        out = data_out(random_generator)
        expected = self.op_numpy(left.to_array(), right.to_array(), out.to_array(), **kw)

        test = op(left, right, out=out, **kw)
        self.check_result(test, expected, out_type)

    def test_incorrect_shape_raises(
        self, op, data_l, data_m, data_r, random_generator, kw,
    ):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        rng = random_generator
        with pytest.raises(ValueError):
            op(data_l(rng), data_m(rng), out=data_r(rng), **kw)

    shapes = [
        (x, y, pytest.param((x.values[0][0], y.values[0][1]), id=""))
        for x, y in itertools.product(shapes_unary(50), repeat=2)
        if x.values[0][1] == y.values[0][0]
    ]
    bad_shapes = [
        (
            pytest.param((2, 2), id=""),
            pytest.param((3, 2), id=""),
            pytest.param((2, 2), id="[2x2@3x2->2x2]")
        ),
        (
            pytest.param((2, 2), id=""),
            pytest.param((2, 3), id=""),
            pytest.param((2, 2), id="[2x2@2x3->2x2]")
        ),
    ]


class TestMatmulDagInPlace(TernaryOpMixin):
    """Test in-place matmul operations that support out parameter."""

    def op_numpy(self, left, right, out, scale=1):
        return scale * np.matmul(left, right.T.conj()) + out

    def test_mathematically_correct(
        self, op,
        data_l, data_r, data_out,
        out_type, random_generator,
        kw,
    ):
        left, right = data_l(random_generator), data_r(random_generator)
        out = data_out(random_generator)
        expected = self.op_numpy(
            left.to_array(),
            right.to_array(),
            out.to_array(),
            **kw
        )

        test = op(left, right, out=out, **kw)
        self.check_result(test, expected, out_type)

    def test_incorrect_shape_raises(
        self, op, data_l, data_m, data_r, random_generator, kw,
    ):
        """
        Test that the operation produces a suitable error if the shapes of the
        given operands are not compatible.
        """
        rng = random_generator
        with pytest.raises(ValueError):
            op(data_l(rng), data_m(rng), out=data_r(rng), **kw)

    shapes = [
        (x, y, pytest.param((x.values[0][0], y.values[0][0]), id=""))
        for x, y in itertools.product(shapes_unary(50), repeat=2)
        if x.values[0][1] == y.values[0][1]
    ]
    bad_shapes = [
        (
            pytest.param((2, 2), id=""),
            pytest.param((2, 3), id=""),
            pytest.param((2, 2), id="[2x2@3x2->2x2]")
        ),
        (
            pytest.param((2, 2), id=""),
            pytest.param((3, 2), id=""),
            pytest.param((2, 2), id="[2x2@2x3->2x2]")
        ),
    ]


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

    def generate_scalar_is_ket(self, metafunc):
        parameters = (
            ['op']
            + [x for x in metafunc.fixturenames
               if x.startswith("data_")]
            + ['out_type']
        )
        cases = []
        cases_map = self.correct_cases or CORRECT_CASES
        for p_op in self.specialisations:
            op, *types, out_type = p_op.values
            args = (op, types, [(self._scalar,) * 3], out_type)
            cases.extend(cases_type_shape_product(cases_map, *args))
        metafunc.parametrize(parameters, cases)
        metafunc.parametrize('scalar_is_ket',
                             [True, False], ids=["ket", "bra"])

    def test_scalar_is_ket(self, op, data_l, data_m, data_r, out_type,
                           scalar_is_ket, random_generator):
        left = data_l(random_generator)
        mid = data_m(random_generator)
        right = data_r(random_generator)
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
