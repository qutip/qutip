import operator

import pytest
from qutip import (
    Qobj, QobjEvo, coefficient, qeye, sigmax, sigmaz, num, rand_stochastic,
    rand_herm, rand_ket, liouvillian, basis, spre, spost, to_choi, expect,
    rand_ket, rand_dm, operator_to_vector, SESolver, MESolver
)
import qutip.core.data as _data
import numpy as np
from numpy.testing import assert_allclose

from qutip.core import data as _data

# prepare coefficient
class Pseudo_qevo:
    # Mimic QobjEvo on __call__
    # and can return parameter to create the equivalent QobjEvo
    # for each coefficient type
    def __init__(self, cte, qobj, func, string, args):
        self.cte = cte
        self.qobj = qobj
        self.func = func
        self.str = string
        self.args = args

    def array(self):
        tlist = np.linspace(0, 10, 10001)
        coeff = self.func(tlist, self.args)
        return ([self.cte, [self.qobj, coeff]], {}, tlist)

    def logarray(self):
        tlist = np.logspace(-3, 1, 10001)
        coeff = self.func(tlist, self.args)
        return ([self.cte, [self.qobj, coeff]], {}, tlist)

    def func_coeff(self):
        return ([self.cte, [self.qobj, self.func]], self.args)

    def string(self):
        return ([self.cte, [self.qobj, self.str]], self.args)

    def func_call(self):
        return (self.__call__, self.args)

    def __call__(self, t, args={}):
        args = args or self.args
        return self.cte + self.qobj * self.func(t, args)

    def __getitem__(self, which):
        return getattr(self, which)()

    @property
    def _dims(self):
        return self.qobj._dims


N = 3
args = {'w1': 1, "w2": 2}
TESTTIMES = np.linspace(0.001, 1.0, 10)


def _real(t, args):
    return np.sin(t*args['w1'])


def _cplx(t, args):
    return np.exp(1j*t*args['w2'])


real_qevo = Pseudo_qevo(
    rand_stochastic(N).to(_data.CSR),
    rand_stochastic(N).to(_data.CSR),
    _real, "sin(t*w1)", args)

herm_qevo = Pseudo_qevo(
    rand_herm(N).to(_data.Dense),
    rand_herm(N).to(_data.Dense),
    _real, "sin(t*w1)", args)

cplx_qevo = Pseudo_qevo(
    rand_stochastic(N).to(_data.Dense),
    rand_stochastic(N).to(_data.CSR) + rand_stochastic(N).to(_data.CSR) * 1j,
    _cplx, "exp(1j*t*w2)", args)


@pytest.fixture(params=['func_coeff', 'string',
                        'array', 'logarray', 'func_call'])
def coeff_type(request):
    # all available QobjEvo types
    return request.param


@pytest.fixture(params=[
    pytest.param(real_qevo, id="real"),
    pytest.param(herm_qevo, id="hermitian"),
    pytest.param(cplx_qevo, id="complex"),
])
def pseudo_qevo(request):
    return request.param


@pytest.fixture
def all_qevo(pseudo_qevo, coeff_type):
    base, args, *tlist = pseudo_qevo[coeff_type]
    if tlist: tlist = tlist[0]
    return QobjEvo(base, args, tlist=tlist)


@pytest.fixture
def other_qevo(all_qevo):
    return all_qevo


def _assert_qobjevo_equivalent(obj1, obj2, tol=1e-8):
    assert obj1._dims == obj1(0)._dims
    assert obj2._dims == obj2(0)._dims
    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj1(t), obj2(t), tol)


def _assert_qobj_almost_eq(obj1, obj2, tol=1e-10):
    assert obj1.dims == obj2.dims
    assert obj1.shape == obj2.shape
    assert obj1.type == obj2.type
    assert _data.iszero((obj1 - obj2).data, tol)


def _assert_qobjevo_different(obj1, obj2):
    assert any(obj1(t) != obj2(t) for t in np.random.rand(10) * .9 + 0.05)


def _div(a, b):
    return a / b


def test_call(pseudo_qevo, coeff_type):
    # test creation of QobjEvo and call
    base, args, *tlist = pseudo_qevo[coeff_type]
    if tlist: tlist = tlist[0]
    qevo = QobjEvo(base, args, tlist=tlist)
    assert isinstance(qevo(0), Qobj)
    assert qevo.isoper
    assert not qevo.isconstant
    assert not qevo.issuper
    _assert_qobjevo_equivalent(pseudo_qevo, qevo)


# Test the QobjEvo.__repr__()
def test_QobjEvo_repr():
    # case_n: cases with Objects of QobjEvo with unique __repr__
    # expected_repr_n: are the Expected result from the __repr__

    case_1 = repr(QobjEvo([qeye(3), lambda t: t]))
    expected_repr_1 = 'QobjEvo: dims = [[3], [3]], shape = (3, 3), '
    expected_repr_1 += 'type = oper, superrep = None, '
    expected_repr_1 += 'isconstant = False, num_elements = 1'
    assert case_1 == expected_repr_1

    case_2 = repr(QobjEvo(qeye(2)))
    expected_repr_2 = 'QobjEvo: dims = [[2], [2]], shape = (2, 2), '
    expected_repr_2 += 'type = oper, superrep = None, '
    expected_repr_2 += 'isconstant = True, num_elements = 1'
    assert case_2 == expected_repr_2

    case_3 = repr(QobjEvo(basis(5, 2)))
    expected_repr_3 = 'QobjEvo: dims = [[5], [1]], shape = (5, 1), '
    expected_repr_3 += 'type = ket, superrep = None, '
    expected_repr_3 += 'isconstant = True, num_elements = 1'
    assert case_3 == expected_repr_3

    X = sigmax()
    S = spre(X) * spost(X.dag())
    case_4 = repr(QobjEvo(to_choi(S)))
    expected_repr_4 = 'QobjEvo: dims = [[[2], [2]], [[2], [2]]], '
    expected_repr_4 += 'shape = (4, 4), type = super, superrep = choi, '
    expected_repr_4 += 'isconstant = True, num_elements = 1'
    assert case_4 == expected_repr_4

    case_5 = repr(QobjEvo([[qeye(4), lambda t: t],
                           [qeye(4), lambda t: t]], compress=False))
    expected_repr_5 = 'QobjEvo: dims = [[4], [4]], shape = (4, 4), '
    expected_repr_5 += 'type = oper, superrep = None, '
    expected_repr_5 += 'isconstant = False, num_elements = 2'
    assert case_5 == expected_repr_5


@pytest.mark.parametrize('coeff_type',
                         ['func_coeff', 'string', 'array', 'logarray'])
def test_product_coeff(pseudo_qevo, coeff_type):
    # test creation of QobjEvo with Qobj * Coefficient
    # Skip pure func: QobjEvo(f(t, args) -> Qobj)
    base = pseudo_qevo[coeff_type]
    cte, [qobj, coeff] = base[0]
    args = base[1] if len(base) >= 2 else {}
    tlist = base[2] if len(base) >= 3 else None
    coeff = coefficient(coeff, args=args, tlist=tlist)
    created = cte + qobj * coeff
    _assert_qobjevo_equivalent(pseudo_qevo, created)

def test_copy(all_qevo):
    qevo = all_qevo
    copy = qevo.copy()
    _assert_qobjevo_equivalent(copy, qevo)
    assert copy is not qevo


@pytest.mark.parametrize('bin_op', [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(lambda a, b: a @ b, id="matmul"),
    pytest.param(lambda a, b: a & b, id="tensor"),
])
def test_binopt(all_qevo, other_qevo, bin_op):
    "QobjEvo arithmetic"
    obj1 = all_qevo
    obj2 = other_qevo
    for t in TESTTIMES:
        as_qevo = bin_op(obj1, obj2)(t)
        as_qobj = bin_op(obj1(t), obj2(t))
        _assert_qobj_almost_eq(as_qevo, as_qobj)


@pytest.mark.parametrize('bin_op', [
    pytest.param(operator.iadd, id="add"),
    pytest.param(operator.isub, id="sub"),
    pytest.param(operator.imul, id="mul"),
    pytest.param(operator.imatmul, id="matmul"),
    pytest.param(operator.iand, id="tensor"),
])
def test_binopt_inplace(all_qevo, other_qevo, bin_op):
    obj1 = all_qevo
    obj2 = other_qevo
    for t in TESTTIMES:
        as_qevo = bin_op(obj1.copy(), obj2)(t)
        as_qobj = bin_op(obj1(t).copy(), obj2(t))
        _assert_qobj_almost_eq(as_qevo, as_qobj)


@pytest.mark.parametrize('bin_op', [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(lambda a, b: a @ b, id="matmul"),
    pytest.param(lambda a, b: a & b, id="tensor"),
])
def test_binopt_qobj(all_qevo, bin_op):
    "QobjEvo arithmetic"
    obj = all_qevo
    qobj = rand_herm(N)
    for t in TESTTIMES:
        as_qevo = bin_op(obj, qobj)(t)
        as_qobj = bin_op(obj(t), qobj)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

        as_qevo = bin_op(qobj, obj)(t)
        as_qobj = bin_op(qobj, obj(t))
        _assert_qobj_almost_eq(as_qevo, as_qobj)


@pytest.mark.parametrize('bin_op', [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(_div, id="div"),
])
def test_binopt_scalar(all_qevo, bin_op):
    "QobjEvo arithmetic"
    obj = all_qevo
    scalar = 0.5 + 1j
    for t in TESTTIMES:
        as_qevo = bin_op(obj, scalar)(t)
        as_qobj = bin_op(obj(t), scalar)
        _assert_qobj_almost_eq(as_qevo, as_qobj)

        if bin_op is not _div:
            as_qevo = bin_op(scalar, obj)(t)
            as_qobj = bin_op(scalar, obj(t))
            _assert_qobj_almost_eq(as_qevo, as_qobj)


def binop_coeff(all_qevo):
    obj = all_qevo
    coeff = coeffient("t")
    created = obj * coeff_t
    for t in TESTTIMES:
        _assert_qobj_almost_eq(created(t), obj(t) * t)


@pytest.mark.parametrize('unary_op', [
    pytest.param(lambda a: a.conj(), id="conj"),
    pytest.param(lambda a: a.dag(), id="dag"),
    pytest.param(lambda a: a.trans(), id="trans"),
    pytest.param(lambda a: -a, id="neg"),
])
def test_unary(all_qevo, unary_op):
    "QobjEvo arithmetic"
    obj = all_qevo
    for t in TESTTIMES:
        transformed = unary_op(obj)
        as_qevo = transformed(t)
        as_qobj = unary_op(obj(t))
        assert transformed._dims == as_qevo._dims
        _assert_qobj_almost_eq(as_qevo, as_qobj)


@pytest.mark.parametrize('unary_op', [
    pytest.param(lambda a: a.conj(), id="conj"),
    pytest.param(lambda a: a.dag(), id="dag"),
    pytest.param(lambda a: a.trans(), id="trans"),
    pytest.param(lambda a: -a, id="neg"),
])
def test_unary_ket(unary_op):
    obj = QobjEvo(rand_ket(5))
    for t in TESTTIMES:
        transformed = unary_op(obj)
        as_qevo = transformed(t)
        as_qobj = unary_op(obj(t))
        assert transformed._dims == as_qevo._dims
        _assert_qobj_almost_eq(as_qevo, as_qobj)


@pytest.mark.parametrize('args_coeff_type',
                         ['func_coeff', 'string', 'func_call'])
def test_args(pseudo_qevo, args_coeff_type):
    obj = QobjEvo(*pseudo_qevo[args_coeff_type])
    args = {'w1': 3, "w2": 3}

    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj(t, args), pseudo_qevo(t, args))
        _assert_qobj_almost_eq(obj(t, **args), pseudo_qevo(t, args))

    # Did it modify original args
    _assert_qobjevo_equivalent(obj, pseudo_qevo)

    obj.arguments(args)
    _assert_qobjevo_different(obj, pseudo_qevo)
    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj(t), pseudo_qevo(t, args))

    args = {'w1': 4, "w2": 4}
    obj.arguments(**args)
    _assert_qobjevo_different(obj, pseudo_qevo)
    for t in TESTTIMES:
        _assert_qobj_almost_eq(obj(t), pseudo_qevo(t, args))


def test_copy_side_effects(all_qevo):
    t = 0.2
    qevo = all_qevo
    copy = qevo.copy()
    before = qevo(t)
    # Ensure inplace modification of the copy do not affect the original
    copy *= 2
    copy += rand_herm(N)
    copy *= rand_herm(N)
    copy.arguments({'w1': 3, "w2": 3})
    after = qevo(t)
    _assert_qobj_almost_eq(before, after)


@pytest.mark.parametrize('coeff_type',
    ['func_coeff', 'string', 'array', 'logarray']
)
def test_tidyup(all_qevo):
    "QobjEvo tidyup"
    obj = all_qevo
    obj *= 1e-12
    obj.tidyup(atol=1e-8)
    t = 0.2
    # check that the Qobj are cleaned
    assert_allclose(obj(t).full(), 0)


def test_QobjEvo_pickle(all_qevo):
    "QobjEvo pickle"
    # used in parallel_map
    import pickle
    obj = all_qevo
    pickled = pickle.dumps(obj, -1)
    recreated = pickle.loads(pickled)
    _assert_qobjevo_equivalent(recreated, obj)


def test_QobjEvo_restore(all_qevo):
    "QobjEvo pickle"
    # used in parallel_map
    obj = all_qevo
    state = obj._getstate()
    recreated = QobjEvo._restore(**state)
    _assert_qobjevo_equivalent(recreated, obj)


def test_mul_vec(all_qevo):
    "QobjEvo matmul ket"
    vec = Qobj(np.arange(N)*.5+.5j)
    op = all_qevo
    for t in TESTTIMES:
        assert_allclose((op(t) @ vec).full(),
                        op.matmul(t, vec).full(), atol=1e-14)


def test_matmul(all_qevo):
    "QobjEvo matmul oper"
    mat = np.random.rand(N, N) + 1 + 1j * np.random.rand(N, N)
    matDense = Qobj(mat).to(_data.Dense)
    matF = Qobj(np.asfortranarray(mat)).to(_data.Dense)
    matCSR = Qobj(mat).to(_data.CSR)
    op = all_qevo
    for t in TESTTIMES:
        Qo1 = op(t)
        assert_allclose((Qo1 @ mat).full(),
                        op.matmul(t, matF).full(), atol=1e-14)
        assert_allclose((Qo1 @ mat).full(),
                        op.matmul(t, matDense).full(), atol=1e-14)
        assert_allclose((Qo1 @ mat).full(),
                        op.matmul(t, matCSR).full(), atol=1e-14)


def test_expect_psi(all_qevo):
    "QobjEvo expect psi"
    vec = _data.dense.fast_from_numpy(np.arange(N)*.5 + .5j)
    qobj = Qobj(vec)
    op = all_qevo
    for t in TESTTIMES:
        Qo1 = op(t)
        assert_allclose(_data.expect(Qo1.data, vec), op.expect(t, qobj),
                        atol=1e-14)


def test_expect_rho(all_qevo):
    "QobjEvo expect rho"
    vec = _data.dense.fast_from_numpy(np.random.rand(N*N) + 1
                                      + 1j * np.random.rand(N*N))
    mat = _data.column_unstack_dense(vec, N)
    qobj = Qobj(mat)
    op = liouvillian(all_qevo)
    for t in TESTTIMES:
        Qo1 = op(t)
        assert abs(_data.expect_super(Qo1.data, vec)
                   - op.expect(t, qobj)) < 1e-14


@pytest.mark.parametrize('dtype',
[pytest.param(dtype, id=dtype.__name__)
     for dtype in _data.to.dtypes])
def test_convert(all_qevo, dtype):
    "QobjEvo expect rho"
    op = all_qevo.to(dtype)
    assert isinstance(op(0.5).data, dtype)


def test_compress():
    "QobjEvo compress"
    obj1 = QobjEvo(
        [[qeye(N), "t"], [qeye(N), "t"], [qeye(N), "t"]])
    assert obj1.num_elements == 1
    obj2 = QobjEvo(
        [[qeye(N), "t"], [qeye(N), "t"], [qeye(N), "t"]], compress=False)
    assert obj2.num_elements == 3
    _assert_qobjevo_equivalent(obj1, obj2)
    obj3 = obj2.copy()
    assert obj3.num_elements == 3
    obj3.compress()
    assert obj3.num_elements == 1
    _assert_qobjevo_equivalent(obj2, obj3)


@pytest.mark.parametrize(['qobjdtype'],
    [pytest.param(dtype, id=dtype.__name__)
     for dtype in _data.to.dtypes])
@pytest.mark.parametrize(['statedtype'],
    [pytest.param(dtype, id=dtype.__name__)
     for dtype in _data.to.dtypes])
def test_layer_support(qobjdtype, statedtype):
    N = 10
    qevo = QobjEvo(rand_herm(N).to(qobjdtype))
    state_dense = rand_ket(N).to(_data.Dense)
    state = state_dense.to(statedtype).data
    state_dense = state_dense.data
    exp_any = qevo.expect_data(0, state)
    exp_dense = qevo.expect_data(0, state_dense)
    assert_allclose(exp_any, exp_dense)
    mul_any = qevo.matmul_data(0, state).to_array()
    mul_dense = qevo.matmul_data(0, state_dense).to_array()
    assert_allclose(mul_any, mul_dense)


def test_QobjEvo_step_coeff():
    "QobjEvo step interpolation"
    coeff1 = np.random.rand(6)
    coeff2 = np.random.rand(6) + np.random.rand(6) * 1.j
    # uniform t
    tlist = np.array([2, 3, 4, 5, 6, 7], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
                      tlist=tlist, order=0)
    assert qobjevo(2.0)[0,0] == coeff1[0]
    assert qobjevo(7.0)[0,0] == coeff1[5]
    assert qobjevo(5.0001)[0,0] == coeff1[3]
    assert qobjevo(3.9999)[0,0] == coeff1[1]

    assert qobjevo(2.0)[0,1] == coeff2[0]
    assert qobjevo(7.0)[0,1] == coeff2[5]
    assert qobjevo(5.0001)[0,1] == coeff2[3]
    assert qobjevo(3.9999)[0,1] == coeff2[1]

    # non-uniform t
    tlist = np.array([1, 2, 4, 5, 6, 8], dtype=float)
    qobjevo = QobjEvo([[sigmaz(), coeff1], [sigmax(), coeff2]],
        tlist=tlist, order=0)
    assert qobjevo(1.0)[0,0] == coeff1[0]
    assert qobjevo(8.0)[0,0] == coeff1[5]
    assert qobjevo(3.9999)[0,0] == coeff1[1]
    assert qobjevo(4.23)[0,0] == coeff1[2]
    assert qobjevo(1.23)[0,0] == coeff1[0]

    assert qobjevo(1.0)[0,1] == coeff2[0]
    assert qobjevo(8.0)[0,1] == coeff2[5]
    assert qobjevo(6.7)[0,1] == coeff2[4]
    assert qobjevo(7.9999)[0,1] == coeff2[4]
    assert qobjevo(3.9999)[0,1] == coeff2[1]


def test_QobjEvo_isherm_flag_knowcase():
    assert QobjEvo(sigmax())(0)._isherm is True
    non_hermitian = sigmax() + 1j
    non_hermitian.isherm  # set flag
    assert QobjEvo(non_hermitian)(0)._isherm is False
    assert QobjEvo([sigmax(), sigmaz()])(0)._isherm is True
    assert QobjEvo([sigmax(), "t"])(0)._isherm is True
    assert QobjEvo([sigmax(), "1j"])(0)._isherm is None
    assert QobjEvo([[sigmax(), "t"], [sigmaz(), "1"]])(0)._isherm is True
    assert QobjEvo([[sigmax(), "t"], [sigmaz(), "1j"]])(0)._isherm is None


@pytest.mark.parametrize(
    "coeff_type",
    ['func_coeff', 'string', 'array', 'logarray']
)
def test_QobjEvo_to_list(coeff_type, pseudo_qevo):
    base, args, *tlist = pseudo_qevo[coeff_type]
    if tlist: tlist = tlist[0]
    qevo = QobjEvo(base, args, tlist=tlist)
    as_list = qevo.to_list()
    assert len(as_list) == 2
    restored = QobjEvo(as_list)
    _assert_qobjevo_equivalent(qevo, restored)


class Feedback_Checker_Coefficient:
    def __init__(self, stacked=True):
        self.state = None
        self.stacked = stacked

    def __call__(self, t, data=None, qobj=None, e_val=None):
        if self.state is not None:
            if data is not None and self.stacked:
                assert data == operator_to_vector(self.state).data
            elif data is not None:
                assert data == self.state.data
            if qobj is not None:
                assert qobj == self.state
            if e_val is not None:
                expected = expect(qeye(self.state.dims[0]), self.state)
                assert e_val == pytest.approx(expected, abs=1e-7)
        return 1.


def test_feedback_oper():
    checker = Feedback_Checker_Coefficient(stacked=False)
    checker.state = basis(2, 1)
    qevo = QobjEvo(
        [qeye(2), checker],
        args={
            "e_val": SESolver.ExpectFeedback(qeye(2), default=1.),
            "data": SESolver.StateFeedback(default=checker.state.data,
                                           raw_data=True),
            "qobj": SESolver.StateFeedback(default=checker.state),
        },
    )

    checker.state = rand_ket(2)
    qevo.expect(0, checker.state)
    checker.state = rand_ket(2)
    qevo.expect(0, checker.state)

    checker.state = rand_ket(2)
    qevo.matmul_data(0, checker.state.data)
    checker.state = rand_ket(2)
    qevo.matmul_data(0, checker.state.data)


def test_feedback_super():
    checker = Feedback_Checker_Coefficient()
    qevo = QobjEvo(
        [spre(qeye(2)), checker],
        args={
            "e_val": MESolver.ExpectFeedback(qeye(2)),
            "data": MESolver.StateFeedback(raw_data=True),
            "qobj": MESolver.StateFeedback(),
        },
    )

    checker.state = rand_dm(2)
    qevo.expect(0, operator_to_vector(checker.state))
    qevo.matmul_data(0, operator_to_vector(checker.state).data)

    qevo.arguments(e_val=MESolver.ExpectFeedback(spre(qeye(2))))

    checker.state = rand_dm(2)
    qevo.expect(0, operator_to_vector(checker.state))
    qevo.matmul_data(0, operator_to_vector(checker.state).data)

    checker = Feedback_Checker_Coefficient(stacked=False)
    qevo = QobjEvo(
        [spre(qeye(2)), checker],
        args={
            "data": MESolver.StateFeedback(raw_data=True, prop=True),
            "qobj": MESolver.StateFeedback(prop=True),
        },
    )

    checker.state = rand_dm(4)
    checker.state.dims = [[[2],[2]], [[2],[2]]]
    qevo.matmul_data(0, checker.state.data)


@pytest.mark.parametrize('dtype', ["CSR", "Dense"])
def test_qobjevo_dtype(dtype):
    obj = QobjEvo([qeye(2, dtype=dtype), [num(2, dtype=dtype), lambda t: t]])
    assert obj.dtype == _data.to.parse(dtype)

    obj = QobjEvo(lambda t: qeye(2, dtype=dtype))
    assert obj.dtype == _data.to.parse(dtype)


def test_qobjevo_mixed():
    obj = QobjEvo([qeye(2, dtype="CSR"), [num(2, dtype="Dense"), lambda t: t]])
    # We test that the output dtype is a know type: accepted by `to.parse`.
    _data.to.parse(obj.dtype)