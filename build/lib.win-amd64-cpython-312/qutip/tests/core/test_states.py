import pytest
import numpy as np
import qutip
from functools import partial
from itertools import combinations


@pytest.mark.parametrize("size, n", [(2, 0), (2, 1), (100, 99)])
def test_basis_simple(size, n):
    qobj = qutip.basis(size, n)
    numpy = np.zeros((size, 1), dtype=complex)
    numpy[n, 0] = 1
    assert np.array_equal(qobj.full(), numpy)


@pytest.mark.parametrize("to_test", [
    qutip.basis, qutip.fock, qutip.fock_dm,
])
@pytest.mark.parametrize("size, n", [([2, 2], [0, 1]), ([2, 3, 4], [1, 2, 0])])
def test_implicit_tensor_basis_like(to_test, size, n):
    implicit = to_test(size, n)
    explicit = qutip.tensor(*[to_test([ss], [nn]) for ss, nn in zip(size, n)])
    assert implicit == explicit


@pytest.mark.parametrize("size, n, offset, msg", [
    ([2, 2], [0, 1, 1], [0, 0], "All list inputs must be the same length."),
    ([2, 2], [1, 1], 1, "All list inputs must be the same length."),
    (-1, 0, 0, "Dimensions must be integers > 0"),
    (1.5, 0, 0, "Dimensions must be integers > 0"),
    (5, 5, 0, "All basis indices must be integers in the range `0 <= n < dimension`."),
    (5, 0, 2, "All basis indices must be integers in the range `offset <= n < dimension+offset`."),
], ids=["n too long", "offset too short", "neg dims",
        "fraction dims", "n too large", "n too small"]
)
def test_basis_error(size, n, offset, msg):
    with pytest.raises(ValueError) as e:
        qutip.basis(size, n, offset)
    assert str(e.value) == msg


def test_basis_error_type():
    with pytest.raises(TypeError) as e:
        qutip.basis(5, 3.5)
    assert str(e.value) == ("Dimensions must be integers")


@pytest.mark.parametrize("size, n, m", [
        ([2, 2], [0, 0], [1, 1]),
        ([2, 3, 4], [1, 2, 0], [0, 1, 3]),
    ])
def test_implicit_tensor_projection(size, n, m):
    implicit = qutip.projection(size, n, m)
    explicit = qutip.tensor(*[qutip.projection(ss, nn, mm)
                              for ss, nn, mm in zip(size, n, m)])
    assert implicit == explicit


@pytest.mark.parametrize("base, operator, args, opargs, eigenval", [
    pytest.param(qutip.basis, qutip.num, (10, 3), (10,), 3,
                 id="basis"),
    pytest.param(qutip.basis, qutip.num, (10, 3, 1), (10, 1), 3,
                 id="basis,offset"),
    pytest.param(qutip.fock, qutip.num, (10, 3), (10,), 3,
                 id="fock"),
    pytest.param(qutip.fock_dm, qutip.num, (10, 3), (10,), 3,
                 id="fock_dm"),
    pytest.param(qutip.fock_dm, qutip.num, (10, 3, 1), (10, 1), 3,
                 id="fock_dm,offset"),
    pytest.param(qutip.coherent, qutip.destroy, (20, 0.75), (20,), 0.75,
                 id="coherent"),
    pytest.param(qutip.coherent, qutip.destroy, (50, 4.25, 1), (50, 1), 4.25,
                 id="coherent,offset"),
    pytest.param(qutip.coherent_dm, qutip.destroy, (25, 1.25), (25,), 1.25,
                 id="coherent_dm"),
    pytest.param(qutip.phase_basis, qutip.phase,
                 (10, 3), (10,), 3 * 2 * np.pi / 10,
                 id="phase_basis"),
    pytest.param(qutip.phase_basis, qutip.phase,
                 (10, 3, 1), (10, 1), 3 * 2 * np.pi / 10 + 1,
                 id="phase_basis,phi0"),
    pytest.param(qutip.spin_state, qutip.spin_Jz, (3, 2), (3,), 2,
                 id="spin_state"),
    pytest.param(qutip.zero_ket, qutip.qeye, (10,), (10,), 0,
                 id="zero_ket"),
])
def test_diverse_basis(base, operator, args, opargs, eigenval):
    # For state which are supposed to eigenvector of an operator
    # Verify that correspondance
    state = base(*args)
    oper = operator(*opargs)
    assert qutip.expect(oper, state) == pytest.approx(eigenval)


@pytest.mark.parametrize('dm', [
    partial(qutip.thermal_dm, n=1.),
    qutip.maximally_mixed_dm,
    partial(qutip.coherent_dm, alpha=0.5),
    partial(qutip.fock_dm, n=1),
    partial(qutip.spin_state, m=2, type='dm'),
    partial(qutip.spin_coherent, theta=1, phi=2, type='dm'),
], ids=[
    'thermal_dm', 'maximally_mixed_dm', 'coherent_dm',
    'fock_dm', 'spin_state', 'spin_coherent'
])
def test_dm(dm):
    N = 5
    rho = dm(N)
    # make sure rho has trace close to 1.0
    assert rho.tr() == pytest.approx(1.0)


def test_CoherentState():
    N = 10
    alpha = 0.5
    c1 = qutip.coherent(N, alpha)  # displacement method
    c2 = qutip.coherent(7, alpha, offset=3)  # analytic method
    c3 = qutip.coherent(N, alpha, offset=0, method="analytic")
    np.testing.assert_allclose(c1.full()[3:], c2.full(), atol=1e-7)
    np.testing.assert_allclose(c1.full(), c3.full(), atol=1e-7)
    with pytest.raises(TypeError) as e:
        qutip.coherent(N, alpha, method="other")
    assert all(method in str(e.value) for method in ["operator", "analytic"])
    with pytest.raises(ValueError) as e:
        qutip.coherent(N, alpha, offset=-1)
    assert str(e.value) == ("Offset must be non-negative")
    with pytest.raises(ValueError) as e:
        qutip.coherent(N, alpha, offset=1, method="operator")
    assert str(e.value) == (
        "The method 'operator' does not support offset != 0. Please"
        " select another method or set the offset to zero."
    )


def test_CoherentDensityMatrix():
    N = 10
    rho = qutip.coherent_dm(N, 1)
    assert rho.tr() == pytest.approx(1.0)
    with pytest.raises(TypeError) as e:
        qutip.coherent_dm(N, 1, method="other")
    assert all(method in str(e.value) for method in ["operator", "analytic"])
    with pytest.raises(ValueError) as e:
        qutip.coherent_dm(N, 1, offset=-1)
    assert str(e.value) == ("Offset must be non-negative")
    with pytest.raises(ValueError) as e:
        qutip.coherent_dm(N, 1, offset=1, method="operator")
    assert str(e.value) == (
        "The method 'operator' does not support offset != 0. Please"
        " select another method or set the offset to zero."
    )


def test_thermal():
    N = 10
    beta = 0.5
    assert qutip.thermal_dm(N, 0) == qutip.fock_dm(N, 0)

    thermal_operator = qutip.thermal_dm(N, beta)
    thermal_analytic = qutip.thermal_dm(N, beta, method="analytic")
    np.testing.assert_allclose(thermal_operator.full(),
                               thermal_analytic.full(), atol=2e-5)

    with pytest.raises(ValueError) as e:
        qutip.thermal_dm(N, beta, method="other")
    assert all(method in str(e.value) for method in ["operator", "analytic"])


@pytest.mark.parametrize('func', [
    qutip.spin_state, partial(qutip.spin_coherent, phi=0.5)
])
def test_spin_output(func):
    assert qutip.isket(func(1.0, 0, type='ket'))
    assert qutip.isbra(func(1.0, 0, type='bra'))
    assert qutip.isoper(func(1.0, 0, type='dm'))

    with pytest.raises(ValueError) as e:
        func(1.0, 0, type='something')
    assert str(e.value).startswith("Invalid value keyword argument")


@pytest.mark.parametrize('N', [2.5, -1])
def test_maximally_mixed_dm_error(N):
    with pytest.raises(ValueError) as e:
        qutip.maximally_mixed_dm(N)
    assert str(e.value) == "Dimensions must be integers > 0"


def test_TripletStateNorm():
    for triplet in qutip.triplet_states():
        assert triplet.norm() == pytest.approx(1.)
    for t1, t2 in combinations(qutip.triplet_states(), 2):
        assert t1.overlap(t2) == pytest.approx(0.)


def test_ket2dm():
    N = 5
    ket = qutip.coherent(N, 2)
    bra = ket.dag()
    oper = qutip.ket2dm(ket)
    oper_from_bra = qutip.ket2dm(bra)
    assert qutip.expect(oper, ket) == pytest.approx(1.)
    assert qutip.isoper(oper)
    assert oper == ket * bra
    assert oper == oper_from_bra
    with pytest.raises(TypeError) as e:
        qutip.ket2dm(oper)
    assert str(e.value) == "Input is not a ket or bra vector."


@pytest.mark.parametrize('state', [[0, 1], [0, 0], [0, 1, 0, 1]])
def test_qstate(state):
    from_basis = qutip.basis([2] * len(state), state)
    from_qstate = qutip.qstate("".join({0: "d", 1: "u"}[i] for i in state))
    assert from_basis == from_qstate


def test_qstate_error():
    with pytest.raises(TypeError) as e:
        qutip.qstate("eeggg")
    assert str(e.value) == ('String input to QSTATE must consist ' +
                            'of "u" and "d" elements only')


@pytest.mark.parametrize('state', ["11000", "eeggg", "dduuu", "VVHHH"])
def test_bra_ket(state):
    from_basis = qutip.basis([2, 2, 2, 2, 2], [1, 1, 0, 0, 0])
    from_ket = qutip.ket(state)
    from_bra = qutip.bra(state).dag()
    assert from_basis == from_ket
    assert from_basis == from_bra


def test_w_states():
    state = (
        qutip.qstate("uddd") +
        qutip.qstate("dudd") +
        qutip.qstate("ddud") +
        qutip.qstate("dddu")
    ) / 2
    assert state == qutip.w_state(4)


def test_ghz_states():
    state = (qutip.qstate("uuu") + qutip.qstate("ddd")).unit()
    assert state == qutip.ghz_state(3)


def test_bell_state():
    states = [
        qutip.bell_state('00'),
        qutip.bell_state('01'),
        qutip.bell_state('10'),
        qutip.bell_state('11')
    ]
    exited = qutip.basis([2, 2], [1, 1])
    for state, overlap in zip(states, [0.5**0.5, -0.5**0.5, 0, 0]):
        assert state.norm() == pytest.approx(1.0)
        assert state.overlap(exited) == pytest.approx(overlap)

    for state1, state2 in combinations(states, 2):
        assert state1.overlap(state2) == pytest.approx(0.0)

    assert qutip.singlet_state() == qutip.bell_state('11')


def _id_func(val):
    "ids generated from the function only"
    if isinstance(val, tuple):
        return ""


# random object accept `str` and base.Data
# Obtain all valid dtype from `to`
dtype_names = list(qutip.data.to._str2type.keys()) + list(qutip.data.to.dtypes)
dtype_types = list(qutip.data.to._str2type.values()) + list(qutip.data.to.dtypes)
dtype_combinations = list(zip(dtype_names, dtype_types))
@pytest.mark.parametrize(['alias', 'dtype'], dtype_combinations,
                         ids=[str(dtype) for dtype in dtype_names])
@pytest.mark.parametrize(['func', 'args'], [
    (qutip.basis, (5, 1)),
    (qutip.fock, (5, 1)),
    (qutip.fock_dm, (5, 1)),
    (qutip.coherent, (5, 1)),
    (qutip.coherent_dm, (5, 1)),
    (qutip.thermal_dm, (5, 1)),
    (qutip.maximally_mixed_dm, (5,)),
    (qutip.phase_basis, (5, 1)),
    (qutip.zero_ket, (5,)),
    (qutip.spin_state, (5, 1)),
    (qutip.spin_coherent, (5, 1, 0.5)),
    (qutip.projection, (5, 1, 2)),
    (qutip.ket, ("001",)),
    (qutip.bra, ('010',)),
    (qutip.qstate, ("uud",)),
    (qutip.w_state, (5,)),
    (qutip.ghz_state, (5,)),
    (qutip.qutrit_basis, ()),
    (qutip.triplet_states, ()),
    (qutip.singlet_state, ()),
    (qutip.bell_state, ('10',)),
    (qutip.enr_fock, ([3, 3, 3], 4, [1, 1, 0])),
    (qutip.enr_thermal_dm, ([3, 3, 3], 4, 2)),
], ids=_id_func)
def test_state_type(func, args, alias, dtype):
    object = func(*args, dtype=alias)
    if isinstance(object, qutip.Qobj):
        assert isinstance(object.data, dtype)
    else:
        for obj in object:
            assert isinstance(obj.data, dtype)

    with qutip.CoreOptions(default_dtype=alias):
        object = func(*args)
        if isinstance(object, qutip.Qobj):
            assert isinstance(object.data, dtype)
        else:
            for obj in object:
                assert isinstance(obj.data, dtype)


@pytest.mark.parametrize(['func', 'args'], [
    (qutip.basis, (None,)),
    (qutip.fock, (None,)),
    (qutip.fock_dm, (None,)),
    (qutip.maximally_mixed_dm, ()),
    (qutip.projection, ([0, 1, 1], [1, 1, 0])),
    (qutip.zero_ket, ()),
], ids=_id_func)
def test_state_space_input(func,  args):
    dims = qutip.dimensions.Space([2, 2, 2])
    assert func(dims, *args) == func([2, 2, 2], *args)
