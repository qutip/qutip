import pytest
import qutip
from qutip.core.einsum import einsum

# Helpers for einsum test:
_cx = qutip.Qobj(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dims=[[2, 2], [2, 2]],
)
_cx_dag = _cx.dag()
_rho_01 = qutip.ket2dm(
    qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 1))
)
_thermal_dm_2q = qutip.tensor(
    qutip.thermal_dm(2, 1), qutip.thermal_dm(2, 1)
)
_L1 = qutip.spre(qutip.tensor(qutip.sigmax(), qutip.sigmay()))
_L2 = qutip.spost(qutip.tensor(qutip.sigmaz(), qutip.sigmax()))
_oper_ket = qutip.operator_to_vector(
    qutip.tensor(qutip.sigmax(), qutip.sigmay())
)
_oper_bra = _oper_ket.dag()


@pytest.mark.parametrize(["subscripts", "operands", "expected", "out_dims"], [
    pytest.param("ii", [qutip.sigmaz()], 0, None),
    pytest.param("ij,ji", [qutip.sigmaz(), qutip.sigmaz()], 2, None),
    pytest.param(
        "ijij", [_thermal_dm_2q], 1, None,
    ),
    pytest.param(
        "ikjl,jm->ikml",
        [qutip.tensor(qutip.sigmaz(), qutip.sigmaz()), qutip.sigmaz()],
        qutip.tensor(qutip.qeye(2), qutip.sigmaz()),
        None,
    ),
    pytest.param(
        "abcd,cdef->abef",
        [_cx, _rho_01],
        _cx @ _rho_01,
        None,
        id="density_matrix_left_multiplication",
    ),
    pytest.param(
        "cdef,efgh->cdgh",
        [_rho_01, _cx_dag],
        _rho_01 @ _cx_dag,
        None,
        id="density_matrix_right_multiplication",
    ),
    pytest.param(
        "abcd,cdef,efgh->abgh",
        [_cx, _rho_01, _cx_dag],
        _cx @ _rho_01 @ _cx_dag,
        None,
        id="density_matrix_conjugation",
    ),
    pytest.param(
        "ijklabcd,abcdmnop->ijklmnop",
        [_L1, _L2],
        _L1 @ _L2,
        _L1.dims,
        id="superoperator_multiplication",
    ),
    pytest.param(
        "yijkl,ijklz->",
        [_oper_bra, _oper_ket],
        complex(_oper_bra @ _oper_ket),
        None,
        id="operator_ket_bra_inner_product",
    ),
])
def test_einsum(subscripts, operands, expected, out_dims):
    res = einsum(subscripts, *operands, out_dims=out_dims)
    assert res == expected


@pytest.mark.parametrize(["subscripts", "operands"], [
    pytest.param(
        "ij", [qutip.sigmax()],
        id="single_operand_no_contraction",
    ),
    pytest.param(
        "ij->ji", [qutip.sigmay()],
        id="single_operand_transpose",
    ),
    pytest.param(
        "ijkl->kjil",
        [qutip.tensor(qutip.sigmam(), qutip.sigmaz())],
        id="single_operand_permutation",
    ),
    pytest.param(
        "cdef,ghef->cdgh",
        [_rho_01, _cx],
        id="col_col_contraction",
    ),
    pytest.param(
        "ij,jk->ki",
        [qutip.sigmax(), qutip.sigmaz()],
        id="output_col_before_row",
    ),
    pytest.param(
        "ikjl,jm->mlik",
        [qutip.tensor(qutip.sigmaz(), qutip.sigmaz()), qutip.sigmaz()],
        id="output_col_before_row_composite",
    ),
])
def test_einsum_rejects_implicit_transpose(subscripts, operands):
    with pytest.raises(ValueError):
        einsum(subscripts, *operands)
