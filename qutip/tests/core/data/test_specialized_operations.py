import numpy as np
import pytest
from qutip import data
from qutip.core.data import CSR, Dense
import qutip


@pytest.mark.parametrize(["op_type", "state_type", "function"], [
    (Dense, Dense, data.herm_matmul_dense),
    (CSR, Dense, data.herm_matmul_csr_dense_dense),
    (Dia, Dense, data.herm_matmul_dia_dense_dense),
    (Dense, CSR, data.herm_matmul_data),
])
def test_herm_matmul(op_type, state_type, function):
    N = 20
    H = qutip.rand_herm(N, dtype=op_type)
    a = qutip.destroy(N, dtype=op_type)
    L = qutip.liouvillian(H, [a]).to(op_type).data

    state = qutip.operator_to_vector(qutip.rand_dm(N, dtype=state_type)).data

    herm_out = function(L, state, N).to_array()
    expected = data.matmul(L, state).to_array()

    np.testing.assert_allclose(herm_out, expected, atol=1e-15)
