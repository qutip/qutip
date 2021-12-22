import pytest
import scipy.sparse
import qutip
from qutip.fastsparse import fast_csr_matrix


class TestOperationEffectsOnType:
    @pytest.mark.parametrize("operation", [
        pytest.param(lambda x: x, id="identity"),
        pytest.param(lambda x: x + x, id="addition"),
        pytest.param(lambda x: x - x, id="subtraction"),
        pytest.param(lambda x: x * x, id="multiplication by op"),
        pytest.param(lambda x: 2*x, id="multiplication by scalar"),
        pytest.param(lambda x: x/3, id="division by scalar"),
        pytest.param(lambda x: -x, id="negation"),
        pytest.param(lambda x: x.copy(), id="copy"),
        pytest.param(lambda x: x.T, id="transpose [.T]"),
        pytest.param(lambda x: x.trans(), id="transpose [.trans()]"),
        pytest.param(lambda x: x.transpose(), id="transpose [.transpose()]"),
        pytest.param(lambda x: x.H, id="adjoint [.H]"),
        pytest.param(lambda x: x.getH(), id="adjoint [.getH()]"),
        pytest.param(lambda x: x.adjoint(), id="adjoint [.adjoint()]"),
    ])
    def test_operations_preserve_type(self, operation):
        op = qutip.rand_herm(5).data
        assert isinstance(operation(op), fast_csr_matrix)

    @pytest.mark.parametrize("operation", [
        pytest.param(lambda x, y: y, id="identity of other"),
        pytest.param(lambda x, y: x + y, id="addition"),
        pytest.param(lambda x, y: y + x, id="r-addition"),
        pytest.param(lambda x, y: x - y, id="subtraction"),
        pytest.param(lambda x, y: y - x, id="r-subtraction"),
        pytest.param(lambda x, y: x * y, id="multiplication"),
        pytest.param(lambda x, y: y * x, id="r-multiplication"),
    ])
    def test_mixed_operations_yield_type(self, operation):
        op = qutip.rand_herm(5).data
        other = scipy.sparse.csr_matrix((op.data, op.indices, op.indptr),
                                        copy=True, shape=op.shape)
        assert not isinstance(operation(op, other), fast_csr_matrix)
