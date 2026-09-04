import itertools
import numpy as np
import pytest
import scipy
import warnings

from .conftest import CORRECT_CASES, WRONG_CASES

from qutip.core import data
from qutip.core.data import Data, Dense, CSR, Dia
from qutip.core.data.dense import OrderEfficiencyWarning
from qutip.testing import mixin, random_data


# Factory methods for generating the cases, mapping type to the function.
# CORRECT_CASES is for getting all the special cases to test, WRONG_CASES is
# for getting just a single case from each.
mixin.CORRECT_CASES = CORRECT_CASES
mixin.WRONG_CASES = WRONG_CASES


class TestAdd(mixin.TestAdd):
    specialisations = [
        pytest.param(data.add_csr, CSR, CSR, CSR),
        pytest.param(data.add_dense, Dense, Dense, Dense),
        pytest.param(data.add_dia, Dia, Dia, Dia),
        pytest.param(data.iadd_dense, Dense, Dense, Dense),
        pytest.param(data.iadd_dense_data_dense, Dense, Dia, Dense),
        pytest.param(data.iadd_data, CSR, Dia, Data),
    ]


class TestAdjoint(mixin.TestAdjoint):
    specialisations = [
        pytest.param(data.adjoint_csr, CSR, CSR),
        pytest.param(data.adjoint_dense, Dense, Dense),
        pytest.param(data.adjoint_dia, Dia, Dia),
    ]


class TestConj(mixin.TestConj):
    specialisations = [
        pytest.param(data.conj_csr, CSR, CSR),
        pytest.param(data.conj_dense, Dense, Dense),
        pytest.param(data.conj_dia, Dia, Dia),
    ]


class TestInner(mixin.TestInner):
    specialisations = [
        pytest.param(data.inner_csr, CSR, CSR, complex),
        pytest.param(data.inner_dia, Dia, Dia, complex),
        pytest.param(data.inner_dense, Dense, Dense, complex),
        pytest.param(data.inner_data, Dense, Dense, complex),
        pytest.param(data.inner_data, CSR, CSR, complex),
    ]


class TestInnerOp(mixin.TestInnerOp):
    specialisations = [
        pytest.param(data.inner_op_csr, CSR, CSR, CSR, complex),
        pytest.param(data.inner_op_dia, Dia, Dia, Dia, complex),
        pytest.param(data.inner_op_dense, Dense, Dense, Dense, complex),
        pytest.param(data.inner_op_data, Dense, CSR, Dense, complex),
    ]


class TestKron(mixin.TestKron):
    specialisations = [
        pytest.param(data.kron_csr, CSR, CSR, CSR),
        pytest.param(data.kron_dense, Dense, Dense, Dense),
        pytest.param(data.kron_dia, Dia, Dia, Dia),
        pytest.param(data.kron_dense_csr_csr, Dense, CSR, CSR),
        pytest.param(data.kron_csr_dense_csr, CSR, Dense, CSR),
        pytest.param(data.kron_dense_dia_dia, Dense, Dia, Dia),
        pytest.param(data.kron_dia_dense_dia, Dia, Dense, Dia),
    ]


class TestKronT(mixin.TestKronT):
    specialisations = [
        pytest.param(data.kron_transpose_data, CSR, CSR, Data),
        pytest.param(data.kron_transpose_dense, Dense, Dense, Dense),
    ]


class TestMatmul(mixin.TestMatmul):
    specialisations = [
        pytest.param(data.matmul_csr, CSR, CSR, CSR),
        pytest.param(data.matmul_csr_dense_dense, CSR, Dense, Dense),
        pytest.param(data.matmul_dense, Dense, Dense, Dense),
        pytest.param(data.matmul_dia, Dia, Dia, Dia),
        pytest.param(data.matmul_dia_dense_dense, Dia, Dense, Dense),
        pytest.param(data.matmul_dense_dia_dense, Dense, Dia, Dense),
    ]


class TestMatmulDag(mixin.TestMatmulDag):
    specialisations = [
        pytest.param(data.matmul_dag_data, CSR, CSR, CSR),
        pytest.param(data.matmul_dag_dense_csr_dense, Dense, CSR, Dense),
        pytest.param(data.matmul_dag_dense_dia_dense, Dense, Dia, Dense),
        pytest.param(data.matmul_dag_dense, Dense, Dense, Dense),
    ]



class TestMatmulInPlace(mixin.TestMatmulInPlace):
    specialisations = [
        pytest.param(data.matmul_csr_dense_dense, CSR, Dense, Dense, Dense),
        pytest.param(data.matmul_dense, Dense, Dense, Dense, Dense),
        pytest.param(data.matmul_dia_dense_dense, Dia, Dense, Dense, Dense),
        pytest.param(data.matmul_dense_dia_dense, Dense, Dia, Dense, Dense),
    ]

    @pytest.mark.filterwarnings(
        "ignore::qutip.core.data.dense.OrderEfficiencyWarning"
    )
    def test_mathematically_correct(
        self, op, data_l, data_r, data_out,
        out_type, random_generator, kw
    ):
        super().test_mathematically_correct(
            op, data_l, data_r, data_out,
            out_type, random_generator, kw
        )


class TestMatmulDagInPlace(mixin.TestMatmulDagInPlace):
    specialisations = [
        pytest.param(data.matmul_dag_dense_csr_dense, Dense, CSR, Dense, Dense),
        pytest.param(data.matmul_dag_dense_dia_dense, Dense, Dia, Dense, Dense),
        pytest.param(data.matmul_dag_dense, Dense, Dense, Dense, Dense),
    ]

    @pytest.mark.filterwarnings(
        "ignore::qutip.core.data.dense.OrderEfficiencyWarning"
    )
    def test_mathematically_correct(
        self, op, data_l, data_r, data_out,
        out_type, random_generator, kw
    ):
        super().test_mathematically_correct(
            op, data_l, data_r, data_out,
            out_type, random_generator, kw
        )


class TestMultiply(mixin.TestMultiply):
    specialisations = [
        pytest.param(data.multiply_csr, CSR, CSR, CSR),
        pytest.param(data.multiply_dense, Dense, Dense, Dense),
        pytest.param(data.multiply_dia, Dia, Dia, Dia),
    ]


class TestMatmul_Outer(mixin.TestMatmul_Outer):
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


class TestMul(mixin.TestMul):
    specialisations = [
        pytest.param(data.mul_csr, CSR, CSR),
        pytest.param(data.mul_dense, Dense, Dense),
        pytest.param(data.mul_dia, Dia, Dia),
    ]


class TestNeg(mixin.TestNeg):
    specialisations = [
        pytest.param(data.neg_csr, CSR, CSR),
        pytest.param(data.neg_dense, Dense, Dense),
        pytest.param(data.neg_dia, Dia, Dia),
    ]


class TestSub(mixin.TestSub):
    specialisations = [
        pytest.param(data.sub_csr, CSR, CSR, CSR),
        pytest.param(data.sub_dense, Dense, Dense, Dense),
        pytest.param(data.sub_dia, Dia, Dia, Dia),
    ]


class TestTrace(mixin.TestTrace):
    specialisations = [
        pytest.param(data.trace_csr, CSR, complex),
        pytest.param(data.trace_dense, Dense, complex),
        pytest.param(data.trace_dia, Dia, complex),
    ]


class TestTrace_oper_ket(mixin.TestTrace_oper_ket):
    specialisations = [
        pytest.param(data.trace_oper_ket_csr, CSR, complex),
        pytest.param(data.trace_oper_ket_dense, Dense, complex),
        pytest.param(data.trace_oper_ket_dia, Dia, complex),
        pytest.param(data.trace_oper_ket_data, CSR, complex),
        pytest.param(data.trace_oper_ket_data, Dense, complex),
    ]


class TestPow(mixin.TestPow):
    specialisations = [
        pytest.param(data.pow_csr, CSR, CSR),
        pytest.param(data.pow_dense, Dense, Dense),
        pytest.param(data.pow_dia, Dia, Dia),
    ]


# Scipy complain went creating full dia matrix.
@pytest.mark.filterwarnings("ignore:Constructing a DIA matrix")
class TestExpm(mixin.TestExpm):
    specialisations = [
        pytest.param(data.expm_csr, CSR, CSR),
        pytest.param(data.expm_csr_dense, CSR, Dense),
        pytest.param(data.expm_dense, Dense, Dense),
        pytest.param(data.expm_dia, Dia, Dia),
    ]


class TestLogm(mixin.TestLogm):
    specialisations = [
        pytest.param(data.logm_dense, Dense, Dense),
    ]


class TestSqrtm(mixin.TestSqrtm):
    specialisations = [
        pytest.param(data.sqrtm_dense, Dense, Dense),
    ]


class TestTranspose(mixin.TestTranspose):
    specialisations = [
        pytest.param(data.transpose_csr, CSR, CSR),
        pytest.param(data.transpose_dense, Dense, Dense),
        pytest.param(data.transpose_dia, Dia, Dia),
    ]


class TestProject(mixin.TestProject):
    specialisations = [
        pytest.param(data.project_csr, CSR, CSR),
        pytest.param(data.project_dia, Dia, Dia),
        pytest.param(data.project_dense_data, Dense, Data),
    ]


def _non_singular(factory):
    def wrapped(*args):
        mat = factory(*args)
        return mat + data.identity_like(mat) * 2

    return wrapped


_non_singular_csr = _non_singular(random_data.random_csr)
_non_singular_dense = _non_singular(random_data.random_dense)


class TestInv(mixin.TestInv):
    specialisations = [
        pytest.param(data.inv_csr, CSR, CSR),
        pytest.param(data.inv_dense, Dense, Dense),
    ]

    correct_cases = {
        CSR:
            lambda shape: [
                pytest.param(
                    lambda gen: _non_singular_csr(shape, 0.8, True, gen),
                    id="Sorted",
                ),
                pytest.param(
                    lambda gen: _non_singular_csr(shape, 0.8, False, gen),
                    id="Unsorted",
                ),
            ],
        Dense:
            lambda shape: [
                pytest.param(
                    lambda gen: _non_singular_dense(shape, True, gen),
                    id="Fortran"
                ),
                pytest.param(
                    lambda gen: _non_singular_dense(shape, False, gen),
                    id="C"
                ),
            ],
    }
    wrong_cases = {
        CSR: lambda shape: [
            lambda gen: _non_singular_csr(shape, 0.5, True, gen)
        ],
        Dense: lambda shape: [
            lambda gen: _non_singular_dense(shape, False, gen)
        ],
    }


class TestZeros_like(mixin.TestZeros_like):
    specialisations = [
        pytest.param(data.zeros_like_data, CSR, CSR),
        pytest.param(data.zeros_like_dense, Dense, Dense),
    ]


class TestIdentity_like(mixin.TestIdentity_like):
    specialisations = [
        pytest.param(data.identity_like_data, CSR, CSR),
        pytest.param(data.identity_like_dense, Dense, Dense),
    ]


class TestWRMN_error(mixin.TestWRMN_error):
    specialisations = [
        pytest.param(data.ode.wrmn_error_csr, CSR, CSR, float),
        pytest.param(data.ode.wrmn_error_dense, Dense, Dense, float),
        pytest.param(data.ode.wrmn_error_dia, Dia, Dia, float),
    ]


class TestPtrace(mixin.TestPtrace):
    specialisations = [
        pytest.param(data.ptrace_csr, CSR, CSR),
        pytest.param(data.ptrace_csr_dense, CSR, Dense),
        pytest.param(data.ptrace_dense, Dense, Dense),
        pytest.param(data.ptrace_dia, Dia, Dia),
    ]
