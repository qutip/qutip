from . import test_mathematics as testing
import numpy as np
import scipy.linalg
import pytest
from qutip import data
from qutip.core.data import CSR, Dense


class TestPtrace(testing.UnaryOpMixin):
    def op_numpy(self, matrix, dims, sel):
        ndims = len(dims)
        dkeep = [dims[x] for x in sel]
        qtrace = list(set(range(ndims)) - set(sel))
        dtrace = [dims[x] for x in qtrace]

        matrix = matrix.reshape(dims+dims)
        matrix = matrix.transpose(qtrace + [ndims + i for i in qtrace] +
                                  sel + [ndims + i for i in sel])
        matrix = matrix.reshape([np.prod(dtrace, dtype=int),
                                 np.prod(dtrace, dtype=int),
                                 np.prod(dkeep, dtype=int),
                                 np.prod(dkeep, dtype=int)])
        return np.trace(matrix)

    # Custom shapes to have also custom dims and sel arguments.
    # These values should not be changed.
    dims = [2]*7
    shapes = [(pytest.param((np.prod(dims), np.prod(dims))),)]
    # bad_shapes = testing.shapes_not_square(np.prod(dims))

    specialisations = [
        pytest.param(data.ptrace_csr, CSR, CSR),
        pytest.param(data.ptrace_csr_dense, CSR, Dense),
        pytest.param(data.ptrace_dense, Dense, Dense),
    ]

    @pytest.mark.parametrize('sel', [[0], [0, 3, 6], list(range(len(dims))), []],
                             ids=['keep_one', 'keep_multiple',
                                  'trace_none', 'trace_all'])
    def test_mathematically_correct(self, op, data_m, out_type, sel):
        """
        Test that the unary operation is mathematically correct for all the
        known type specialisations.
        """
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), self.dims, sel)
        test = op(matrix, self.dims, sel)
        assert isinstance(test, out_type)
        if issubclass(out_type, data.Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.atol, rtol=self.rtol)
        else:
            np.testing.assert_allclose(test, expected, atol=self.atol,
                                       rtol=self.rtol)


