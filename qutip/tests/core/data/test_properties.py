import numpy as np
import pytest

from qutip import data as _data
from qutip import CoreOptions

@pytest.fixture(params=[_data.CSR, _data.Dense, _data.Dia], ids=["CSR", "Dense", "Dia"])
def datatype(request):
    return request.param


class Test_isherm:
    tol = 1e-12

    @pytest.mark.repeat(20)
    @pytest.mark.parametrize("density", (0.1, 0.8))
    @pytest.mark.parametrize("size", (10, 100))
    def test_random_equal_structure(self, datatype, size, density):
        # This is only going to be approximate, as entries generated onto the
        # diagonal will fill only one slot, not two, after addition with the
        # transpose.  The densities are arbitrary, though, so this doesn't
        # matter much.
        nnz = int(0.5 * size*size * density) or 1
        indices = np.triu_indices(size)
        choice = np.random.choice(indices[0].size, nnz, replace=False)
        indices = (indices[0][choice], indices[1][choice])

        # Real-symmetric matrices.
        base = np.zeros((size, size), dtype=np.complex128)
        base[indices] = np.random.rand(nnz)
        base += base.T.conj()
        base = _data.to(datatype, _data.create(base))
        assert _data.isherm(base)

        # Complex Hermitian matrices
        base = np.zeros((size, size), dtype=np.complex128)
        base[indices] = np.random.rand(nnz) + 1j*np.random.rand(nnz)
        base += base.T.conj()
        base = _data.to(datatype, _data.create(base))
        assert _data.isherm(base)

        # Complex skew-Hermitian matrices
        base = np.zeros((size, size), dtype=np.complex128)
        base[indices] = np.random.rand(nnz) + 1j*np.random.rand(nnz)
        base += base.T
        base = _data.to(datatype, _data.create(base))
        assert not _data.isherm(base)

    @pytest.mark.parametrize("cols", (2, 4))
    @pytest.mark.parametrize("rows", (1, 5))
    def test_nonsquare_shapes(self, datatype, rows, cols):
        real = _data.to(datatype, _data.create(np.random.rand(rows, cols)))
        assert not _data.isherm(real, self.tol)
        assert not _data.isherm(real.transpose(), self.tol)

        imag = _data.to(
            datatype,
            _data.create(np.random.rand(rows, cols) + 1j * np.random.rand(rows, cols)),
        )
        assert not _data.isherm(imag, self.tol)
        assert not _data.isherm(imag.transpose(), self.tol)

    def test_diagonal_elements(self, datatype):
        n = 10
        base = _data.to(datatype, _data.create(np.diag(np.random.rand(n))))
        assert _data.isherm(base, tol=self.tol)
        assert not _data.isherm(_data.mul(base, 1j), tol=self.tol)

    def test_compare_implicit_zero_structure(self, datatype):
        """
        Regression test for gh-1350, comparing explicitly stored values in the
        matrix (but below the tolerance for allowable Hermicity) to implicit
        zeros.
        """

        with CoreOptions(auto_tidyup=False):
            base = _data.to(
                datatype,
                _data.create(np.array([[1, self.tol * 1e-3j], [0, 1]]))
            )
        # If this first line fails, the zero has been stored explicitly and so
        # the test is invalid.
        assert np.count_nonzero(base.to_array()) == 3
        assert _data.isherm(base, tol=self.tol)
        assert _data.isherm(base.transpose(), tol=self.tol)

        # A similar test if the structures are different, but it's not
        # Hermitian.
        base = _data.to(datatype, _data.create(np.array([[1, 1j], [0, 1]])))
        assert np.count_nonzero(base.to_array()) == 3
        assert not _data.isherm(base, tol=self.tol)
        assert not _data.isherm(base.transpose(), tol=self.tol)

        # Catch possible edge case where it shouldn't be Hermitian, but faulty
        # loop logic doesn't fully compare all rows.
        base = _data.to(
            datatype,
            _data.create(
                np.array(
                    [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.complex128,
                )
            ),
        )
        assert np.count_nonzero(base.to_array()) == 1
        assert not _data.isherm(base, tol=self.tol)
        assert not _data.isherm(base.transpose(), tol=self.tol)

    @pytest.mark.parametrize("density", np.linspace(0.2, 1, 17))
    def test_compare_implicit_zero_random(self, datatype, density):
        """
        Regression test of gh-1350.

        Larger matrices where all off-diagonal elements are below the
        absolute tolerance, so everything should always appear Hermitian, but
        with random patterns of non-zero elements.  It doesn't matter that it
        isn't Hermitian if scaled up; everything is below absolute tolerance,
        so it should appear so.  We also set the diagonal to be larger to the
        tolerance to ensure isherm can't just compare everything to zero.
        """
        n = 10
        base = self.tol * 1e-2 * (np.random.rand(n, n) + 1j * np.random.rand(n, n))
        # Mask some values out to zero.
        base[np.random.rand(n, n) > density] = 0
        np.fill_diagonal(base, self.tol * 1000)
        nnz = np.count_nonzero(base)
        with CoreOptions(auto_tidyup=False):
            base = _data.to(datatype, _data.create(base))
        assert np.count_nonzero(base.to_array()) == nnz
        assert _data.isherm(base, tol=self.tol)
        assert _data.isherm(base.transpose(), tol=self.tol)

        # Similar test when it must be non-Hermitian.  We set the diagonal
        # to be real because we want to test off-diagonal implicit zeros,
        # and having an imaginary first element would automatically fail.
        nnz = 0
        while nnz <= n:
            # Ensure that we don't just have the real diagonal.
            base = self.tol * 1000j * np.random.rand(n, n)
            # Mask some values out to zero.
            base[np.random.rand(n, n) > density] = 0
            np.fill_diagonal(base, self.tol * 1000)
            nnz = np.count_nonzero(base)
        with CoreOptions(auto_tidyup=False):
            base = _data.to(datatype, _data.create(base))
        assert np.count_nonzero(base.to_array()) == nnz
        assert not _data.isherm(base, tol=self.tol)
        assert not _data.isherm(base.transpose(), tol=self.tol)

    def test_structure_detection(self, datatype):
        base = np.array([[1,1,0],
                         [0,1,1],
                         [1,0,1]])
        base = _data.to(datatype, _data.create(base))
        assert not _data.isherm(base, tol=self.tol)


class Test_isdiag:
    @pytest.mark.parametrize("shape",
        [(10, 1), (2, 5), (5, 2), (5, 5)]
    )
    def test_isdiag(self, shape, datatype):
        mat = np.zeros(shape)
        data = _data.to(datatype, _data.Dense(mat))
        # empty matrices are diagonal
        assert _data.isdiag(data)

        mat[0, 0] = 1
        data = _data.to(datatype, _data.Dense(mat))
        assert _data.isdiag(data)

        mat[1, 0] = 1
        data = _data.to(datatype, _data.Dense(mat))
        assert not _data.isdiag(data)
