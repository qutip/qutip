import numpy as np
from numpy.testing import (assert_, assert_equal, assert_almost_equal)
import scipy.sparse as sp

from qutip.fastsparse import fast_csr_matrix, fast_identity
from qutip.random_objects import (rand_dm, rand_herm,
                                  rand_ket, rand_unitary)
from qutip.cy.spmath import (zcsr_kron, zcsr_transpose, zcsr_adjoint,
                            zcsr_isherm)


def test_csr_kron():
    "spmath: zcsr_kron"
    num_test = 5
    for _ in range(num_test):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_herm(ra,0.5).data
        B = rand_herm(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)

    for _ in range(num_test):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = rand_herm(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)

    for _ in range(num_test):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_dm(ra,0.5).data
        B = rand_herm(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)

    for _ in range(num_test):
        ra = np.random.randint(2,100)
        rb = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = rand_ket(rb,0.5).data
        As = A.tocsr(1)
        Bs = B.tocsr(1)
        C = sp.kron(As,Bs, format='csr')
        D = zcsr_kron(A, B)
        assert_almost_equal(C.data, D.data)
        assert_equal(C.indices, D.indices)
        assert_equal(C.indptr, D.indptr)


def test_zcsr_transpose():
    "spmath: zcsr_transpose"
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)

    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_herm(5,1.0/ra).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)

    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_dm(ra,1.0/ra).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)

    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_unitary(ra,1.0/ra).data
        B = A.T.tocsr()
        C = A.trans()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)


def test_zcsr_adjoint():
    "spmath: zcsr_adjoint"
    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_ket(ra,0.5).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)

    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_herm(5,1.0/ra).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)

    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_dm(ra,1.0/ra).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)

    for k in range(50):
        ra = np.random.randint(2,100)
        A = rand_unitary(ra,1.0/ra).data
        B = A.conj().T.tocsr()
        C = A.adjoint()
        x = np.all(B.data == C.data)
        y = np.all(B.indices == C.indices)
        z = np.all(B.indptr == C.indptr)
        assert_(x*y*z)


def test_zcsr_mult():
    "spmath: zcsr_mult"
    for k in range(50):
        A = rand_ket(10,0.5).data
        B = rand_herm(10,0.5).data

        C = A.tocsr(1)
        D = B.tocsr(1)

        ans1 = B*A
        ans2 = D*C
        ans2.sort_indices()
        x = np.all(ans1.data == ans2.data)
        y = np.all(ans1.indices == ans2.indices)
        z = np.all(ans1.indptr == ans2.indptr)
        assert_(x*y*z)

    for k in range(50):
        A = rand_ket(10,0.5).data
        B = rand_ket(10,0.5).dag().data

        C = A.tocsr(1)
        D = B.tocsr(1)

        ans1 = B*A
        ans2 = D*C
        ans2.sort_indices()
        x = np.all(ans1.data == ans2.data)
        y = np.all(ans1.indices == ans2.indices)
        z = np.all(ans1.indptr == ans2.indptr)
        assert_(x*y*z)

        ans1 = A*B
        ans2 = C*D
        ans2.sort_indices()
        x = np.all(ans1.data == ans2.data)
        y = np.all(ans1.indices == ans2.indices)
        z = np.all(ans1.indptr == ans2.indptr)
        assert_(x*y*z)

    for k in range(50):
        A = rand_dm(10,0.5).data
        B = rand_dm(10,0.5).data

        C = A.tocsr(1)
        D = B.tocsr(1)

        ans1 = B*A
        ans2 = D*C
        ans2.sort_indices()
        x = np.all(ans1.data == ans2.data)
        y = np.all(ans1.indices == ans2.indices)
        z = np.all(ans1.indptr == ans2.indptr)
        assert_(x*y*z)

    for k in range(50):
        A = rand_dm(10,0.5).data
        B = rand_herm(10,0.5).data

        C = A.tocsr(1)
        D = B.tocsr(1)

        ans1 = B*A
        ans2 = D*C
        ans2.sort_indices()
        x = np.all(ans1.data == ans2.data)
        y = np.all(ans1.indices == ans2.indices)
        z = np.all(ans1.indptr == ans2.indptr)
        assert_(x*y*z)


def test_zcsr_isherm():
    "spmath: zcsr_isherm"
    N = 100
    for kk in range(100):
        A = rand_herm(N, 0.1)
        B = rand_herm(N, 0.05) + 1j*rand_herm(N, 0.05)
        assert_(zcsr_isherm(A.data))
        assert_(zcsr_isherm(B.data)==0)


def test_zcsr_isherm_compare_implicit_zero():
    """
    Regression test for gh-1350, comparing explicitly stored values in the
    matrix (but below the tolerance for allowable Hermicity) to implicit zeros.
    """
    tol = 1e-12
    n = 10

    base = sp.csr_matrix(np.array([[1, tol * 1e-3j], [0, 1]]))
    base = fast_csr_matrix((base.data, base.indices, base.indptr), base.shape)
    # If this first line fails, the zero has been stored explicitly and so the
    # test is invalid.
    assert base.data.size == 3
    assert zcsr_isherm(base, tol=tol)
    assert zcsr_isherm(base.T, tol=tol)

    # A similar test if the structures are different, but it's not
    # Hermitian.
    base = sp.csr_matrix(np.array([[1, 1j], [0, 1]]))
    base = fast_csr_matrix((base.data, base.indices, base.indptr), base.shape)
    assert base.data.size == 3
    assert not zcsr_isherm(base, tol=tol)
    assert not zcsr_isherm(base.T, tol=tol)

    # Catch possible edge case where it shouldn't be Hermitian, but faulty loop
    # logic doesn't fully compare all rows.
    base = sp.csr_matrix(np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.complex128))
    base = fast_csr_matrix((base.data, base.indices, base.indptr), base.shape)
    assert base.data.size == 1
    assert not zcsr_isherm(base, tol=tol)
    assert not zcsr_isherm(base.T, tol=tol)

    # Pure diagonal matrix.
    base = fast_identity(n)
    base.data *= np.random.rand(n)
    assert zcsr_isherm(base, tol=tol)
    assert not zcsr_isherm(base * 1j, tol=tol)

    # Larger matrices where all off-diagonal elements are below the absolute
    # tolerance, so everything should always appear Hermitian, but with random
    # patterns of non-zero elements.  It doesn't matter that it isn't Hermitian
    # if scaled up; everything is below absolute tolerance, so it should appear
    # so.  We also set the diagonal to be larger to the tolerance to ensure
    # isherm can't just compare everything to zero.
    for density in np.linspace(0.2, 1, 21):
        base = tol * 1e-2 * (np.random.rand(n, n) + 1j*np.random.rand(n, n))
        # Mask some values out to zero.
        base[np.random.rand(n, n) > density] = 0
        np.fill_diagonal(base, tol * 1000)
        nnz = np.count_nonzero(base)
        base = sp.csr_matrix(base)
        base = fast_csr_matrix((base.data, base.indices, base.indptr), (n, n))
        assert base.data.size == nnz
        assert zcsr_isherm(base, tol=tol)
        assert zcsr_isherm(base.T, tol=tol)

        # Similar test when it must be non-Hermitian.  We set the diagonal to
        # be real because we want to test off-diagonal implicit zeros, and
        # having an imaginary first element would automatically fail.
        nnz = 0
        while nnz <= n:
            # Ensure that we don't just have the real diagonal.
            base = tol * 1000j*np.random.rand(n, n)
            # Mask some values out to zero.
            base[np.random.rand(n, n) > density] = 0
            np.fill_diagonal(base, tol * 1000)
            nnz = np.count_nonzero(base)
        base = sp.csr_matrix(base)
        base = fast_csr_matrix((base.data, base.indices, base.indptr), (n, n))
        assert base.data.size == nnz
        assert not zcsr_isherm(base, tol=tol)
        assert not zcsr_isherm(base.T, tol=tol)


def test_issue_1998():
    tol = 1e-12
    base = sp.csr_matrix(np.array([[1,1,0],
                                   [0,1,1],
                                   [1,0,1]],
                                  dtype=np.complex128))
    base = fast_csr_matrix((base.data, base.indices, base.indptr), base.shape)
    assert not zcsr_isherm(base, tol=tol)
    assert not zcsr_isherm(base.T, tol=tol)
