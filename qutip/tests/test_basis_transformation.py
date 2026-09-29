import numpy as np
import pytest
import qutip


def real_hermitian(n_levels, random_generator):
    qobj = qutip.Qobj(0.5 - random_generator.random((n_levels, n_levels)))
    return qobj + qobj.dag()


def imaginary_hermitian(n_levels, random_generator):
    qobj = qutip.Qobj(1j*(0.5 - random_generator.random((n_levels, n_levels))))
    return qobj + qobj.dag()


def complex_hermitian(n_levels, random_generator):
    return (real_hermitian(n_levels, random_generator)
            + imaginary_hermitian(n_levels, random_generator))


def rand_bra(n_levels, random_generator):
    return qutip.rand_ket(n_levels, seed=random_generator).dag()


@pytest.mark.parametrize("hermitian_constructor", [real_hermitian,
                                                   imaginary_hermitian,
                                                   complex_hermitian])
@pytest.mark.parametrize("n_levels", [2, 10])
def test_transformation_to_eigenbasis_is_reversible(hermitian_constructor,
                                                    n_levels, random_generator):
    """Transform n-level real-values to eigenbasis and back"""
    H1 = hermitian_constructor(n_levels, random_generator)
    _, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)  # In the eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # Back to original basis
    assert (H1 - H2).norm() < 1e-6


@pytest.mark.parametrize("n_levels", [4])
def test_ket_and_dm_transformations_equivalent(n_levels, random_generator):
    """Consistency between transformations of kets and density matrices."""
    psi0 = qutip.rand_ket(n_levels, seed=random_generator)
    # Generate a random basis
    _, rand_basis = qutip.rand_dm(
        n_levels, density=1, seed=random_generator
    ).eigenstates()
    rho1 = qutip.ket2dm(psi0).transform(rand_basis, True)
    rho2 = qutip.ket2dm(psi0.transform(rand_basis, True))
    assert (rho1 - rho2).norm() < 1e-6


def test_eigenbasis_transformation_makes_diagonal_operator(random_generator):
    """Check diagonalization via eigenbasis transformation."""
    cx, cy, cz = random_generator.random(3)
    H = cx*qutip.sigmax() + cy*qutip.sigmay() + cz*qutip.sigmaz()
    _, ekets = H.eigenstates()
    Heb = H.transform(ekets).tidyup()  # Heb should be diagonal
    assert abs(Heb.full() - np.diag(Heb.full().diagonal())).max() < 1e-6
