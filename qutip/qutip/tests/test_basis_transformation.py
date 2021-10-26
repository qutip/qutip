# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
import pytest
import qutip


def real_hermitian(n_levels):
    qobj = qutip.Qobj(0.5 - np.random.random_sample((n_levels, n_levels)))
    return qobj + qobj.dag()


def imaginary_hermitian(n_levels):
    qobj = qutip.Qobj(1j*(0.5 - np.random.random_sample((n_levels, n_levels))))
    return qobj + qobj.dag()


def complex_hermitian(n_levels):
    return real_hermitian(n_levels) + imaginary_hermitian(n_levels)


def rand_bra(n_levels):
    return qutip.rand_ket(n_levels).dag()


@pytest.mark.parametrize("hermitian_constructor", [real_hermitian,
                                                   imaginary_hermitian,
                                                   complex_hermitian])
@pytest.mark.parametrize("n_levels", [2, 10])
def test_transformation_to_eigenbasis_is_reversible(hermitian_constructor,
                                                    n_levels):
    """Transform n-level real-values to eigenbasis and back"""
    H1 = hermitian_constructor(n_levels)
    _, ekets = H1.eigenstates()
    Heb = H1.transform(ekets)  # In the eigenbasis (should be diagonal)
    H2 = Heb.transform(ekets, True)  # Back to original basis
    assert (H1 - H2).norm() < 1e-6


@pytest.mark.parametrize("n_levels", [4])
def test_ket_and_dm_transformations_equivalent(n_levels):
    """Consistency between transformations of kets and density matrices."""
    psi0 = qutip.rand_ket(n_levels)
    # Generate a random basis
    _, rand_basis = qutip.rand_dm(n_levels, density=1).eigenstates()
    rho1 = qutip.ket2dm(psi0).transform(rand_basis, True)
    rho2 = qutip.ket2dm(psi0.transform(rand_basis, True))
    assert (rho1 - rho2).norm() < 1e-6


def test_eigenbasis_transformation_makes_diagonal_operator():
    """Check diagonalization via eigenbasis transformation."""
    cx, cy, cz = np.random.random_sample((3,))
    H = cx*qutip.sigmax() + cy*qutip.sigmay() + cz*qutip.sigmaz()
    _, ekets = H.eigenstates()
    Heb = H.transform(ekets).tidyup()  # Heb should be diagonal
    assert abs(Heb.full() - np.diag(Heb.full().diagonal())).max() < 1e-6


_state_constructors = [qutip.rand_herm, qutip.rand_ket, rand_bra]
@pytest.mark.parametrize("constructor", _state_constructors)
@pytest.mark.parametrize("n_levels", [2, 10])
@pytest.mark.parametrize("inverse", [True, False])
def test_transformations_from_qobj_and_direct_eigenbases_match(constructor,
                                                               n_levels,
                                                               inverse):
    """
    Check transformations from the Qobj-calculated eigenbasis matches the one
    from the scipy-calculated basis.
    """
    qobj = constructor(n_levels)
    # Generate a random basis
    random_dm = qutip.rand_dm(n_levels, density=1)
    _, qobj_basis = random_dm.eigenstates()
    _, direct_basis = qutip.sparse.sp_eigs(random_dm.data, isherm=1)
    H1 = qobj.transform(qobj_basis, inverse=inverse)
    H2 = qobj.transform(direct_basis, inverse=inverse)
    assert (H1 - H2).norm() < 1e-6
