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

from numpy.testing import assert_, run_module_suite

from qutip import (destroy, enr_destroy, identity, tensor, enr_fock,
                   enr_identity, enr_thermal_dm, thermal_dm,
                   state_number_enumerate)


def test_enr_destory_full():
    "Excitation-number-restricted state-space: full state space"
    a1, a2 = enr_destroy([4, 4], 4**2)
    b1, b2 = tensor(destroy(4), identity(4)), tensor(identity(4), destroy(4))

    assert_(a1 == b1)
    assert_(a2 == b2)


def test_enr_destory_single():
    "Excitation-number-restricted state space: single excitations"
    a1, a2 = enr_destroy([4, 4], 1)
    assert_(a1.shape == (3, 3))

    a1, a2, a3 = enr_destroy([4, 4, 4], 1)
    assert_(a1.shape == (4, 4))

    a1, a2, a3, a4 = enr_destroy([4, 4, 4, 4], 1)
    assert_(a1.shape == (5, 5))


def test_enr_destory_double():
    "Excitation-number-restricted state space: two excitations"
    a1, a2 = enr_destroy([4, 4], 2)
    assert_(a1.shape == (6, 6))

    a1, a2, a3 = enr_destroy([4, 4, 4], 2)
    assert_(a1.shape == (10, 10))

    a1, a2, a3, a4 = enr_destroy([4, 4, 4, 4], 2)
    assert_(a1.shape == (15, 15))


def test_enr_fock_state():
    "Excitation-number-restricted state space: fock states"
    dims, excitations = [4, 4], 2

    a1, a2 = enr_destroy(dims, excitations)

    psi = enr_fock(dims, excitations, [0, 2])
    assert_(abs((a1.dag()*a1).matrix_element(psi.dag(), psi) - 0) < 1e-10)
    assert_(abs((a2.dag()*a2).matrix_element(psi.dag(), psi) - 2) < 1e-10)

    psi = enr_fock(dims, excitations, [2, 0])
    assert_(abs((a1.dag()*a1).matrix_element(psi.dag(), psi) - 2) < 1e-10)
    assert_(abs((a2.dag()*a2).matrix_element(psi.dag(), psi) - 0) < 1e-10)

    psi = enr_fock(dims, excitations, [1, 1])
    assert_(abs((a1.dag()*a1).matrix_element(psi.dag(), psi) - 1) < 1e-10)
    assert_(abs((a2.dag()*a2).matrix_element(psi.dag(), psi) - 1) < 1e-10)


def test_enr_identity():
    "Excitation-number-restricted state space: identity operator"
    dims, excitations = [4, 4], 2

    i = enr_identity(dims, excitations)
    assert_((i.diag() == 1).all())
    assert_(i.dims[0] == dims)
    assert_(i.dims[1] == dims)


def test_enr_thermal_dm1():
    "Excitation-number-restricted state space: thermal density operator (I)"
    dims, excitations = [3, 4, 5, 6], 3

    n_vec = [0.01, 0.05, 0.1, 0.15]

    rho = enr_thermal_dm(dims, excitations, n_vec)

    rho_ref = tensor([thermal_dm(d, n_vec[idx])
                      for idx, d in enumerate(dims)])
    gonners = [idx for idx, state in enumerate(state_number_enumerate(dims))
               if sum(state) > excitations]
    rho_ref = rho_ref.eliminate_states(gonners)
    rho_ref = rho_ref / rho_ref.tr()

    assert_(abs((rho.data - rho_ref.data).data).max() < 1e-12)


def test_enr_thermal_dm2():
    "Excitation-number-restricted state space: thermal density operator (II)"
    dims, excitations = [3, 4, 5], 2

    n_vec = 0.1

    rho = enr_thermal_dm(dims, excitations, n_vec)

    rho_ref = tensor([thermal_dm(d, n_vec) for idx, d in enumerate(dims)])
    gonners = [idx for idx, state in enumerate(state_number_enumerate(dims))
               if sum(state) > excitations]
    rho_ref = rho_ref.eliminate_states(gonners)
    rho_ref = rho_ref / rho_ref.tr()

    assert_(abs((rho.data - rho_ref.data).data).max() < 1e-12)


if __name__ == "__main__":
    run_module_suite()
