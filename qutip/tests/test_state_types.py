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
from qutip import (basis, bell_state, bra, ket, coherent, coherent_dm, fock,
                   fock_dm, ghz_state, w_state, ket2dm, maximally_mixed_dm,
                   thermal_dm, spin_state, spin_coherent)
from numpy.testing import assert_equal, run_module_suite
from scipy.sparse import isspmatrix_csr


def test_basis_type():
    "State CSR Type: basis"
    st = basis(5, 1)
    assert_equal(isspmatrix_csr(st.data), True)


def test_bell_type():
    "State CSR Type: bell_state"
    for k in ['00', '01', '10', '11']:
        st = bell_state(k)
        assert_equal(isspmatrix_csr(st.data), True)


def test_bra_type():
    "State CSR Type: bra"
    st = bra('10')
    assert_equal(isspmatrix_csr(st.data), True)


def test_coherent_type():
    "State CSR Type: coherent"
    st = coherent(25, 2+2j)
    assert_equal(isspmatrix_csr(st.data), True)
    st = coherent(25, 2+2j, method='analytic')
    assert_equal(isspmatrix_csr(st.data), True)


def test_coherentdm_type():
    "State CSR Type: coherent_dm"
    st = coherent_dm(25, 2+2j)
    assert_equal(isspmatrix_csr(st.data), True)


def test_fock_type():
    "State CSR Type: fock"
    st = fock(5, 1)
    assert_equal(isspmatrix_csr(st.data), True)


def test_fockdm_type():
    "State CSR Type: fock_dm"
    st = fock_dm(5, 3)
    assert_equal(isspmatrix_csr(st.data), True)


def test_ghz_type():
    "State CSR Type: ghz_state"
    st = ghz_state(3)
    assert_equal(isspmatrix_csr(st.data), True)


def test_ket_type():
    "State CSR Type: ket"
    st = ket('10')
    assert_equal(isspmatrix_csr(st.data), True)


def test_ket2dm_type():
    "State CSR Type: ket2dm"
    st = ket2dm(basis(5, 1))
    assert_equal(isspmatrix_csr(st.data), True)


def test_maxmixed_type():
    "State CSR Type: maximall_mixed_dm"
    st = maximally_mixed_dm(10)
    assert_equal(isspmatrix_csr(st.data), True)


def test_spincoherent_type():
    "State CSR Type: spin_coherent"
    st = spin_coherent(5, np.pi/4, np.pi/4)
    assert_equal(isspmatrix_csr(st.data), True)


def test_spinstate_type():
    "State CSR Type: spin_state"
    st = spin_state(5, 3)
    assert_equal(isspmatrix_csr(st.data), True)


def test_thermal_type():
    "State CSR Type: thermal_dm"
    st = thermal_dm(25, 5)
    assert_equal(isspmatrix_csr(st.data), True)
    st = thermal_dm(25, 5, method='analytic')
    assert_equal(isspmatrix_csr(st.data), True)


def test_wstate_type():
    "State CSR Type: w_state"
    st = w_state(3)
    assert_equal(isspmatrix_csr(st.data), True)

if __name__ == "__main__":
    run_module_suite()
