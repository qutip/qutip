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
from numpy.testing import assert_equal, run_module_suite
from scipy.sparse import isspmatrix_csr
from qutip import (commutator, position, momentum, create, destroy, displace,
                   jmat, num, phase, qdiags, qeye, identity, qutrit_ops,
                   squeeze, squeezing, qzero, charge, tunneling)


def test_commutator_type():
    "Operator CSR Type: commutator"
    op = commutator(position(5), momentum(5))
    assert_equal(isspmatrix_csr(op.data), True)


def test_create_type():
    "Operator CSR Type: create"
    op = create(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_destroy_type():
    "Operator CSR Type: destroy"
    op = destroy(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_displace_type():
    "Operator CSR Type: displace"
    op = displace(5, 0.1)
    assert_equal(isspmatrix_csr(op.data), True)


def test_jmat_type():
    "Operator CSR Type: spin ops"
    for k in ['x', 'y', 'z', '+', '-']:
        op = jmat(1/2, k)
        assert_equal(isspmatrix_csr(op.data), True)


def test_momentum_type():
    "Operator CSR Type: momentum"
    op = momentum(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_num_type():
    "Operator CSR Type: num"
    op = num(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_phase_type():
    "Operator CSR Type: phase"
    op = phase(5, np.pi)
    assert_equal(isspmatrix_csr(op.data), True)


def test_position_type():
    "Operator CSR Type: position"
    op = position(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_qdiags_type():
    "Operator CSR Type: qdiags"
    op = qdiags(np.sqrt(range(1, 4)), 1)
    assert_equal(isspmatrix_csr(op.data), True)


def test_qeye_type():
    "Operator CSR Type: qeye/identity"
    op = qeye(5)
    assert_equal(isspmatrix_csr(op.data), True)
    op = identity(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_qtrit_type():
    "Operator CSR Type: qutrit ops"
    ops = qutrit_ops()
    for k in ops:
        assert_equal(isspmatrix_csr(k.data), True)


def test_squeeze_type():
    "Operator CSR Type: squeeze"
    op = squeeze(5, 0.1j)
    assert_equal(isspmatrix_csr(op.data), True)


def test_squeezing_type():
    "Operator CSR Type: squeezing"
    op = squeezing(destroy(5), qeye(5), -0.1j)
    assert_equal(isspmatrix_csr(op.data), True)


def test_zero_type():
    "Operator CSR Type: zero_oper"
    op = qzero(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_charge_type():
    "Operator CSR Type: charge"
    op = charge(5)
    assert_equal(isspmatrix_csr(op.data), True)


def test_tunneling_type():
    "Operator CSR Type: tunneling"
    op = tunneling(5)
    assert_equal(isspmatrix_csr(op.data), True)

if __name__ == "__main__":
    run_module_suite()
