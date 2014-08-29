# -*- coding: utf-8 -*-
"""
Simple tests for metrics and pseudometrics implemented in
the qutip.metrics module.
"""

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

from numpy import abs
from numpy.linalg import norm
from numpy.testing import assert_, run_module_suite, assert_raises
import scipy

from qutip.qobj import Qobj
from qutip.states import basis
from qutip.operators import create, destroy, jmat, identity
from qutip.propagator import propagator
from qutip.random_objects import rand_herm
from qutip.superop_reps import (to_super, to_choi, to_kraus)
from qutip.metrics import average_gate_fidelity


class TestMetrics(object):
    """
    A test class for the metrics and pseudo-metrics included with QuTiP.
    """

    def rand_super(self):
        h_5 = rand_herm(5)
        return propagator(h_5, scipy.rand(), [
            create(5), destroy(5), jmat(2, 'z')
        ])

    def test_average_gate_fidelity(self):
        """
        Metrics: Checks that average gate fidelities are sensible for random
        maps, and are equal to 1 for identity maps.
        """
        for dims in range(2, 5):
            assert_(abs(average_gate_fidelity(identity(dims)) - 1) <= 1e-12)
        assert_(0 <= average_gate_fidelity(self.rand_super()) <= 1)

if __name__ == "__main__":
    run_module_suite()
