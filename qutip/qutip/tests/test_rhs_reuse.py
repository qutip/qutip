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
from numpy.testing import assert_, assert_equal, run_module_suite
import qutip as qt
from qutip.solver import config

"""
def test_rhs_reuse():
    "" "
    rhs_reuse : pyx filenames match for rhs_reus= True
    "" "
    N = 10
    a = qt.destroy(N)
    H = [a.dag()*a, [a+a.dag(), 'sin(t)']]
    psi0 = qt.fock(N,3)
    tlist = np.linspace(0,10,10)
    e_ops = [a.dag()*a]
    c_ops = [0.25*a]

    # Test sesolve
    out1 = qt.mesolve(H, psi0,tlist, e_ops=e_ops)

    _temp_config_name = config.tdname

    out2 = qt.mesolve(H, psi0,tlist, e_ops=e_ops)

    assert_(config.tdname != _temp_config_name)
    _temp_config_name = config.tdname

    out3 = qt.mesolve(H, psi0,tlist, e_ops=e_ops,
                        options=qt.Options(rhs_reuse=True))

    assert_(config.tdname == _temp_config_name)

    # Test mesolve

    out1 = qt.mesolve(H, psi0,tlist, c_ops=c_ops, e_ops=e_ops)

    _temp_config_name = config.tdname

    out2 = qt.mesolve(H, psi0,tlist, c_ops=c_ops, e_ops=e_ops)

    assert_(config.tdname != _temp_config_name)
    _temp_config_name = config.tdname

    out3 = qt.mesolve(H, psi0,tlist, e_ops=e_ops, c_ops=c_ops,
                        options=qt.Options(rhs_reuse=True))

    assert_(config.tdname == _temp_config_name)

if __name__ == "__main__":
    run_module_suite()
"""
