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
import qutip


def test_dqd_current():
    "Counting statistics: current and current noise in a DQD model"

    G = 0
    L = 1
    R = 2

    sz = qutip.projection(3, L, L) - qutip.projection(3, R, R)
    sx = qutip.projection(3, L, R) + qutip.projection(3, R, L)
    sR = qutip.projection(3, G, R)
    sL = qutip.projection(3, G, L)

    w0 = 1
    tc = 0.6 * w0
    GammaR = 0.0075 * w0
    GammaL = 0.0075 * w0
    nth = 0.00
    eps_vec = np.linspace(-1.5*w0, 1.5*w0, 20)

    J_ops = [GammaR * qutip.sprepost(sR, sR.dag())]

    c_ops = [np.sqrt(GammaR * (1 + nth)) * sR,
             np.sqrt(GammaR * (nth)) * sR.dag(),
             np.sqrt(GammaL * (nth)) * sL,
             np.sqrt(GammaL * (1 + nth)) * sL.dag()]

    current = np.zeros(len(eps_vec))
    noise = np.zeros(len(eps_vec))

    for n, eps in enumerate(eps_vec):
        H = (eps/2 * sz + tc * sx)
        L = qutip.liouvillian(H, c_ops)
        rhoss = qutip.steadystate(L)
        current[n], noise[n] = qutip.countstat_current_noise(L, [],
                                                             rhoss=rhoss,
                                                             J_ops=J_ops)

        current2 = qutip.countstat_current(L, rhoss=rhoss, J_ops=J_ops)
        assert abs(current[n] - current2) < 1e-8

        current2 = qutip.countstat_current(L, c_ops, J_ops=J_ops)
        assert abs(current[n] - current2) < 1e-8

    current_target = (tc**2 * GammaR
                      / (tc**2 * (2+GammaR/GammaL) + GammaR**2/4 + eps_vec**2))
    noise_target = current_target * (
        1 - (8*GammaL*tc**2*(4 * eps_vec**2 * (GammaR - GammaL)
                             + GammaR*(3*GammaL*GammaR + GammaR**2 + 8*tc**2))
             / (4*tc**2*(2*GammaL + GammaR) + GammaL*GammaR**2
                + 4*eps_vec**2*GammaL)**2)
    )

    np.testing.assert_allclose(current, current_target, atol=1e-4)
    np.testing.assert_allclose(noise, noise_target, atol=1e-4)
