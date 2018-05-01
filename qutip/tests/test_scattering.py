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

"""
This module includes a collection of testing functions for the QuTiP scattering
module. Tests are approximate with low resolution to minimize runtime.
"""

# Author:  Ben Bartlett
# Contact: benbartlett@stanford.edu

import numpy as np
from numpy.testing import assert_, run_module_suite
from qutip.operators import create, destroy
from qutip.states import basis
from qutip.scattering import *


class TestScattering:
    """
    A test class for the QuTiP quantum optical scattering module. These tests
    only use the two-level system for comparison, since larger systems can
    take a long time to run.
    """

    def testScatteringProbability(self):
        """
        Asserts that pi pulse in TLS has P0 ~ 0 and P0+P1+P2 ~ 1
        """
        w0 = 1.0 * 2 * np.pi
        gamma = 1.0
        sm = np.sqrt(gamma) * destroy(2)
        pulseArea = np.pi
        pulseLength = 0.2 / gamma
        RabiFreq = pulseArea / (2 * pulseLength)
        psi0 = basis(2, 0)
        tlist = np.geomspace(gamma, 10 * gamma, 40) - gamma
        # Define TLS Hamiltonian
        H0S = w0 * create(2) * destroy(2)
        H1S1 = lambda t, args: \
            RabiFreq * 1j * np.exp(-1j * w0 * t) * (t < pulseLength)
        H1S2 = lambda t, args: \
            RabiFreq * -1j * np.exp(1j * w0 * t) * (t < pulseLength)
        Htls = [H0S, [sm.dag(), H1S1], [sm, H1S2]]
        # Run the test
        P0 = scattering_probability(Htls, psi0, 0, [sm], tlist)
        P1 = scattering_probability(Htls, psi0, 1, [sm], tlist)
        P2 = scattering_probability(Htls, psi0, 2, [sm], tlist)
        assert_(P0 < 1e-3)
        assert_(np.abs(P0 + P1 + P2 - 1) < 1e-3)

    def testScatteringAmplitude(self):
        """
        Asserts that a 2pi pulse in TLS has ~0 amplitude after pulse
        """
        w0 = 1.0 * 2 * np.pi
        gamma = 1.0
        sm = np.sqrt(gamma) * destroy(2)
        pulseArea = 2 * np.pi
        pulseLength = 0.2 / gamma
        RabiFreq = pulseArea / (2 * pulseLength)
        psi0 = basis(2, 0)
        T = 50
        tlist = np.linspace(0, 1 / gamma, T)
        # Define TLS Hamiltonian
        H0S = w0 * create(2) * destroy(2)
        H1S1 = lambda t, args: \
            RabiFreq * 1j * np.exp(-1j * w0 * t) * (t < pulseLength)
        H1S2 = lambda t, args: \
            RabiFreq * -1j * np.exp(1j * w0 * t) * (t < pulseLength)
        Htls = [H0S, [sm.dag(), H1S1], [sm, H1S2]]
        # Run the test
        state = temporal_scattered_state(Htls, psi0, 1, [sm], tlist)
        basisVec = temporal_basis_vector([[40]], T)
        amplitude = np.abs((basisVec.dag() * state).full().item())
        assert_(amplitude < 1e-3)

    def testWaveguideSplit(self):
        """
        Checks that a trivial splitting of a waveguide collapse operator like
        [sm] -> [sm/sqrt2, sm/sqrt2] doesn't affect the normalization or result
        """
        gamma = 1.0
        sm = np.sqrt(gamma) * destroy(2)
        pulseArea = np.pi
        pulseLength = 0.2 / gamma
        RabiFreq = pulseArea / (2 * pulseLength)
        psi0 = basis(2, 0)
        tlist = np.geomspace(gamma, 10 * gamma, 40) - gamma
        # Define TLS Hamiltonian with rotating frame transformation
        Htls = [[sm.dag() + sm, lambda t, args: RabiFreq * (t < pulseLength)]]
        # Run the test
        c_ops = [sm]
        c_ops_split = [sm / np.sqrt(2), sm / np.sqrt(2)]
        P1 = scattering_probability(Htls, psi0, 1, c_ops, tlist)
        P1_split = scattering_probability(Htls, psi0, 1, c_ops_split, tlist)
        tolerance = 1e-7
        assert_(1 - tolerance < P1 / P1_split < 1 + tolerance)


if __name__ == "__main__":
    run_module_suite()

