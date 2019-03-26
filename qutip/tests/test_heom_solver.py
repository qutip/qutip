# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford, Neill Lambert
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
Tests for main control.pulseoptim methods
"""

from __future__ import division

import os
import numpy as np
from numpy import pi, real, cos, tanh
from numpy.testing import (
    assert_, assert_almost_equal, run_module_suite, assert_equal,
    assert_array_almost_equal, assert_raises)
from scipy.integrate import quad, IntegrationWarning
from qutip import Qobj, sigmaz, basis, expect
from qutip.nonmarkov.heom import (HSolverDL, Heom,
                                  HSolverUB)
from qutip.solver import Options
import warnings
from qutip.operators import sigmax, sigmaz


warnings.simplefilter('ignore', IntegrationWarning)
    
class TestHSolver:
    """
    A test class for the hierarchy model solver
    """
    
    def test_pure_dephasing(self):
        """
        HSolverDL: Compare with pure-dephasing analytical
        assert that the analytical result and HEOM produce the 
        same time dephasing evoltion.
        """
        resid_tol = 1e-4
        
        def spectral_density(omega, lam_c, omega_c):
            return 2.0*lam_c*omega*omega_c / (omega_c**2 + omega**2)
    
        def integrand(omega, lam_c, omega_c, Temp, t):
            J = spectral_density(omega, lam_c, omega_c)
            return (-4.0*J*(1.0 - cos(omega*t)) / 
                        (tanh(omega/(2.0*Temp))*omega**2))
                        
                        
        cut_freq = 0.05 
        coup_strength = 0.025
        temperature = 1.0/0.95
        tlist = np.linspace(0, 10, 21)
        
        # Calculate the analytical results by numerical integration
        lam_c = coup_strength/pi
        PEG_DL = [0.5*np.exp(quad(integrand, 0, np.inf, 
                              args=(lam_c, cut_freq, temperature, t))[0])
                              for t in tlist]
        
      
        H_sys = Qobj(np.zeros((2, 2)))
        Q = sigmaz()
        initial_state = 0.5*Qobj(np.ones((2, 2)))
        P12p = basis(2,0)*basis(2,1).dag()

        integ_options = Options(nsteps=15000, store_states=True)
        
        test_desc = "renorm, bnd_cut_approx, and stats"
        hsolver = HSolverDL(H_sys, Q, coup_strength, temperature, 
                            20, 2, cut_freq, 
                         renorm=True, bnd_cut_approx=True, 
                         options=integ_options, stats=True)
        
        result = hsolver.run(initial_state, tlist)
        P12_result1 = expect(result.states, P12p)
        resid = abs(real(P12_result1 - PEG_DL))
        max_resid = max(resid)
        assert_(max_resid < resid_tol, "Max residual {} outside tolerence {}, "
                "for hsolve with {}".format(max_resid, resid_tol, test_desc))
        
        resid_tol = 1e-3
        test_desc = "renorm"
        hsolver.configure(H_sys, Q, coup_strength, temperature, 
                            20, 2, cut_freq, 
                         renorm=True, bnd_cut_approx=False, 
                         options=integ_options, stats=False)
        assert_(hsolver.stats == None, "Failed to unset stats")
        result = hsolver.run(initial_state, tlist)
        P12_result1 = expect(result.states, P12p)
        resid = abs(real(P12_result1 - PEG_DL))
        max_resid = max(resid)
        assert_(max_resid < resid_tol, "Max residual {} outside tolerence {}, "
                "for hsolve with {}".format(max_resid, resid_tol, test_desc))
        
        resid_tol = 1e-4
        test_desc = "bnd_cut_approx"
        hsolver.configure(H_sys, Q, coup_strength, temperature, 
                            20, 2, cut_freq, 
                         renorm=False, bnd_cut_approx=True, 
                         options=integ_options, stats=False)
        assert_(hsolver.stats == None, "Failed to unset stats")
        result = hsolver.run(initial_state, tlist)
        P12_result1 = expect(result.states, P12p)
        resid = abs(real(P12_result1 - PEG_DL))
        max_resid = max(resid)
        assert_(max_resid < resid_tol, "Max residual {} outside tolerence {}, "
                "for hsolve with {}".format(max_resid, resid_tol, test_desc))
        

def test_heom():
    """
    Test the HEOM method.
    """
    Q = sigmax()
    wq = 1.
    wrc = 1.
    w0 = 1.
    temp = 0.
    beta = np.inf
    cut_freq = 0.5

    Hsys = 0.5*wq*sigmaz()
    initial_ket = basis(2, 1)
    Nc = 9
    rho0 = initial_ket*initial_ket.dag()
    Nexp = 2
    lam, gamma, w0 = 0.2, 0.05, 1.
    omega = np.sqrt(w0**2 - (gamma/2.)**2)
    a = omega + 1j*gamma/2.
    wcav = omega

    lam_coeff = lam**2/(2*(omega))

    tlist = np.linspace(0, 200, 1000)

    options = Options(nsteps=1500, store_states=True, atol=1e-12, rtol=1e-12)
    solver1 = HSolverUB(Hsys, Q, lam, temp, Nc, Nexp,
                         cut_freq, cav_freq = w0, cav_broad = gamma,
                         tlist=tlist, options=options)

    ck1, vk1 = solver1._calc_nonmatsubara_params(lam, gamma, w0, beta)
    ck2, vk2 = solver1._calc_matsubara_params(lam, gamma, w0, beta, Nexp, tlist)

    solver2 = Heom(Hsys, Q, lam_coeff, np.concatenate([ck1, ck2]),
        np.concatenate([vk1, vk2]),
        ncut=Nc)

    output1 = solver1.run(rho0, tlist)
    result1 = np.real(expect(output1.states, sigmaz()))
    
    output2 = solver2.run(rho0, tlist)
    result2 = np.real(expect(output2.states, sigmaz()))
    max_resid = np.max(np.abs((result1+1)/2. - (result2+1)/2))

    resid_tol = 1e-2
    assert_(max_resid < resid_tol)


if __name__ == "__main__":
    run_module_suite()
