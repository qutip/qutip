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
from numpy.testing import assert_, run_module_suite
from qutip import fsesolve, sigmax, sigmaz, rand_ket, num, mesolve
from qutip import sigmap, sigmam, floquet_master_equation_rates, expect, Qobj
from qutip import floquet_modes, floquet_modes_table, fmmesolve
from qutip import floquet_modes_t_lookup


class TestFloquet:
    """
    A test class for the QuTiP functions for Floquet formalism.
    """

    def testFloquetUnitary(self):
        """
        Floquet: test unitary evolution of time-dependent two-level system
        """

        delta = 1.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.5 * 2 * np.pi
        omega = np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = [H0, [H1, lambda t, args: np.sin(args['w'] * t)]]
        e_ops = [num(2)]

        # Solve schrodinger equation with floquet solver
        sol = fsesolve(H, psi0, tlist, e_ops, T, args)

        # Compare with results from standard schrodinger equation
        sol_ref = mesolve(H, psi0, tlist, [], e_ops, args)

        assert_(max(abs(sol.expect[0] - sol_ref.expect[0])) < 1e-4)

    def testFloquetMasterEquation1(self):
        """
        Test Floquet-Markov Master Equation for a driven two-level system
        without dissipation.
        """

        delta = 1.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.5 * 2 * np.pi
        omega = np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = [H0, [H1, lambda t, args: np.sin(args['w'] * t)]]
        e_ops = [num(2)]
        gamma1 = 0

        # Collapse operator for Floquet-Markov Master Equation
        c_op_fmmesolve = sigmax()

        # Collapse operators for Lindblad Master Equation
        def noise_spectrum(omega):
            if omega > 0:
                return 0.5 * gamma1 * omega/(2*np.pi)
            else:
                return 0

        ep, vp = H0.eigenstates()
        op0 = vp[0]*vp[0].dag()
        op1 = vp[1]*vp[1].dag()

        c_op_mesolve = []
        gamma = np.zeros([2, 2], dtype=complex)
        for i in range(2):
            for j in range(2):
                if i != j:
                    gamma[i][j] = 2*np.pi*c_op_fmmesolve.matrix_element(
                        vp[j], vp[i])*c_op_fmmesolve.matrix_element(
                        vp[i], vp[j])*noise_spectrum(ep[j]-ep[i])

        for i in range(2):
            for j in range(2):
                c_op_mesolve.append(np.sqrt(gamma[i][j])*(vp[i]*vp[j].dag()))

        # Find the floquet modes
        f_modes_0, f_energies = floquet_modes(H, T, args)

        # Precalculate mode table
        f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                              np.linspace(0, T, 500 + 1),
                                              H, T, args)

        # Solve the floquet-markov master equation
        output1 = fmmesolve(H, psi0, tlist, [c_op_fmmesolve], [],
                            [noise_spectrum], T, args, floquet_basis=True)

        # Calculate expectation values in the computational basis
        p_ex = np.zeros(np.shape(tlist), dtype=complex)
        for idx, t in enumerate(tlist):
            f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
            f_states_t = [np.exp(-1j*t*f_energies[0])*f_modes_t[0],
                          np.exp(-1j*t*f_energies[1])*f_modes_t[1]]
            p_ex[idx] = expect(num(2), output1.states[idx].transform(
                f_states_t, True))

        # Compare with mesolve
        output2 = mesolve(H, psi0, tlist, c_op_mesolve, [], args)
        p_ex_ref = expect(num(2), output2.states)

        assert_(max(abs(np.real(p_ex)-np.real(p_ex_ref))) < 1e-4)

    def testFloquetMasterEquation2(self):
        """
        Test Floquet-Markov Master Equation for a two-level system
        subject to dissipation.
        """

        delta = 1.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.5 * 2 * np.pi
        omega = np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = [H0, [H1, lambda t, args: np.sin(args['w'] * t)]]
        e_ops = [num(2)]
        gamma1 = 1

        A = 0. * 2 * np.pi
        psi0 = rand_ket(2)
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = [H0, [H1, lambda t, args: np.sin(args['w'] * t)]]

        # Collapse operator for Floquet-Markov Master Equation
        c_op_fmmesolve = sigmax()

        # Collapse operator for Lindblad Master Equation
        def noise_spectrum(omega):
            if omega > 0:
                return 0.5 * gamma1 * omega/(2*np.pi)
            else:
                return 0

        ep, vp = H0.eigenstates()
        op0 = vp[0]*vp[0].dag()
        op1 = vp[1]*vp[1].dag()

        c_op_mesolve = []
        gamma = np.zeros([2, 2], dtype=complex)
        for i in range(2):
            for j in range(2):
                if i != j:
                    gamma[i][j] = 2*np.pi*c_op_fmmesolve.matrix_element(
                        vp[j], vp[i])*c_op_fmmesolve.matrix_element(
                        vp[i], vp[j])*noise_spectrum(ep[j]-ep[i])

        for i in range(2):
            for j in range(2):
                c_op_mesolve.append(np.sqrt(gamma[i][j])*(vp[i]*vp[j].dag()))

        # Find the floquet modes
        f_modes_0, f_energies = floquet_modes(H, T, args)

        # Precalculate mode table
        f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                              np.linspace(0, T, 500 + 1),
                                              H, T, args)

        # Solve the floquet-markov master equation
        output1 = fmmesolve(
                            H, psi0, tlist, [c_op_fmmesolve], [],
                            [noise_spectrum], T, args, floquet_basis=True)
        # Calculate expectation values in the computational basis
        p_ex = np.zeros(np.shape(tlist), dtype=complex)
        for idx, t in enumerate(tlist):
            f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
            f_states_t = [np.exp(-1j*t*f_energies[0])*f_modes_t[0],
                          np.exp(-1j*t*f_energies[1])*f_modes_t[1]]
            p_ex[idx] = expect(num(2), output1.states[idx].transform(
                                f_states_t,
                                True))

        # Compare with mesolve
        output2 = mesolve(H, psi0, tlist, c_op_mesolve, [], args)
        p_ex_ref = expect(num(2), output2.states)

        assert_(max(abs(np.real(p_ex)-np.real(p_ex_ref))) < 1e-4)

    def testFloquetRates(self):
        """
        Compare rate transition and frequency transitions to analytical
        results for a driven two-level system, for different drive amplitudes.
        """

        # Parameters
        wq = 5.4 * 2 * np.pi
        wd = 6.7 * 2 * np.pi
        delta = wq - wd
        T = 2 * np.pi / wd
        tlist = np.linspace(0.0, 2 * T, 101)
        array_A = np.linspace(0.001, 3 * 2 * np.pi, 10, endpoint=False)
        H0 = Qobj(wq/2 * sigmaz())
        arg = {'wd': wd}
        c_ops = sigmax()
        gamma1 = 1

        delta_ana_deltas = []
        array_ana_E0 = [-np.sqrt((delta/2)**2 + a**2) for a in array_A]
        array_ana_E1 = [np.sqrt((delta/2)**2 + a**2) for a in array_A]
        array_ana_delta = [2*np.sqrt((delta/2)**2 + a**2) for a in array_A]

        def noise_spectrum(omega):
            if omega > 0:
                return 0.5 * gamma1 * omega/(2*np.pi)
            else:
                return 0

        idx = 0
        for a in array_A:
            # Hamiltonian
            H1_p = Qobj(a * sigmap())
            H1_m = Qobj(a * sigmam())
            H = [H0, [H1_p, lambda t, args: np.exp(-1j * arg['wd'] * t)],
                     [H1_m, lambda t, args: np.exp(1j * arg['wd'] * t)]]

            # Floquet modes
            fmodes0, fenergies = floquet_modes(H, T, args={}, sort=True)
            f_modes_table_t = floquet_modes_table(fmodes0, fenergies,
                                                  tlist, H, T,
                                                  args={})
            # Get X delta
            DeltaMatrix, X, frates, Amat = floquet_master_equation_rates(
                                            fmodes0, fenergies,
                                            c_ops, H, T, {},
                                            noise_spectrum, 0, 5)
            # Check energies
            deltas = np.ndarray.flatten(DeltaMatrix)

            assert_(min(abs(deltas-array_ana_delta[idx])) < 1e-4)

            # Check matrix elements
            Xs = np.ndarray.flatten(X)

            normPlus = np.sqrt(a**2 + (array_ana_E1[idx] - delta/2)**2)
            normMinus = np.sqrt(a**2 + (array_ana_E0[idx] - delta/2)**2)

            Xpp_p1 = (a/normPlus**2)*(array_ana_E1[idx]-delta/2)
            assert_(min(abs(Xs-Xpp_p1)) < 1e-4)
            Xpp_m1 = (a/normPlus**2)*(array_ana_E1[idx]-delta/2)
            assert_(min(abs(Xs-Xpp_m1)) < 1e-4)
            Xmm_p1 = (a/normMinus**2)*(array_ana_E0[idx]-delta/2)
            assert_(min(abs(Xs-Xmm_p1)) < 1e-4)
            Xmm_m1 = (a/normMinus**2)*(array_ana_E0[idx]-delta/2)
            assert_(min(abs(Xs-Xmm_m1)) < 1e-4)
            Xpm_p1 = (a/(normMinus*normPlus))*(array_ana_E0[idx]-delta/2)
            assert_(min(abs(Xs-Xmm_p1)) < 1e-4)
            Xpm_m1 = (a/(normMinus*normPlus))*(array_ana_E1[idx]-delta/2)
            assert_(min(abs(Xs-Xpm_m1)) < 1e-4)
            idx += 1

if __name__ == "__main__":
    run_module_suite()
