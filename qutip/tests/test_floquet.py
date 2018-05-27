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
from qutip import (basis, expect, fsesolve, sigmax, sigmaz, rand_ket, num,
                   mesolve)
from qutip.floquet import (floquet_modes, floquet_modes_table,
                           floquet_modes_t_lookup, fmmesolve)


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

        # solve schrodinger equation with floquet solver
        sol = fsesolve(H, psi0, tlist, e_ops, T, args)

        # compare with results from standard schrodinger equation
        sol_ref = mesolve(H, psi0, tlist, [], e_ops, args)

        assert_(max(abs(sol.expect[0] - sol_ref.expect[0])) < 1e-4)

    def testFloquetDissipativeEOps(self):
        """
        Floquet: test dissipative evolution of time-dependent two-level system
        using expectation values computed in fmmesolve.
        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.25 * 2 * np.pi
        omega = 1.0 * 2 * np.pi
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 20 * T, 101)
        psi0 = basis(2, 0)

        H0 = - delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = [H0, [H1, lambda t, args: np.sin(args['w'] * t)]]

        # noise power spectrum
        gamma1 = 0.1
        def noise_spectrum(omega):
            return 0.5 * gamma1 * omega / (2 * np.pi)

        # find the floquet modes for the time-dependent hamiltonian
        f_modes_0, f_energies = floquet_modes(H, T, args)

        # solve the floquet-markov master equation
        output = fmmesolve(H, psi0, tlist,
                           [sigmax()], [num(2)], [noise_spectrum], T, args)

        # calculate expectation values in the computational basis
        p_ex = output.expect[0]

        # For reference: calculate the same thing with mesolve
        output = mesolve(H, psi0, tlist, [np.sqrt(gamma1) * sigmax()],
                         [num(2)], args)
        p_ex_ref = output.expect[0]

        assert_(max(abs(np.real(p_ex) - np.real(p_ex_ref))) < 1e-1)

    def testFloquetDissipative(self):
        """
        Floquet: test dissipative evolution of time-dependent two-level system
        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.25 * 2 * np.pi
        omega = 1.0 * 2 * np.pi
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 20 * T, 101)
        psi0 = basis(2, 0)

        H0 = - delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = [H0, [H1, lambda t, args: np.sin(args['w'] * t)]]

        # noise power spectrum
        gamma1 = 0.1
        def noise_spectrum(omega):
            return 0.5 * gamma1 * omega / (2 * np.pi)

        # find the floquet modes for the time-dependent hamiltonian
        f_modes_0, f_energies = floquet_modes(H, T, args)

        # precalculate mode table
        f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                              np.linspace(0, T, 500 + 1),
                                              H, T, args)

        # solve the floquet-markov master equation
        output = fmmesolve(H, psi0, tlist,
                           [sigmax()], [], [noise_spectrum], T, args)

        # calculate expectation values in the computational basis
        p_ex = np.zeros(np.shape(tlist), dtype=np.complex)
        for idx, t in enumerate(tlist):
            f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
            p_ex[idx] = expect(num(2), output.states[idx].transform(f_modes_t,
                                                                    True))

        # For reference: calculate the same thing with mesolve
        output = mesolve(H, psi0, tlist, [np.sqrt(gamma1) * sigmax()],
                         [num(2)], args)
        p_ex_ref = output.expect[0]

        assert_(max(abs(np.real(p_ex) - np.real(p_ex_ref))) < 1e-1)


if __name__ == "__main__":
    run_module_suite()
