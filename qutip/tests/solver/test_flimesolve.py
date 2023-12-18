import numpy as np
from qutip import (
    sigmax, sigmay, sigmaz,  sigmap, sigmam,
    rand_ket, num, destroy,
    mesolve, expect, sesolve,
    Qobj, QobjEvo, coefficient
)

from qutip.solver.floquet import (
    FloquetBasis, floquet_tensor, fmmesolve, FMESolver,
    _floquet_delta_tensor, _floquet_X_matrices, fsesolve
)

from qutip.solver.flimesolve import (
    FLiMESolver, flimesolve
)
import pytest


# Writing this script to closely mirror test_floquet.py, but without
#    anything already tested there (e.g. unitary evolution)


class TestFlimesolve:
    """
    A test class for the QuTiP functions for Floquet formalism.
    """

    def testFloquetLindbladMasterEquation1(self):
        """
        Test Floquet-Lindblad Master Equation for a driven two-level system
        without dissipation.
        """
        delta = 1.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.5 * 2 * np.pi
        omega = 10*np.sqrt(delta ** 2 + eps0 ** 2)
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
        c_op = sigmax()

        # Solve the floquet-markov master equation
        p_ex = flimesolve(H, psi0, tlist, T,
                          c_ops_and_rates=[[c_op, gamma1]],
                          e_ops=e_ops, args=args).expect[0]

        # Compare with mesolve
        p_ex_ref = mesolve(
            H, psi0, tlist, [np.sqrt(gamma1)*c_op], e_ops, args
        ).expect[0]

        np.testing.assert_allclose(
            np.real(p_ex), np.real(p_ex_ref), atol=1e-4)

    def testFloquetMasterEquation2(self):
        """
        Test Floquet-Lindblad Master Equation for a two-level system
        subject to dissipation.

        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 10*np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 2**6+1)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]],
                    args=args)
        e_ops = [num(2)]
        gamma1 = 0.01

        # Collapse operator for Floquet-Markov Master Equation
        c_op = sigmax()

        # Solve the floquet-lindblad master equation
        p_ex = flimesolve(H, psi0, tlist, T,
                          c_ops_and_rates=[[c_op, gamma1]],
                          e_ops=e_ops, args=args).expect[0]

        # Compare with mesolve
        p_ex_ref = mesolve(
            H, psi0, tlist, np.sqrt(gamma1)*c_op, e_ops=e_ops, args=args
        ).expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-4)

    def testFloquetMasterEquation3(self):
        """
        Test Floquet-Lindblad Master Equation for a two-level system
        subject to dissipation with internal transform of flimesolve
        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 10*np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 2**6+1)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]],
                    args=args)
        e_ops = [num(2)]
        gamma1 = 0.01

        # Collapse operator for Floquet-Markov Master Equation
        c_op = sigmax()

        # Solve the floquet-markov master equation
        floquet_basis = FloquetBasis(H, T)
        solver = FLiMESolver(
            floquet_basis, [[c_op, gamma1]], args
        )
        solver.start(psi0, tlist[0])
        p_ex = [expect(e_ops, solver.step(t))[0] for t in tlist]

        # Compare with mesolve
        output2 = mesolve(H, psi0, tlist, np.sqrt(gamma1)*c_op, e_ops, args)
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref),
                                   atol=5 * 1e-4)

    def testFloquetMasterEquation_multiple_coupling(self):
        """
        Test Floquet-Lindblad Master Equation for a two-level system
        subject to dissipation with multiple coupling operators
        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 10*np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 2**6+1)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]],
                    args=args)
        e_ops = [num(2)]
        gamma1 = 0.01

        # Solve the floquet-lindblad master equation

        output1 = flimesolve(H, psi0, tlist, T=T,
                             c_ops_and_rates=[[sigmax(), gamma1],
                                              [sigmay(), gamma1]],
                             e_ops=e_ops, args=args)
        p_ex = output1.expect[0]

        # Compare with mesolve
        output2 = mesolve(H, psi0, tlist,
                          [np.sqrt(gamma1)*sigmax(), np.sqrt(gamma1)*sigmay()],
                          e_ops, args)
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-4)

    def testFloquetMasterEquation_multiple_coupling(self):
        """
        Test Floquet-Lindblad Master Equation for a two-level system
        subject to dissipation with multiple coupling operators
        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 10*np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 2**6+1)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]],
                    args=args)
        e_ops = [num(2)]
        gamma1 = 0.01

        # Collapse operator for Floquet-Markov Master Equation
        c_ops_fmmesolve = [sigmax(), sigmay()]
        c_ops_flimesolve = [[sigmax(), gamma1],
                            [sigmay(), gamma1]]

        # Collapse operator for Lindblad Master Equation
        def noise_spectrum1(omega):
            return (omega > 0) * 0.5 * gamma1 * omega/(2*np.pi)

        def noise_spectrum2(omega):
            return (omega > 0) * 0.5 * gamma1 / (2 * np.pi)

        noise_spectra = [noise_spectrum1, noise_spectrum2]

        # Solve the floquet-lindblad master equation

        output1 = flimesolve(H, psi0, tlist, T=T,
                             c_ops_and_rates=c_ops_flimesolve,
                             e_ops=e_ops, args=args)
        p_ex = output1.expect[0]

        # Compare with mesolve
        output2 = fmmesolve(
            H, psi0, tlist, c_ops_fmmesolve, e_ops, noise_spectra, T
        )
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(
            np.real(p_ex), np.real(p_ex_ref), atol=1e-20)
