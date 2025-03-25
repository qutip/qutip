import numpy as np
from qutip import (
    sigmax,
    sigmay,
    sigmaz,
    rand_ket,
    num,
    destroy,
    mesolve,
    expect,
    Qobj,
    QobjEvo,
    basis,
    correlation,
    brmesolve,
)

from qutip.solver.floquet import FloquetBasis

from qutip.solver.flimesolve import FLiMESolver, flimesolve
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
        omega = 100 * np.sqrt(delta**2 + eps0**2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = -eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {"w": omega}
        H = [H0, [H1, lambda t, args: np.sin(args["w"] * t)]]
        e_ops = [num(2)]
        gamma1 = 0

        # Collapse operator for Floquet-Markov Master Equation
        c_op = sigmax()

        # Solve the floquet-Lindblad master equation

        p_ex = flimesolve(
            H,
            psi0,
            tlist,
            T,
            c_ops=[c_op * np.sqrt(gamma1)],
            e_ops=e_ops,
            args=args,
            options={"Nt": 2**5},
        ).expect[0]
        # Compare with mesolve
        p_ex_ref = mesolve(
            H, psi0, tlist, [np.sqrt(gamma1) * c_op], e_ops, args
        ).expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-5)

    def testFloquetMasterEquation2(self):
        """
        Test Floquet-Lindblad Master Equation for a two-level system
        subject to dissipation.

        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 100 * np.sqrt(delta**2 + eps0**2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 2**6 + 1)
        psi0 = rand_ket(2)
        H0 = -eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {"w": omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]], args=args)
        e_ops = [num(2)]
        gamma1 = 0.01

        # Collapse operator for Floquet-Markov Master Equation
        c_op = sigmax()

        # Solve the floquet-lindblad master equation
        p_ex = flimesolve(
            H,
            psi0,
            tlist,
            T,
            c_ops=[c_op * np.sqrt(gamma1)],
            e_ops=e_ops,
            args=args,
            options={"Nt": 2**5},
        ).expect[0]

        # Compare with mesolve
        p_ex_ref = mesolve(
            H, psi0, tlist, np.sqrt(gamma1) * c_op, e_ops=e_ops, args=args
        ).expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-5)

    def testFloquetMasterEquation3(self):
        """
        Test Floquet-Lindblad Master Equation for a two-level system
        subject to dissipation with internal transform of flimesolve
        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 100 * np.sqrt(delta**2 + eps0**2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 2**6 + 1)
        psi0 = rand_ket(2)
        H0 = -eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {"w": omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]], args=args)
        e_ops = [num(2)]
        gamma1 = 0.01

        # Collapse operator for Floquet-Markov Master Equation
        c_op = sigmax()

        # Solve the floquet-markov master equation
        floquet_basis = FloquetBasis(H, T, args=args)
        solver = FLiMESolver(
            floquet_basis,
            c_ops=[c_op * np.sqrt(gamma1)],
            rsa=0,
            options={"Nt": 2**5},
        )
        solver.start(psi0, tlist[0])
        p_ex = [expect(e_ops, solver.step(t))[0] for t in tlist]

        # Compare with mesolve
        output2 = mesolve(H, psi0, tlist, np.sqrt(gamma1) * c_op, e_ops, args)
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(
            np.real(p_ex), np.real(p_ex_ref), atol=5 * 1e-5
        )

    def testFloquetMasterEquation_multiple_coupling(self):
        """
        Test Floquet-Lindblad Master Equation for a two-level system
        subject to dissipation with multiple coupling operators
        """
        delta = 0.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 100 * np.sqrt(delta**2 + eps0**2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 2**6 + 1)
        psi0 = rand_ket(2)
        H0 = -eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {"w": omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]], args=args)
        e_ops = [num(2)]
        gamma1 = 0.01

        # Solve the floquet-lindblad master equation
        p_ex = flimesolve(
            H,
            psi0,
            tlist,
            T,
            c_ops=[sigmax() * np.sqrt(gamma1), sigmay() * np.sqrt(gamma1)],
            e_ops=e_ops,
            args=args,
            options={"Nt": 2**5},
        ).expect[0]

        # Compare with mesolve
        output2 = mesolve(
            H,
            psi0,
            tlist,
            [np.sqrt(gamma1) * sigmax(), np.sqrt(gamma1) * sigmay()],
            e_ops,
            args,
        )
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-5)

    def testFLiMEtimesense(self):
        """
        Test Floquet-Lindblad Master Equation with nonzero timesense values.

        """
        delta = 0.2 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        gamma1 = 0.05
        T = 2 * np.pi / delta
        H = -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz()

        tlist = np.linspace(0, 15.0, 15 * 2**4)

        psi0 = basis(2, 1)
        e_ops = [sigmax()]

        output = flimesolve(
            H,
            psi0,
            tlist,
            T,
            c_ops=[np.sqrt(gamma1) * sigmax()],
            e_ops=e_ops,
            args=[],
            options={"Nt": 2**8},
            rsa=1e5,
        )
        output1 = brmesolve(
            H,
            psi0,
            tlist,
            c_ops=[np.sqrt(gamma1) * sigmax()],
            e_ops=e_ops,
            sec_cutoff=-1,
        )

        np.testing.assert_allclose(
            output.expect[0], output1.expect[0], atol=1e-2
        )

    def testFLiMEPDS(self):
        """
        Testing Power Density Spectra for FLiMESolve
        """

        # driving field
        def f(t):
            return np.sin(omega_d * t)

        # Hamiltonian parameters
        Delta = 2 * np.pi  # qubit splitting

        # Bath parameters
        gamma = 0.05 * Delta / (2 * np.pi)  # dissipation strength
        temp = 0  # temperature

        # Simulation parameters
        psi0 = basis(2, 0)  # initial state
        e_ops = [sigmaz()]

        # Hamiltonian
        omega_d = 0.05 * Delta  # drive frequency
        A = Delta  # drive amplitude
        H_adi = [[A / 2.0 * sigmaz(), f]]

        # Simulation parameters
        T = 2 * np.pi / omega_d  # period length
        tlist = np.linspace(0, 5 * T, (5) * 2**4)

        H_adi = [[A / 2.0 * sigmaz(), f]]

        # Bose einstein distribution
        def nth(w):
            if temp > 0:
                return 1 / (np.exp(w / temp) - 1)
            else:
                return 0

        # Power spectrum
        def power_spectrum(w):
            if w > 0:
                return gamma / 2 * (nth(w) + 1)
            elif w == 0:
                return 0
            else:
                return 0  # gamma * nth(-w)

        a_ops = [[sigmax(), power_spectrum]]
        brme_result2 = brmesolve(H_adi, psi0, tlist, a_ops=a_ops, e_ops=e_ops)

        timesense = 1e10
        c_ops_fme = [Qobj(sigmax())]
        adi_fme_pow = flimesolve(
            H_adi,
            psi0,
            tlist,
            T,
            c_ops=c_ops_fme,
            e_ops=e_ops,
            power_spectra=[power_spectrum],
            rsa=timesense,
            Nt=2**6,
            options={"rtol": 1e-12, "atol": 1e-12},
        )

        np.testing.assert_allclose(
            brme_result2.expect[0], adi_fme_pow.expect[0], atol=2e-2
        )
