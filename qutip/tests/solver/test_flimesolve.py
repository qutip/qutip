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

        # Solve the floquet-lindblad master equation

        p_ex = flimesolve(
            H,
            psi0,
            tlist,
            T,
            c_ops=[c_op * np.sqrt(gamma1)],
            e_ops=e_ops,
            args=args,
        ).expect[0]
        # Compare with mesolve
        p_ex_ref = mesolve(
            H, psi0, tlist, [np.sqrt(gamma1) * c_op], e_ops=e_ops, args=args
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
            floquet_basis, c_ops=[c_op * np.sqrt(gamma1)], time_sense=0
        )
        solver.start(psi0, tlist[0])
        p_ex = [expect(e_ops, solver.step(t))[0] for t in tlist]

        # Compare with mesolve
        output2 = mesolve(
            H, psi0, tlist, np.sqrt(gamma1) * c_op, e_ops=e_ops, args=args
        )
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-5)

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
        ).expect[0]

        # Compare with mesolve
        output2 = mesolve(
            H,
            psi0,
            tlist,
            [np.sqrt(gamma1) * sigmax(), np.sqrt(gamma1) * sigmay()],
            e_ops=e_ops,
            args=args,
        )
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-5)

    def testFLiMEtimesense(self):
        """
        Test Floquet-Lindblad Master Equation with nonzero timesense values.

        """
        delta = 0.0 * 2 * np.pi
        eps0 = 10.0 * 2 * np.pi
        A = 1 * 2 * np.pi
        omega = 10.0 * 2 * np.pi
        T = 2 * np.pi / omega

        tlist = np.linspace(0.0, 50 * T, 2**8)
        psi0 = basis(2, 0)

        H0 = -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz()
        H1 = A / 2.0 * sigmax()
        args = {"w": omega}
        H = [H0, [H1, lambda t, w: np.sin(w * t)]]
        gamma1 = 0.1

        # setting the value of the time_sense argument arbitrarily high
        t_sensitivity = 1e10

        # solve the floquet-lindblad master equation
        output = flimesolve(
            H,
            psi0,
            tlist,
            T,
            c_ops=[np.sqrt(gamma1) * sigmax()],
            args=args,
            options={"store_floquet_states": True},
            time_sense=t_sensitivity,
            e_ops=num(2),
        )
        p_ex = output.expect[0]

        output = mesolve(
            H,
            psi0,
            tlist,
            [np.sqrt(gamma1) * sigmax()],
            e_ops=[num(2)],
            args=args,
        )
        p_ex_ref = output.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-4)

    def testFLiMECorrelation(self):
        """
        Test Floquet-Lindblad Master Equation with correlation functions.

        """
        Om1 = 2 * np.pi * 0.0072992700729927

        T = 1 / 280
        Gamma = 2 * np.pi * 0.00025

        Nt = 20
        timef = 2 * T
        dt = timef / Nt
        tlist = np.linspace(0, timef - dt, Nt)

        H = -(Om1 / 2) * sigmax()

        rho0 = Qobj([[0.5001, 0], [0, 0.4999]])
        kwargs = {"T": T, "time_sense": 1e5}
        testg1F = correlation.correlation_2op_1t(
            H,
            rho0,
            taulist=tlist,
            c_ops=[np.sqrt(Gamma) * destroy(2)],
            a_op=destroy(2).dag(),
            b_op=destroy(2),
            solver="fme",
            reverse=True,
            **kwargs,
        )

        testg1M = correlation.correlation_2op_1t(
            H,
            rho0,
            taulist=tlist,
            c_ops=[np.sqrt(Gamma) * destroy(2)],
            a_op=destroy(2).dag(),
            b_op=destroy(2),
            solver="me",
            reverse=True,
        )

        np.testing.assert_allclose(testg1F, testg1M, atol=1e-5)
