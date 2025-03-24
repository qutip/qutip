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
            Nt=50,
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
            Nt=2**5,
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
            time_sense=0,
            Nt=2**5,
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
            Nt=2**5,
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
            Nt=2**8,
            time_sense=1e5,
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

    def testFLiMECorrelation(self):
        """
        Test Floquet-Lindblad Master Equation with nonzero timesense values.

        """

        E1mag = 2 * np.pi * 0.072992700729927
        E1pol = np.sqrt(1 / 2) * np.array([1, 1, 0])
        E1 = E1mag * E1pol

        dmag = 1
        d = dmag * np.sqrt(1 / 2) * np.array([1, 1, 0])

        Om1 = np.dot(d, E1)
        Om1t = np.dot(d, np.conj(E1))

        wlas = 2 * np.pi * 280
        wres = 2 * np.pi * 280

        T = 2 * np.pi / abs(1)  # period of the Hamiltonian
        Hargs = {"l": (wlas)}
        w = Hargs["l"]
        Gamma = 2 * np.pi * 0.0025  # in THz, roughly equivalent to 1 micro eV

        Nt = 2**8
        timef = 10 * T
        dt = timef / Nt
        tlist = np.linspace(0, timef - dt, Nt)

        H_atom = ((wres - wlas) / 2) * np.array([[-1, 0], [0, 1]])
        Hf1 = -(1 / 2) * np.array([[0, Om1], [np.conj(Om1), 0]])

        H0 = Qobj(H_atom)  # Time independant Term
        Hf1 = Qobj(Hf1)  # Forward Rotating Term

        H = [
            H0 + Hf1
        ]  # Full Hamiltonian in string format, a form acceptable to QuTiP

        rho0 = Qobj([[0.5001, 0], [0, 0.4999]])
        kwargs = {"T": T, "time_sense": 0}
        testg1F = correlation.correlation_2op_1t(
            H,
            rho0,
            taulist=tlist,
            c_ops=[np.sqrt(Gamma) * destroy(2)],
            a_op=destroy(2).dag(),
            b_op=destroy(2),
            solver="fme",
            reverse=True,
            args=Hargs,
            Nt=Nt,
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
            args=Hargs,
            **kwargs,
        )

        np.testing.assert_allclose(testg1F, testg1M, atol=1e-2)
