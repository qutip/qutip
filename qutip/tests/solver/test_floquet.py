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
import pytest


def _convert_c_ops(c_op_fmmesolve, noise_spectrum, vp, ep):
    """
    Convert e_ops for fmmesolve to mesolve
    """
    c_op_mesolve = []
    N = len(vp)
    for i in range(N):
        for j in range(N):
            if i != j:
                # caclculate the rate
                gamma = 2 * np.pi * c_op_fmmesolve.matrix_element(
                    vp[j], vp[i]) * c_op_fmmesolve.matrix_element(
                    vp[i], vp[j]) * noise_spectrum(ep[j] - ep[i])

                # add c_op for mesolve
                c_op_mesolve.append(
                    np.sqrt(gamma) * (vp[i] * vp[j].dag())
                )
    return c_op_mesolve


class TestFloquet:
    """
    A test class for the QuTiP functions for Floquet formalism.
    """

    def testFloquetBasis(self):
        N = 10
        a = destroy(N)
        H = num(N) + (a+a.dag()) * coefficient(lambda t: np.cos(t))
        T = 2 * np.pi
        floquet_basis = FloquetBasis(H, T)
        psi0 = rand_ket(N)
        tlist = np.linspace(0, 10, 11)
        floquet_psi0 = floquet_basis.to_floquet_basis(psi0)
        states = sesolve(H, psi0, tlist).states
        for t, state in zip(tlist, states):
            from_floquet = floquet_basis.from_floquet_basis(floquet_psi0, t)
            assert state.overlap(from_floquet) == pytest.approx(1., abs=8e-5)

    def testFloquetUnitary(self):
        N = 10
        a = destroy(N)
        H = num(N) + (a+a.dag()) * coefficient(lambda t: np.cos(t))
        T = 2 * np.pi
        psi0 = rand_ket(N)
        tlist = np.linspace(0, 10, 11)
        states_se = sesolve(H, psi0, tlist).states
        states_fse = fsesolve(H, psi0, tlist, T=T).states
        for state_se, state_fse in zip(states_se, states_fse):
            assert state_se.overlap(state_fse) == pytest.approx(1., abs=5e-5)


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
        def spectrum(omega):
            return (omega > 0) * omega * 0.5 * gamma1 / (2 * np.pi)

        ep, vp = H0.eigenstates()
        op0 = vp[0]*vp[0].dag()
        op1 = vp[1]*vp[1].dag()

        c_op_mesolve = _convert_c_ops(c_op_fmmesolve, spectrum, vp, ep)

        # Solve the floquet-markov master equation
        p_ex = fmmesolve(H, psi0, tlist, [c_op_fmmesolve], [spectrum], T,
                         e_ops=[num(2)], args=args).expect[0]

        # Compare with mesolve
        p_ex_ref = mesolve(
            H, psi0, tlist, c_op_mesolve, e_ops=[num(2)], args=args
        ).expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-4)

    def testFloquetMasterEquation2(self):
        """
        Test Floquet-Markov Master Equation for a two-level system
        subject to dissipation.
        """
        delta = 1.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.0 * 2 * np.pi
        omega = np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]],
                    args=args)
        e_ops = [num(2)]
        gamma1 = 1

        # Collapse operator for Floquet-Markov Master Equation
        c_op_fmmesolve = sigmax()

        # Collapse operator for Lindblad Master Equation
        def spectrum(omega):
            return (omega > 0) * omega * 0.5 * gamma1 / (2 * np.pi)

        ep, vp = H0.eigenstates()
        op0 = vp[0]*vp[0].dag()
        op1 = vp[1]*vp[1].dag()

        c_op_mesolve = _convert_c_ops(c_op_fmmesolve, spectrum, vp, ep)

        # Solve the floquet-markov master equation
        p_ex = fmmesolve(
            H, psi0, tlist, [c_op_fmmesolve], [spectrum], T,
            e_ops=[num(2)], args=args
        ).expect[0]

        # Compare with mesolve
        p_ex_ref = mesolve(
            H, psi0, tlist, c_op_mesolve, e_ops=[num(2)], args=args
        ).expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-4)

    @pytest.mark.parametrize("kmax", [5, 25, 100])
    def testFloquetMasterEquation3(self, kmax):
        """
        Test Floquet-Markov Master Equation for a two-level system
        subject to dissipation with internal transform of fmmesolve
        """
        delta = 1.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.0 * 2 * np.pi
        omega = np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]],
                    args=args)
        e_ops = [num(2)]
        gamma1 = 1

        # Collapse operator for Floquet-Markov Master Equation
        c_op_fmmesolve = sigmax()

        # Collapse operator for Lindblad Master Equation
        def spectrum(omega):
            return (omega > 0) * 0.5 * gamma1 * omega / (2 * np.pi)

        ep, vp = H0.eigenstates()
        op0 = vp[0]*vp[0].dag()
        op1 = vp[1]*vp[1].dag()

        c_op_mesolve = _convert_c_ops(c_op_fmmesolve, spectrum, vp, ep)

        # Solve the floquet-markov master equation
        floquet_basis = FloquetBasis(H, T)
        solver = FMESolver(
            floquet_basis, [(c_op_fmmesolve, spectrum)], kmax=kmax
        )
        solver.start(psi0, tlist[0])
        p_ex = [expect(num(2), solver.step(t)) for t in tlist]

        # Compare with mesolve
        output2 = mesolve(
            H, psi0, tlist, c_op_mesolve, e_ops=[num(2)], args=args
        )
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref),
                                   atol=5 * 1e-4)

    def testFloquetMasterEquation_multiple_coupling(self):
        """
        Test Floquet-Markov Master Equation for a two-level system
        subject to dissipation with multiple coupling operators
        """
        delta = 1.0 * 2 * np.pi
        eps0 = 1.0 * 2 * np.pi
        A = 0.0 * 2 * np.pi
        omega = np.sqrt(delta ** 2 + eps0 ** 2)
        T = (2 * np.pi) / omega
        tlist = np.linspace(0.0, 2 * T, 101)
        psi0 = rand_ket(2)
        H0 = - eps0 / 2.0 * sigmaz() - delta / 2.0 * sigmax()
        H1 = A / 2.0 * sigmax()
        args = {'w': omega}
        H = QobjEvo([H0, [H1, lambda t, w: np.sin(w * t)]],
                    args=args)
        e_ops = [num(2)]
        gamma1 = 1

        # Collapse operator for Floquet-Markov Master Equation
        c_ops_fmmesolve = [sigmax(), sigmay()]

        # Collapse operator for Lindblad Master Equation
        def noise_spectrum1(omega):
            return (omega > 0) * 0.5 * gamma1 * omega/(2*np.pi)

        def noise_spectrum2(omega):
            return (omega > 0) * 0.5 * gamma1 / (2 * np.pi)

        noise_spectra = [noise_spectrum1, noise_spectrum2]

        ep, vp = H0.eigenstates()
        op0 = vp[0]*vp[0].dag()
        op1 = vp[1]*vp[1].dag()

        c_op_mesolve = []

        # Convert the c_ops for fmmesolve to c_ops for mesolve
        for c_op_fmmesolve, noise_spectrum in zip(c_ops_fmmesolve,
                                                   noise_spectra):
            c_op_mesolve += _convert_c_ops(
                c_op_fmmesolve, noise_spectrum, vp, ep
            )

        # Solve the floquet-markov master equation
        output1 = fmmesolve(
            H, psi0, tlist, c_ops_fmmesolve, noise_spectra, T, e_ops=e_ops,
        )
        p_ex = output1.expect[0]
        # Compare with mesolve
        output2 = mesolve(H, psi0, tlist, c_op_mesolve, e_ops=e_ops, args=args)
        p_ex_ref = output2.expect[0]

        np.testing.assert_allclose(np.real(p_ex), np.real(p_ex_ref), atol=1e-4)

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
        H0 = Qobj(wq / 2 * sigmaz())
        args = {'wd': wd}
        c_ops = sigmax()
        gamma1 = 1
        kmax = 5

        array_ana_E0 = [-np.sqrt((delta / 2)**2 + a**2) for a in array_A]
        array_ana_E1 = [np.sqrt((delta / 2)**2 + a**2) for a in array_A]
        array_ana_delta = [
            2 * np.sqrt((delta / 2)**2 + a**2)
            for a in array_A
        ]

        def noise_spectrum(omega):
            return (omega > 0) * 0.5 * gamma1 * omega/(2*np.pi)

        idx = 0
        for a in array_A:
            # Hamiltonian
            H1_p = a * sigmap()
            H1_m = a * sigmam()
            H = QobjEvo(
                [H0, [H1_p, lambda t, wd: np.exp(-1j * wd * t)],
                [H1_m, lambda t, wd: np.exp(1j * wd * t)]], args=args
            )

            floquet_basis = FloquetBasis(H, T)
            DeltaMatrix = _floquet_delta_tensor(floquet_basis.e_quasi, kmax, T)
            X = _floquet_X_matrices(floquet_basis, [c_ops], kmax)

            # Check energies
            deltas = np.ndarray.flatten(DeltaMatrix)

            # deltas and array_ana_delta have at least 1 value in common.
            assert (min(abs(deltas - array_ana_delta[idx])) < 1e-4)

            # Check matrix elements
            Xs = np.concatenate(
                [X[0][k].to_array() for k in range(-kmax, kmax+1)]
            ).flatten()

            normPlus = np.sqrt(a**2 + (array_ana_E1[idx] - delta / 2)**2)
            normMinus = np.sqrt(a**2 + (array_ana_E0[idx] - delta / 2)**2)

            Xpp_p1 = (a / normPlus**2) * (array_ana_E1[idx] - delta / 2)
            assert (min(abs(Xs - Xpp_p1)) < 1e-4)
            Xpp_m1 = (a / normPlus**2) * (array_ana_E1[idx] - delta / 2)
            assert (min(abs(Xs - Xpp_m1)) < 1e-4)
            Xmm_p1 = (a / normMinus**2) * (array_ana_E0[idx] - delta / 2)
            assert (min(abs(Xs - Xmm_p1)) < 1e-4)
            Xmm_m1 = (a / normMinus**2) * (array_ana_E0[idx] - delta / 2)
            assert (min(abs(Xs - Xmm_m1)) < 1e-4)
            Xpm_p1 = (a / (normMinus * normPlus)
                      * (array_ana_E0[idx]- delta / 2))
            assert (min(abs(Xs - Xpm_p1)) < 1e-4)
            Xpm_m1 = (a / (normMinus * normPlus)
                      * (array_ana_E1[idx] - delta / 2))
            assert (min(abs(Xs - Xpm_m1)) < 1e-4)
            idx += 1


def test_fsesolve_fallback():
    H = [sigmaz(), lambda t: np.sin(t * 2 * np.pi)]
    psi0 = rand_ket(2)
    ffstate = fmmesolve(H, psi0, [0, 1], T=1.).final_state
    fstate = sesolve(H, psi0, [0, 1]).final_state
    assert (ffstate - fstate).norm() < 1e-5
