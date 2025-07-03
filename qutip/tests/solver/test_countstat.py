import numpy as np
import qutip
import pytest


@pytest.mark.filterwarnings("ignore::FutureWarning")
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
    eps_vec = np.linspace(-1.5 * w0, 1.5 * w0, 20)

    J_ops = [GammaR * qutip.sprepost(sR, sR.dag())]

    c_ops = [
        np.sqrt(GammaR * (1 + nth)) * sR,
        np.sqrt(GammaR * (nth)) * sR.dag(),
        np.sqrt(GammaL * (nth)) * sL,
        np.sqrt(GammaL * (1 + nth)) * sL.dag(),
    ]

    current = np.zeros(len(eps_vec))
    noise = np.zeros(len(eps_vec))

    for n, eps in enumerate(eps_vec):
        H = eps / 2 * sz + tc * sx
        L0 = qutip.liouvillian(H)
        L = qutip.liouvillian(H, c_ops)
        rhoss = qutip.steadystate(L)
        current_1, noise_1, skewness_1 = qutip.countstat_current_noise(
            L, [], wlist=[0, 1], rhoss=rhoss, J_ops=J_ops
        )
        current[n] = current_1[0]
        noise[n] = noise_1[0, 0, 0]

        current_2 = qutip.countstat_current(L, rhoss=rhoss, J_ops=J_ops)
        assert abs(current_1[0] - current_2[0]) < 1e-8

        current_2 = qutip.countstat_current(L0, c_ops, J_ops=J_ops)
        assert abs(current_1[0] - current_2[0]) < 1e-8

        current_2 = qutip.countstat_current(H, c_ops)
        assert abs(current_1[0] - current_2[0]) < 1e-8

        current_3, noise_3, skewness_3 = qutip.countstat_current_noise(
            L, c_ops, wlist=[0, 1], sparse=False
        )
        assert abs(current_1[0] - current_3[0]) < 1e-6
        np.testing.assert_allclose(noise_1[0, 0, :], noise_3[0, 0, :],
                                   atol=1e-6)

    current_target = (
        tc**2
        * GammaR
        / (tc**2 * (2 + GammaR / GammaL) + GammaR**2 / 4 + eps_vec**2)
    )
    noise_target = current_target * (
        1
        - (
            8
            * GammaL
            * tc**2
            * (
                4 * eps_vec**2 * (GammaR - GammaL)
                + GammaR * (3 * GammaL * GammaR + GammaR**2 + 8 * tc**2)
            )
            / (
                4 * tc**2 * (2 * GammaL + GammaR)
                + GammaL * GammaR**2
                + 4 * eps_vec**2 * GammaL
            )
            ** 2
        )
    )

    np.testing.assert_allclose(current, current_target, atol=1e-4)
    np.testing.assert_allclose(noise, noise_target, atol=1e-4)

def compute_analytical_cumulants(Gamma_r, Gamma_l):
    """Compute the analytical values of the first three cumulants."""
    current = (Gamma_l * Gamma_r) / (Gamma_l + Gamma_r)
    
    noise = current * (Gamma_r**2 + Gamma_l**2) / (Gamma_r + Gamma_l)**2
    
    skewness = (current * (Gamma_r**4 - 2 * Gamma_r**3 * Gamma_l + 
                6 * Gamma_r**2 * Gamma_l**2 - 2 * Gamma_r * Gamma_l**3 + 
                Gamma_l**4) / (Gamma_r + Gamma_l)**4)

    return current, noise, skewness

@pytest.mark.parametrize("method", ["pinv", "direct"])
def test_three_cumulants(method):
    """Counting statistics: Test the calculation of 
    the three first cummulat for the direct and pseudoinv methods"""
    Gamma_r = 0.5
    Gamma_l = 0.1

    d = qutip.destroy(2).dag()
    L_s = qutip.liouvillian(0*d.dag()*d, [np.sqrt(Gamma_l) *d, np.sqrt(Gamma_r) *d.dag()])
    rho_ss = qutip.steadystate(L_s)
    I_s = Gamma_r * qutip.sprepost(d.dag(),d)

    current_num, noise_num, skw_num = qutip.countstat_current_noise(L_s, [], rhoss=rho_ss, J_ops=[I_s], I_ops=[I_s], sparse=True, method=method)

    current_ana, noise_ana, skewness_ana = compute_analytical_cumulants(Gamma_r, Gamma_l)

    np.testing.assert_allclose(current_num[0], current_ana, atol=1e-4)
    np.testing.assert_allclose(noise_num[0,0,0], noise_ana, atol=1e-4)
    np.testing.assert_allclose(skw_num[0,0], skewness_ana, atol=1e-4)
