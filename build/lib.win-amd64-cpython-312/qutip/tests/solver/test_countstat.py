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
        current_1, noise_1 = qutip.countstat_current_noise(
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

        current_3, noise_3 = qutip.countstat_current_noise(
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
