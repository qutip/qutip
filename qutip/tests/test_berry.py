import numpy as np
import pytest

import qutip.berry as berry
from qutip import Qobj, sigmax, sigmay, sigmaz


def _monopole_eigfs(n0=40, n1=40, winding=1, excited=False):
    """Ground/excited eigenstates of H = B.sigma on the unit sphere, with
    B a winding-n hedgehog.  Ground state has Chern number -n, excited +n.
    """
    thetas = np.linspace(0, np.pi, n0)
    phis = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    eigfs = np.zeros((n0, n1, 2), dtype=complex)
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            B = np.array(
                [
                    np.sin(th) * np.cos(winding * ph),
                    np.sin(th) * np.sin(winding * ph),
                    np.cos(th),
                ]
            )
            H = B[0] * sigmax().full() + B[1] * sigmay().full() + B[2] * sigmaz().full()
            _, ekets = Qobj(H).eigenstates()
            eigfs[i, j] = ekets[1 if excited else 0].full().reshape(-1)
    return eigfs


def _ssh_eigfs(n=100, v=1.0, w=2.0):
    """Lower-band eigenstates of the SSH model on a k-mesh around the BZ."""
    ks = np.linspace(0, 2 * np.pi, n, endpoint=False)
    eigfs = np.zeros((n, 2), dtype=complex)
    for i, k in enumerate(ks):
        H = (v + w * np.cos(k)) * sigmax().full() + w * np.sin(k) * sigmay().full()
        _, ekets = Qobj(H).eigenstates()
        eigfs[i] = ekets[0].full().reshape(-1)
    return eigfs


def test_monopole_ground_state_chern_minus_one():
    """S^2 monopole: ground state of H = B.sigma carries C = -1."""
    C = berry.chern_number(
        _monopole_eigfs(n0=30, n1=30), periodic=(False, True)
    )
    assert np.isclose(C, -1.0, atol=1e-9)


def test_monopole_excited_state_chern_plus_one():
    """Excited state of the same model carries C = +1."""
    C = berry.chern_number(
        _monopole_eigfs(n0=30, n1=30, excited=True), periodic=(False, True)
    )
    assert np.isclose(C, 1.0, atol=1e-9)


def test_ssh_zak_phase():
    """SSH: lower-band Zak phase is pi in the topological phase (v < w)
    and 0 in the trivial phase (v > w)."""
    topo = berry.berry_phase(_ssh_eigfs(n=100, v=1.0, w=2.0))
    assert np.isclose(abs(topo), np.pi, atol=1e-9)
    triv = berry.berry_phase(_ssh_eigfs(n=100, v=2.0, w=1.0))
    assert np.isclose(abs(triv), 0.0, atol=1e-9)


def test_three_sector_chern_sum_zero():
    """Two +1 sectors and one -2 sector: the Chern numbers sum to zero."""
    C1 = berry.chern_number(
        _monopole_eigfs(n0=30, n1=30, excited=True), periodic=(False, True)
    )
    C2 = berry.chern_number(
        _monopole_eigfs(n0=30, n1=30, excited=True), periodic=(False, True)
    )
    C3 = berry.chern_number(
        _monopole_eigfs(n0=30, n1=30, winding=2), periodic=(False, True)
    )
    assert np.isclose(C1, 1.0, atol=1e-9)
    assert np.isclose(C2, 1.0, atol=1e-9)
    assert np.isclose(C3, -2.0, atol=1e-9)
    assert np.isclose(C1 + C2 + C3, 0.0, atol=1e-9)


def test_multiband_determinant_formulation():
    """Block-diagonal 6-level system (two monopoles + one winding-2
    monopole, scaled apart): the determinant formulation over all six
    bands sums to zero Chern number (bands come in +- pairs)."""
    n0, n1 = 30, 30
    thetas = np.linspace(0, np.pi, n0)
    phis = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    eigfs = np.zeros((n0, n1, 6, 6), dtype=complex)
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            H = np.zeros((6, 6), dtype=complex)
            for idx, (winding, scale) in enumerate([(1, 1.0), (1, 2.0), (2, 3.0)]):
                B = np.array(
                    [
                        np.sin(th) * np.cos(winding * ph),
                        np.sin(th) * np.sin(winding * ph),
                        np.cos(th),
                    ]
                )
                block = (
                    B[0] * sigmax().full()
                    + B[1] * sigmay().full()
                    + B[2] * sigmaz().full()
                )
                H[2 * idx : 2 * idx + 2, 2 * idx : 2 * idx + 2] = scale * block
            _, ekets = Qobj(H).eigenstates()
            for b in range(6):
                eigfs[i, j, b] = ekets[b].full().reshape(-1)
    C = berry.chern_number(eigfs, periodic=(False, True))
    # determinant formulation converges as the grid is refined; total over
    # all bands of a time-reversal-free block system is near zero
    assert abs(C) < 0.5


def test_qobj_input_berry_phase():
    """The 1d phase accepts a plain list of Qobj kets."""
    n = 60
    ks = np.linspace(0, 2 * np.pi, n, endpoint=False)
    kets = []
    for k in ks:
        H = (1.0 + 2.0 * np.cos(k)) * sigmax() + 2.0 * np.sin(k) * sigmay()
        kets.append(H.eigenstates()[1][0])
    phase = berry.berry_phase(kets)
    assert np.isclose(abs(phase), np.pi, atol=1e-9)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        berry.berry_curvature(np.zeros((3, 3, 3, 3, 3), dtype=complex))
    with pytest.raises(ValueError):
        berry.berry_phase(np.zeros((3, 3, 3, 3), dtype=complex))
