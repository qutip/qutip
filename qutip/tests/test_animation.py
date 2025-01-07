import pytest
import qutip
import numpy as np
from qutip.wigner import sph_harm_y

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

def test_result_state():
    H = qutip.rand_dm(2)
    tlist = np.linspace(0, 3*np.pi, 2)
    results = qutip.mesolve(H, H, tlist)

    fig, ani = qutip.anim_fock_distribution(results)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_result_state_ValueError():
    H = qutip.rand_dm(2)
    tlist = np.linspace(0, 3*np.pi, 2)
    results = qutip.mesolve(H, H, tlist, options={"store_states": False})

    text = 'Nothing to visualize. You might have forgotten ' +\
           'to set options={"store_states": True}.'
    with pytest.raises(ValueError) as exc_info:
        fig, ani = qutip.anim_fock_distribution(results)
    assert str(exc_info.value) == text


def test_anim_wigner_sphere():
    psi = qutip.rand_ket(5)
    wigner = qutip.wigner_transform(psi, 2, False, 50, ["x"])

    fig, ani = qutip.anim_wigner_sphere([wigner]*2)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_anim_hinton():
    rho = qutip.rand_dm(5)
    rhos = [rho]*2

    fig, ani = qutip.anim_hinton(rhos)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_anim_sphereplot():
    theta = np.linspace(0, np.pi, 90)
    phi = np.linspace(0, 2 * np.pi, 60)
    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    values = sph_harm_y(2, -1, theta_mesh, phi_mesh).T
    fig, ani = qutip.anim_sphereplot([values]*2, theta, phi)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_anim_matrix_histogram():
    rho = qutip.rand_dm(5)
    rhos = [rho]*2

    fig, ani = qutip.anim_matrix_histogram(rhos)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_anim_fock_distribution():
    rho = qutip.rand_dm(5)
    rhos = [rho]*2

    fig, ani = qutip.anim_fock_distribution(rhos)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_anim_wigner():
    rho = qutip.rand_dm(5)
    rhos = [rho]*2

    fig, ani = qutip.anim_wigner(rhos)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


@pytest.mark.filterwarnings(
    "ignore:The input coordinates to pcolor:UserWarning"
)
def test_anim_spin_distribution():
    j = 5
    psi = qutip.spin_state(j, -j)
    psi = qutip.spin_coherent(j, np.random.rand() * np.pi,
                              np.random.rand() * 2 * np.pi)
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    Q, THETA, PHI = qutip.spin_q_function(psi, theta, phi)

    fig, ani = qutip.anim_spin_distribution([Q]*2, THETA, PHI)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_anim_qubism():
    state = qutip.ket("01")

    fig, ani = qutip.anim_qubism([state]*2)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)


def test_anim_schmidt():
    state = qutip.ket("01")

    fig, ani = qutip.anim_schmidt([state]*2)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ani, mpl.animation.ArtistAnimation)
