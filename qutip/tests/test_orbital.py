import pytest
from scipy.special import sph_harm
from qutip.orbital import orbital
import qutip
import numpy as np


def test_orbital_single_ket():
    """ Checks output for a single ket as input"""
    # setup mesh for theta and phi
    theta_list = np.linspace(0, np.pi, num=50)
    phi_list = np.linspace(0, 2 * np.pi, num=100)
    for theta, phi in zip(theta_list, phi_list):
        # set l and m
        for l in range(0, 5):
            for m in range(-l, l + 1):
                q = qutip.basis(2 * l + 1, l + m)
                # check that outputs are the same,
                # note that theta and phi are interchanged for scipy
                assert sph_harm(m, l, phi, theta) == orbital(theta, phi, q)


def test_orbital_multiple_ket():
    """ Checks if the combination of multiple kets works """
    theta_list = np.linspace(0, np.pi, num=50)
    phi_list = np.linspace(0, 2 * np.pi, num=100)
    l, m = 5, 2
    q1 = qutip.basis(2 * l + 1, l + m)
    l, m = 3, -1
    q2 = qutip.basis(2 * l + 1, l + m)
    for theta, phi in zip(theta_list, phi_list):
        exp = sph_harm(2, 5, phi, theta) + sph_harm(-1, 3, phi, theta)
        assert orbital(theta, phi, q1, q2) ==  exp


def test_orbital_explicit():
    """ Checks explicit configurations of orbital functions"""
    theta_list = np.linspace(0, np.pi, num=50)
    phi_list = np.linspace(0, 2 * np.pi, num=100)
    # Constant function
    l, m = 0, 0
    q = qutip.basis(2 * l + 1, l + m)
    assert orbital(0, 0, q) == 0.5 * np.sqrt(1 / np.pi)
    # cosine function
    l, m = 1, 0
    q = qutip.basis(2 * l + 1, l + m)
    assert np.allclose(orbital(theta_list, 0, q),
                       0.5 * np.sqrt(3 / np.pi) * np.cos(theta_list))
    # cosine with phase
    l, m = 1, 1
    q = qutip.basis(2 * l + 1, l + m)
    phi_mesh, theta_mesh = np.meshgrid(phi_list, theta_list)
    assert np.allclose(orbital(theta_list, phi_list, q),
                       -0.5 * np.sqrt(3 / (2 * np.pi)) * np.sin(
                           theta_mesh) * np.exp(1j * phi_mesh))
