import numpy as np
import qutip as qt
from jaynes_cummings import jaynes_cummings


def test_jaynes_cummings_basic():
    jc = jaynes_cummings(omega_c=1.0, omega_a=1.0, g=0.1, n_cavity=3)
    assert jc.hamiltonian.isherm
    assert "a" in jc.operators
    assert "sigma_z" in jc.operators
    assert jc.dimension == 6
    assert len(jc.eigenvalues) == 6


def test_jaynes_cummings_non_rwa():
    jc = jaynes_cummings(rotating_wave=False, n_cavity=3)
    assert "(\\sigma_+ + \\sigma_-)" in jc.latex


def test_jaynes_cummings_with_dissipation():
    jc = jaynes_cummings(cavity_decay=0.5, atomic_decay=0.1, thermal_photons=0.1)
    assert len(jc.c_ops) > 0
    for op in jc.c_ops:
        assert isinstance(op, qt.Qobj)
