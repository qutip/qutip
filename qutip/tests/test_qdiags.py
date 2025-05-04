import numpy as np
from qutip import gates, Qobj

def test_phasegate_hermitian():
    phi = 4 * np.pi / 3  # Angle where bug occurs
    pg = gates.phasegate(phi)
    expected = Qobj([[1, 0], [0, np.exp(1j * phi)]])
    assert pg.isherm is False, "Phasegate should not be Hermitian"
    assert np.allclose(pg.dag().full(), expected.dag().full()), "Phasegate dagger should conjugate phase"
    assert np.allclose(pg.diag(), [1, np.exp(1j * phi)]), "Phasegate diag should include imaginary part"
