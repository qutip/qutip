import numpy as np
from qutip import gates, Qobj

def test_phasegate_hermitian():
    phi = 4 * np.pi / 3  # Angle where bug occurs
    pg = gates.phasegate(phi)
    expected = Qobj([[1, 0], [0, np.exp(1j * phi)]])
    assert not pg.isherm, "Phasegate should not be Hermitian"
    assert np.allclose(pg.dag().full(), expected.dag().full()), "Phasegate dagger should conjugate phase"
    diag_elements = np.diag(pg.full())
    expected_diag = [1, np.exp(1j * phi)]
    assert np.allclose(diag_elements, expected_diag), "Phasegate diagonal elements incorrect"
