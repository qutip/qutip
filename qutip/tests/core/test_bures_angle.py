import numpy as np

from qutip import basis
from qutip.core import metrics


def test_bures_angle_clamps_fidelity_above_one(monkeypatch):
    # Roundoff can make fidelity(A, A) slightly greater than one, which is
    # outside arccos's domain.
    monkeypatch.setattr(metrics, "fidelity", lambda A, B: 1 + np.finfo(float).eps)

    state = basis(2, 0)
    angle = metrics.bures_angle(state, state)

    assert np.isfinite(angle)
    assert angle == 0
