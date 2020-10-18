"""
Tests for the Bosonic HEOM solvers.
"""
import numpy as np
from numpy.linalg import eigvalsh


from scipy.integrate import quad


from qutip import Qobj, sigmaz, sigmax, basis, expect, Options, destroy


from bofin.heom import add_at_idx


def test_add_at_idx():
    """
    Tests the function to add at hierarchy index.
    """
    seq = (2, 3, 4)
    assert (add_at_idx(seq, 2, 1), (2, 3, 5))

    seq = (2, 3, 4)
    assert (add_at_idx(seq, 0, -1), (1, 3, 4))
