import numpy as np
import qutip as qt
from qubit import qubit


def test_qubit_basic_creation():
    qb = qubit(omega=2.0)
    assert qb.name == "Qubit"
    assert "sigma_z" in qb.operators
    assert qb.hamiltonian.shape == (2, 2)
    assert r"\omega" in qb.latex
    vals = qb.eigenvalues
    assert np.allclose(vals, [-1.0, 1.0])


def test_qubit_with_dissipation():
    qb = qubit(omega=1.0, decay_rate=0.5, dephasing_rate=0.2)
    assert len(qb.c_ops) == 2
    assert all(isinstance(op, qt.Qobj) for op in qb.c_ops)


def test_qubit_pretty_print_and_repr(capsys):
    qb = qubit()
    qb.pretty_print()
    captured = capsys.readouterr()
    assert "Quantum System" in captured.out
    assert "Qubit" in repr(qb)
