import sys

import pytest


@pytest.fixture
def without_qutip_qip(monkeypatch):
    monkeypatch.setitem(sys.modules, "qutip_qip", None)
    monkeypatch.delitem(sys.modules, "qutip.qip", raising=False)


@pytest.fixture
def with_qutip_qip_stub(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "qutip_qip"
    pkg_dir.mkdir()
    init_file = pkg_dir / "__init__.py"
    init_file.write_text("__version__ = 'x.y.z'")
    circuit_file = pkg_dir / "circuit.py"
    circuit_file.write_text("class QubitCircuit:\n    pass")

    monkeypatch.syspath_prepend(tmp_path)
    # Make sure the stub modules is the one imported
    monkeypatch.delitem(sys.modules, "qutip_qip", raising=False)
    monkeypatch.delitem(sys.modules, "qutip.qip", raising=False)


def test_failed_import(without_qutip_qip):
    # Ensure 'qutip.qip' is not imported yet
    assert "qutip.qip" not in sys.modules
    with pytest.raises(
        ImportError,
        match="Importing 'qutip.qip' requires the 'qutip_qip' package.",
    ):
        import qutip.qip


def test_with_qip(with_qutip_qip_stub):
    import qutip.qip
    import qutip.qip.circuit as circuit
    from qutip.qip.circuit import QubitCircuit
    import qutip_qip

    assert qutip.qip.__version__ == "x.y.z"
    assert qutip.qip is qutip_qip
    assert circuit is qutip_qip.circuit
    assert QubitCircuit is qutip_qip.circuit.QubitCircuit
