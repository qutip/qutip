import sys

import pytest


class QutipQipStub:
    class CircuitModuleStub:
        QubitCircuit = "FakeQubitCircuit"

    circuit = CircuitModuleStub()


@pytest.fixture
def without_qutip_qip(monkeypatch):
    monkeypatch.setitem(sys.modules, "qutip_qip", None)
    monkeypatch.delitem(sys.modules, "qutip.qip", raising=False)


@pytest.fixture
def with_qutip_qip_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "qutip_qip", QutipQipStub())
    monkeypatch.delitem(sys.modules, "qutip.qip", raising=False)


def test_failed_import(without_qutip_qip):
    # Ensure 'qutip.qip' is not imported yet
    assert "qutip.qip" not in sys.modules
    with pytest.raises(
        ImportError,
        match="Importing 'qutip.qip' requires the 'qutip_qip' package.",
    ):
        import qutip.qip


def test_with_qip(monkeypatch, with_qutip_qip_stub):
    import qutip.qip

    monkeypatch.setitem(sys.modules, "qutip.qip.circuit", qutip.qip.circuit)
    import qutip.qip.circuit as circuit
    from qutip.qip.circuit import QubitCircuit
    import qutip_qip

    assert qutip.qip is qutip_qip
    assert circuit is qutip_qip.circuit
    assert QubitCircuit is qutip_qip.circuit.QubitCircuit
