import sys
import unittest

import pytest


class TestWithoutQip(unittest.TestCase):
    def setUp(self):
        self._temp_qip = None
        if sys.modules.get("qutip_qip"):
            self._temp_qip = sys.modules["qutip_qip"]
        sys.modules["qutip_qip"] = None

    def tearDown(self):
        if self._temp_qip:
            sys.modules["qutip_qip"] = self._temp_qip
        else:
            del sys.modules["qutip_qip"]

    def test_failed_import(self):
        # Ensure 'qutip.qip' is not imported yet
        assert "qutip.qip" not in sys.modules
        with pytest.raises(
            ImportError,
            match="'qutip.qip' imports require the 'qutip_qip' package.",
        ):
            import qutip.qip


def test_with_qip():
    # Skips test if 'qutip_qip' is not installed
    qutip_qip = pytest.importorskip("qutip_qip")
    import qutip.qip
    import qutip.qip.circuit as circuit
    from qutip.qip.circuit import QubitCircuit

    assert qutip.qip is qutip_qip
    assert circuit is qutip_qip.circuit
    assert QubitCircuit is qutip_qip.circuit.QubitCircuit
