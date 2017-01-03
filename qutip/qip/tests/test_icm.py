"""
Tests for the ICM module.
"""

import numpy as np
from numpy.testing import assert_equal, assert_, run_module_suite

from qutip.qip.circuit import Gate
from qutip.qip.icm import Icm
from qutip import QubitCircuit


def test_decomposition():
    """
    Test if initial decomposition in terms of T, V, P and CNOT is
    correct.
    """
    qcirc = QubitCircuit(5, reverse_states=False)
    qcirc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
    qcirc.add_gate("SNOT", targets=[3])
    qcirc.add_gate("RX", targets=[5], arg_value=np.pi, arg_label=r'\pi')

    icm_model = Icm(qcirc)
    decomposed = icm_model.decompose_gates()

    decomposed_gates = set([(gate.name, gate.arg_label)
                            for gate in decomposed.gates])

    icm_gate_set = [("CNOT", None),
                    ("SNOT", None),
                    ("TOFFOLI", None),
                    ("RZ", r"\pi/2"),
                    ("RZ", r"\pi/4"),
                    ("RX", r"\pi/2"),
                    ("RZ", r"-\pi/2"),
                    ("RZ", r"-\pi/4"),
                    ("RX", r"-\pi/2")]

    assert_(decomposed.gates[0].name == "TOFFOLI")
    assert_(decomposed.gates[1].name == "SNOT")
    assert_(decomposed.gates[2].name == "RX")
    assert_(decomposed.gates[2].arg_label == r"\pi/2")
    assert_(decomposed.gates[2].arg_value == np.pi / 2)

    assert_(decomposed.gates[3].name == "RX")
    assert_(decomposed.gates[3].arg_label == r"\pi/2")
    assert_(decomposed.gates[3].arg_value == np.pi / 2)
    
    for gate in decomposed_gates:
        assert_(gate in icm_gate_set)

if __name__ == "__main__":
    run_module_suite()
