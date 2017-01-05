"""
Tests for the ICM module.
"""

import numpy as np
from numpy.testing import (assert_equal, assert_, run_module_suite,
                           assert_raises)

from qutip.qip.circuit import Gate
from qutip.qip.icm import (Icm, pgate, tgate, vgate,
                           decompose_toffoli, decompose_SNOT)
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


def test_non_icm_gates():
    qc_non_icm = QubitCircuit(2)
    qc_non_icm.add_gate("RY", targets=[1], arg_value=np.pi / 2,
                        arg_label=r'\pi/2')
    qc_non_icm.add_gate("SNOT", targets=[1])
    qc_non_icm.add_gate("RZ", targets=[1], arg_value=np.pi / 2,
                        arg_label=r'\pi/2')
    qc_non_icm.add_gate("RX", targets=[0], arg_value=np.pi / 10,
                        arg_label=r"\pi/10")
    non_icm_model = Icm(qc_non_icm)
    # Need appropiate error message
    assert_raises(KeyError, non_icm_model.decompose_gates)


def test_ancilla_cost():
    """
    Test if ancilla cost calculation is correct.
    correct.
    """
    qcirc = QubitCircuit(5, reverse_states=False)
    qcirc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
    qcirc.add_gate("SNOT", targets=[3])
    qcirc.add_gate("RX", targets=[5], arg_value=np.pi, arg_label=r'\pi')
    qcirc.add_gate("TOFFOLI", controls=[1, 2], targets=[0])
    qcirc.add_gate("SNOT", targets=[2])

    icm_model = Icm(qcirc)
    ancilla = icm_model.ancilla_cost()

    assert_(ancilla["TOFFOLI"] == 84)
    assert_(ancilla["SNOT"] == 6)
    assert_(ancilla["V"] == 2)


def test_toffoli_decomposition():
    """
    Test if the Toffoli decomposition works correctly.
    """
    qcirc = QubitCircuit(5, reverse_states=False)

    # Use RX gates with target=[4] to distinguish start and end of
    # TOFFOLI gates
    qcirc.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcirc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
    qcirc.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcirc.add_gate("TOFFOLI", controls=[2, 3], targets=[4])
    qcirc.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcirc.add_gate("SNOT", targets=[4])
    qcirc.add_gate("RX", targets=[4], arg_value=np.pi / 2, arg_label=r'\pi/2')
    qcirc.add_gate("RZ", targets=[4], arg_value=np.pi / 2, arg_label=r'\pi/2')

    toffoli_decomposed = decompose_toffoli(qcirc)

    # Check gate sequence is proper for non Toffoli gates
    assert_(toffoli_decomposed.gates[0].name == "RX")
    assert_(toffoli_decomposed.gates[17].name == "RX")
    assert_(toffoli_decomposed.gates[34].name == "RX")
    assert_(toffoli_decomposed.gates[35].name == "SNOT")
    assert_(toffoli_decomposed.gates[36].name == "RX")
    assert_(toffoli_decomposed.gates[37].name == "RZ")

    # Check Toffoli gates

    assert_(toffoli_decomposed.gates[1].name == "SNOT")
    assert_(toffoli_decomposed.gates[2].name == "CNOT")
    assert_(toffoli_decomposed.gates[3].name == "RZ")
    assert_(toffoli_decomposed.gates[14].name == "CNOT")
    assert_(toffoli_decomposed.gates[15].name == "RZ")
    assert_(toffoli_decomposed.gates[16].name == "RZ")

    assert_(toffoli_decomposed.gates[18].name == "SNOT")
    assert_(toffoli_decomposed.gates[19].name == "CNOT")
    assert_(toffoli_decomposed.gates[20].name == "RZ")
    assert_(toffoli_decomposed.gates[31].name == "CNOT")
    assert_(toffoli_decomposed.gates[32].name == "RZ")
    assert_(toffoli_decomposed.gates[33].name == "RZ")


def test_SNOT_decomposition():
    """
    Test SNOT gate decomposition.
    """
    qcirc = QubitCircuit(5, reverse_states=False)

    # Use RX gates with target=[4] to distinguish start and end

    qcirc.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcirc.add_gate("SNOT", targets=[0])
    qcirc.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcirc.add_gate("SNOT", targets=[1])
    qcirc.add_gate("RZ", targets=[4], arg_value=np.pi / 2, arg_label=r'\pi/2')

    SNOT_decomposed = decompose_SNOT(qcirc)

    # Check other gates are sequenced properly

    assert_(SNOT_decomposed.gates[0].name == "RX")
    assert_(SNOT_decomposed.gates[4].name == "RX")
    assert_(SNOT_decomposed.gates[8].name == "RZ")

    # Check SNOT as PVP

    assert_(SNOT_decomposed.gates[1].name == "RZ")
    assert_(SNOT_decomposed.gates[2].name == "RX")
    assert_(SNOT_decomposed.gates[3].name == "RZ")

    assert_(SNOT_decomposed.gates[5].name == "RZ")
    assert_(SNOT_decomposed.gates[6].name == "RX")
    assert_(SNOT_decomposed.gates[7].name == "RZ")


if __name__ == "__main__":
    run_module_suite()
