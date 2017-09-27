"""
Tests for the ICM module.
"""
import numpy as np
from numpy.testing import (assert_equal, assert_, run_module_suite,
                           assert_raises)

from qutip.qip.circuit import Gate
from qutip.qip.icm import *
from qutip import QubitCircuit


def test_label():
    """
    Test if corect ICM label is returned
    """
    assert_equal(icm_label("RZ", r"\pi/2"), "P")
    assert_equal(icm_label("RZ", r"-\pi/2"), "P_dagger")
    assert_equal(icm_label("RZ", r"\pi/4"), "T")
    assert_equal(icm_label("RY", r"pi/2"), None)
    assert_equal(icm_label("RZ", r"pi/10"), None)
    assert_equal(icm_label("RZ", r"10"), None)


def test_toffoli():
    """
    Test is the Toffoli decomposition is correct.
    """
    qcircuituit = QubitCircuit(5, reverse_states=False)
    # Use RX gates with target=[4] to distinguish start and end of
    # TOFFOLI gates
    qcircuituit.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcircuituit.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
    qcircuituit.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcircuituit.add_gate("TOFFOLI", controls=[2, 3], targets=[4])
    qcircuituit.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcircuituit.add_gate("SNOT", targets=[4])
    qcircuituit.add_gate("RX", targets=[4],
                         arg_value=np.pi / 2, arg_label=r'\pi/2')
    qcircuituit.add_gate("RZ", targets=[4],
                         arg_value=np.pi / 2, arg_label=r'\pi/2')

    toffoli_decomposed = decompose_toffoli(qcircuituit)

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


def test_SNOT():
    """
    Test SNOT gate decomposition.
    """
    qcircuituit = QubitCircuit(5, reverse_states=False)
    # Use RX gates with target=[4] to distinguish start and end
    qcircuituit.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcircuituit.add_gate("SNOT", targets=[0])
    qcircuituit.add_gate("RX", targets=[4], arg_value=np.pi, arg_label=r'\pi')
    qcircuituit.add_gate("SNOT", targets=[1])
    qcircuituit.add_gate("RZ", targets=[4], arg_value=np.pi / 2,
                         arg_label=r'\pi/2')

    SNOT_decomposed = decompose_SNOT(qcircuituit)
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


def test_ancilla_cost():
    """
    Test if ancilla cost calculation is correct.
    correct.
    """
    qcirc = QubitCircuit(5, reverse_states=False)
    qcirc.add_gate("TOFFOLI", controls=[0, 1], targets=[2])
    qcirc.add_gate("SNOT", targets=[3])
    qcirc.add_gate("RX", targets=[5], arg_value=np.pi/2, arg_label=r'\pi/2')
    qcirc.add_gate("TOFFOLI", controls=[1, 2], targets=[0])
    qcirc.add_gate("SNOT", targets=[2])

    icm_model = Icm(qcirc)
    ancilla = icm_model.ancilla_cost()

    assert_(ancilla["TOFFOLI"] == 84)
    assert_(ancilla["SNOT"] == 6)
    assert_(ancilla["V"] == 1)


def test_icm_P():
    """
    Test for the P gate conversion.
    """
    qcirc = QubitCircuit(2, reverse_states=False)
    qcirc.add_gate("CNOT", controls=[0], targets=[1])
    qcirc.add_gate("CNOT", controls=[1], targets=[0])
    qcirc.add_gate("RZ", targets=[0], arg_value=np.pi/2, arg_label=r"\pi/2")
    qcirc.add_gate("CNOT", controls=[1], targets=[0])

    icm_representation = replace_P(qcirc, 2)

    assert_(icm_representation.gates[0].name == "CNOT")
    assert_(icm_representation.gates[0].controls[0] == 0)
    assert_(icm_representation.gates[0].targets[0] == 2)
    assert_(icm_representation.gates[-1].name == "CNOT")
    assert_(icm_representation.gates[-1].controls[0] == 2)
    assert_(icm_representation.gates[-1].targets[0] == 1)

    assert_(icm_representation.gates[2].arg_label == "ancilla")
    assert_(icm_representation.gates[3].controls[0] == 1)
    assert_(icm_representation.gates[3].targets[0] == 0)


def test_icm_V():
    """
    Test for the V gate conversion.
    """
    qcirc = QubitCircuit(2, reverse_states=False)
    qcirc.add_gate("CNOT", controls=[0], targets=[1])
    qcirc.add_gate("CNOT", controls=[1], targets=[0])
    qcirc.add_gate("RX", targets=[0], arg_value=np.pi/2, arg_label=r"\pi/2")
    qcirc.add_gate("CNOT", controls=[1], targets=[0])

    icm_representation = replace_V(qcirc, 2)

    assert_(icm_representation.gates[0].name == "CNOT")
    assert_(icm_representation.gates[0].controls[0] == 0)
    assert_(icm_representation.gates[0].targets[0] == 2)
    assert_(icm_representation.gates[1].name == "CNOT")
    assert_(icm_representation.gates[1].controls[0] == 2)
    assert_(icm_representation.gates[1].targets[0] == 0)
    assert_(icm_representation.gates[-1].name == "CNOT")
    assert_(icm_representation.gates[-1].controls[0] == 2)
    assert_(icm_representation.gates[-1].targets[0] == 1)

    assert_(icm_representation.gates[2].arg_label == "ancilla")
    assert_(icm_representation.gates[3].controls[0] == 0)
    assert_(icm_representation.gates[3].targets[0] == 1)


def test_icm_T():
    """
    Test for the T gate conversion.
    """
    qcirc = QubitCircuit(2, reverse_states=False)
    qcirc.add_gate("CNOT", controls=[0], targets=[1])
    qcirc.add_gate("CNOT", controls=[1], targets=[0])
    qcirc.add_gate("RZ", targets=[0], arg_value=np.pi/4, arg_label=r"\pi/4")
    qcirc.add_gate("CNOT", controls=[1], targets=[0])

    icm_representation = replace_T(qcirc, 2)
    assert_(icm_representation.gates[0].name == "CNOT")
    assert_(icm_representation.gates[0].controls[0] == 0)
    assert_(icm_representation.gates[0].targets[0] == 6)
    assert_(icm_representation.gates[1].name == "CNOT")
    assert_(icm_representation.gates[1].controls[0] == 6)
    assert_(icm_representation.gates[1].targets[0] == 0)
    assert_(icm_representation.gates[-1].name == "CNOT")
    assert_(icm_representation.gates[-1].controls[0] == 6)
    assert_(icm_representation.gates[-1].targets[0] == 5)

    assert_(icm_representation.gates[2].arg_label == "ancilla")
    assert_(icm_representation.gates[3].arg_label == "ancilla")
    assert_(icm_representation.gates[4].name == "y")
    assert_(icm_representation.gates[5].name == "+")
    assert_(icm_representation.gates[6].name == "0")

    assert_(icm_representation.gates[7].controls[0] == 1)
    assert_(icm_representation.gates[7].targets[0] == 0)
    assert_(icm_representation.gates[12].controls[0] == 4)
    assert_(icm_representation.gates[12].targets[0] == 5)


def test_icm():
    """
    Test the complete ICM tranformation
    """
    qcirc = QubitCircuit(2, reverse_states=False)
    qcirc.add_gate("CNOT", controls=[0], targets=[1])
    qcirc.add_gate("SNOT", targets=[0])
    qcirc.add_gate("RZ", targets=[0], arg_value=np.pi/2, arg_label=r"\pi/2")
    qcirc.add_gate("RX", targets=[0], arg_value=np.pi/2, arg_label=r"\pi/2")
    qcirc.add_gate("CNOT", controls=[0], targets=[1])

    model = Icm(qcirc)
    icm_representation = model.to_icm()

    assert_(icm_representation.gates[0].name == "IN")
    assert_(icm_representation.gates[2].name == "CNOT")
    assert_(icm_representation.gates[2].controls[0] == 0)
    assert_(icm_representation.gates[2].targets[0] == 6)

    assert_(icm_representation.gates[-1].name == "OUT")
    assert_(icm_representation.gates[-2].name == "OUT")

    assert_(icm_representation.gates[-3].controls[0] == 0)
    assert_(icm_representation.gates[-3].targets[0] == 6)

    assert_(icm_representation.gates[3].arg_label == "ancilla")

    assert_(icm_representation.gates[3].name == "Y")
    assert_(icm_representation.gates[5].arg_label == "measurement")
    assert_(icm_representation.gates[6].arg_label == "correction")

    assert_(icm_representation.gates[7].arg_label == "ancilla")
    assert_(icm_representation.gates[7].targets[0] == 2)

    assert_(icm_representation.gates[11].arg_label == "ancilla")
    assert_(icm_representation.gates[11].targets[0] == 3)

if __name__ == "__main__":
    run_module_suite()
