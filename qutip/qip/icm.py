"""
A representation of a quantum circuit consiting of qubit initialisation (I),
controlled NOT gates (C) and measurements (M) with respect to different bases.
"""
import numpy as np

from qutip import QubitCircuit, Qobj
from qutip.qip.circuit import Gate
from qutip.qip.gates import rx, ry, rz, snot, phasegate, cnot, toffoli

_icm_gate_dict = {("RZ", r"\pi/2"): "P",
                  ("RZ", r"\pi/4"): "T",
                  ("RX", r"\pi/2"): "V",
                  ("RZ", r"-\pi/2"): "P_dagger",
                  ("RZ", r"-\pi/4"): "T_dagger",
                  ("RX", r"-\pi/2"): "V_dagger"}

class Icm(QubitCircuit):
    """
    A representation of a quantum circuit consisting entirely of qubit
    initialisations (I), a network of controlled (C) NOT gates and measurement
    (M) in different basis. According to [1], the transformation to
    the ICM representation provides a cannonical form for an exact and
    fault-tolerant, error corrected circuit needed for optimization prior to
    final implementation on a hardware model.

    References
    ----------
    .. [1] arXiv:1509.03962v1 [quant-ph]
    """
    def __init__(self, qcircuit):
        """
        Parameters
        ----------
        qcircuit: QubitCircuit
            A quantum circuit which is an instance of a QubitCircuit class.
        """
        self.qcircuit = qcircuit
        self.gates = qcircuit.gates
        self.icm = None

    def decompose_gates(self):
        """
        A function to decompose `qcircuit` in terms of the P, V, T, SNOT, CNOT
        and TOFFOLI gates which forms the starting point of ICM conversion.
        First we will use the `resolve_gate` function to get all the gates in
        terms of CNOT, SNOT, TOFFOLI, RX and RZ. Then we will re-name the
        rotation gates as (P (P_dagger), T (T_dagger), V (V_dagger)).

        Returns
        -------
        decomposed_circuit: QubitCircuit
        """
        resolved_circuit = self.qcircuit.resolve_gates(["RX", "RZ", "CNOT",
                                                        "SNOT", "TOFFOLI"])
        decomposed_circuit = QubitCircuit(resolved_circuit.N)

        # Check all gates and set argument label with necessary ICM gates

        for gate in resolved_circuit.gates:

            if gate.arg_label == r"\pi":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=np.pi / 2,
                                            arg_label=r'\pi/2')

                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=np.pi / 2,
                                            arg_label=r'\pi/2')

            elif gate.name == "GLOBALPHASE":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)

            elif gate.name == "TOFFOLI":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)
            elif gate.name == "CNOT":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)

            elif gate.name == "SNOT":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)

            else:
                icm_label = _icm_gate_dict[(gate.name, gate.arg_label)]
                decomposed_circuit.add_gate(icm_label,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)
        return (decomposed_circuit)

    def ancilla_cost(self):
        """
        Determines the number of ancilla qubits and additional gates required 
        for ICM representation of a given quantum circuit decomposed into P, T, V
        SNOT and Toffoli gates. The P, T, V gates are implemented using ancilla
        qubits and gate teleportation requiring CNOT and measurement. Each T gate
        requires 5 ancillae and 6 CNOT gates. The P and V gates each require
        1 ancilla and 1 CNOT gate. Each Hadamard gate is implemented
        using a sequence of P and V gates requiring 3 extra ancillae and gates. The
        Toffoli gate requires 55 extra gates and 42 ancillae.

        Returns
        -------
        ancilla_cost: dict
            A dictionary which gives the ancilla count for each type of gate from
            the set (P (P_dagger), T (T_dagger), V (V_dagger), SNOT, TOFFOLI)

        References
        ----------
        .. [1] arXiv:1509.03962v1 [quant-ph]
        """
        decomposed_circuit = self.decompose_gates()
        cost = dict({"P": 0, "T": 0, "V": 0,
                             "SNOT": 0, "TOFFOLI": 0})

        for gate in decomposed_circuit.gates:
            if gate.name == "CNOT":
                continue

            elif gate.name == "SNOT":
                cost["SNOT"] += 3

            elif gate.name == "TOFFOLI":
                cost["TOFFOLI"] += 42

            elif ((_icm_gate_dict[(gate.name, gate.arg_label)] == "P")
                or (_icm_gate_dict[(gate.name, gate.arg_label)] == "P_dagger")):
                cost["P"] += 1

            elif ((_icm_gate_dict[(gate.name, gate.arg_label)] == "T")
                   or (_icm_gate_dict[(gate.name, gate.arg_label)] == "T_dagger")):
                cost["T"] += 5

            elif ((_icm_gate_dict[(gate.name, gate.arg_label)] == "V")
                or (_icm_gate_dict[(gate.name, gate.arg_label)] == "V_dagger")):
                cost["V"] += 1

            else:
                raise ValueError("Gate decomposition is not in correct ICM basis")

        return (cost)
