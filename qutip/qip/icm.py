"""
A representation of a quantum circuit consiting of qubit initialisation (I),
controlled NOT gates (C) and measurements (M) with respect to different bases.
"""
import numpy as np

from qutip import QubitCircuit, Qobj
from qutip.qip.circuit import Gate
from qutip.qip.gates import rx, ry, rz, snot, phasegate, cnot, toffoli


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

        replace_dict = {("RZ", r"\pi/2"): "P", ("RZ", r"\pi/4"): "T",
                        ("RX", r"\pi/2"): "V", ("RZ", r"-\pi/2"): "P_dagger",
                        ("RZ", r"-\pi/4"): "T_dagger",
                        ("RX", r"-\pi/2"): "V_dagger"}

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
                icm_label = replace_dict[(gate.name, gate.arg_label)]
                decomposed_circuit.add_gate(icm_label,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)
        return (decomposed_circuit)
