"""
A representation of a quantum circuit consiting of qubit initialisation (I),
controlled NOT gates (C) and measurements (M) with respect to different bases.
"""
import numpy as np

from qutip import QubitCircuit, Qobj
from qutip.qip.circuit import Gate
from qutip.qip.gates import rx, ry, rz, snot, phasegate, cnot, toffoli


def replace_gate(gate):
    """
    A function to replace the rotation gates with the corresponding
    gate from the set {P, T, V, P_dagger, T_dagger, V_dagger} in the
    ICM model.

    Parameters
    ----------
    gate: Gate
        Gate from the set {RZ, RZ or CNOT}

    arg_label: str
        Argument of the gate. Valid arguments are (+/- pi) and (+/- pi/2)

    Returns
    -------
    icm_gate: Gate
        A gate from the set {P, T, V, P_dagger, T_dagger, V_dagger}.
        It is an instance of `qutip.qip.circuit.Gate` class.
    """
    replace_dict = {("RZ", r"\pi/2"): "P", ("RZ", r"\pi/4"): "T",
                    ("RX", r"\pi/2"): "V", ("RZ", r"-\pi/2"): "P_dagger",
                    ("RZ", r"-\pi/4"): "T_dagger", 
                    ("RX", r"-\pi/2"): "V_dagger"}

    try: 
        gate.name = replace_dict[(gate.name, gate.arg_label)]
        return (gate)

    except:
        raise ValueError("Gate decomposition is not in correct basis.\n"
                         "Initial decomposotion should be in interms of "
                         "P, T, V gates")

class Icm(QubitCircuit):
    """
    A representation of a quantum circuit consisting entirely of qubit
    initialisations (I), a network of controlled (C) NOT gates and measurement
    (M) in different basis. According to [1], the transformation of a circuit to
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
        A function to decompose `qcircuit` in terms of the P, V and T gates
        which forms the starting point of ICM conversion. First we will use
        the `resolve_gate` function to get all the gates in terms of CNOT, rx,
        ry and rz. Then we will represent the rotation gates in terms of the
        P T and V gates defined here. 
        """
        q_resolved = self.qcircuit.resolve_gates(["CNOT", "RX", "RZ"])
        q_icm = QubitCircuit(q_resolved.N)

        # Check all gates and replace with P, T, V
        
        for gate in q_resolved.gates:
            if gate.targets is not None:
                if gate.name == "CNOT":
                    q_icm.add_gate(gate.name,
                                   targets=gate.targets,
                                   controls=gate.controls,
                                   arg_value=gate.arg_value,
                                   arg_label=gate.arg_label)

                elif gate.arg_label == r"\pi":
                    if gate.name == "RX":
                        name = "V"
                    elif gate.name == "RZ":
                        name = "T"
                    q_icm.add_gate(name, targets=gate.targets,
                                   controls=gate.controls, arg_value=np.pi / 2,
                                   arg_label=r"\pi/2")
                    q_icm.add_gate(name, targets=gate.targets,
                                   controls=gate.controls, arg_value=np.pi / 2,
                                   arg_label=r"\pi/2")

                else :                
                    replaced_gate = replace_gate(gate) 
                    q_icm.add_gate(replaced_gate.name, targets=replaced_gate.targets,
                                   controls=replaced_gate.controls,
                                   arg_value=replaced_gate.arg_value,
                                   arg_label=replaced_gate.arg_label)
        return (q_icm)