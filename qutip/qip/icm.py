"""
A representation of a quantum circuit consiting of qubit initialisation (I),
controlled NOT gates (C) and measurements (M) with respect to different bases.
"""
import numpy as np

from qutip import QubitCircuit
from qutip.qip.circuit import Gate


def icm_label(gate_name, arg_label):
    """
    A function to get the ICM labels for rotation gates based on the argument
    label which defines the rotation angle.

    Parameters
    ----------
    gate_name: str
        The rotation axis for the gate (RX, RZ)

    arg_label: str
        The angle of rotation.
        Specify the angle as a raw string - r"\pi/2", r"\pi/4".

    Returns
    -------
    icm_label: str
        A string specifying the corresponding ICM gate. If the gate is not an
        ICM gate then None is returned.
    """
    # Define a dictionary for ICM gates
    icm_gates = {("RZ", r"\pi/2"): "P", ("RZ", r"\pi/4"): "T",
                 ("RX", r"\pi/2"): "V", ("RZ", r"-\pi/2"): "P_dagger",
                 ("RZ", r"-\pi/4"): "T_dagger", ("RX", r"-\pi/2"): "V_dagger"}
    icm_label = None
    if (gate_name, arg_label) in icm_gates.keys():
        icm_label = icm_gates[(gate_name, arg_label)]
    return (icm_label)


def pgate(targets=None, dagger=False):
    """
    The P gate. It is a rotation about RZ by pi/2.

    Parameters
    ----------
    targets: list
        A list of targets. Although this will be just one value but
        to maintain consistency with the `Gate` class, we pass a list.
    dagger : bool
        Return the P or P_dagger gate
        default: False

    Returns
    -------
    P: Gate
        The P gate
    """
    arg_value = np.pi / 2
    arg_label = r"\pi/2"

    if dagger:
        arg_value = - np.pi / 2
        arg_label = r"-\pi/2"

    return Gate("RZ", targets=targets,
                arg_value=arg_value, arg_label=arg_label)


def tgate(targets=None, dagger=False):
    """
    The T gate. It is a rotation about RZ by pi/4.

    Parameters
    ----------
    targets: list
        A list of targets. Although this will be just one value but
        to maintain consistency with the `Gate` class, we pass a list.
    dagger : bool
        Return the T or T_dagger gate
        default: False

    Returns
    -------
    T: Gate
        The T or T_dagger gate.
    """
    arg_value = np.pi / 4
    arg_label = r"\pi/4"

    if dagger:
        arg_value = - np.pi / 4
        arg_label = r"-\pi/4"
    return Gate("RZ", targets=targets,
                arg_value=arg_value, arg_label=arg_label)


def vgate(targets=None, dagger=False):
    """
    The V gate. It is a rotation about RX by pi/2.

    Parameters
    ----------
    targets: list
        A list of targets. Although this will be just one value but
        to maintain consistency with the `Gate` class, we pass a list.
    dagger : bool
        Return the V or V_dagger gate.
        default: False

    Returns
    -------
    V: Gate
        The V or V_dagger gate.
    """
    arg_value = np.pi / 2
    arg_label = r"\pi/2"

    if dagger:
        arg_value = - np.pi / 2
        arg_label = r"-\pi/2"
    return Gate("RX", targets=targets,
                arg_value=arg_value, arg_label=arg_label)


def decompose_toffoli(qcircuit):
    """
    Decompose the TOFFOLI gates in qcircuit to CNOT, H, P and T gates.

    Parameters
    ----------
    qcircuit: QubitCircuit
        The circuit containing `TOFFOLI` gates.

    Returns
    -------
    decomposed: QubitCircuit
        The circuit with TOFFOLI gates decomposed.
    """
    decomposed = QubitCircuit(qcircuit.N, reverse_states=False)

    for idx, gate in enumerate(qcircuit.gates):
        if gate.name == "TOFFOLI":
            c1 = gate.controls[0]
            c2 = gate.controls[1]
            t = gate.targets[0]

            decomposed.add_gate(Gate("SNOT", targets=[t]))
            decomposed.add_gate(Gate("CNOT", targets=[t], controls=[c2]))
            decomposed.add_gate(tgate(targets=[t], dagger=True))
            decomposed.add_gate(Gate("CNOT", targets=[t], controls=[c1]))
            decomposed.add_gate(tgate(targets=[t], dagger=False))
            decomposed.add_gate(Gate("CNOT", targets=[t], controls=[c2]))
            decomposed.add_gate(tgate(targets=[t], dagger=True))
            decomposed.add_gate(Gate("CNOT", targets=[t], controls=[c1]))
            decomposed.add_gate(tgate(targets=[c2], dagger=True))
            decomposed.add_gate(tgate(targets=[t], dagger=False))
            decomposed.add_gate(Gate("SNOT", targets=[t]))
            decomposed.add_gate(Gate("CNOT", targets=[c2], controls=[c1]))
            decomposed.add_gate(tgate(targets=[c2], dagger=True))
            decomposed.add_gate(Gate("CNOT", targets=[c2], controls=[c1]))
            decomposed.add_gate(tgate(targets=[c1], dagger=False))
            decomposed.add_gate(pgate(targets=[c2], dagger=False))

        else:
            decomposed.add_gate(gate)

    return decomposed


def decompose_SNOT(qcircuit):
    """
    Decompose the SNOT gates in qcircuit to PVP.

    Parameters
    ----------
    qcircuit: QubitCircuit
        The circuit containing `SNOT` gates.

    Returns
    -------
    decomposed: QubitCircuit
        The circuit with SNOT gates decomposed as PVP
    """
    decomposed = QubitCircuit(qcircuit.N, reverse_states=False)

    for gate in qcircuit.gates:
        if gate.name == "SNOT":
            target = gate.targets
            decomposed.add_gate(pgate(targets=target))
            decomposed.add_gate(vgate(targets=target))
            decomposed.add_gate(pgate(targets=target))
        else:
            decomposed.add_gate(gate)

    return decomposed


def replace_P(cicm, idx):
    """
    Convert P gates in the given quantum circuit to the ICM implementation.
    Use gate teleportation.

    Parameters
    ----------
    cicm: QubitCircuit

    Returns
    -------
    cicm: QubitCircuit
        The quantum circuit with P gates implemeted using Gate teleportation.
    """
    gate = cicm.gates[idx]
    if icm_label(gate.name, gate.arg_label) == "P":
        t = gate.targets[0]
        # Move targets and controls of all gates before the P gate
        cicm.gates[idx] = Gate("Y", targets=[t + 1],
                               arg_label=r"ancilla")

        for gate in cicm.gates:
            if gate.arg_label in ["ancilla",
                                  "measurement",
                                  "correction", "teleported"]:
                continue

            if gate.targets[0] > t:
                gate.targets[0] += 1

            if gate.name == "CNOT" and gate.controls[0] > t:
                gate.controls[0] += 1

        CNOT = Gate("CNOT", targets=[t], controls=[t + 1],
                    arg_label="teleported")
        measurement = Gate("z", targets=[t], arg_label=r"measurement")
        correction = Gate("xz", controls=[t], targets=[t + 1],
                          arg_label=r"correction")

        cicm.gates.insert(idx + 1, CNOT)
        cicm.gates.insert(idx + 2, measurement)
        cicm.gates.insert(idx + 3, correction)

        # Move targets and controls of all gates after the P gate

        for gate in cicm.gates[idx+4:]:
            if gate.arg_label in ["ancilla", "measurement",
                                  "correction", "teleported"]:
                continue

            if gate.targets[0] == t:
                gate.targets[0] += 1

        cicm.N += 1
    return (cicm)


def replace_V(cicm, idx):
    """
    Convert V gates in the given quantum circuit to the ICM implementation.
    Use gate teleportation.

    Parameters
    ----------
    cicm: QubitCircuit

    Returns
    -------
    cicm: QubitCircuit
        The quantum circuit with P gates implemeted using Gate teleportation.
    """
    gate = cicm.gates[idx]
    if icm_label(gate.name, gate.arg_label) == "V":
        t = gate.targets[0]
        # Move targets and controls of all gates before the P gate
        cicm.gates[idx] = Gate("Y", targets=[t + 1],
                               arg_label=r"ancilla")
        for gate in cicm.gates:
            if gate.arg_label in ["ancilla",
                                  "measurement",
                                  "correction", "teleported"]:
                continue

            if gate.targets[0] > t:
                gate.targets[0] += 1

            if gate.name == "CNOT" and gate.controls[0] > t:
                gate.controls[0] += 1

        CNOT = Gate("CNOT", targets=[t + 1], controls=[t],
                    arg_label="teleported")
        measurement = Gate("x", targets=[t], arg_label=r"measurement")
        correction = Gate("x/z", controls=[t], targets=[t + 1],
                          arg_label=r"correction")

        cicm.gates.insert(idx + 1, CNOT)
        cicm.gates.insert(idx + 2, measurement)
        cicm.gates.insert(idx + 3, correction)

        # Move targets and controls of all gates after the P gate
        for gate in cicm.gates[idx+4:]:
            if gate.arg_label in ["ancilla", "measurement",
                                  "correction", "teleported"]:
                continue

            if gate.targets[0] == t:
                gate.targets[0] += 1

        cicm.N += 1
    return (cicm)


def replace_T(cicm, idx):
    """
    Convert T gates in the given quantum circuit to the ICM implementation.
    Use gate teleportation.

    Parameters
    ----------
    cicm: QubitCircuit

    Returns
    -------
    cicm: QubitCircuit
        The quantum circuit with P gates implemeted using Gate teleportation.
    """
    gate = cicm.gates[idx]
    if icm_label(gate.name, gate.arg_label) == "T":
        t = gate.targets[0]
        # Move targets and controls of all gates before the T gate
        for gate in cicm.gates:
            if gate.arg_label in ["ancilla", "measurement",
                                  "correction", "teleported"]:
                continue

            if gate.targets[0] > t:
                gate.targets[0] += 5

            if gate.name == "CNOT" and gate.controls[0] > t:
                gate.controls[0] += 5

        cicm.gates[idx] = Gate("a", targets=[t + 1],
                               arg_label=r"ancilla")
        cicm.gates.insert(idx + 1, Gate("0", targets=[t + 2],
                          arg_label=r"ancilla"))
        cicm.gates.insert(idx + 2, Gate("y", targets=[t + 3],
                          arg_label=r"ancilla"))
        cicm.gates.insert(idx + 3, Gate("+", targets=[t + 4],
                          arg_label=r"ancilla"))
        cicm.gates.insert(idx + 4, Gate("0", targets=[t + 5],
                          arg_label=r"ancilla"))

        cicm.gates.insert(idx + 5, Gate("CNOT", targets=[t],
                          controls=[t+1], arg_label="teleported"))
        cicm.gates.insert(idx + 6, Gate("CNOT", targets=[t + 2],
                          controls=[t + 1], arg_label="teleported"))
        cicm.gates.insert(idx + 7, Gate("CNOT", targets=[t + 1],
                          controls=[t + 3], arg_label="teleported"))
        cicm.gates.insert(idx + 8, Gate("CNOT", targets=[t + 2],
                          controls=[t + 4], arg_label="teleported"))
        cicm.gates.insert(idx + 9, Gate("CNOT", targets=[t + 5],
                          controls=[t + 3], arg_label="teleported"))
        cicm.gates.insert(idx + 10, Gate("CNOT", targets=[t + 5],
                          controls=[t + 4], arg_label="teleported"))
        cicm.gates.insert(idx + 11, Gate("z", targets=[t],
                          arg_label="measurement"))
        cicm.gates.insert(idx + 12, Gate("z/x", targets=[t + 1],
                          controls=[t], arg_label="correction"))
        cicm.gates.insert(idx + 13, Gate("x/z", targets=[t + 2],
                          controls=[t+1], arg_label="correction"))
        cicm.gates.insert(idx + 14, Gate("x/z", targets=[t + 3],
                          controls=[t+2], arg_label="correction"))
        cicm.gates.insert(idx + 15, Gate("z/x", targets=[t + 4],
                          controls=[t+3], arg_label="correction"))

        # Move targets and controls of all gates after the T gate

        for gate in cicm.gates[idx+16:]:
            if gate.arg_label in ["ancilla", "measurement",
                                  "correction", "teleported"]:
                continue
            if gate.targets[0] == t:
                gate.targets[0] += 5

        cicm.N += 5
    return (cicm)


def visualise(icm_circuit):
    """
    Push all inputs and ancilla to the beginning of circuit, CNOT's in
    the middle, measurement, correction and output in the end.

    Parameters
    ----------
    icm_circuit: QubitCircuit
        The icm circuit after conversion.

    Returns
    -------
    cicm: QubitCircuit
        The ICM converted circuit with inputs and ancilla in the beginning,
        CNOT's in the middle, measurements, corrections and outputs in the end

    icm_dict: dict
        The icm circuit as a dictionary. This is used to generate the geometric
        description.
    """
    bits = [i for i in range(icm_circuit.N)]

    inputs = []
    outputs = []

    initializations = []
    cnots = []
    measurements = []

    corrections = []

    cicm = QubitCircuit(icm_circuit.N, reverse_states=False)

    for gate in icm_circuit.gates:
        if gate.name == "CNOT":
            cnots += [(gate.controls[0], gate.targets[0])]
        if gate.arg_label == "ancilla":
            initializations += [(gate.targets[0], gate.name)]
        if gate.arg_label == "measurement":
            measurements += [(gate.targets[0], gate.name)]
        if gate.arg_label == "correction":
            corrections += [(gate.targets[0], gate.name, gate.controls[0])]
        if gate.arg_label == "input":
            inputs += [gate.targets[0]]
        if gate.arg_label == "output":
            outputs += [gate.targets[0]]

    for item in inputs:
        cicm.add_gate("IN", arg_label="input", targets=[item])

    for item in initializations:
        cicm.add_gate(item[1], arg_label="ancilla", targets=item[0])

    for item in cnots:
        cicm.add_gate("CNOT", controls=[item[0]], targets=[item[1]])

    for item in measurements:
        cicm.add_gate(item[1], arg_label="measurement", targets=[item[0]])

    for item in corrections:
        cicm.add_gate(item[1], arg_label="correction", controls=[item[0]-1],
                       targets=[item[0]])

    for item in outputs:
        cicm.add_gate("OUT", arg_label="output", targets=[item])

    _inits = [{"bit": item[0], "type": item[1]} for item in initializations]
    _measures = [{"bit": item[0], "type": item[1]} for item in measurements]
    _cnots = [{"controls": [item[0]], "targets": [item[1]]} for item in cnots]

    icm_dict = {"bits": bits, "inputs": inputs, "outputs": outputs,
                "initializations": _inits,
                "measurements": _measures,
                "cnots": _cnots}

    return (cicm, icm_dict)


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

    def ancilla_cost(self):
        """
        for ICM representation of a given quantum circuit decomposed into
        P, T, V SNOT and Toffoli gates. The P, T, V gates are implemented using
        ancilla qubits and gate teleportation requiring CNOT and measurement.
        Each T gate requires 5 ancillae and 6 CNOT gates. The P and V gates
        require 1 ancilla and 1 CNOT gate. Each Hadamard gate is implemented
        using a sequence of P and V gates requiring 3 extra ancillae and gates.
        The Toffoli gate requires 55 extra gates and 42 ancillae.

        Returns
        -------
        ancilla_cost: dict
            A dictionary which gives the ancilla count for each type of gate.

        References
        ----------
        .. [1] arXiv:1509.03962v1 [quant-ph]
        """
        decomposed_circuit = self.qcircuit
        cost = dict({"P": 0, "T": 0, "V": 0,
                             "SNOT": 0, "TOFFOLI": 0})

        for gate in decomposed_circuit.gates:
            if gate.name == "CNOT":
                continue

            elif gate.name == "SNOT":
                cost["SNOT"] += 3

            elif gate.name == "TOFFOLI":
                cost["TOFFOLI"] += 42

            else:
                icm_gate = icm_label(gate.name, gate.arg_label)
                if icm_gate is None:
                    raise ValueError("Gate decomposition is not in ICM basis")

                if icm_gate == "P" or icm_gate == "P_dagger":
                    cost["P"] += 1

                if icm_gate == "T" or icm_gate == "T_dagger":
                    cost["T"] += 5

                if icm_gate == "V" or icm_gate == "V_dagger":
                    cost["V"] += 1

        return (cost)

    def to_icm(self):
        """
        A function to convert the initially decomposed circuit to the ICM
        model.

        We first resolve all the gates into TOFFOLI, SNOT, RX, RZ and CNOT
        gates. This is done by using `decompose_gates`. Then we use the
        algorithm outlined in [1] to implement each gate using ancilla qubits
        and gate teleportation. Ancilla cost can be calculated by the
        function `ancilla_cost`.

        Returns
        -------
        icm_circuit: QubitCircuit
            The ICM representation of the given quantum circuit.
            Converts `self.qcircuit` into a ICM representation.
        """
        # Replace "TOFFOLI" and "SNOT" with their equivalent ICM representation
        decomposed_toffoli = decompose_toffoli(self.qcircuit)
        decomposed_circuit = decompose_SNOT(decomposed_toffoli)
        icm_circuit = QubitCircuit(decomposed_circuit.N, reverse_states=False)

        for i in range(decomposed_circuit.N):
            icm_circuit.add_gate("IN", arg_label="input", targets=[i])

        for gate in decomposed_circuit.gates:
            icm_circuit.add_gate(gate)

        for i in range(decomposed_circuit.N):
            icm_circuit.add_gate("OUT", arg_label="output", targets=[i])

        idx = 0
        while idx < len(icm_circuit.gates):
            name = icm_circuit.gates[idx].name
            arg_label = icm_circuit.gates[idx].arg_label

            if icm_label(name, arg_label) == "P":
                icm_circuit = replace_P(icm_circuit, idx)
            if icm_label(name, arg_label) == "V":
                icm_circuit = replace_V(icm_circuit, idx)
            if icm_label(name, arg_label) == "T":
                icm_circuit = replace_T(icm_circuit, idx)

            idx += 1

        return icm_circuit
