# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

from collections.abc import Iterable
import warnings
import inspect

import numpy as np

import numpy as np

from qutip.qip.circuit_latex import _latex_compile
from qutip.qip.gates import *
from qutip.qip.qubits import qubit_states

__all__ = ['Gate', 'QubitCircuit']


class Gate(object):
    """
    Representation of a quantum gate, with its required parametrs, and target
    and control qubits.

    Parameters
    ----------
    name : string
        Gate name.
    targets : list or int
        Gate targets.
    controls : list or int
        Gate controls.
    arg_value : float
        Argument value(phi).
    arg_label : string
        Label for gate representation.
    """

    def __init__(self, name, targets=None, controls=None, arg_value=None,
                 arg_label=None):
        """
        Create a gate with specified parameters.
        """
        self.name = name
        self.targets = None
        self.controls = None

        if not isinstance(targets, Iterable) and targets is not None:
            self.targets = [targets]
        else:
            self.targets = targets

        if not isinstance(controls, Iterable) and controls is not None:
            self.controls = [controls]
        else:
            self.controls = controls

        for ind_list in [self.targets, self.controls]:
            if isinstance(ind_list, Iterable):
                all_integer = all(
                    [isinstance(ind, np.int) for ind in ind_list])
                if not all_integer:
                    raise ValueError("Index of a qubit must be an integer")

        if name in ["SWAP", "ISWAP", "SQRTISWAP", "SQRTSWAP", "BERKELEY",
                    "SWAPalpha"]:
            if (self.targets is None) or (len(self.targets) != 2):
                raise ValueError("Gate %s requires two targets" % name)
            if self.controls is not None:
                raise ValueError("Gate %s cannot have a control" % name)

        elif name in ["CNOT", "CSIGN", "CRX", "CRY", "CRZ"]:
            if self.targets is None or len(self.targets) != 1:
                raise ValueError("Gate %s requires one target" % name)
            if self.controls is None or len(self.controls) != 1:
                raise ValueError("Gate %s requires one control" % name)

        elif name in ["SNOT", "RX", "RY", "RZ", "PHASEGATE"]:
            if self.controls is not None:
                raise ValueError("Gate %s does not take controls" % name)

        elif name in ["RX", "RY", "RZ", "CPHASE", "SWAPalpha", "PHASEGATE",
                            "GLOBALPHASE", "CRX", "CRY", "CRZ"]:
            if arg_value is None:
                raise ValueError("Gate %s requires an argument value" % name)

        self.arg_value = arg_value
        self.arg_label = arg_label

    def __str__(self):
        s = "Gate(%s, targets=%s, controls=%s)" % (self.name,
                                                   self.targets,
                                                   self.controls)
        return s

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)


_gate_name_to_label = {
    'RX': r'R_x',
    'RY': r'R_y',
    'RZ': r'R_z',
    'CRX': r'R_x',
    'CRY': r'R_y',
    'CRZ': r'R_z',
    'SQRTNOT': r'\sqrt{\rm NOT}',
    'SNOT': r'{\rm H}',
    'PHASEGATE': r'{\rm PHASE}',
    'CPHASE': r'{\rm R}',
    'CNOT': r'{\rm CNOT}',
    'CSIGN': r'{\rm Z}',
    'BERKELEY': r'{\rm BERKELEY}',
    'SWAPalpha': r'{\rm SWAPalpha}',
    'SWAP': r'{\rm SWAP}',
    'ISWAP': r'{i}{\rm SWAP}',
    'SQRTSWAP': r'\sqrt{\rm SWAP}',
    'SQRTISWAP': r'\sqrt{{i}\rm SWAP}',
    'FREDKIN': r'{\rm FREDKIN}',
    'TOFFOLI': r'{\rm TOFFOLI}',
    'GLOBALPHASE': r'{\rm Ph}',
}


def _gate_label(name, arg_label):

    if name in _gate_name_to_label:
        gate_label = _gate_name_to_label[name]
    else:
        warnings.warn("Unknown gate %s" % name)
        gate_label = name

    if arg_label:
        return r'%s(%s)' % (gate_label, arg_label)
    else:
        return r'%s' % gate_label


class QubitCircuit(object):
    """
    Representation of a quantum program/algorithm, maintaining a sequence
    of gates.

    Parameters
    ----------
    N : int
        Number of qubits in the system.
    user_gates : dict
        Define a dictionary of the custom gates. See examples for detail.
    input_states : list
        A list of string such as `0`,'+', "A", "Y". Only used for latex.

    Examples
    --------
    >>> def user_gate():
    ...     mat = np.array([[1.,   0],
    ...                     [0., 1.j]])
    ...     return Qobj(mat, dims=[[2], [2]])
    >>> qc.QubitCircuit(2, user_gates={"T":user_gate})
    >>> qc.add_gate("T", targets=[0])
    """

    def __init__(self, N, input_states=None, output_states=None,
                 reverse_states=True, user_gates=None):
        # number of qubits in the register
        self.N = N
        self.reverse_states = reverse_states
        self.gates = []
        self.U_list = []
        self.input_states = [None for i in range(N)]
        self.output_states = [None for i in range(N)]
        if user_gates is None:
            self.user_gates = {}
        else:
            if isinstance(user_gates, dict):
                self.user_gates = user_gates
            else:
                raise ValueError(
                    "`user_gate` takes a python dictionary of the form"
                    "{{str: gate_function}}, not {}".format(user_gates))

    def add_state(self, state, targets=None, state_type="input"):
        """
        Add an input or ouput state to the circuit. By default all the input
        and output states will be initialized to `None`. A particular state can
        be added by specifying the state and the qubit where it has to be added
        along with the type as input or output.

        Parameters
        ----------
        state: str
            The state that has to be added. It can be any string such as `0`,
            '+', "A", "Y"
        targets: list
            A list of qubit positions where the given state has to be added.
        state_type: str
            One of either "input" or "output". This specifies whether the state
            to be added is an input or output.
            default: "input"

        """
        if state_type == "input":
            for i in targets:
                self.input_states[i] = state
        if state_type == "output":
            for i in targets:
                self.output_states[i] = state

    def add_gate(self, gate, targets=None, controls=None, arg_value=None,
                 arg_label=None, index=None):
        """
        Adds a gate with specified parameters to the circuit.

        Parameters
        ----------
        gate: string or `Gate`
            Gate name. If gate is an instance of `Gate`, parameters are
            unpacked and added.
        targets: list
            Gate targets.
        controls: list
            Gate controls.
        arg_value: float
            Argument value(phi).
        arg_label: string
            Label for gate representation.
        index : list
            Positions to add the gate.
        """
        if isinstance(gate, Gate):
            name = gate.name
            targets = gate.targets
            controls = gate.controls
            arg_value = gate.arg_value
            arg_label = gate.arg_label

        else:
            name = gate

        if index is None:
            self.gates.append(Gate(name, targets=targets, controls=controls,
                                   arg_value=arg_value, arg_label=arg_label))

        else:
            for position in index:
                self.gates.insert(position, Gate(name, targets=targets,
                                                 controls=controls,
                                                 arg_value=arg_value,
                                                 arg_label=arg_label))

    def add_1q_gate(self, name, start=0, end=None, qubits=None,
                    arg_value=None, arg_label=None):
        """
        Adds a single qubit gate with specified parameters on a variable
        number of qubits in the circuit. By default, it applies the given gate
        to all the qubits in the register.

        Parameters
        ----------
        name : string
            Gate name.
        start : int
            Starting location of qubits.
        end : int
            Last qubit for the gate.
        qubits : list
            Specific qubits for applying gates.
        arg_value : float
            Argument value(phi).
        arg_label : string
            Label for gate representation.
        """
        if name not in ["RX", "RY", "RZ", "SNOT", "SQRTNOT", "PHASEGATE"]:
            raise ValueError("%s is not a single qubit gate" % name)

        if qubits is not None:
            for i in range(len(qubits)):
                self.gates.append(Gate(name, targets=qubits[i], controls=None,
                                       arg_value=arg_value,
                                       arg_label=arg_label))

        else:
            if end is None:
                end = self.N - 1
            for i in range(start, end):
                self.gates.append(Gate(name, targets=i, controls=None,
                                       arg_value=arg_value,
                                       arg_label=arg_label))

    def add_circuit(self, qc, start=0):
        """
        Adds a block of a qubit circuit to the main circuit.
        Globalphase gates are not added.

        Parameters
        ----------
        qc : QubitCircuit
            The circuit block to be added to the main circuit.
        start : int
            The qubit on which the first gate is applied.
        """
        if self.N - start < qc.N:
            raise NotImplementedError("Targets exceed number of qubits.")

        for gate in qc.gates:
            if gate.name in ["RX", "RY", "RZ", "SNOT", "SQRTNOT", "PHASEGATE"]:
                self.add_gate(gate.name, gate.targets[0] + start, None,
                              gate.arg_value, gate.arg_label)
            elif gate.name in ["CPHASE", "CNOT", "CSIGN", "CRX", "CRY", "CRZ"]:
                self.add_gate(gate.name, gate.targets[0] + start,
                              gate.controls[0] + start, gate.arg_value,
                              gate.arg_label)
            elif gate.name in ["BERKELEY", "SWAPalpha", "SWAP", "ISWAP",
                               "SQRTSWAP", "SQRTISWAP"]:
                self.add_gate(gate.name, None,
                              [gate.controls[0] + start,
                               gate.controls[1] + start], None, None)
            elif gate.name in ["TOFFOLI"]:
                self.add_gate(gate.name, gate.targets[0] + start,
                              [gate.controls[0] + start,
                               gate.controls[1] + start], None, None)
            elif gate.name in ["FREDKIN"]:
                self.add_gate(gate.name,
                              [gate.targets[0] + start,
                               gate.targets[1] + start],
                              gate.controls + start, None, None)
            elif gate.name in self.user_gates:
                self.add_gate(
                              gate.name, targets=gate.targets,
                              arg_value=gate.arg_value)

    def remove_gate(self, index=None, end=None, name=None, remove="first"):
        """
        Remove a gate from a specific index or between two indexes or the
        first, last or all instances of a particular gate.

        Parameters
        ----------
        index : int
            Location of gate to be removed.
        name : string
            Gate name to be removed.
        remove : string
            If first or all gate are to be removed.
        """
        if index is not None and index <= self.N:
            if end is not None and end <= self.N:
                for i in range(end - index):
                    self.gates.pop(index + i)
            elif end is not None and end > self.N:
                raise ValueError("End target exceeds number of gates.")
            else:
                self.gates.pop(index)

        elif name is not None and remove == "first":
            for gate in self.gates:
                if name == gate.name:
                    self.gates.remove(gate)
                    break

        elif name is not None and remove == "last":
            for i in range(self.N + 1):
                if name == self.gates[self.N - i].name:
                    self.gates.remove(self.gates[self.N - i])
                    break

        elif name is not None and remove == "all":
            for j in range(self.N + 1):
                if name == self.gates[self.N - j].name:
                    self.gates.remove(self.gates[self.N - j])

        else:
            self.gates.pop()

    def reverse_circuit(self):
        """
        Reverse an entire circuit of unitary gates.

        Returns
        ----------
        qc : QubitCircuit
            Return QubitCircuit of resolved gates for the qubit circuit in the
            reverse order.

        """
        temp = QubitCircuit(self.N, self.reverse_states)

        for gate in reversed(self.gates):
            temp.add_gate(gate)

        return temp

    def resolve_gates(self, basis=["CNOT", "RX", "RY", "RZ"]):
        """
        Unitary matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right in the specified
        basis.

        Parameters
        ----------
        basis : list.
            Basis of the resolved circuit.

        Returns
        -------
        qc : QubitCircuit
            Return QubitCircuit of resolved gates for the qubit circuit in the
            desired basis.
        """
        qc_temp = QubitCircuit(self.N, self.reverse_states)
        temp_resolved = []

        basis_1q_valid = ["RX", "RY", "RZ"]
        basis_2q_valid = ["CNOT", "CSIGN", "ISWAP", "SQRTSWAP", "SQRTISWAP"]

        if isinstance(basis, list):
            basis_1q = []
            basis_2q = []
            for gate in basis:
                if gate in basis_2q_valid:
                    basis_2q.append(gate)
                elif gate in basis_1q_valid:
                    basis_1q.append(gate)
                else:
                    raise NotImplementedError(
                        "%s is not a valid basis gate" % gate)
            if len(basis_1q) == 1:
                raise ValueError("Not sufficient single-qubit gates in basis")
            elif len(basis_1q) == 0:
                basis_1q = ["RX", "RY", "RZ"]

        else:  # only one 2q gate is given as basis
            basis_1q = ["RX", "RY", "RZ"]
            if basis in basis_2q_valid:
                basis_2q = [basis]
            else:
                raise ValueError("%s is not a valid two-qubit basis gate"
                                 % basis)

        for gate in self.gates:
            if gate.name == "RX":
                temp_resolved.append(gate)
            elif gate.name == "RY":
                temp_resolved.append(gate)
            elif gate.name == "RZ":
                temp_resolved.append(gate)
            elif gate.name == "SQRTNOT":
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi / 4,
                                          arg_label=r"\pi/4"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
            elif gate.name == "SNOT":
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
            elif gate.name == "PHASEGATE":
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=gate.arg_value / 2,
                                          arg_label=gate.arg_label))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          gate.arg_value, gate.arg_label))
            elif gate.name in basis_2q:  # ignore all gate in 2q basis
                temp_resolved.append(gate)
            elif gate.name == "CPHASE":
                raise NotImplementedError("Cannot be resolved in this basis")
            elif gate.name == "CNOT":
                temp_resolved.append(gate)
            elif gate.name == "CSIGN":
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("CNOT", gate.targets, gate.controls))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
            elif gate.name == "BERKELEY":
                raise NotImplementedError("Cannot be resolved in this basis")
            elif gate.name == "SWAPalpha":
                raise NotImplementedError("Cannot be resolved in this basis")
            elif gate.name == "SWAP":
                if "ISWAP" in basis_2q:  # dealed with separately
                    temp_resolved.append(gate)
                else:
                    temp_resolved.append(
                        Gate("CNOT", gate.targets[0], gate.targets[1]))
                    temp_resolved.append(
                        Gate("CNOT", gate.targets[1], gate.targets[0]))
                    temp_resolved.append(
                        Gate("CNOT", gate.targets[0], gate.targets[1]))
            elif gate.name == "ISWAP":
                temp_resolved.append(Gate("CNOT", gate.targets[0],
                                          gate.targets[1]))
                temp_resolved.append(Gate("CNOT", gate.targets[1],
                                          gate.targets[0]))
                temp_resolved.append(Gate("CNOT", gate.targets[0],
                                          gate.targets[1]))
                temp_resolved.append(Gate("RZ", gate.targets[0], None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets[0], None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("CNOT", gate.targets[0],
                                          gate.targets[1]))
                temp_resolved.append(Gate("RY", gate.targets[0], None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
            elif gate.name == "SQRTSWAP":
                raise NotImplementedError("Cannot be resolved in this basis")
            elif gate.name == "SQRTISWAP":
                raise NotImplementedError("Cannot be resolved in this basis")
            elif gate.name == "FREDKIN":
                temp_resolved.append(Gate("CNOT", gate.targets[0],
                                          gate.targets[1]))
                temp_resolved.append(Gate("CNOT", gate.targets[0],
                                          gate.controls))
                temp_resolved.append(Gate("RZ", gate.controls, None,
                                          arg_value=np.pi / 8,
                                          arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", [gate.targets[0]], None,
                                          arg_value=-np.pi / 8,
                                          arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets[0],
                                          gate.controls))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets[1], None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=-np.pi / 2,
                                          arg_label=r"-\pi/2"))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RZ", gate.targets[0], None,
                                          arg_value=np.pi / 8,
                                          arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                          arg_value=np.pi / 8,
                                          arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets[1],
                                          gate.controls))
                temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                          arg_value=-np.pi / 8,
                                          arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets[1],
                                          gate.targets[0]))
                temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                          arg_value=np.pi / 8,
                                          arg_label=r"\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets[1],
                                          gate.controls))
                temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                          arg_value=-np.pi / 8,
                                          arg_label=r"-\pi/8"))
                temp_resolved.append(Gate("CNOT", gate.targets[1],
                                          gate.targets[0]))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets[1], None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=-np.pi / 2,
                                          arg_label=r"-\pi/2"))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("CNOT", gate.targets[0],
                                          gate.targets[1]))

            elif gate.name == "TOFFOLI":
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=1 * np.pi / 8,
                                          arg_label=r"\pi/8"))
                temp_resolved.append(Gate("RZ", gate.controls[1], None,
                                          arg_value=np.pi/2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RZ", gate.controls[0], None,
                                          arg_value=np.pi / 4,
                                          arg_label=r"\pi/4"))
                temp_resolved.append(Gate("CNOT", gate.controls[1],
                                          gate.controls[0]))
                temp_resolved.append(Gate("RZ", gate.controls[1], None,
                                          arg_value=-np.pi / 4,
                                          arg_label=r"-\pi/4"))
                temp_resolved.append(Gate("CNOT", gate.controls[1],
                                          gate.controls[0]))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))
                temp_resolved.append(Gate("RZ", gate.controls[1], None,
                                          arg_value=-np.pi / 4,
                                          arg_label=r"-\pi/4"))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi / 4,
                                          arg_label=r"\pi/4"))
                temp_resolved.append(Gate("CNOT", gate.targets,
                                          gate.controls[0]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=-np.pi / 4,
                                          arg_label=r"-\pi/4"))
                temp_resolved.append(Gate("CNOT", gate.targets,
                                          gate.controls[1]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=np.pi / 4,
                                          arg_label=r"\pi/4"))
                temp_resolved.append(Gate("CNOT", gate.targets,
                                          gate.controls[0]))
                temp_resolved.append(Gate("RZ", gate.targets, None,
                                          arg_value=-np.pi / 4,
                                          arg_label=r"-\pi/4"))
                temp_resolved.append(Gate("CNOT", gate.targets,
                                          gate.controls[1]))
                temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RY", gate.targets, None,
                                          arg_value=np.pi / 2,
                                          arg_label=r"\pi/2"))
                temp_resolved.append(Gate("RX", gate.targets, None,
                                          arg_value=np.pi, arg_label=r"\pi"))

            elif gate.name == "GLOBALPHASE":
                temp_resolved.append(Gate(gate.name, gate.targets,
                                          gate.controls,
                                          gate.arg_value, gate.arg_label))
            else:
                raise NotImplementedError(
                    "Gate {} "
                    "cannot be resolved.".format(gate.name))

        if "CSIGN" in basis_2q:
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("CSIGN", gate.targets,
                                              gate.controls))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        elif "ISWAP" in basis_2q:
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=np.pi / 4,
                                              arg_label=r"\pi/4"))
                    qc_temp.gates.append(Gate("ISWAP", [gate.controls[0],
                                                        gate.targets[0]],
                                              None))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RY", gate.controls, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("ISWAP", [gate.controls[0],
                                                        gate.targets[0]],
                                              None))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                elif gate.name == "SWAP":
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=np.pi / 4,
                                              arg_label=r"\pi/4"))
                    qc_temp.gates.append(Gate("ISWAP", gate.targets, None))
                    qc_temp.gates.append(Gate("RX", gate.targets[0], None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("ISWAP", gate.targets, None))
                    qc_temp.gates.append(Gate("RX", gate.targets[1], None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("ISWAP", [gate.targets[1],
                                                        gate.targets[0]],
                                              None))
                    qc_temp.gates.append(Gate("RX", gate.targets[0], None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        elif "SQRTSWAP" in basis_2q:
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("SQRTSWAP", [gate.controls[0],
                                                           gate.targets[0]],
                                              None))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=np.pi,
                                              arg_label=r"\pi"))
                    qc_temp.gates.append(Gate("SQRTSWAP", [gate.controls[0],
                                                           gate.targets[0]],
                                              None))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        elif "SQRTISWAP" in basis_2q:
            for gate in temp_resolved:
                if gate.name == "CNOT":
                    qc_temp.gates.append(Gate("RY", gate.controls, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RX", gate.controls, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("SQRTISWAP", [gate.controls[0],
                                                            gate.targets[0]],
                                              None))
                    qc_temp.gates.append(Gate("RX", gate.controls, None,
                                              arg_value=np.pi,
                                              arg_label=r"\pi"))
                    qc_temp.gates.append(Gate("SQRTISWAP", [gate.controls[0],
                                                            gate.targets[0]],
                                              None))
                    qc_temp.gates.append(Gate("RY", gate.controls, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=np.pi / 4,
                                              arg_label=r"\pi/4"))
                    qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                              arg_value=np.pi,
                                              arg_label=r"\pi"))
                    qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                              arg_value=3 * np.pi / 2,
                                              arg_label=r"3\pi/2"))
                else:
                    qc_temp.gates.append(gate)
        else:
            qc_temp.gates = temp_resolved

        if len(basis_1q) == 2:
            temp_resolved = qc_temp.gates
            qc_temp.gates = []
            for gate in temp_resolved:
                if gate.name == "RX" and "RX" not in basis_1q:
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                elif gate.name == "RY" and "RY" not in basis_1q:
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                elif gate.name == "RZ" and "RZ" not in basis_1q:
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=-np.pi / 2,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=np.pi / 2,
                                              arg_label=r"\pi/2"))
                else:
                    qc_temp.gates.append(gate)

        return qc_temp

    def adjacent_gates(self):
        """
        Method to resolve two qubit gates with non-adjacent control/s or
        target/s in terms of gates with adjacent interactions.

        Returns
        -------
        qc : QubitCircuit
            Return QubitCircuit of the gates for the qubit circuit with the
            resolved non-adjacent gates.

        """
        temp = QubitCircuit(self.N, reverse_states=self.reverse_states)
        swap_gates = ["SWAP", "ISWAP", "SQRTISWAP", "SQRTSWAP", "BERKELEY",
                      "SWAPalpha"]

        for gate in self.gates:
            if gate.name == "CNOT" or gate.name == "CSIGN":
                start = min([gate.targets[0], gate.controls[0]])
                end = max([gate.targets[0], gate.controls[0]])
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        # Apply required gate if control, target are adjacent
                        # to each other, provided |control-target| is even.
                        if end == gate.controls[0]:
                            temp.gates.append(Gate(gate.name, targets=[i],
                                                   controls=[i + 1]))
                        else:
                            temp.gates.append(Gate(gate.name, targets=[i + 1],
                                                   controls=[i]))
                    elif (start + end - i - i == 2 and
                          (end - start + 1) % 2 == 1):
                        # Apply a swap between i and its adjacent gate, then
                        # the required gate if and then another swap if control
                        # and target have one qubit between them, provided
                        # |control-target| is odd.
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        if end == gate.controls[0]:
                            temp.gates.append(Gate(gate.name, targets=[i + 1],
                                                   controls=[i + 2]))
                        else:
                            temp.gates.append(Gate(gate.name, targets=[i + 2],
                                                   controls=[i + 1]))
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        i += 1
                    else:
                        # Swap the target/s and/or control with their adjacent
                        # qubit to bring them closer.
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        temp.gates.append(Gate("SWAP",
                                               targets=[start + end - i - 1,
                                                        start + end - i]))
                    i += 1

            elif gate.name in swap_gates:
                start = min([gate.targets[0], gate.targets[1]])
                end = max([gate.targets[0], gate.targets[1]])
                i = start
                while i < end:
                    if start + end - i - i == 1 and (end - start + 1) % 2 == 0:
                        temp.gates.append(Gate(gate.name, targets=[i, i + 1]))
                    elif ((start + end - i - i) == 2 and
                          (end - start + 1) % 2 == 1):
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        temp.gates.append(
                            Gate(gate.name, targets=[i + 1, i + 2]))
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        i += 1
                    else:
                        temp.gates.append(Gate("SWAP", targets=[i, i + 1]))
                        temp.gates.append(Gate("SWAP",
                                               targets=[start + end - i - 1,
                                                        start + end - i]))
                    i += 1

            else:
                raise NotImplementedError(
                    "`adjacent_gates` is not defined for "
                    "gate {}.".format(gate.name))

        return temp

    def propagators(self):
        """
        Propagator matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right.

        Returns
        -------
        U_list : list
            Return list of unitary matrices for the qubit circuit.

        """
        self.U_list = []

        for gate in self.gates:
            if gate.name == "RX":
                self.U_list.append(rx(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RY":
                self.U_list.append(ry(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RZ":
                self.U_list.append(rz(gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "SQRTNOT":
                self.U_list.append(sqrtnot(self.N, gate.targets[0]))
            elif gate.name == "SNOT":
                self.U_list.append(snot(self.N, gate.targets[0]))
            elif gate.name == "PHASEGATE":
                self.U_list.append(phasegate(gate.arg_value, self.N,
                                             gate.targets[0]))
            elif gate.name == "CRX":
                self.U_list.append(controlled_gate(rx(gate.arg_value),
                                                   N=self.N,
                                                   control=gate.controls[0],
                                                   target=gate.targets[0]))
            elif gate.name == "CRY":
                self.U_list.append(controlled_gate(ry(gate.arg_value),
                                                   N=self.N,
                                                   control=gate.controls[0],
                                                   target=gate.targets[0]))
            elif gate.name == "CRZ":
                self.U_list.append(controlled_gate(rz(gate.arg_value),
                                                   N=self.N,
                                                   control=gate.controls[0],
                                                   target=gate.targets[0]))
            elif gate.name == "CPHASE":
                self.U_list.append(cphase(gate.arg_value, self.N,
                                          gate.controls[0], gate.targets[0]))
            elif gate.name == "CNOT":
                self.U_list.append(cnot(self.N,
                                        gate.controls[0], gate.targets[0]))
            elif gate.name == "CSIGN":
                self.U_list.append(csign(self.N,
                                         gate.controls[0], gate.targets[0]))
            elif gate.name == "BERKELEY":
                self.U_list.append(berkeley(self.N, gate.targets))
            elif gate.name == "SWAPalpha":
                self.U_list.append(swapalpha(gate.arg_value, self.N,
                                             gate.targets))
            elif gate.name == "SWAP":
                self.U_list.append(swap(self.N, gate.targets))
            elif gate.name == "ISWAP":
                self.U_list.append(iswap(self.N, gate.targets))
            elif gate.name == "SQRTSWAP":
                self.U_list.append(sqrtswap(self.N, gate.targets))
            elif gate.name == "SQRTISWAP":
                self.U_list.append(sqrtiswap(self.N, gate.targets))
            elif gate.name == "FREDKIN":
                self.U_list.append(fredkin(self.N, gate.controls[0],
                                           gate.targets))
            elif gate.name == "TOFFOLI":
                self.U_list.append(toffoli(self.N, gate.controls,
                                           gate.targets[0]))
            elif gate.name == "GLOBALPHASE":
                self.U_list.append(globalphase(gate.arg_value, self.N))
            elif gate.name in self.user_gates:
                if gate.controls is not None:
                    raise ValueError(
                        "A user defined gate {} takes only  "
                        "`targets` variable.".format(gate.name))
                func = self.user_gates[gate.name]
                para_num = len(inspect.getfullargspec(func)[0])
                if para_num == 0:
                    oper = func()
                elif para_num == 1:
                    oper = func(gate.arg_value)
                else:
                    raise ValueError(
                        "gate function takes at most one parameters.")
                self.U_list.append(expand_oper(oper, self.N, gate.targets))

            else:
                raise NotImplementedError(
                    "{} gate is an unknown gate.".format(gate.name))

        return self.U_list

    def latex_code(self):
        rows = []

        gates = self.gates

        for gate in gates:
            col = []
            for n in range(self.N):
                if gate.targets and n in gate.targets:

                    if len(gate.targets) > 1:
                        if gate.name == "SWAP":
                            col.append(r" \qswap \qwx ")

                        elif ((self.reverse_states and
                                n == max(gate.targets)) or
                                (not self.reverse_states and
                                    n == min(gate.targets))):
                            col.append(r" \multigate{%d}{%s} " %
                                       (len(gate.targets) - 1,
                                        _gate_label(gate.name,
                                                    gate.arg_label)))
                        else:
                            col.append(r" \ghost{%s} " %
                                       (_gate_label(gate.name,
                                                    gate.arg_label)))

                    elif gate.name == "CNOT":
                        col.append(r" \targ ")
                    elif gate.name == "TOFFOLI":
                        col.append(r" \targ ")
                    else:
                        col.append(r" \gate{%s} " %
                                   _gate_label(gate.name, gate.arg_label))

                elif gate.controls and n in gate.controls:
                    m = (gate.targets[0] - n) * (-1 if self.reverse_states
                                                 else 1)
                    col.append(r" \ctrl{%d} " % m)

                elif (not gate.controls and not gate.targets):
                    # global gate
                    if ((self.reverse_states and n == self.N - 1) or
                            (not self.reverse_states and n == 0)):
                        col.append(r" \multigate{%d}{%s} " %
                                   (self.N - 1,
                                    _gate_label(gate.name, gate.arg_label)))
                    else:
                        col.append(r" \ghost{%s} " %
                                   (_gate_label(gate.name, gate.arg_label)))

                else:
                    col.append(r" \qw ")

            col.append(r" \qw ")
            rows.append(col)

        input_states = ["\lstick{\ket{" + x + "}}" if x is not None
                        else "" for x in self.input_states]

        code = ""
        n_iter = (reversed(range(self.N)) if self.reverse_states
                  else range(self.N))
        for n in n_iter:
            code += r" & %s" % input_states[n]
            for m in range(len(gates)):
                code += r" & %s" % rows[m][n]
            code += r" & \qw \\ " + "\n"

        return code

    def _repr_png_(self):
        return _latex_compile(self.latex_code(), format="png")

    def _repr_svg_(self):
        return _latex_compile(self.latex_code(), format="svg")

    @property
    def png(self):
        from IPython.display import Image
        return Image(self._repr_png_(), embed=True)

    @property
    def svg(self):
        from IPython.display import SVG
        return SVG(self._repr_svg_())

    def qasm(self):

        code = "# qasm code generated by QuTiP\n\n"

        for n in range(self.N):
            code += "\tqubit\tq%d\n" % n

        code += "\n"

        for gate in self.gates:
            code += "\t%s\t" % gate.name
            qtargets = ["q%d" %
                        t for t in gate.targets] if gate.targets else []
            qcontrols = (["q%d" % c for c in gate.controls] if gate.controls
                         else [])
            code += ",".join(qtargets + qcontrols)
            code += "\n"

        return code
