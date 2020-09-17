# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in sourc e and binary forms, with or without
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
from collections import defaultdict
from itertools import product

import warnings
import inspect

import numpy as np
from copy import deepcopy

from qutip.qip import circuit_latex as _latex
from qutip.qip.operations.gates import (rx, ry, rz, sqrtnot, snot, phasegate,
                                        x_gate, y_gate, z_gate, cy_gate,
                                        cz_gate, s_gate, t_gate, cs_gate,
                                        qasmu_gate, ct_gate, cphase, cnot,
                                        csign, berkeley, swapalpha, swap, iswap,
                                        sqrtswap, sqrtiswap, fredkin,
                                        toffoli, controlled_gate, globalphase,
                                        expand_operator)
from qutip import tensor, basis, identity, fidelity
from qutip.qobj import Qobj
from qutip.measurement import measurement_statistics


try:
    from IPython.display import Image as DisplayImage, SVG as DisplaySVG
except ImportError:
    # If IPython doesn't exist, then we set the nice display hooks to be simple
    # pass-throughs.
    def DisplayImage(data, *args, **kwargs):
        return data

    def DisplaySVG(data, *args, **kwargs):
        return data

__all__ = ['Gate', 'QubitCircuit', 'Measurement']

_single_qubit_gates = ["RX", "RY", "RZ", "SNOT", "SQRTNOT", "PHASEGATE",
                       "X", "Y", "Z", "S", "T", "QASMU"]
_para_gates = ["RX", "RY", "RZ", "CPHASE", "SWAPalpha", "PHASEGATE",
               "GLOBALPHASE", "CRX", "CRY", "CRZ", "QASMU"]
_ctrl_gates = ["CNOT", "CSIGN", "CRX", "CRY", "CRZ", "CY", "CZ",
               "CS", "CT"]
_swap_like = ["SWAP", "ISWAP", "SQRTISWAP", "SQRTSWAP", "BERKELEY",
              "SWAPalpha"]
_toffoli_like = ["TOFFOLI"]
_fredkin_like = ["FREDKIN"]


class Gate:
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
    classical_controls : int or list of int, optional
        indices of classical bits to control gate on.
    control_value : int, optional
        value of classical bits to control on, the classical controls are
        interpreted as an integer with lowest bit being the first one.
        If not specified, then the value is interpreted to be
        2 ** len(classical_controls) - 1 (i.e. all classical controls are 1).
    """

    def __init__(self, name, targets=None, controls=None,
                 arg_value=None, arg_label=None,
                 classical_controls=None, control_value=None):
        """
        Create a gate with specified parameters.
        """

        self.name = name
        self.targets = None
        self.controls = None
        self.classical_controls = None
        self.control_value = None

        if not isinstance(targets, Iterable) and targets is not None:
            self.targets = [targets]
        else:
            self.targets = targets

        if not isinstance(controls, Iterable) and controls is not None:
            self.controls = [controls]
        else:
            self.controls = controls

        if (not isinstance(classical_controls, Iterable) and
                classical_controls is not None):
            self.classical_controls = [classical_controls]
        else:
            self.classical_controls = classical_controls

        if (control_value is not None
                and control_value < 2 ** len(classical_controls)):
            self.control_value = control_value

        for ind_list in [self.targets, self.controls, self.classical_controls]:
            if isinstance(ind_list, Iterable):
                all_integer = all(
                    [isinstance(ind, np.int) for ind in ind_list])
                if not all_integer:
                    raise ValueError("Index of a qubit must be an integer")

        if name in _single_qubit_gates:
            if self.targets is None or len(self.targets) != 1:
                raise ValueError("Gate %s requires one target" % name)
            if self.controls is not None:
                raise ValueError("Gate %s cannot have a control" % name)
        elif name in _swap_like:
            if (self.targets is None) or (len(self.targets) != 2):
                raise ValueError("Gate %s requires two targets" % name)
            if self.controls is not None:
                raise ValueError("Gate %s cannot have a control" % name)
        elif name in _ctrl_gates:
            if self.targets is None or len(self.targets) != 1:
                raise ValueError("Gate %s requires one target" % name)
            if self.controls is None or len(self.controls) != 1:
                raise ValueError("Gate %s requires one control" % name)
        elif name in _fredkin_like:
            if self.targets is None or len(self.targets) != 2:
                raise ValueError("Gate %s requires one target" % name)
            if self.controls is None or len(self.controls) != 1:
                raise ValueError("Gate %s requires two control" % name)
        elif name in _toffoli_like:
            if self.targets is None or len(self.targets) != 1:
                raise ValueError("Gate %s requires one target" % name)
            if self.controls is None or len(self.controls) != 2:
                raise ValueError("Gate %s requires two control" % name)

        if name in _para_gates:
            if arg_value is None:
                raise ValueError("Gate %s requires an argument value" % name)
        else:
            if (name in _GATE_NAME_TO_LABEL) and (arg_value is not None):
                raise ValueError("Gate %s does not take argument value" % name)

        self.arg_value = arg_value
        self.arg_label = arg_label

    def __str__(self):
        str_name = (("Gate(%s, targets=%s, controls=%s,"
                    " classical controls=%s, control_value=%s)")
                    % (self.name, self.targets,
                       self.controls, self.classical_controls,
                       self.control_value))
        return str_name

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)

    def _to_qasm(self, qasm_out):
        """
        Pipe output of gate signature and application to QasmOutput object.

        Parameters
        ----------
        qasm_out: QasmOutput
            object to store QASM output.
        """

        qasm_gate = qasm_out.qasm_name(self.name)

        if not qasm_gate:
            error_str = "{} gate's qasm defn is not specified".format(self.name)
            raise NotImplementedError(error_str)

        if self.classical_controls:
            err_msg = "Exporting controlled gates is not implemented yet."
            raise NotImplementedError(err_msg)
        else:
            qasm_out.output(qasm_out._qasm_str(qasm_gate,
                                               self.controls,
                                               self.targets,
                                               self.arg_value))


_GATE_NAME_TO_LABEL = {
    'X': r'X',
    'Y': r'Y',
    'CY': r'C_y',
    'Z': r'Z',
    'CZ': r'C_z',
    'S': r'S',
    'CS': r'C_s',
    'T': r'T',
    'CT': r'C_t',
    'RX': r'R_x',
    'RY': r'R_y',
    'RZ': r'R_z',
    'CRX': r'R_x',
    'CRY': r'R_y',
    'CRZ': r'R_z',
    'SQRTNOT': r'\sqrt{\rm NOT}',
    'SNOT': r'{\rm H}',
    'PHASEGATE': r'{\rm PHASE}',
    'QASMU': r'{\rm QASM-U}',
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

    if name in _GATE_NAME_TO_LABEL:
        gate_label = _GATE_NAME_TO_LABEL[name]
    else:
        warnings.warn("Unknown gate %s" % name)
        gate_label = name

    if arg_label:
        return r'%s(%s)' % (gate_label, arg_label)
    return r'%s' % gate_label


class Measurement:
    """
    Representation of a quantum measurement, with its required parameters,
    and target qubits.

    Parameters
    ----------
    name : string
        Measurement name.
    targets : list or int
        Gate targets.
    classical_store : int
        Result of the measurment is stored in this
        classical register of the circuit.
    """

    def __init__(self, name, targets=None, index=None, classical_store=None):
        """
        Create a measurement with specified parameters.
        """

        self.name = name
        self.targets = None
        self.classical_store = classical_store
        self.index = index

        if not isinstance(targets, Iterable) and targets is not None:
            self.targets = [targets]
        else:
            self.targets = targets

        for ind_list in [self.targets]:
            if isinstance(ind_list, Iterable):
                all_integer = all(
                    [isinstance(ind, np.int) for ind in ind_list])
                if not all_integer:
                    raise ValueError("Index of a qubit must be an integer")

    def measurement_comp_basis(self, state):
        '''
        Measures a particular qubit (determined by the target)
        whose ket vector/ density matrix is specified in the
        computational basis and returns collapsed_states and probabilities
        (retains full dimension).

        Parameters
        ----------
        state : ket or oper
                state to be measured on specified by
                ket vector or density matrix

        Returns
        -------
        collapsed_states : List of Qobjs
                        the collapsed state obtained after measuring the qubits
                        and obtaining the qubit specified by the target in the
                        state specified by the index.
        probabilities : List of floats
                        the probability of measuring a state in a the state
                        specified by the index.
        '''

        n = int(np.log2(state.shape[0]))
        target = self.targets[0]

        if target < n:
            op0 = basis(2, 0) * basis(2, 0).dag()
            op1 = basis(2, 1) * basis(2, 1).dag()
            measurement_ops = [op0, op1]
        else:
            raise ValueError("target is not valid")

        return measurement_statistics(state, measurement_ops,
                                      targets=self.targets)

    def __str__(self):
        str_name = (("Measurement(%s, target=%s, classical_store=%s)") %
                    (self.name, self.targets, self.classical_store))
        return str_name

    def __repr__(self):
        return str(self)

    def _repr_latex_(self):
        return str(self)

    def _to_qasm(self, qasm_out):
        """
        Pipe output of measurement to QasmOutput object.

        Parameters
        ----------
        qasm_out: QasmOutput
            object to store QASM output.
        """

        qasm_out.output("measure q[{}] -> c[{}]".format(self.targets[0],
                                                        self.classical_store))


class QubitCircuit:
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
    dims : list
        A list of integer for the dimension of each composite system.
        e.g [2,2,2,2,2] for 5 qubits system. If None, qubits system
        will be the default option.

    Examples
    --------
    >>> def user_gate():
    ...     mat = np.array([[1.,   0],
    ...                     [0., 1.j]])
    ...     return Qobj(mat, dims=[[2], [2]])
    >>> qubit_circuit = QubitCircuit(2, user_gates={"T":user_gate})
    >>> qubit_circuit.add_gate("T", targets=[0])
    """

    def __init__(self, N, input_states=None, output_states=None,
                 reverse_states=True, user_gates=None, dims=None, num_cbits=0):
        # number of qubits in the register
        self.N = N
        self.reverse_states = reverse_states
        self.gates = []
        self.U_list = []
        self.dims = dims
        self.num_cbits = num_cbits

        if input_states:
            self.input_states = input_states
        else:
            self.input_states = [None for i in range(N+num_cbits)]

        if output_states:
            self.output_states = output_states
        else:
            self.output_states = [None for i in range(N+num_cbits)]

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

    def add_measurement(self, measurement, targets=None, index=None,
                        classical_store=None):
        """
        Adds a measurement with specified parameters to the circuit.

        Parameters
        ----------
        name: string
            Measurement name. If name is an instance of `Measuremnent`,
            parameters are unpacked and added.
        targets: list
            Gate targets
        index : list
            Positions to add the gate.
        classical_store : int
            Classical register where result of measurement is stored.
        """

        if isinstance(measurement, Measurement):
            name = measurement.name
            targets = measurement.targets
            classical_store = measurement.classical_store

        else:
            name = measurement

        if index is None:
            self.gates.append(
                    Measurement(name, targets=targets,
                                classical_store=classical_store))

        else:
            for position in index:
                self.gates.insert(
                    position,
                    Measurement(name, targets=targets,
                                classical_store=classical_store))

    def add_gate(self, gate, targets=None, controls=None, arg_value=None,
                 arg_label=None, index=None,
                 classical_controls=None, control_value=None):
        """
        Adds a gate with specified parameters to the circuit.

        Parameters
        ----------
        gate: string or :class:`.Gate`
            Gate name. If gate is an instance of :class:`.Gate`, parameters are
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
        classical_controls : int or list of int, optional
            indices of classical bits to control gate on.
        control_value : int, optional
            value of classical bits to control on, the classical controls are
            interpreted as an integer with lowest bit being the first one.
            If not specified, then the value is interpreted to be
            2 ** len(classical_controls) - 1
            (i.e. all classical controls are 1).
        """

        if isinstance(gate, Gate):
            name = gate.name
            targets = gate.targets
            controls = gate.controls
            arg_value = gate.arg_value
            arg_label = gate.arg_label
            classical_controls = gate.classical_controls
            control_value = gate.control_value

        else:
            name = gate

        if index is None:
            gate = Gate(name, targets=targets, controls=controls,
                        arg_value=arg_value, arg_label=arg_label,
                        classical_controls=classical_controls,
                        control_value=control_value)
            self.gates.append(gate)

        else:
            for position in index:
                num_mes = (sum(isinstance(op, Measurement) for op
                               in self.gates[:position]))
                gate = Gate(name, targets=targets, controls=controls,
                            arg_value=arg_value, arg_label=arg_label,
                            classical_controls=classical_controls,
                            control_value=control_value)
                self.gates.insert(position, gate)

    def add_1q_gate(self, name, start=0, end=None, qubits=None,
                    arg_value=None, arg_label=None,
                    classical_controls=None, control_value=None):
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
        if name not in ["RX", "RY", "RZ", "SNOT", "SQRTNOT", "PHASEGATE",
                        "X", "Y", "Z", "S", "T", "QASMU"]:
            raise ValueError("%s is not a single qubit gate" % name)

        if qubits is not None:
            for _, i in enumerate(qubits):
                gate = Gate(name, targets=qubits[i], controls=None,
                            arg_value=arg_value, arg_label=arg_label,
                            classical_controls=classical_controls,
                            control_value=control_value)
                self.gates.append(gate)

        else:
            if end is None:
                end = self.N - 1
            for i in range(start, end+1):
                gate = Gate(name, targets=i, controls=None,
                            arg_value=arg_value, arg_label=arg_label,
                            classical_controls=classical_controls,
                            control_value=control_value)
                self.gates.append(gate)

    def add_circuit(self, qc, start=0):
        """
        Adds a block of a qubit circuit to the main circuit.
        Globalphase gates are not added.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            The circuit block to be added to the main circuit.
        start : int
            The qubit on which the first gate is applied.
        """
        if self.N - start < qc.N:
            raise NotImplementedError("Targets exceed number of qubits.")

        for circuit_op in qc.gates:

            if isinstance(circuit_op, Gate):
                gate = circuit_op

                if gate.name in ["RX", "RY", "RZ",
                                 "SNOT", "SQRTNOT", "PHASEGATE", "QASMU"]:
                    self.add_gate(gate.name, gate.targets[0] + start, None,
                                  gate.arg_value, gate.arg_label)
                elif gate.name in ["X", "Y", "Z", "S", "T"]:
                    self.add_gate(gate.name, gate.targets[0] + start, None,
                                  None, gate.arg_label)
                elif gate.name in ["CPHASE", "CNOT", "CSIGN", "CRX", "CRY",
                                   "CRZ", "CY", "CZ", "CS", "CT"]:
                    self.add_gate(gate.name, gate.targets[0] + start,
                                  gate.controls[0] + start, gate.arg_value,
                                  gate.arg_label)
                elif gate.name in ["BERKELEY", "SWAPalpha", "SWAP", "ISWAP",
                                   "SQRTSWAP", "SQRTISWAP"]:
                    self.add_gate(gate.name,
                                  [gate.targets[0] + start,
                                   gate.targets[1] + start])
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
            else:
                measurement = circuit_op
                self.add_measurement(
                                measurement.name,
                                targets=[measurement.targets[0] + start],
                                classical_store=measurement.classical_store)

    def remove_gate_or_measurement(self, index=None, end=None,
                                   name=None, remove="first"):
        """
        Remove a gate from a specific index or between two indexes or the
        first, last or all instances of a particular gate.

        Parameters
        ----------
        index : int
            Location of gate or measurement to be removed.
        name : string
            Gate or Measurement name to be removed.
        remove : string
            If first or all gates/measurements are to be removed.
        """
        if index is not None:
            if index > len(self.gates):
                raise ValueError("Index exceeds number \
                                    of gates + measurements.")
            if end is not None and end <= len(self.gates):
                for i in range(end - index):
                    self.gates.pop(index + i)
            elif end is not None and end > self.N:
                raise ValueError("End target exceeds number \
                                    of gates + measurements.")
            else:
                self.gates.pop(index)

        elif name is not None and remove == "first":
            for circuit_op in self.gates:
                if name == circuit_op.name:
                    self.gates.remove(circuit_op)
                    break

        elif name is not None and remove == "last":
            for i in reversed(range(len(self.gates))):
                if name == self.gates[i].name:
                    self.gates.pop(i)
                    break

        elif name is not None and remove == "all":
            for i in reversed(range(len(self.gates))):
                if name == self.gates[i].name:
                    self.gates.pop(i)

        else:
            self.gates.pop()

    def reverse_circuit(self):
        """
        Reverse an entire circuit of unitary gates.

        Returns
        -------
        qubit_circuit : :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of resolved gates for the
            qubit circuit in the reverse order.

        """
        temp = QubitCircuit(self.N, reverse_states=self.reverse_states,
                            num_cbits=self.num_cbits,
                            input_states=self.input_states,
                            output_states=self.output_states)

        for circuit_op in reversed(self.gates):
            if isinstance(circuit_op, Gate):
                temp.add_gate(circuit_op)
            else:
                temp.add_measurement(circuit_op)

        return temp

    def _resolve_to_universal(self, gate, temp_resolved, basis_1q, basis_2q):
        """A dispatch method"""
        if gate.name in basis_2q:
            method = getattr(self, '_gate_basis_2q')
        else:
            if gate.name == "SWAP" and "ISWAP" in basis_2q:
                method = getattr(self, '_gate_IGNORED')
            else:
                method = getattr(self, '_gate_' + str(gate.name))
        method(gate, temp_resolved)

    def _gate_IGNORED(self, gate, temp_resolved):
        temp_resolved.append(gate)
    _gate_RY = _gate_RZ = _gate_basis_2q = _gate_IGNORED
    _gate_CNOT = _gate_RX = _gate_IGNORED

    def _gate_SQRTNOT(self, gate, temp_resolved):
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=np.pi / 4, arg_label=r"\pi/4"))
        temp_resolved.append(Gate("RX", gate.targets, None,
                                  arg_value=np.pi / 2, arg_label=r"\pi/2"))

    def _gate_SNOT(self, gate, temp_resolved):
        half_pi = np.pi / 2
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RX", gate.targets, None,
                                  arg_value=np.pi, arg_label=r"\pi"))

    def _gate_PHASEGATE(self, gate, temp_resolved):
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=gate.arg_value / 2,
                                  arg_label=gate.arg_label))
        temp_resolved.append(Gate("RZ", gate.targets, None,
                                  gate.arg_value, gate.arg_label))

    def _gate_NOTIMPLEMENTED(self, gate, temp_resolved):
        raise NotImplementedError("Cannot be resolved in this basis")

    _gate_PHASEGATE = _gate_BERKELEY = _gate_SWAPalpha = _gate_NOTIMPLEMENTED
    _gate_SQRTSWAP = _gate_SQRTISWAP = _gate_NOTIMPLEMENTED

    def _gate_CSIGN(self, gate, temp_resolved):
        half_pi = np.pi / 2
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RX", gate.targets, None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("CNOT", gate.targets, gate.controls))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RX", gate.targets, None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=np.pi, arg_label=r"\pi"))

    def _gate_SWAP(self, gate, temp_resolved):
        temp_resolved.append(
            Gate("CNOT", gate.targets[0], gate.targets[1]))
        temp_resolved.append(
            Gate("CNOT", gate.targets[1], gate.targets[0]))
        temp_resolved.append(
            Gate("CNOT", gate.targets[0], gate.targets[1]))

    def _gate_ISWAP(self, gate, temp_resolved):
        half_pi = np.pi / 2
        temp_resolved.append(Gate("CNOT", gate.targets[0],
                                  gate.targets[1]))
        temp_resolved.append(Gate("CNOT", gate.targets[1],
                                  gate.targets[0]))
        temp_resolved.append(Gate("CNOT", gate.targets[0],
                                  gate.targets[1]))
        temp_resolved.append(Gate("RZ", gate.targets[0], None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets[0], None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RX", gate.targets[0], None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("CNOT", gate.targets[0],
                                  gate.targets[1]))
        temp_resolved.append(Gate("RY", gate.targets[0], None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RX", gate.targets[0], None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))

    def _gate_FREDKIN(self, gate, temp_resolved):
        half_pi = np.pi / 2
        eigth_pi = np.pi / 8
        temp_resolved.append(Gate("CNOT", gate.targets[0],
                                  gate.targets[1]))
        temp_resolved.append(Gate("CNOT", gate.targets[0],
                                  gate.controls))
        temp_resolved.append(Gate("RZ", gate.controls, None,
                                  arg_value=eigth_pi,
                                  arg_label=r"\pi/8"))
        temp_resolved.append(Gate("RZ", [gate.targets[0]], None,
                                  arg_value=-eigth_pi,
                                  arg_label=r"-\pi/8"))
        temp_resolved.append(Gate("CNOT", gate.targets[0],
                                  gate.controls))
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets[1], None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=-half_pi,
                                  arg_label=r"-\pi/2"))
        temp_resolved.append(Gate("RZ", gate.targets, None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RZ", gate.targets[0], None,
                                  arg_value=eigth_pi,
                                  arg_label=r"\pi/8"))
        temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                  arg_value=eigth_pi,
                                  arg_label=r"\pi/8"))
        temp_resolved.append(Gate("CNOT", gate.targets[1],
                                  gate.controls))
        temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                  arg_value=-eigth_pi,
                                  arg_label=r"-\pi/8"))
        temp_resolved.append(Gate("CNOT", gate.targets[1],
                                  gate.targets[0]))
        temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                  arg_value=eigth_pi,
                                  arg_label=r"\pi/8"))
        temp_resolved.append(Gate("CNOT", gate.targets[1],
                                  gate.controls))
        temp_resolved.append(Gate("RZ", gate.targets[1], None,
                                  arg_value=-eigth_pi,
                                  arg_label=r"-\pi/8"))
        temp_resolved.append(Gate("CNOT", gate.targets[1],
                                  gate.targets[0]))
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets[1], None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=-half_pi,
                                  arg_label=r"-\pi/2"))
        temp_resolved.append(Gate("RZ", gate.targets, None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("CNOT", gate.targets[0],
                                  gate.targets[1]))

    def _gate_TOFFOLI(self, gate, temp_resolved):
        half_pi = np.pi / 2
        quarter_pi = np.pi / 4
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=np.pi / 8,
                                  arg_label=r"\pi/8"))
        temp_resolved.append(Gate("RZ", gate.controls[1], None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RZ", gate.controls[0], None,
                                  arg_value=quarter_pi,
                                  arg_label=r"\pi/4"))
        temp_resolved.append(Gate("CNOT", gate.controls[1],
                                  gate.controls[0]))
        temp_resolved.append(Gate("RZ", gate.controls[1], None,
                                  arg_value=-quarter_pi,
                                  arg_label=r"-\pi/4"))
        temp_resolved.append(Gate("CNOT", gate.controls[1],
                                  gate.controls[0]))
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RX", gate.targets, None,
                                  arg_value=np.pi, arg_label=r"\pi"))
        temp_resolved.append(Gate("RZ", gate.controls[1], None,
                                  arg_value=-quarter_pi,
                                  arg_label=r"-\pi/4"))
        temp_resolved.append(Gate("RZ", gate.targets, None,
                                  arg_value=quarter_pi,
                                  arg_label=r"\pi/4"))
        temp_resolved.append(Gate("CNOT", gate.targets,
                                  gate.controls[0]))
        temp_resolved.append(Gate("RZ", gate.targets, None,
                                  arg_value=-quarter_pi,
                                  arg_label=r"-\pi/4"))
        temp_resolved.append(Gate("CNOT", gate.targets,
                                  gate.controls[1]))
        temp_resolved.append(Gate("RZ", gate.targets, None,
                                  arg_value=quarter_pi,
                                  arg_label=r"\pi/4"))
        temp_resolved.append(Gate("CNOT", gate.targets,
                                  gate.controls[0]))
        temp_resolved.append(Gate("RZ", gate.targets, None,
                                  arg_value=-quarter_pi,
                                  arg_label=r"-\pi/4"))
        temp_resolved.append(Gate("CNOT", gate.targets,
                                  gate.controls[1]))
        temp_resolved.append(Gate("GLOBALPHASE", None, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RY", gate.targets, None,
                                  arg_value=half_pi,
                                  arg_label=r"\pi/2"))
        temp_resolved.append(Gate("RX", gate.targets, None,
                                  arg_value=np.pi, arg_label=r"\pi"))

    def _gate_GLOBALPHASE(self, gate, temp_resolved):
        temp_resolved.append(Gate(gate.name, gate.targets,
                                  gate.controls,
                                  gate.arg_value, gate.arg_label))

    def _resolve_2q_basis(self, basis, qc_temp, temp_resolved):
        """Dispatch method"""
        method = getattr(self, '_basis_' + str(basis), temp_resolved)
        method(qc_temp, temp_resolved)

    def _basis_CSIGN(self, qc_temp, temp_resolved):
        half_pi = np.pi / 2
        for gate in temp_resolved:
            if gate.name == "CNOT":
                qc_temp.gates.append(Gate("RY", gate.targets, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("CSIGN", gate.targets,
                                          gate.controls))
                qc_temp.gates.append(Gate("RY", gate.targets, None,
                                          arg_value=half_pi,
                                          arg_label=r"\pi/2"))
            else:
                qc_temp.gates.append(gate)

    def _basis_ISWAP(self, qc_temp, temp_resolved):
        half_pi = np.pi / 2
        quarter_pi = np.pi / 4
        for gate in temp_resolved:
            if gate.name == "CNOT":
                qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=quarter_pi,
                                          arg_label=r"\pi/4"))
                qc_temp.gates.append(Gate("ISWAP", [gate.controls[0],
                                          gate.targets[0]],
                                          None))
                qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("RY", gate.controls, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                          arg_value=half_pi,
                                          arg_label=r"\pi/2"))
                qc_temp.gates.append(Gate("ISWAP", [gate.controls[0],
                                          gate.targets[0]], None))
                qc_temp.gates.append(Gate("RY", gate.targets, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                          arg_value=half_pi,
                                          arg_label=r"\pi/2"))
            elif gate.name == "SWAP":
                qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=quarter_pi,
                                          arg_label=r"\pi/4"))
                qc_temp.gates.append(Gate("ISWAP", gate.targets, None))
                qc_temp.gates.append(Gate("RX", gate.targets[0], None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("ISWAP", gate.targets, None))
                qc_temp.gates.append(Gate("RX", gate.targets[1], None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("ISWAP", [gate.targets[1],
                                          gate.targets[0]], None))
                qc_temp.gates.append(Gate("RX", gate.targets[0], None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
            else:
                qc_temp.gates.append(gate)

    def _basis_SQRTSWAP(self, qc_temp, temp_resolved):
        half_pi = np.pi / 2
        for gate in temp_resolved:
            if gate.name == "CNOT":
                qc_temp.gates.append(Gate("RY", gate.targets, None,
                                          arg_value=half_pi,
                                          arg_label=r"\pi/2"))
                qc_temp.gates.append(Gate("SQRTSWAP", [gate.controls[0],
                                          gate.targets[0]],
                                          None))
                qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                          arg_value=np.pi,
                                          arg_label=r"\pi"))
                qc_temp.gates.append(Gate("SQRTSWAP", [gate.controls[0],
                                          gate.targets[0]], None))
                qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("RY", gate.targets, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
            else:
                qc_temp.gates.append(gate)

    def _basis_SQRTISWAP(self, qc_temp, temp_resolved):
        half_pi = np.pi / 2
        quarter_pi = np.pi / 4
        for gate in temp_resolved:
            if gate.name == "CNOT":
                qc_temp.gates.append(Gate("RY", gate.controls, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("RX", gate.controls, None,
                                          arg_value=half_pi,
                                          arg_label=r"\pi/2"))
                qc_temp.gates.append(Gate("RX", gate.targets, None,
                                          arg_value=-half_pi,
                                          arg_label=r"-\pi/2"))
                qc_temp.gates.append(Gate("SQRTISWAP", [gate.controls[0],
                                          gate.targets[0]],
                                          None))
                qc_temp.gates.append(Gate("RX", gate.controls, None,
                                          arg_value=np.pi,
                                          arg_label=r"\pi"))
                qc_temp.gates.append(Gate("SQRTISWAP", [gate.controls[0],
                                          gate.targets[0]], None))
                qc_temp.gates.append(Gate("RY", gate.controls, None,
                                          arg_value=half_pi,
                                          arg_label=r"\pi/2"))
                qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=quarter_pi,
                                          arg_label=r"\pi/4"))
                qc_temp.gates.append(Gate("RZ", gate.controls, None,
                                          arg_value=np.pi,
                                          arg_label=r"\pi"))
                qc_temp.gates.append(Gate("GLOBALPHASE", None, None,
                                          arg_value=3 * half_pi,
                                          arg_label=r"3\pi/2"))
            else:
                qc_temp.gates.append(gate)

    def run(self, state, cbits=None, U_list=None, measure_results=()):
        '''
        This is the primary circuit run function for 1 run, must be called
        after adding all the gates and measurements on the circuit and returns
        the resultant state after evolution.

        Parameters
        ----------
        state : ket
                state to be observed on specified by density matrix.
        cbits : List of ints, optional
                initialization of the classical bits
        U_list : List of unitaries, optional
                optional parameter to pass in the propagator list
        measure_results : tuple of ints, optional
                optional specification of each measurement result to enable
                post-selection. If specified, the measurement results are
                set to the tuple of bits (sequentially) instead of being
                chosen at random.

        Returns
        -------
        state : returns the ket of the output state after running the circuit.
        '''

        if cbits and len(cbits) == self.num_cbits:
            self.cbits = cbits
        else:
            self.cbits = [0] * self.num_cbits

        if state.shape[0] != 2 ** self.N:
            return ValueError("dimension of state is incorrect")

        if U_list:
            self.U_list = U_list
        else:
            self.U_list = self.propagators()

        ulistindex = 0
        probability = 1
        measure_ind = 0

        for operation in self.gates:

            if isinstance(operation, Measurement):

                states, probabilities = operation.measurement_comp_basis(state)
                if measure_results:
                    i = int(measure_results[measure_ind])
                    measure_ind += 1
                else:
                    i = np.random.choice([0, 1], p=probabilities)
                probability *= probabilities[i]
                state = states[i]
                if operation.classical_store is not None:
                    self.cbits[operation.classical_store] = i

            else:

                if (operation.classical_controls):
                    if operation.control_value is None:
                        apply_gate = all([self.cbits[i] for i
                                         in operation.classical_controls])
                    else:
                        bitstring = "".join([str(
                                        self.cbits[i]) for i
                                        in operation.classical_controls[::-1]])
                        register_value = int(bitstring, 2)
                        apply_gate = (register_value == operation.control_value)
                    if apply_gate:
                        state = self.U_list[ulistindex] * state
                        ulistindex += 1
                    else:
                        ulistindex += 1
                        continue
                else:
                    state = self.U_list[ulistindex] * state
                    ulistindex += 1

        return state, probability

    def run_statistics(self, state, cbits=None):
        '''
        This is the circuit run function for num_runs run, must be called after
        adding all the gates and measurements on the circuit and returns the
        probability with which each output state is observed.

        Parameters
        ----------
        state : ket
                state to be observed on specified by density matrix.
        cbits : List of ints, optional
                initialization of the classical bits.

        Returns
        -------
        states : List of kets
                returns a list of possible circuit states output by run.
        state_probs : List of floats
                returns probabilities of getting above output states.
        '''

        state_probs = []
        states = []

        U_list = self.propagators()

        num_measurements = len(list(filter(
                                lambda x: isinstance(x, Measurement),
                                self.gates)))

        for measure_results in product("01", repeat=num_measurements):
            found = 0
            final_state, probability = self.run(
                                            state, cbits=cbits, U_list=U_list,
                                            measure_results=measure_results)
            states.append(final_state)
            state_probs.append(probability)

        return states, state_probs

    def resolve_gates(self, basis=["CNOT", "RX", "RY", "RZ"]):
        """
        Unitary matrix calculator for N qubits returning the individual
        steps as unitary matrices operating from left to right in the specified
        basis.
        Calls '_resolve_to_universal' for each gate, this function maps
        each 'GATENAME' with its corresponding '_gate_basis_2q'
        Subsequently calls _resolve_2q_basis for each basis, this function maps
        each '2QGATENAME' with its corresponding '_basis_'

        Parameters
        ----------
        basis : list.
            Basis of the resolved circuit.

        Returns
        -------
        qc : :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of resolved gates
            for the qubit circuit in the desired basis.
        """
        qc_temp = QubitCircuit(self.N, reverse_states=self.reverse_states,
                               num_cbits=self.num_cbits)
        temp_resolved = []

        basis_1q_valid = ["RX", "RY", "RZ"]
        basis_2q_valid = ["CNOT", "CSIGN", "ISWAP", "SQRTSWAP", "SQRTISWAP"]

        num_measurements = len(list(filter(
                                lambda x: isinstance(x, Measurement),
                                self.gates)))
        if num_measurements > 0:
            raise NotImplementedError("adjacent_gates must be called before \
            measurements are added to the circuit")

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
            if len(basis_1q) == 0:
                basis_1q = ["RX", "RY", "RZ"]

        else:  # only one 2q gate is given as basis
            basis_1q = ["RX", "RY", "RZ"]
            if basis in basis_2q_valid:
                basis_2q = [basis]
            else:
                raise ValueError("%s is not a valid two-qubit basis gate"
                                 % basis)

        for gate in self.gates:
            if gate.name in ("X", "Y", "Z"):
                qc_temp.gates.append(Gate("GLOBALPHASE", arg_value=np.pi/2))
                gate = Gate(
                    "R" + gate.name, targets=gate.targets, arg_value=np.pi)
            try:
                self._resolve_to_universal(gate, temp_resolved,
                                           basis_1q, basis_2q)
            except AttributeError:
                exception = f"Gate {gate.name} cannot be resolved."
                raise NotImplementedError(exception)

        match = False
        for basis_unit in ["CSIGN", "ISWAP", "SQRTSWAP", "SQRTISWAP"]:
            if basis_unit in basis_2q:
                match = True
                self._resolve_2q_basis(basis_unit, qc_temp, temp_resolved)
                break
        if not match:
            qc_temp.gates = temp_resolved

        if len(basis_1q) == 2:
            temp_resolved = qc_temp.gates
            qc_temp.gates = []
            half_pi = np.pi / 2
            for gate in temp_resolved:
                if gate.name == "RX" and "RX" not in basis_1q:
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=-half_pi,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              arg_value=half_pi,
                                              arg_label=r"\pi/2"))
                elif gate.name == "RY" and "RY" not in basis_1q:
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=-half_pi,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RZ", gate.targets, None,
                                              arg_value=half_pi,
                                              arg_label=r"\pi/2"))
                elif gate.name == "RZ" and "RZ" not in basis_1q:
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=-half_pi,
                                              arg_label=r"-\pi/2"))
                    qc_temp.gates.append(Gate("RY", gate.targets, None,
                                              gate.arg_value, gate.arg_label))
                    qc_temp.gates.append(Gate("RX", gate.targets, None,
                                              arg_value=half_pi,
                                              arg_label=r"\pi/2"))
                else:
                    qc_temp.gates.append(gate)

        qc_temp.gates = deepcopy(qc_temp.gates)

        return qc_temp

    def adjacent_gates(self):
        """
        Method to resolve two qubit gates with non-adjacent control/s or
        target/s in terms of gates with adjacent interactions.

        Returns
        -------
        qubit_circuit: :class:`.QubitCircuit`
            Return :class:`.QubitCircuit` of the gates
            for the qubit circuit with the resolved non-adjacent gates.

        """
        temp = QubitCircuit(self.N, reverse_states=self.reverse_states,
                            num_cbits=self.num_cbits)
        swap_gates = ["SWAP", "ISWAP", "SQRTISWAP", "SQRTSWAP", "BERKELEY",
                      "SWAPalpha"]
        num_measurements = len(list(filter(
                                lambda x: isinstance(x, Measurement),
                                self.gates)))
        if num_measurements > 0:
            raise NotImplementedError("adjacent_gates must be called before \
            measurements are added to the circuit")

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
                            temp.gates.append(Gate(gate.name,
                                                   targets=[i + 1],
                                                   controls=[i + 2]))
                        else:
                            temp.gates.append(Gate(gate.name,
                                                   targets=[i + 2],
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
                        temp.gates.append(Gate(gate.name,
                                               targets=[i + 1, i + 2]))
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

        temp.gates = deepcopy(temp.gates)

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

        gates = filter(lambda x: isinstance(x, Gate), self.gates)

        for gate in gates:
            if gate.name == "RX":
                self.U_list.append(rx(
                    gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RY":
                self.U_list.append(ry(
                    gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "RZ":
                self.U_list.append(rz(
                    gate.arg_value, self.N, gate.targets[0]))
            elif gate.name == "X":
                self.U_list.append(x_gate(self.N, gate.targets[0]))
            elif gate.name == "Y":
                self.U_list.append(y_gate(self.N, gate.targets[0]))
            elif gate.name == "CY":
                self.U_list.append(cy_gate(
                    self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "Z":
                self.U_list.append(z_gate(self.N, gate.targets[0]))
            elif gate.name == "CZ":
                self.U_list.append(cz_gate(
                    self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "T":
                self.U_list.append(t_gate(self.N, gate.targets[0]))
            elif gate.name == "CT":
                self.U_list.append(ct_gate(
                    self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "S":
                self.U_list.append(s_gate(self.N, gate.targets[0]))
            elif gate.name == "CS":
                self.U_list.append(cs_gate(
                    self.N, gate.controls[0], gate.targets[0]))
            elif gate.name == "SQRTNOT":
                self.U_list.append(sqrtnot(self.N, gate.targets[0]))
            elif gate.name == "SNOT":
                self.U_list.append(snot(self.N, gate.targets[0]))
            elif gate.name == "PHASEGATE":
                self.U_list.append(phasegate(gate.arg_value, self.N,
                                             gate.targets[0]))
            elif gate.name == "QASMU":
                self.U_list.append(qasmu_gate(gate.arg_value, self.N,
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
                    raise ValueError("A user defined gate {} takes only  "
                                     "`targets` variable.".format(gate.name))
                func_or_oper = self.user_gates[gate.name]
                if inspect.isfunction(func_or_oper):
                    func = func_or_oper
                    para_num = len(inspect.getfullargspec(func)[0])
                    if para_num == 0:
                        oper = func()
                    elif para_num == 1:
                        oper = func(gate.arg_value)
                    else:
                        raise ValueError(
                                "gate function takes at most one parameters.")
                elif isinstance(func_or_oper, Qobj):
                    oper = func_or_oper
                else:
                    raise ValueError("gate is neither function nor operator")
                self.U_list.append(expand_operator(
                    oper, N=self.N, targets=gate.targets, dims=self.dims))
            else:
                raise NotImplementedError(
                    "{} gate is an unknown gate.".format(gate.name))

        return self.U_list

    def latex_code(self):
        rows = []

        ops = self.gates
        col = []
        for op in ops:
            if isinstance(op, Gate):
                gate = op
                col = []
                for n in range(self.N+self.num_cbits):

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
                        elif gate.name == "CY":
                            col.append(r" \targ ")
                        elif gate.name == "CZ":
                            col.append(r" \targ ")
                        elif gate.name == "CS":
                            col.append(r" \targ ")
                        elif gate.name == "CT":
                            col.append(r" \targ ")
                        elif gate.name == "TOFFOLI":
                            col.append(r" \targ ")
                        else:
                            col.append(r" \gate{%s} " %
                                       _gate_label(gate.name, gate.arg_label))

                    elif gate.controls and n in gate.controls:
                        control_tag = (-1 if
                                       self.reverse_states
                                       else 1) * (gate.targets[0] - n)
                        col.append(r" \ctrl{%d} " % control_tag)

                    elif (gate.classical_controls and
                            (n - self.N) in gate.classical_controls):
                        control_tag = n - gate.targets[0]
                        col.append(r" \ctrl{%d} " % control_tag)

                    elif (not gate.controls and not gate.targets):
                        # global gate
                        if ((self.reverse_states and n == self.N - 1) or
                                (not self.reverse_states and n == 0)):
                            col.append(r" \multigate{%d}{%s} " %
                                       (self.N - 1,
                                        _gate_label(gate.name,
                                                    gate.arg_label)))
                        else:
                            col.append(r" \ghost{%s} " %
                                       (_gate_label(gate.name,
                                                    gate.arg_label)))
                    else:
                        col.append(r" \qw ")

            else:
                measurement = op
                col = []
                for n in range(self.N+self.num_cbits):

                    if n in measurement.targets:
                        col.append(r" \meter")
                    elif (n-self.N) == measurement.classical_store:
                        store_tag = n - measurement.targets[0]
                        col.append(r" \cwx[%d] " % store_tag)
                    else:
                        col.append(r" \qw ")

            col.append(r" \qw ")
            rows.append(col)

        input_states_quantum = [r"\lstick{\ket{" + x + "}}" if x is not None
                                else "" for x in self.input_states[:self.N]]
        input_states_classical = [r"\lstick{" + x + "}" if x is not None
                                  else "" for x in self.input_states[self.N:]]
        input_states = input_states_quantum + input_states_classical

        code = ""
        n_iter = (reversed(range(self.N+self.num_cbits)) if self.reverse_states
                  else range(self.N+self.num_cbits))
        for n in n_iter:
            code += r" & %s" % input_states[n]
            for m in range(len(ops)):
                code += r" & %s" % rows[m][n]
            code += r" & \qw \\ " + "\n"

        return code

    # This slightly convoluted dance with the conversion formats is because
    # image conversion has optional dependencies.  We always want the `png` and
    # `svg` methods to be available so that they are discoverable by the user,
    # however if one is called without the required dependency, then they'll
    # get a `RuntimeError` explaining the problem.  We only want the IPython
    # magic methods `_repr_xxx_` to be defined if we know that the image
    # conversion is available, so the user doesn't get exceptions on display
    # because IPython tried to do something behind their back.

    def _raw_png(self):
        return _latex.image_from_latex(self.latex_code(), "png")

    if 'png' in _latex.CONVERTERS:
        _repr_png_ = _raw_png

    @property
    def png(self):
        return DisplayImage(self._raw_png(), embed=True)

    def _raw_svg(self):
        return _latex.image_from_latex(self.latex_code(), "svg")

    if 'svg' in _latex.CONVERTERS:
        _repr_svg_ = _raw_svg

    @property
    def svg(self):
        return DisplaySVG(self._raw_svg())

    def _to_qasm(self, qasm_out):
        """
        Pipe output of circuit object to QasmOutput object.

        Parameters
        ----------
        qasm_out: QasmOutput
            object to store QASM output.
        """

        qasm_out.output("qreg q[{}];".format(self.N))
        if self.num_cbits:
            qasm_out.output("creg c[{}];".format(self.num_cbits))
        qasm_out.output(n=1)

        for op in self.gates:
            if ((not isinstance(op, Measurement))
                    and not qasm_out.is_defined(op.name)):
                qasm_out._qasm_defns(op)

        for op in self.gates:
            op._to_qasm(qasm_out)
