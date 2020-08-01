import re
import os
from itertools import chain
from copy import deepcopy
import warnings

from math import pi
import numpy as np

from qutip.qip import gate_sequence_product
from qutip.qip.circuit import QubitCircuit
from qutip.qip.operations.gates import controlled_gate, qasmu_gate, rz, snot


__all__ = ["read_qasm", "save_qasm", "print_qasm", "circuit_to_qasm_str"]


class QasmGate:
    '''
    Class which stores the gate definitions as specified in the QASM file.
    '''

    def __init__(self, name, gate_args, gate_regs):
        self.name = name
        self.gate_args = gate_args
        self.gate_regs = gate_regs
        self.gates_inside = []


def _get_qiskit_gates():
    '''
    Create and return a dictionary containing custom gates needed
    for "qiskit" mode. These include a subset of gates usually defined
    in the file "qelib1.inc".

    Returns a dictionary mapping gate names to QuTiP gates.
    '''

    def u2(args):
        return qasmu_gate([np.pi/2, args[0], args[1]])

    def id():
        return qasmu_gate([0, 0, 0])

    def sdg():
        return rz(-1 * np.pi/2)

    def tdg():
        return rz(-1 * np.pi/4)

    def cu3(args):
        return controlled_gate(qasmu_gate(args))

    def ch():
        return controlled_gate(snot())

    return {"ch": ch, "tdg": tdg, "id": id, "u2": u2, "sdg": sdg, "cu3": cu3}


def _tokenize_line(command):
    '''
    Tokenize (break into several parts a string of) a single line of QASM code.

    Parameters
    ----------
    command : str
        One line of QASM code to be broken into "tokens".

    Returns
    -------
    tokens : list of str
        The tokens (parts) corresponding to the qasm line taken as input.
    '''

    # for gates without arguments
    if "(" not in command:
        tokens = list(chain(*[a.split() for a in command.split(",")]))
        tokens = [token.strip() for token in tokens]
    # for classically controlled gates
    elif re.match(r"\s*if\s*\(", command):
        groups = re.match(r"\s*if\s*\((.*)\)\s*(.*)\s+\((.*)\)(.*)", command)
        # for classically controlled gates with arguments
        if groups:
            tokens = ["if", "(", groups.group(1), ")"]
            tokens_gate = _tokenize_line("{} ({}) {}".format(groups.group(2),
                                                             groups.group(3),
                                                             groups.group(4)))
            tokens += tokens_gate
        # for classically controlled gates without arguments
        else:
            groups = re.match(r"\s*if\s*\((.*)\)(.*)", command)
            tokens = ["if", "(", groups.group(1), ")"]
            tokens_gate = _tokenize_line(groups.group(2))
            tokens += tokens_gate
        tokens = [token.strip() for token in tokens]
    # for gates with arguments
    else:
        groups = re.match(r"(^.*?)\((.*)\)(.*)", command)
        if not groups:
            raise SyntaxError("QASM: Incorrect bracket formatting")
        tokens = groups.group(1).split()
        tokens.append("(")
        tokens += groups.group(2).split(",")
        tokens.append(")")
        tokens += groups.group(3).split(",")
        tokens = [token.strip() for token in tokens]

    return tokens


def _tokenize(token_cmds):
    '''
    Tokenize QASM code for processing, i.e. break it into several parts.

    Parameters
    ----------
    token_cmds : list of str
        Lines of QASM code.

    Returns
    -------
    tokens : list of (list of str)
        List of tokens corresponding to each QASM line taken as input.
    '''

    processed_commands = []

    for line in token_cmds:

        # carry out some pre-processing for convenience
        for c in "[]()":
            line = line.replace(c, " " + c + " ")
        for c in "{}":
            line = line.replace(c, " ; " + c + " ; ")
        line_commands = line.split(";")
        line_commands = list(filter(lambda x: x != "", line_commands))

        for command in line_commands:
            tokens = _tokenize_line(command)
            processed_commands.append(tokens)

    return list(filter(lambda x: x != [], processed_commands))


def _gate_processor(command):
    '''
    Process tokens for a gate call statement separating them into args and regs.
    Processes tokens from a "gate call" (e.g. rx(pi) q[0]) and returns the
    tokens for the arguments and registers separately.
    '''

    gate_args = []
    gate_regs = []
    tokens = command[1:]
    reg_start = 0

    # extract arguments
    if "(" in tokens and ")" in tokens:
        bopen = tokens.index("(")
        bclose = tokens.index(")")
        gate_args = tokens[bopen+1:bclose]
        reg_start = bclose+1

    # extract registers
    gate_regs = tokens[reg_start:]

    return gate_args, gate_regs


class QasmProcessor:
    '''
    Class which holds variables used in processing QASM code.
    '''

    def __init__(self, commands, mode="qiskit", version="2.0"):
        self.qubit_regs = {}
        self.cbit_regs = {}
        self.num_qubits = 0
        self.num_cbits = 0
        self.qasm_gates = {}
        self.mode = mode
        self.version = version
        self.predefined_gates = set(["CX", "U"])

        if self.mode == "qiskit":
            self.qiskitgates = set(["u3", "u2", "u1", "cx",  "id", "x", "y",
                                    "z", "h", "s", "sdg", "t", "tdg", "rx",
                                    "ry", "rz", "cz", "cy", "ch", "ccx", "crz",
                                    "cu1", "cu3"])
            self.predefined_gates = self.predefined_gates.union(self.qiskitgates)

        self.gate_names = deepcopy(self.predefined_gates)
        for gate in self.predefined_gates:
            self.qasm_gates[gate] = QasmGate("U",
                                             ["alpha", "beta", "gamma"],
                                             ["q"])
        self.commands = commands

    def _process_includes(self):
        '''
        QASM allows for code to be specified in additional files with the
        ".inc" extension, especially to specify gate definitions in terms of
        the built-in gates. Process into tokens all the
        additional files and insert it into previously processed list.
        '''

        prev_index = 0
        expanded_commands = []

        for curr_index, command in enumerate(self.commands):

            if command[0] != "include":
                continue

            filename = command[1].strip('"')

            if self.mode == "qiskit" and filename == "qelib1.inc":
                continue

            if os.path.exists(filename):
                with open(filename, "r") as f:
                    qasm_lines = [line.strip() for line
                                  in f.read().splitlines()]
                    qasm_lines = list(filter(
                                      lambda x: x[:2] != "//" and x != "",
                                      qasm_lines))

                    expanded_commands = (expanded_commands
                                         + command[prev_index:curr_index]
                                         + _tokenize(qasm_lines))
                    prev_index = curr_index + 1
            else:
                raise ValueError(command[1] + ": such a file does not exist")

        expanded_commands += self.commands[prev_index:]
        self.commands = expanded_commands

    def _initialize_pass(self):
        '''
        Passes through the tokenized commands, create QasmGate objects for
        each user-defined gate, process register declarations.
        '''

        gate_defn_mode = False
        open_bracket_mode = False

        unprocessed = []

        for num, command in enumerate(self.commands):
            if gate_defn_mode:
                if command[0] == "{":
                    gate_defn_mode = False
                    open_bracket_mode = True
                    gate_elems = []
                    continue
                else:
                    raise SyntaxError("QASM: incorrect bracket formatting")
            elif open_bracket_mode:
                if command[0] == "{":
                    raise SyntaxError("QASM: incorrect bracket formatting")
                elif command[0] == "}":
                    if not curr_gate.gates_inside:
                        raise NotImplementedError("QASM: opaque gate {} are  \
                                                   not allowed, please define \
                                                   or omit \
                                                   them".format(curr_gate.name))
                    open_bracket_mode = False
                    self.gate_names.add(curr_gate.name)
                    self.qasm_gates[curr_gate.name] = curr_gate
                    continue
                elif command[0] in self.gate_names:
                    name = command[0]
                    gate_args, gate_regs = _gate_processor(command)
                    gate_added = self.qasm_gates[name]
                    curr_gate.gates_inside.append([name,
                                                   gate_args,
                                                   gate_regs])
            elif command[0] == "gate":
                gate_name = command[1]
                gate_args, gate_regs = _gate_processor(command[1:])
                curr_gate = QasmGate(gate_name, gate_args, gate_regs)
                gate_defn_mode = True

            elif command[0] == "qreg":
                groups = re.match(r"(.*)\[(.*)\]", "".join(command[1:]))
                if groups:
                    qubit_name = groups.group(1)
                    num_regs = int(groups.group(2))
                    self.qubit_regs[qubit_name] = list(range(
                                                self.num_qubits,
                                                self.num_qubits + num_regs))
                    self.num_qubits += num_regs
                else:
                    raise SyntaxError("QASM: incorrect bracket formatting")

            elif command[0] == "creg":
                groups = re.match(r"(.*)\[(.*)\]", "".join(command[1:]))
                if groups:
                    cbit_name = groups.group(1)
                    num_regs = int(groups.group(2))
                    self.cbit_regs[cbit_name] = list(range(self.num_cbits,
                                                     self.num_cbits + num_regs))
                    self.num_cbits += num_regs
                else:
                    raise SyntaxError("QASM: incorrect bracket formatting")
            elif command[0] == "reset":
                raise NotImplementedError(("QASM: reset functionality "
                                           "is not supported."))
            elif command[0] in ["barrier", "include"]:
                continue
            else:
                unprocessed.append(num)
                continue

        if open_bracket_mode:
            raise SyntaxError("QASM: incorrect bracket formatting")

        self.commands = [self.commands[i] for i in unprocessed]

    def _custom_gate(self, qc_temp, gate_call):
        '''
        Recursively process a custom-defined gate with specified arguments
        to produce a dummy circuit with all the gates in the custom-defined
        gate.

        Parameters
        ----------

        qc_temp: :class:`.QubitCircuit`
            temporary circuit to process custom gate
        gate_call: list of str
            tokens corresponding to gate signature/call
        '''

        gate_name, args, regs = gate_call
        gate = self.qasm_gates[gate_name]
        args_map = {}
        regs_map = {}

        # maps variables to supplied arguments, registers
        for i, arg in enumerate(gate.gate_args):
            args_map[arg] = eval(str(args[i]))
        for i, reg in enumerate(gate.gate_regs):
            regs_map[reg] = regs[i]
        # process all the constituent gates with supplied arguments, registers
        for call in gate.gates_inside:

            # create function call for the constituent gate
            name, com_args, com_regs = call

            for arg, real_arg in args_map.items():
                com_args = [command.replace(arg.strip(), str(real_arg))
                            for command in com_args]
            for reg, real_reg in regs_map.items():
                com_regs = [command.replace(reg.strip(), str(real_reg))
                            for command in com_regs]
            com_args = [eval(arg) for arg in com_args]

            if name in self.predefined_gates:
                qc_temp.user_gates = _get_qiskit_gates()
                com_regs = [int(reg) for reg in com_regs]
                self._add_predefined_gates(qc_temp, name, com_regs, com_args)
            else:
                self._custom_gate(qc_temp, [name, com_args, com_regs])

    def _regs_processor(self, regs, reg_type):
        '''
        Process register tokens: map them to the :class:`.QubitCircuit` indices
        of the respective registers.

        Parameters
        ----------
        regs : list of str
            Token list corresponding to qubit/cbit register invocations.
        reg_type : str
            reg_type can be "measure" or "gate" to specify type of required
            processing.

        Returns
        -------

        regs list : list of (list of regs)
                list of register sets to which circuit operations
                are applied.

        '''

        # turns messy tokens into appropriate form
        # ['q', 'p', '[', '0', ']'] -> ['q', 'p[0]']
        regs = [reg.replace(" ", "") for reg in regs]
        if "[" in regs:
            prev_token = ""
            new_regs = []
            open_bracket_mode = False
            for token in regs:
                if token == "[":
                    open_bracket_mode = True
                elif open_bracket_mode:
                    if token == "]":
                        open_bracket_mode = False
                        reg_name = new_regs.pop()
                        new_regs.append(reg_name + "[" + reg_num + "]")
                    elif token.isdigit():
                        reg_num = token
                    else:
                        raise SyntaxError("QASM: incorrect bracket formatting")
                else:
                    new_regs.append(token)
            if open_bracket_mode:
                raise SyntaxError("QASM: incorrect bracket formatting")
            regs = new_regs

        if reg_type == "measure":

            # processes register tokens of the form q[i] -> c[i]
            groups = re.match(r"(.*)\[(.*)\]->(.*)\[(.*)\]", "".join(regs))
            if groups:
                qubit_name = groups.group(1)
                qubit_ind = int(groups.group(2))
                qubit_lst = self.qubit_regs[qubit_name]
                if qubit_ind < len(qubit_lst):
                    qubit = qubit_lst[0] + qubit_ind
                else:
                    raise ValueError("QASM: qubit index out of bounds")
                cbit_name = groups.group(3)
                cbit_ind = int(groups.group(4))
                cbit_lst = self.cbit_regs[cbit_name]
                if cbit_ind < len(cbit_lst):
                    cbit = cbit_lst[0] + cbit_ind
                else:
                    raise ValueError("QASM: cbit index out of bounds")
                return [(qubit, cbit)]
            # processes register tokens of the form q -> c
            else:
                qubit_name = regs[0]
                cbit_name = regs[2]
                qubits = self.qubit_regs[qubit_name]
                cbits = self.cbit_regs[cbit_name]
                if len(qubits) == len(cbits):
                    return zip(qubits, cbits)
                else:
                    raise ValueError("QASM: qubit and cbit \
                                     register sizes are different")
        else:
            # processes gate tokens to create sets of registers to
            # which the gates are applied.
            new_regs = []
            expand = 0
            for reg in regs:
                if "[" in reg:
                    groups = re.match(r"(.*)\[(.*)\]", "".join(reg))
                    qubit_name = groups.group(1)
                    qubit_ind = int(groups.group(2))
                    qubit_lst = self.qubit_regs[qubit_name]
                    if qubit_ind < len(qubit_lst):
                        qubit = qubit_lst[0] + qubit_ind
                else:
                    qubit_name = reg
                    qubit = self.qubit_regs[qubit_name]
                    expand = len(qubit)
                new_regs.append(qubit)
            if expand:
                return zip(*list(map(
                        lambda x: x if isinstance(x, list) else [x] * expand,
                        new_regs)))
            else:
                return [new_regs]

    def _add_qiskit_gates(self, qc, name, regs, args=None,
                          classical_controls=None, control_value=None):
        """
        Add any gates that are pre-defined in qiskit-style exported
        qasm file with included "qelib1.inc".

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            the circuit to which the gate is added.
        name : str
            name of gate to be added.
        regs : list of int
            list of qubit register indices to add gates to.
        args : float, optional
            value of args supplied to the gate.
        classical_controls : list of int, optional
            indices of classical bits to control gate on.
        control_value : int, optional
            value of classical bits to control on, the classical controls are
            interpreted as an integer with lowest bit being the first one.
            If not specified, then the value is interpreted to be
            2 ** len(classical_controls) - 1
            (i.e. all classical controls are 1).
        """

        gate_name_map_1q = {"x": "X", "y": "Y", "z": "Z", "h": "SNOT",
                            "t": "T", "s": "S", "sdg": "sdg", "tdg": "tdg",
                            "rx": "RX", "ry": "RY", "rz": "RZ"}
        if len(args) == 0:
            args = None
        elif len(args) == 1:
            args = args[0]

        if name == "u3":
            qc.add_gate("QASMU", targets=regs[0], arg_value=args,
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "u2":
            qc.add_gate("u2", targets=regs[0], arg_value=args,
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "u1":
            qc.add_gate("RZ", targets=regs[0], arg_value=args,
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "cz":
            qc.add_gate("CZ", targets=regs[1], controls=regs[0],
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "cy":
            qc.add_gate("CY", targets=regs[1], controls=regs[0],
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "ch":
            qc.add_gate("ch", targets=regs,
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "ccx":
            qc.add_gate("TOFFOLI", targets=regs[2], controls=regs[:2],
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "crz":
            qc.add_gate("CRZ", targets=regs[1], controls=regs[0],
                        arg_value=args,
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "cu1":
            qc.add_gate("CPHASE", targets=regs[1], controls=regs[0],
                        arg_value=args,
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "cu3":
            qc.add_gate("cu3", targets=[regs[0], regs[1]],
                        arg_value=args,
                        classical_controls=classical_controls,
                        control_value=control_value)
        if name == "cx":
            qc.add_gate("CNOT",  targets=int(regs[1]), controls=int(regs[0]),
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name in gate_name_map_1q:
            if args == []:
                args = None
            qc.add_gate(gate_name_map_1q[name], targets=regs[0], arg_value=args,
                        classical_controls=classical_controls,
                        control_value=control_value)

    def _add_predefined_gates(self, qc, name, com_regs, com_args,
                              classical_controls=None, control_value=None):
        """
        Add any gates that are pre-defined and/or inbuilt
        in our circuit.

        Parameters
        ----------
        qc : :class:`.QubitCircuit`
            the circuit to which the gate is added.
        name : str
            name of gate to be added.
        regs : list of int
            list of qubit register indices to add gates to.
        args : float, optional
            value of args supplied to the gate.
        classical_controls : list of int, optional
            indices of classical bits to control gate on.
        control_value : int, optional
            value of classical bits to control on, the classical controls are
            interpreted as an integer with lowest bit being the first one.
            If not specified, then the value is interpreted to be
            2 ** len(classical_controls) - 1
            (i.e. all classical controls are 1).
        """

        if name == "CX":
            qc.add_gate("CNOT",
                        targets=int(com_regs[1]),
                        controls=int(com_regs[0]),
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name == "U":
            qc.add_gate("QASMU",
                        targets=int(com_regs[0]),
                        arg_value=[float(arg) for arg in com_args],
                        classical_controls=classical_controls,
                        control_value=control_value)
        elif name in self.qiskitgates and self.mode == "qiskit":
            self._add_qiskit_gates(qc, name, com_regs, com_args,
                                   classical_controls, control_value)

    def _gate_add(self, qc, command, custom_gates,
                  classical_controls=None, control_value=None):
        '''
        Add gates to :class:`.QubitCircuit` from processed tokens,
        define custom gates if necessary.

        Parameters
        ----------
        qc: :class:`.QubitCircuit`
            circuit object to which gate is added
        command: list of str
            list of tokens corresponding to gate application
        custom_gates: {gate name : gate function or unitary}
            dictionary of user gates defined for qutip
        classical_controls : int or list of int, optional
            indices of classical bits to control gate on.
        control_value : int, optional
            value of classical bits to control on, the classical controls are
            interpreted as an integer with lowest bit being the first one.
            If not specified, then the value is interpreted to be
            2 ** len(classical_controls) - 1
            (i.e. all classical controls are 1).
        '''

        args, regs = _gate_processor(command)
        reg_set = self._regs_processor(regs, "gate")

        if args:
            gate_name = "{}({})".format(command[0], ",".join(args))
        else:
            gate_name = "{}".format(command[0])

        # creates custom-gate (if required) using gate defn and provided args
        if (command[0] not in self.predefined_gates
                and command[0] not in custom_gates):
            n = len(reg_set[0])
            qc_temp = QubitCircuit(n)
            self._custom_gate(qc_temp,
                              [command[0], args, [str(i) for i in range(n)]])
            unitary_mat = gate_sequence_product(qc_temp.propagators())
            custom_gates[gate_name] = unitary_mat

        qc.user_gates = custom_gates

        # adds gate to the QubitCircuit
        for regs in reg_set:
            regs = [int(i) for i in regs]
            if command[0] in self.predefined_gates:
                args = [eval(arg) for arg in args]
                self._add_predefined_gates(
                                    qc,
                                    command[0],
                                    regs,
                                    args,
                                    classical_controls=classical_controls,
                                    control_value=control_value)
            else:
                if not isinstance(regs, list):
                    regs = [regs]
                qc.add_gate(gate_name,
                            targets=regs,
                            classical_controls=classical_controls,
                            control_value=control_value)

    def _final_pass(self, qc):
        '''
        Take a blank circuit, add all the gates and measurements specified
        by QASM.
        '''

        custom_gates = {}
        if self.mode == "qiskit":
            custom_gates = _get_qiskit_gates()

        for command in self.commands:
            if command[0] in self.gate_names:
                # adds gates to the QubitCircuit
                self._gate_add(qc, command, custom_gates)
            elif command[0] == "measure":
                # adds measurement to the QubitCircuit
                reg_set = self._regs_processor(command[1:], "measure")
                for regs in reg_set:
                    qc.add_measurement("M", targets=[regs[0]],
                                       classical_store=regs[1])
            elif command[0] == "if":
                warnings.warn(("Information about individual registers"
                              " is not preserved in QubitCircuit"))
                # adds classically controlled gates to the QubitCircuit
                cbit_reg, control_value = command[2].split("==")
                cbit_inds = self.cbit_regs[cbit_reg]
                control_value = int(control_value)
                self._gate_add(qc, command[4:], custom_gates,
                               cbit_inds, control_value)
            else:
                err = "QASM: {} is not a valid QASM command.".format(command[0])
                raise SyntaxError(err)


def read_qasm(qasm_input, mode="qiskit", version="2.0", strmode=False):
    '''
    Read OpenQASM intermediate representation
    (https://github.com/Qiskit/openqasm) and return
    a :class:`.QubitCircuit` and state inputs as specified in the
    QASM file.

    Parameters
    ----------
    qasm_input : str
        File location or String Input for QASM file to be imported. In case of
        string input, the parameter strmode must be True.
    mode : str
        QASM mode to be read in. When mode is "qiskit",
        the "qelib1.inc" include is automatically included,
        without checking externally. Otherwise, each include is
        processed.
    version : str
        QASM version of the QASM file. Only version 2.0 is currently supported.
    strmode : bool
        if specified as True, indicates that qasm_input is in string format
        rather than from file.

    Returns
    -------
    qc : :class:`.QubitCircuit`
        Returns a :class:`.QubitCircuit` object specified in the QASM file.
    '''

    if strmode:
        qasm_lines = qasm_input.splitlines()
    else:
        f = open(qasm_input, "r")
        qasm_lines = f.read().splitlines()
        f.close()

    # split input into lines and ignore comments
    qasm_lines = [line.strip() for line in qasm_lines]
    qasm_lines = list(filter(lambda x: x[:2] != "//" and x != "", qasm_lines))

    if version != "2.0":
        raise NotImplementedError("QASM: Only OpenQASM 2.0 \
                                  is currently supported.")

    if qasm_lines.pop(0) != "OPENQASM 2.0;":
        raise SyntaxError("QASM: File does not contain QASM 2.0 header")

    qasm_obj = QasmProcessor(qasm_lines, mode=mode, version=version)
    qasm_obj.commands = _tokenize(qasm_obj.commands)

    qasm_obj._process_includes()

    qasm_obj._initialize_pass()
    qc = QubitCircuit(qasm_obj.num_qubits, num_cbits=qasm_obj.num_cbits)

    qasm_obj._final_pass(qc)

    return qc


_GATE_NAME_TO_QASM_NAME = {
    "QASMU": "U",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "SNOT": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "T": "t",
    "CRZ": "crz",
    "CNOT": "cx",
    "TOFFOLI": "ccx"
}


class QasmOutput():
    """
    Class for QASM export.

    Parameters
    ----------
    version: str, optional
        OpenQASM version, currently must be "2.0" necessarily.
    """

    def __init__(self, version="2.0"):
        self.version = version
        self.lines = []
        self.gate_name_map = deepcopy(_GATE_NAME_TO_QASM_NAME)

    def output(self, line="", n=0):
        '''
        Pipe QASM output string to QasmOutput's lines variable.

        Parameters
        ----------
        line: str, optional
            string to be appended to QASM output.
        n: int, optional
            number of blank lines to be appended to QASM output.
        '''

        if line:
            self.lines.append(line)
        self.lines = self.lines + [""] * n

    def _flush(self):
        '''
        Resets QasmOutput variables.
        '''

        self.lines = []
        self.gate_name_map = deepcopy(_GATE_NAME_TO_QASM_NAME)

    def _qasm_str(self, q_name, q_controls, q_targets, q_args):
        '''
        Returns QASM string for gate definition or gate application given
        name, registers, arguments.
        '''

        if not q_controls:
            q_controls = []
        q_regs = q_controls + q_targets

        if isinstance(q_targets[0], int):
            q_regs = ",".join(['q[{}]'.format(reg) for reg in q_regs])
        else:
            q_regs = ",".join(q_regs)

        if q_args:
            if isinstance(q_args, list):
                q_args = ",".join([str(arg) for arg in q_args])
            return "{}({}) {};".format(q_name, q_args, q_regs)
        else:
            return "{} {};".format(q_name, q_regs)

    def _qasm_defn_from_resolved(self, curr_gate, gates_lst):
        '''
        Resolve QASM definition of QuTiP gate in terms of component gates.

        Parameters
        ----------
        curr_gate: :class:`.Gate`
            QuTiP gate which needs to be resolved into component gates.
        gates_lst: list of :class:`.Gate`
            list of gate that constitute QASM definition of self.
        '''

        forbidden_gates = ["GLOBALPHASE", "PHASEGATE"]
        reg_map = ['a', 'b', 'c']

        q_controls = None
        if curr_gate.controls:
            q_controls = [reg_map[i] for i in curr_gate.controls]
        q_targets = [reg_map[i] for i in curr_gate.targets]
        arg_name = None
        if curr_gate.arg_value:
            arg_name = "theta"

        self.output("gate {} {{".format(self._qasm_str(curr_gate.name.lower(),
                                                       q_controls,
                                                       q_targets,
                                                       arg_name)[:-1]))

        for gate in gates_lst:
            if gate.name in self.gate_name_map:
                gate.targets = [reg_map[i] for i in gate.targets]
                if gate.controls:
                    gate.controls = [reg_map[i] for i in gate.controls]
                self.output(self._qasm_str(
                                self.gate_name_map[gate.name],
                                gate.controls,
                                gate.targets,
                                gate.arg_value))
            elif gate.name in forbidden_gates:
                continue
            else:
                raise ValueError(("The given resolved gate {} cannot be defined"
                                  " in QASM format").format(curr_gate.name))
        self.output("}")

    def _qasm_defn_resolve(self, gate):
        '''
        Resolve QASM definition of QuTiP gate if possible.

        Parameters
        ----------
        gate: :class:`.Gate`
            QuTiP gate which needs to be resolved into component gates.

        '''

        qc = QubitCircuit(3)
        gates_lst = []
        if gate.name == "CSIGN":
            qc._gate_CSIGN(gate, gates_lst)
        else:
            err_msg = "No definition specified for {} gate".format(gate.name)
            raise NotImplementedError(err_msg)

        self._qasm_defn_from_resolved(gate, gates_lst)
        self.gate_name_map[gate.name] = gate.name.lower()

    def _qasm_defns(self, gate):
        '''
        Define QASM gates for QuTiP gates that do not have QASM counterparts.

        Parameters
        ----------
        gate: :class:`.Gate`
            QuTiP gate which needs to be defined in QASM format.
        '''

        if gate.name == "CRY":
            gate_def = "gate cry(theta) a,b { cu3(theta,0,0) a,b; }"
        elif gate.name == "CRX":
            gate_def = "gate crx(theta) a,b { cu3(theta,-pi/2,pi/2) a,b; }"
        elif gate.name == "SQRTNOT":
            gate_def = "gate sqrtnot a {h a; u1(-pi/2) a; h a; }"
        elif gate.name == "CS":
            gate_def = "gate cs a,b { cu1(pi/2) a,b; }"
        elif gate.name == "CT":
            gate_def = "gate ct a,b { cu1(pi/4) a,b; }"
        elif gate.name == "SWAP":
            gate_def = "gate swap a,b { cx a,b; cx b,a; cx a,b; }"
        else:
            self._qasm_defn_resolve(gate)
            return

        self.output("// QuTiP definition for gate {}".format(gate.name))
        self.output(gate_def)
        self.gate_name_map[gate.name] = gate.name.lower()

    def qasm_name(self, gate_name):
        '''
        Return QASM gate name for corresponding QuTiP gate.

        Parameters
        ----------
        gate_name: str
            QuTiP gate name.
        '''

        if gate_name in self.gate_name_map:
            return self.gate_name_map[gate_name]
        else:
            return None

    def is_defined(self, gate_name):
        '''
        Check if QASM gate definition exists for QuTiP gate.

        Parameters
        ----------
        gate_name: str
            QuTiP gate name.
        '''

        return gate_name in self.gate_name_map

    def _qasm_output(self, qc):
        '''
        QASM output handler.

        Parameters
        ----------
        qc: :class:`.QubitCircuit`
            circuit object to produce QASM output for.
        '''

        self._flush()

        self.output("// QASM 2.0 file generated by QuTiP", 1)

        if self.version == "2.0":
            self.output("OPENQASM 2.0;")
        else:
            raise NotImplementedError("QASM: Only OpenQASM 2.0 \
                                      is currently supported.")

        self.output('include "qelib1.inc";', 1)

        qc._to_qasm(self)

        return self.lines


def print_qasm(qc):
    '''
    Print QASM output of circuit object.

    Parameters
    ----------
    qc: :class:`.QubitCircuit`
        circuit object to produce QASM output for.
    '''

    qasm_out = QasmOutput("2.0")
    lines = qasm_out._qasm_output(qc)
    for line in lines:
        print(line)


def circuit_to_qasm_str(qc):
    '''
    Return QASM output of circuit object as string

    Parameters
    ----------
    qc: :class:`.QubitCircuit`
        circuit object to produce QASM output for.

    Returns
    -------
    output: str
        string corresponding to QASM output.
    '''

    qasm_out = QasmOutput("2.0")
    lines = qasm_out._qasm_output(qc)
    output = ""
    for line in lines:
        output += line + "\n"
    return output


def save_qasm(qc, file_loc):
    '''
    Save QASM output of circuit object to file.

    Parameters
    ----------
    qc: :class:`.QubitCircuit`
        circuit object to produce QASM output for.
    '''

    qasm_out = QasmOutput("2.0")
    lines = qasm_out._qasm_output(qc)
    with open(file_loc, "w") as f:
        for line in lines:
            f.write("{}\n".format(line))
