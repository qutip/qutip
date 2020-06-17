from qutip.qip.circuit import QubitCircuit
import re
import os
from itertools import chain
from qutip.qip import gate_sequence_product
from math import pi


class QASMGate:
    '''
    Class which stores the gate definitions as specified in QASM.
    '''

    def __init__(self, name, gate_args, gate_regs):
        self.name = name
        self.gate_args = gate_args
        self.gate_regs = gate_regs
        self.gates_inside = []


def _tokenize(tok_commands):
    '''
    Tokenizes QASM code for processing
    '''

    processed_commands = []

    for line in tok_commands:

        # carry out some pre-processing for convenience
        for c in "[]()":
            line = line.replace(c, " " + c + " ")
        for c in "{}":
            line = line.replace(c, " ; " + c + " ; ")
        line_commands = line.split(";")
        line_commands = list(filter(lambda x: x != "", line_commands))

        for command in line_commands:
            # for gates specified without arguments and other statements
            if "(" not in command:
                f = list(chain(*[a.split() for a in command.split(",")]))
                f = [a.strip() for a in f]
            # for gates specified with arguments
            else:
                groups = re.match(r"(^.*?)\((.*)\)(.*)", command)
                if not groups:
                    raise SyntaxError("bracket error")
                f = groups.group(1).split()
                f.append("(")
                f += groups.group(2).split(",")
                f.append(")")
                f += groups.group(3).split(",")
                f = [a.strip() for a in f]
            processed_commands.append(f)

    return list(filter(lambda x: x != [], processed_commands))


def _gate_processor(command):
    '''
    Procsesses tokens for a gate call statement separating into args and regs
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


class QASMProcessor:
    '''
    Class which holds variables used in processing QASM code.
    '''

    def __init__(self, commands):
        self.qubit_regs = {}
        self.cbit_regs = {}
        self.num_qubits = 0
        self.num_cbits = 0
        self.gate_names = set(["CX", "U"])
        self.qasm_gates = {}
        self.qasm_gates["U"] = QASMGate("U", ["alpha", "beta", "gamma"], ["q"])
        self.qasm_gates["CX"] = QASMGate("CX", [], ["c", "t"])
        self.commands = commands

    def _process_includes(self):
        '''
        QASM allows for code to be specified in additional files with the
        ".inc" extension, espcially to specify gate definitions in terms of
        the built-in gates. This function processes into tokens all the additional
        files and inserts it into previously processed tokens
        '''

        prev_index = 0
        expanded_commands = []

        for curr_index, command in enumerate(self.commands):

            if command[0] != "include":
                continue

            filename = command[1].strip('"')
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    qasm_lines = [line.strip() for line in f.read().splitlines()]
                    qasm_lines = list(filter(lambda x: x[:2] != "//" and x != "",
                                    qasm_lines))

                    expanded_commands = (expanded_commands
                                        + command[prev_index:curr_index]
                                        + _tokenize(qasm_lines))
                    prev_index = curr_index + 1
            else:
                raise ValueError(command[1] + ": such a file does not exist")

        expanded_commands += self.commands[prev_index:]
        self.commands = expanded_commands

    def _custom_gate(self, qc_temp, gate_call):
        '''
        Recursively processes a custom-defined gate with specified arguments
        to produce a dummy circuit with all the gates constituting it.
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
                com_args = [command.replace(arg.strip(),
                                            str(real_arg)) for command
                                                            in com_args]
            for reg, real_reg in regs_map.items():
                com_regs = [command.replace(reg.strip(),
                                            str(real_reg)) for command
                                                            in com_regs]
            com_args = [eval(arg) for arg in com_args]

            if name == "CX":
                qc_temp.add_gate("CNOT",  targets=int(com_regs[1]),
                            controls=int(com_regs[0]))
            elif name == "U":
                target = int(com_regs[0])
                qc_temp.add_gate("RZ", targets=target,
                            arg_value=float(com_args[2]))
                qc_temp.add_gate("RY", targets=target,
                            arg_value=float(com_args[0]))
                qc_temp.add_gate("RZ", targets=target,
                            arg_value=float(com_args[1]))
            else:
                self._custom_gate(qc_temp, [name, com_args, com_regs])

    def _regs_processor(self, regs, reg_type):
        '''
        Processes register tokens mapping them to the QubitCircuit index
        of the respective registers.
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
                        raise SyntaxError("syntax")
                else:
                    new_regs.append(token)
            if open_bracket_mode:
                raise SyntaxError("Incorrect formatting")
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
                    raise SyntaxError()
                cbit_name = groups.group(3)
                cbit_ind = int(groups.group(4))
                cbit_lst = self.cbit_regs[cbit_name]
                if cbit_ind < len(cbit_lst):
                    cbit = cbit_lst[0] + cbit_ind
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
                    raise SyntaxError("some")
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
                return ip(*list(map(
                        lambda x: x if isinstance(x, list) else [x] * expand,
                        new_regs)))
            else:
                return [new_regs]

    def _initialize_pass(self):
        '''
        Passes through the tokenized commands to create QASMGate objects for
        each user-defined gate, as well as process register declarations.
        '''

        gate_defn_mode = False
        open_bracket_mode = False

        unprocessed = set()

        for num, command in enumerate(self.commands):
            if gate_defn_mode:
                if command[0] == "{":
                    gate_defn_mode = False
                    open_bracket_mode = True
                    gate_elems = []
                    continue
                else:
                    raise SyntaxError("unmatched brackets")
            elif open_bracket_mode:
                if command[0] == "}":
                    open_bracket_mode = False
                    self.gate_names.add(curr_gate.name)
                    self.qasm_gates[curr_gate.name] = curr_gate
                    continue
                elif command[0] in self.gate_names:
                    name = command[0]
                    gate_args, gate_regs = _gate_processor(command)
                    gate_added = self.qasm_gates[name]
                    if (len(gate_added.gate_args) == len(gate_args)
                            and len(gate_added.gate_regs) == len(gate_regs)):
                        curr_gate.gates_inside.append([name,
                                                    gate_args, gate_regs])
            elif command[0] == "gate":
                gate_name = command[1]
                gate_args, gate_regs = _gate_processor(command[1:])
                curr_gate = QASMGate(gate_name, gate_args, gate_regs)
                gate_defn_mode = True

            elif command[0] == "qreg":
                groups = re.match(r"(.*)\[(.*)\]", "".join(command[1:]))
                if groups:
                    qubit_name = groups.group(1)
                    num_regs = int(groups.group(2))
                    self.qubit_regs[qubit_name] = list(range(self.num_qubits,
                                                self.num_qubits + num_regs))
                    self.num_qubits += num_regs
                else:
                    raise SyntaxError("Improper Formatting")

            elif command[0] == "creg":
                groups = re.match(r"(.*)\[(.*)\]", "".join(command[1:]))
                if groups:
                    cbit_name = groups.group(1)
                    num_regs = int(groups.group(2))
                    self.cbit_regs[cbit_name] = list(range(self.num_cbits,
                                                    self.num_cbits + num_regs))
                    self.num_cbits += num_regs
                else:
                    raise SyntaxError("Improper Formatting")
            else:
                unprocessed.add(num)
                continue

        if open_bracket_mode:
            raise SyntaxError("Scope for gate definition poorly formed")

        self.commands = [self.commands[i] for i in unprocessed]

    def _final_pass(self, qc):
        '''
        Takes a blank circuit and adds all the gates and measurements specified.
        '''

        custom_gates = {}
        for command in self.commands:

            if command[0] in self.gate_names:

                args, regs = _gate_processor(command)
                reg_set = self._regs_processor(regs, "gate")

                gate_name = command[0] + "(" + ",".join(args) + ")"

                # creates custom-gate using gate defn and provided args
                if gate_name not in custom_gates:
                    n = len(reg_set[0])
                    qc_temp = QubitCircuit(n)
                    self._custom_gate(qc_temp, [command[0], args,
                                            [str(i) for i in range(n)]])
                    p = gate_sequence_product(qc_temp.propagators())
                    custom_gates[gate_name] = p
                    qc.user_gates = custom_gates

                # adds gate to the QubitCircuit
                for regs in reg_set:
                    regs = [int(i) for i in regs]
                    if command[0] == "CX":
                        qc.add_gate("CNOT",
                                    targets=[regs[1]],
                                    controls=[regs[0]])
                    else:
                        if not isinstance(regs, list):
                            regs = [regs]
                        qc.add_gate(gate_name, targets=regs)

            elif command[0] == "measure":
                # adds measurement to the QubitCircuit
                reg_set = self._regs_processor(command[1:], "measure")
                for regs in reg_set:
                    # qc.add_measurement("M", targets = [regs[0]], classical_store = regs[1])
                    continue


def read_qasm(file):
    '''
    Read OpenQASM intermediate representation
    (https://github.com/Qiskit/openqasm) and return
    a QubitCircuit and state inputs as specified in the
    QASM file.
    '''

    f = open("f2.qasm", "r")
    # split input into lines and ignore comments
    qasm_lines = [line.strip() for line in f.read().splitlines()]
    qasm_lines = list(filter(lambda x: x[:2] != "//" and x != "", qasm_lines))
    f.close()

    if qasm_lines.pop(0) != "OPENQASM 2.0;":
        raise SyntaxError("File does not contain QASM header")

    qasm_obj = QASMProcessor(qasm_lines)
    qasm_obj.commands = _tokenize(qasm_obj.commands)
    qasm_obj._process_includes()

    qasm_obj._initialize_pass()
    qc = QubitCircuit(qasm_obj.num_qubits, num_cbits=qasm_obj.num_cbits)
    qasm_obj._final_pass(qc)
    return qc
