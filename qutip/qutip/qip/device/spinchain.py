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
import warnings

import numpy as np

from qutip.operators import sigmax, sigmay, sigmaz, identity
from qutip.tensor import tensor
from qutip.qip.circuit import QubitCircuit
from qutip.qip.device.processor import Processor
from qutip.qip.device.modelprocessor import ModelProcessor
from qutip.qip.pulse import Pulse
from qutip.qip.compiler.gatecompiler import GateCompiler
from qutip.qip.compiler.spinchaincompiler import SpinChainCompiler


__all__ = ['SpinChain', 'LinearSpinChain', 'CircularSpinChain']


class SpinChain(ModelProcessor):
    """
    The processor based on the physical implementation of
    a spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.
    It is a base class and should not be used directly, please
    refer the the subclasses :class:`qutip.qip.LinearSpinChain` and
    :class:`qutip.qip.CircularSpinChain`.
    (Only additional attributes are documented here, for others please
    refer to the parent class :class:`qutip.qip.device.ModelProcessor`)

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `N` or a float for all qubits.

    t2: list of float
        Characterize the decoherence of dephasing for
        each qubit. A list of size `N` or a float for all qubits.

    Attributes
    ----------
    sx: list
        The delta for each of the qubits in the system.

    sz: list
        The epsilon for each of the qubits in the system.

    sxsy: list
        The interaction strength for each of the qubit pair in the system.

    sx_ops: list
        A list of sigmax Hamiltonians for each qubit.

    sz_ops: list
        A list of sigmaz Hamiltonians for each qubit.

    sxsy_ops: list
        A list of tensor(sigmax, sigmay)
        interacting Hamiltonians for each qubit.

    sx_u: array_like
        Pulse matrix for sigmax Hamiltonians.

    sz_u: array_like
        Pulse matrix for sigmaz Hamiltonians.

    sxsy_u: array_like
        Pulse matrix for tensor(sigmax, sigmay) interacting Hamiltonians.
    """
    def __init__(self, N, correct_global_phase,
                 sx, sz, sxsy, t1, t2):
        super(SpinChain, self).__init__(
            N, correct_global_phase=correct_global_phase, t1=t1, t2=t2)
        self.correct_global_phase = correct_global_phase
        self.spline_kind = "step_func"
        # params and ops are set in the submethods

    def set_up_ops(self, N):
        """
        Generate the Hamiltonians for the spinchain model and save them in the
        attribute `ctrls`.

        Parameters
        ----------
        N: int
            The number of qubits in the system.
        """
        # sx_ops
        for m in range(N):
            self.pulses.append(
                Pulse(sigmax(), m, spline_kind=self.spline_kind))
        # sz_ops
        for m in range(N):
            self.pulses.append(
                Pulse(sigmaz(), m, spline_kind=self.spline_kind))
        # sxsy_ops
        operator = tensor([sigmax(), sigmax()]) + tensor([sigmay(), sigmay()])
        for n in range(N - 1):
            self.pulses.append(
                Pulse(operator, [n, n+1], spline_kind=self.spline_kind))

    def set_up_params(self, sx, sz):
        """
        Save the parameters in the attribute `params` and check the validity.

        Parameters
        ----------
        sx: float or list
            The coefficient of sigmax in the model

        sz: flaot or list
            The coefficient of sigmaz in the model

        Notes
        -----
        The coefficient of sxsy is defined in the submethods.
        All parameters will be multiplied by 2*pi for simplicity
        """
        sx_para = 2 * np.pi * self.to_array(sx, self.N)
        self._params["sx"] = sx_para
        sz_para = 2 * np.pi * self.to_array(sz, self.N)
        self._params["sz"] = sz_para

    @property
    def sx_ops(self):
        return self.ctrls[: self.N]

    @property
    def sz_ops(self):
        return self.ctrls[self.N: 2*self.N]

    @property
    def sxsy_ops(self):
        return self.ctrls[2*self.N:]

    @property
    def sx_u(self):
        return self.coeffs[: self.N]

    @property
    def sz_u(self):
        return self.coeffs[self.N: 2*self.N]

    @property
    def sxsy_u(self):
        return self.coeffs[2*self.N:]

    def load_circuit(self, qc, setup):
        """
        Decompose a :class:`qutip.QubitCircuit` in to the control
        amplitude generating the corresponding evolution.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        setup: string
            "linear" or "circular" for two sub-calsses.

        Returns
        -------
        tlist: array_like
            A NumPy array specifies the time of each coefficient

        coeffs: array_like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.
        """
        gates = self.optimize_circuit(qc).gates

        compiler = SpinChainCompiler(
            self.N, self._params, setup=setup,
            global_phase=0., num_ops=len(self.ctrls))
        tlist, self.coeffs, self.global_phase = compiler.decompose(gates)
        self.set_all_tlist(tlist)
        return tlist, self.coeffs

    def adjacent_gates(self, qc, setup="linear"):
        """
        Method to resolve 2 qubit gates with non-adjacent control/s or target/s
        in terms of gates with adjacent interactions for linear/circular spin
        chain system.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            The circular spin chain circuit to be resolved

        setup: Boolean
            Linear of Circular spin chain setup

        Returns
        -------
        qc: :class:`qutip.QubitCircuit`
            Returns QubitCircuit of resolved gates for the qubit circuit in the
            desired basis.
        """
        # FIXME This huge block has been here for a long time.
        # It could be moved to the new compiler section and carefully
        # splitted into smaller peaces.
        qc_t = QubitCircuit(qc.N, qc.reverse_states)
        swap_gates = ["SWAP", "ISWAP", "SQRTISWAP", "SQRTSWAP", "BERKELEY",
                      "SWAPalpha"]
        N = qc.N

        for gate in qc.gates:
            if gate.name == "CNOT" or gate.name == "CSIGN":
                start = min([gate.targets[0], gate.controls[0]])
                end = max([gate.targets[0], gate.controls[0]])

                if (setup == "linear" or
                        (setup == "circular" and (end - start) <= N // 2)):
                    i = start
                    while i < end:
                        if (start + end - i - i == 1 and
                                (end - start + 1) % 2 == 0):
                            # Apply required gate if control and target are
                            # adjacent to each other, provided |control-target|
                            # is even.
                            if end == gate.controls[0]:
                                qc_t.add_gate(gate.name, targets=[i],
                                              controls=[i + 1])
                            else:
                                qc_t.add_gate(gate.name, targets=[i + 1],
                                              controls=[i])

                        elif (start + end - i - i == 2 and
                              (end - start + 1) % 2 == 1):
                            # Apply a swap between i and its adjacent gate,
                            # then the required gate if and then another swap
                            # if control and target have one qubit between
                            # them, provided |control-target| is odd.
                            qc_t.add_gate("SWAP", targets=[i, i + 1])
                            if end == gate.controls[0]:
                                qc_t.add_gate(gate.name, targets=[i + 1],
                                              controls=[i + 2])
                            else:
                                qc_t.add_gate(gate.name, targets=[i + 2],
                                              controls=[i + 1])
                            qc_t.add_gate("SWAP", [i, i + 1])
                            i += 1

                        else:
                            # Swap the target/s and/or control with their
                            # adjacent qubit to bring them closer.
                            qc_t.add_gate("SWAP", [i, i + 1])
                            qc_t.add_gate("SWAP", [start + end - i - 1,
                                                   start + end - i])
                        i += 1

                elif (end - start) < N - 1:
                    """
                    If the resolving has to go backwards, the path is first
                    mapped to a separate circuit and then copied back to the
                    original circuit.
                    """

                    temp = QubitCircuit(N - end + start)
                    i = 0
                    while i < (N - end + start):

                        if (N + start - end - i - i == 1 and
                                (N - end + start + 1) % 2 == 0):
                            if end == gate.controls[0]:
                                temp.add_gate(gate.name, targets=[i],
                                              controls=[i + 1])
                            else:
                                temp.add_gate(gate.name, targets=[i + 1],
                                              controls=[i])

                        elif (N + start - end - i - i == 2 and
                              (N - end + start + 1) % 2 == 1):
                            temp.add_gate("SWAP", targets=[i, i + 1])
                            if end == gate.controls[0]:
                                temp.add_gate(gate.name, targets=[i + 2],
                                              controls=[i + 1])
                            else:
                                temp.add_gate(gate.name, targets=[i + 1],
                                              controls=[i + 2])
                            temp.add_gate("SWAP", [i, i + 1])
                            i += 1

                        else:
                            temp.add_gate("SWAP", [i, i + 1])
                            temp.add_gate("SWAP",
                                          [N + start - end - i - 1,
                                           N + start - end - i])
                        i += 1

                    j = 0
                    for gate in temp.gates:
                        if (j < N - end - 2):
                            if gate.name in ["CNOT", "CSIGN"]:
                                qc_t.add_gate(gate.name, end + gate.targets[0],
                                              end + gate.controls[0])
                            else:
                                qc_t.add_gate(gate.name,
                                              [end + gate.targets[0],
                                               end + gate.targets[1]])
                        elif (j == N - end - 2):
                            if gate.name in ["CNOT", "CSIGN"]:
                                qc_t.add_gate(gate.name, end + gate.targets[0],
                                              (end + gate.controls[0]) % N)
                            else:
                                qc_t.add_gate(gate.name,
                                              [end + gate.targets[0],
                                               (end + gate.targets[1]) % N])
                        else:
                            if gate.name in ["CNOT", "CSIGN"]:
                                qc_t.add_gate(gate.name,
                                              (end + gate.targets[0]) % N,
                                              (end + gate.controls[0]) % N)
                            else:
                                qc_t.add_gate(gate.name,
                                              [(end + gate.targets[0]) % N,
                                               (end + gate.targets[1]) % N])
                        j = j + 1

                elif (end - start) == N - 1:
                    qc_t.add_gate(gate.name, gate.targets, gate.controls)

            elif gate.name in swap_gates:
                start = min([gate.targets[0], gate.targets[1]])
                end = max([gate.targets[0], gate.targets[1]])

                if (setup == "linear" or
                        (setup == "circular" and (end - start) <= N // 2)):
                    i = start
                    while i < end:
                        if (start + end - i - i == 1 and
                                (end - start + 1) % 2 == 0):
                            qc_t.add_gate(gate.name, [i, i + 1])
                        elif ((start + end - i - i) == 2 and
                              (end - start + 1) % 2 == 1):
                            qc_t.add_gate("SWAP", [i, i + 1])
                            qc_t.add_gate(gate.name, [i + 1, i + 2])
                            qc_t.add_gate("SWAP", [i, i + 1])
                            i += 1
                        else:
                            qc_t.add_gate("SWAP", [i, i + 1])
                            qc_t.add_gate("SWAP", [start + end - i - 1,
                                                   start + end - i])
                        i += 1

                else:
                    temp = QubitCircuit(N - end + start)
                    i = 0
                    while i < (N - end + start):

                        if (N + start - end - i - i == 1 and
                                (N - end + start + 1) % 2 == 0):
                            temp.add_gate(gate.name, [i, i + 1])

                        elif (N + start - end - i - i == 2 and
                              (N - end + start + 1) % 2 == 1):
                            temp.add_gate("SWAP", [i, i + 1])
                            temp.add_gate(gate.name, [i + 1, i + 2])
                            temp.add_gate("SWAP", [i, i + 1])
                            i += 1

                        else:
                            temp.add_gate("SWAP", [i, i + 1])
                            temp.add_gate("SWAP", [N + start - end - i - 1,
                                                   N + start - end - i])
                        i += 1

                    j = 0
                    for gate in temp.gates:
                        if(j < N - end - 2):
                            qc_t.add_gate(gate.name, [end + gate.targets[0],
                                                      end + gate.targets[1]])
                        elif(j == N - end - 2):
                            qc_t.add_gate(gate.name,
                                          [end + gate.targets[0],
                                           (end + gate.targets[1]) % N])
                        else:
                            qc_t.add_gate(gate.name,
                                          [(end + gate.targets[0]) % N,
                                           (end + gate.targets[1]) % N])
                        j = j + 1

            else:
                qc_t.add_gate(gate.name, gate.targets, gate.controls,
                              gate.arg_value, gate.arg_label)

        return qc_t

    def eliminate_auxillary_modes(self, U):
        return U

    def optimize_circuit(self, qc):
        """
        Take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.

        Parameters
        ----------
        qc: :class:`qutip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        Returns
        -------
        qc: :class:`qutip.QubitCircuit`
            The circuit representation with elementary gates
            that can be implemented in this model.
        """
        self.qc0 = qc
        self.qc1 = self.adjacent_gates(self.qc0)
        self.qc2 = self.qc1.resolve_gates(
            basis=["SQRTISWAP", "ISWAP", "RX", "RZ"])
        return self.qc2


class LinearSpinChain(SpinChain):
    """
    A processor based on the physical implementation of
    a linear spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, N, correct_global_phase=True,
                 sx=0.25, sz=1.0, sxsy=0.1, t1=None, t2=None):

        super(LinearSpinChain, self).__init__(
            N, correct_global_phase=correct_global_phase,
            sx=sx, sz=sz, sxsy=sxsy, t1=t1, t2=t2)
        self.set_up_params(sx=sx, sz=sz, sxsy=sxsy)
        self.set_up_ops(N)

    def set_up_ops(self, N):
        super(LinearSpinChain, self).set_up_ops(N)

    def set_up_params(self, sx, sz, sxsy):
        # Doc same as in the parent class
        super(LinearSpinChain, self).set_up_params(sx, sz)
        sxsy_para = 2 * np.pi * self.to_array(sxsy, self.N-1)
        self._params["sxsy"] = sxsy_para

    @property
    def sxsy_ops(self):
        return self.ctrls[2*self.N: 3*self.N-1]

    @property
    def sxsy_u(self):
        return self.coeffs[2*self.N: 3*self.N-1]

    def load_circuit(self, qc):
        return super(LinearSpinChain, self).load_circuit(qc, "linear")

    def get_operators_labels(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method``plot_pulses``.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        return ([[r"$\sigma_x^%d$" % n for n in range(self.N)],
                [r"$\sigma_z^%d$" % n for n in range(self.N)],
                [r"$\sigma_x^%d\sigma_x^{%d} + \sigma_y^%d\sigma_y^{%d}$"
                 % (n, n + 1, n, n + 1) for n in range(self.N - 1)],
                 ])

    def adjacent_gates(self, qc):
        return super(LinearSpinChain, self).adjacent_gates(qc, "linear")


class CircularSpinChain(SpinChain):
    """
    A processor based on the physical implementation of
    a circular spin chain qubits system.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.

    Parameters
    ----------
    N: int
        The number of qubits in the system.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.

    sx: int or list
        The delta for each of the qubits in the system.

    sz: int or list
        The epsilon for each of the qubits in the system.

    sxsy: int or list
        The interaction strength for each of the qubit pair in the system.

    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, N, correct_global_phase=True,
                 sx=0.25, sz=1.0, sxsy=0.1, t1=None, t2=None):
        if N <= 1:
            raise ValueError(
                "Circuit spin chain must have at least 2 qubits. "
                "The number of qubits is increased to 2.")
        super(CircularSpinChain, self).__init__(
            N, correct_global_phase=correct_global_phase,
            sx=sx, sz=sz, sxsy=sxsy, t1=t1, t2=t2)
        self.set_up_params(sx=sx, sz=sz, sxsy=sxsy)
        self.set_up_ops(N)

    def set_up_ops(self, N):
        super(CircularSpinChain, self).set_up_ops(N)
        operator = tensor([sigmax(), sigmax()]) + tensor([sigmay(), sigmay()])
        self.pulses.append(
            Pulse(operator, [N-1, 0], spline_kind=self.spline_kind))

    def set_up_params(self, sx, sz, sxsy):
        # Doc same as in the parent class
        super(CircularSpinChain, self).set_up_params(sx, sz)
        sxsy_para = 2 * np.pi * self.to_array(sxsy, self.N)
        self._params["sxsy"] = sxsy_para

    @property
    def sxsy_ops(self):
        return self.ctrls[2*self.N: 3*self.N]

    @property
    def sxsy_u(self):
        return self.coeffs[2*self.N: 3*self.N]

    def load_circuit(self, qc):
        return super(CircularSpinChain, self).load_circuit(qc, "circular")

    def get_operators_labels(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method``plot_pulses``.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        return ([[r"$\sigma_x^%d$" % n for n in range(self.N)],
                [r"$\sigma_z^%d$" % n for n in range(self.N)],
                [r"$\sigma_x^%d\sigma_x^{%d} + \sigma_y^%d\sigma_y^{%d}$"
                 % (n, (n + 1) % self.N, n, (n + 1) % self.N)
                 for n in range(self.N)]])

    def adjacent_gates(self, qc):
        return super(CircularSpinChain, self).adjacent_gates(qc, "circular")
