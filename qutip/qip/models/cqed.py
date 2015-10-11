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
import numpy as np
import warnings
from qutip import tensor, identity, destroy, sigmax, sigmaz, basis
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.qip.models.circuitprocessor import CircuitProcessor


class DispersivecQED(CircuitProcessor):
    """
    Representation of the physical implementation of a quantum
    program/algorithm on a dispersive cavity-QED system.
    """

    def __init__(self, N, correct_global_phase=True, Nres=None, deltamax=None,
                 epsmax=None, w0=None, wq=None, eps=None, delta=None, g=None):
        """
        Parameters
        ----------
        Nres: Integer
            The number of energy levels in the resonator.

        deltamax: Integer/List
            The sigma-x coefficient for each of the qubits in the system.

        epsmax: Integer/List
            The sigma-z coefficient for each of the qubits in the system.

        wo: Integer
            The base frequency of the resonator.

        wq: Integer/List
            The frequency of the qubits.

        eps: Integer/List
            The epsilon for each of the qubits in the system.

        delta: Integer/List
            The epsilon for each of the qubits in the system.

        g: Integer/List
            The interaction strength for each of the qubit with the resonator.
        """

        super(DispersivecQED, self).__init__(N, correct_global_phase)

        # user definable
        if Nres is None:
            self.Nres = 10
        else:
            self.Nres = Nres

        if deltamax is None:
            self.sx_coeff = np.array([1.0 * 2 * np.pi] * N)
        elif not isinstance(deltamax, list):
            self.sx_coeff = np.array([deltamax * 2 * np.pi] * N)
        else:
            self.sx_coeff = np.array(deltamax)

        if epsmax is None:
            self.sz_coeff = np.array([9.5 * 2 * np.pi] * N)
        elif not isinstance(epsmax, list):
            self.sz_coeff = np.array([epsmax * 2 * np.pi] * N)
        else:
            self.sz_coeff = np.array(epsmax)

        if w0 is None:
            self.w0 = 10 * 2 * np.pi
        else:
            self.w0 = w0

        if eps is None:
            self.eps = np.array([9.5 * 2 * np.pi] * N)
        elif not isinstance(eps, list):
            self.eps = np.array([eps * 2 * np.pi] * N)
        else:
            self.eps = np.array(eps)

        if delta is None:
            self.delta = np.array([0.0 * 2 * np.pi] * N)
        elif not isinstance(delta, list):
            self.delta = np.array([delta * 2 * np.pi] * N)
        else:
            self.delta = np.array(delta)

        if g is None:
            self.g = np.array([0.01 * 2 * np.pi] * N)
        elif not isinstance(g, list):
            self.g = np.array([g * 2 * np.pi] * N)
        else:
            self.g = np.array(g)

        if wq is not None:
            if not isinstance(wq, list):
                self.wq = np.array([wq] * N)
            else:
                self.wq = np.array(wq)

        if wq is None:
            if eps is None:
                self.eps = np.array([9.5 * 2 * np.pi] * N)
            elif not isinstance(eps, list):
                self.eps = np.array([eps] * N)
            else:
                self.eps = np.array(eps)

            if delta is None:
                self.delta = np.array([0.0 * 2 * np.pi] * N)
            elif not isinstance(delta, list):
                self.delta = np.array([delta] * N)
            else:
                self.delta = np.array(delta)

        # computed
        self.wq = np.sqrt(self.eps ** 2 + self.delta ** 2)
        self.Delta = self.wq - self.w0

        # rwa/dispersive regime tests
        if any(self.g / (self.w0 - self.wq) > 0.05):
            warnings.warn("Not in the dispersive regime")

        if any((self.w0 - self.wq) / (self.w0 + self.wq) > 0.05):
            warnings.warn(
                "The rotating-wave approximation might not be valid.")

        self.sx_ops = [tensor([identity(self.Nres)] +
                              [sigmax() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]
        self.sz_ops = [tensor([identity(self.Nres)] +
                              [sigmaz() if m == n else identity(2)
                               for n in range(N)])
                       for m in range(N)]

        self.a = tensor([destroy(self.Nres)] + [identity(2) for n in range(N)])

        self.cavityqubit_ops = []
        for n in range(N):
            sm = tensor([identity(self.Nres)] +
                        [destroy(2) if m == n else identity(2)
                         for m in range(N)])
            self.cavityqubit_ops.append(self.a.dag() * sm + self.a * sm.dag())

        self.psi_proj = tensor([basis(self.Nres, 0)] +
                               [identity(2) for n in range(N)])

    def get_ops_and_u(self):
        H0 = self.a.dag() * self.a
        return ([H0] + self.sx_ops + self.sz_ops + self.cavityqubit_ops,
                np.hstack((self.w0 * np.zeros((self.sx_u.shape[0], 1)),
                          self.sx_u, self.sz_u, self.g_u)))

    def get_ops_labels(self):
        return ([r"$a^\dagger a$"] +
                [r"$\sigma_x^%d$" % n for n in range(self.N)] +
                [r"$\sigma_z^%d$" % n for n in range(self.N)] +
                [r"$g_{%d}$" % (n) for n in range(self.N)])

    def optimize_circuit(self, qc):
        self.qc0 = qc
        self.qc1 = self.qc0.resolve_gates(basis=["ISWAP", "RX", "RZ"])
        self.qc2 = self.dispersive_gate_correction(self.qc1)

        return self.qc2

    def eliminate_auxillary_modes(self, U):
        return self.psi_proj.dag() * U * self.psi_proj

    def dispersive_gate_correction(self, qc1, rwa=True):
        """
        Method to resolve ISWAP and SQRTISWAP gates in a cQED system by adding
        single qubit gates to get the correct output matrix.

        Parameters
        ----------
        qc: Qobj
            The circular spin chain circuit to be resolved

        rwa: Boolean
            Specify if RWA is used or not.

        Returns
        ----------
        qc: QubitCircuit
            Returns QubitCircuit of resolved gates for the qubit circuit in the
            desired basis.
        """
        qc = QubitCircuit(qc1.N, qc1.reverse_states)

        for gate in qc1.gates:
            qc.gates.append(gate)
            if rwa:
                if gate.name == "SQRTISWAP":
                    qc.gates.append(Gate("RZ", [gate.targets[0]], None,
                                         arg_value=-np.pi / 4,
                                         arg_label=r"-\pi/4"))
                    qc.gates.append(Gate("RZ", [gate.targets[1]], None,
                                         arg_value=-np.pi / 4,
                                         arg_label=r"-\pi/4"))
                    qc.gates.append(Gate("GLOBALPHASE", None, None,
                                         arg_value=-np.pi / 4,
                                         arg_label=r"-\pi/4"))
                elif gate.name == "ISWAP":
                    qc.gates.append(Gate("RZ", [gate.targets[0]], None,
                                         arg_value=-np.pi / 2,
                                         arg_label=r"-\pi/2"))
                    qc.gates.append(Gate("RZ", [gate.targets[1]], None,
                                         arg_value=-np.pi / 2,
                                         arg_label=r"-\pi/2"))
                    qc.gates.append(Gate("GLOBALPHASE", None, None,
                                         arg_value=-np.pi / 2,
                                         arg_label=r"-\pi/2"))

        return qc

    def load_circuit(self, qc):

        gates = self.optimize_circuit(qc).gates

        self.global_phase = 0
        self.sx_u = np.zeros((len(gates), len(self.sx_ops)))
        self.sz_u = np.zeros((len(gates), len(self.sz_ops)))
        self.g_u = np.zeros((len(gates), len(self.cavityqubit_ops)))
        self.T_list = []

        n = 0
        for gate in gates:

            if gate.name == "ISWAP":
                t0, t1 = gate.targets[0], gate.targets[1]
                self.sz_u[n, t0] = self.wq[t0] - self.w0
                self.sz_u[n, t1] = self.wq[t1] - self.w0
                self.g_u[n, t0] = self.g[t0]
                self.g_u[n, t1] = self.g[t1]

                J = self.g[t0] * self.g[t1] * (1 / self.Delta[t0] +
                                               1 / self.Delta[t1]) / 2
                T = (4 * np.pi / abs(J)) / 4
                self.T_list.append(T)
                n += 1

            elif gate.name == "SQRTISWAP":
                t0, t1 = gate.targets[0], gate.targets[1]
                self.sz_u[n, t0] = self.wq[t0] - self.w0
                self.sz_u[n, t1] = self.wq[t1] - self.w0
                self.g_u[n, t0] = self.g[t0]
                self.g_u[n, t1] = self.g[t1]

                J = self.g[t0] * self.g[t1] * (1 / self.Delta[t0] +
                                               1 / self.Delta[t1]) / 2
                T = (4 * np.pi / abs(J)) / 8
                self.T_list.append(T)
                n += 1

            elif gate.name == "RZ":
                g = self.sz_coeff[gate.targets[0]]
                self.sz_u[n, gate.targets[0]] = np.sign(gate.arg_value) * g
                T = abs(gate.arg_value) / (2 * g)
                self.T_list.append(T)
                n += 1

            elif gate.name == "RX":
                g = self.sx_coeff[gate.targets[0]]
                self.sx_u[n, gate.targets[0]] = np.sign(gate.arg_value) * g
                T = abs(gate.arg_value) / (2 * g)
                self.T_list.append(T)
                n += 1

            elif gate.name == "GLOBALPHASE":
                self.global_phase += gate.arg_value

            else:
                raise ValueError("Unsupported gate %s" % gate.name)
