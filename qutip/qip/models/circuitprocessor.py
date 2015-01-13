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
from qutip.qip.gates import globalphase


class CircuitProcessor(object):
    """
    Base class for representation of the physical implementation of a quantum
    program/algorithm on a specified qubit system.
    """

    def __init__(self, N, correct_global_phase):
        """
        Parameters
        ----------
        N: Integer
            The number of qubits in the system.

        correct_global_phase: Boolean
            Check if the global phases should be included in the final result.
        """
        self.N = N
        self.correct_global_phase = correct_global_phase

    def optimize_circuit(self, qc):
        """
        Function to take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.

        Parameters
        ----------
        qc: QubitCircuit
            Takes the quantum circuit to be implemented.

        Returns
        --------
        qc: QubitCircuit
            The optimal circuit representation.
        """
        raise NotImplemented("Use the function in the sub-class")

    def adjacent_gates(self, qc, setup):
        """
        Function to take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.

        Parameters
        ----------
        qc: QubitCircuit
            Takes the quantum circuit to be implemented.

        setup: String
            Takes the nature of the spin chain; linear or circular.

        Returns
        --------
        qc: QubitCircuit
            The resolved circuit representation.
        """
        raise NotImplemented("Use the function in the sub-class")

    def load_circuit(self, qc):
        """
        Translates an abstract quantum circuit to its corresponding Hamiltonian
        for a specific model.

        Parameters
        ----------
        qc: QubitCircuit
            Takes the quantum circuit to be implemented.
        """
        raise NotImplemented("Use the function in the sub-class")

    def get_ops_and_u(self):
        """
        Returns the Hamiltonian operators and corresponding values by stacking
        them together.
        """
        raise NotImplemented("Use the function in the sub-class")

    def get_ops_labels(self):
        """
        Returns the Hamiltonian operators and corresponding labels by stacking
        them together.
        """
        pass

    def eliminate_auxillary_modes(self, U):
        return U

    def run(self, qc=None):
        """
        Generates the propagator matrix by running the Hamiltonian for the
        appropriate time duration for the desired physical system.

        Parameters
        ----------
        qc: QubitCircuit
            Takes the quantum circuit to be implemented.

        Returns
        --------
        U_list: list
            The propagator matrix obtained from the physical implementation.
        """
        if qc:
            self.load_circuit(qc)
        U_list = []
        H_ops, H_u = self.get_ops_and_u()

        for n in range(len(self.T_list)):
            H = sum([H_u[n, m] * H_ops[m] for m in range(len(H_ops))])
            U = (-1j * H * self.T_list[n]).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        if self.correct_global_phase and self.global_phase != 0:
            U_list.append(globalphase(self.global_phase, N=self.N))

        return U_list

    def run_state(self, qc=None, states=None):
        """
        Generates the propagator matrix by running the Hamiltonian for the
        appropriate time duration for the desired physical system with the
        given initial state of the qubit register.

        Parameters
        ----------
        qc: QubitCircuit
            Takes the quantum circuit to be implemented.

        states: Qobj
            Initial state of the qubits in the register.

        Returns
        --------
        U_list: list
            The propagator matrix obtained from the physical implementation.
        """
        if states is None:
            raise NotImplementedError("Qubit state not defined.")
        if qc:
            self.load_circuit(qc)
        U_list = [states]
        H_ops, H_u = self.get_ops_and_u()

        for n in range(len(self.T_list)):
            H = sum([H_u[n, m] * H_ops[m] for m in range(len(H_ops))])
            U = (-1j * H * self.T_list[n]).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        if self.correct_global_phase and self.global_phase != 0:
            U_list.append(globalphase(self.global_phase, N=self.N))

        return U_list

    def pulse_matrix(self):
        """
        Generates the pulse matrix for the desired physical system.

        Returns
        --------
        t, u, labels:
            Returns the total time and label for every operation.
        """
        dt = 0.01
        H_ops, H_u = self.get_ops_and_u()

        t_tot = sum(self.T_list)
        n_t = int(np.ceil(t_tot / dt))
        n_ops = len(H_ops)

        t = np.linspace(0, t_tot, n_t)
        u = np.zeros((n_ops, n_t))

        t_start = 0
        for n in range(len(self.T_list)):

            t_idx_len = int(np.floor(self.T_list[n] / dt))

            mm = 0
            for m in range(len(H_ops)):
                u[mm, t_start:(t_start + t_idx_len)] = (np.ones(t_idx_len) *
                                                        H_u[n, m])
                mm += 1

            t_start += t_idx_len

        return t, u, self.get_ops_labels()

    def plot_pulses(self):
        """
        Maps the physical interaction between the circuit components for the
        desired physical system.

        Returns
        --------
        fig, ax: Figure
            Maps the physical interaction between the circuit components.
        """
        import matplotlib.pyplot as plt
        t, u, u_labels = self.pulse_matrix()
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for n, uu in enumerate(u):
            ax.plot(t, u[n], label=u_labels[n])

        ax.axis('tight')
        ax.set_ylim(-1.5 * 2 * np.pi, 1.5 * 2 * np.pi)
        ax.legend(loc='center left',
                  bbox_to_anchor=(1, 0.5), ncol=(1 + len(u) // 16))
        fig.tight_layout()

        return fig, ax
