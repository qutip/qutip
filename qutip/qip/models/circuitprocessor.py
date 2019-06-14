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
import numbers

import numpy as np
import matplotlib.pyplot as plt

from qutip.qobj import Qobj
import qutip.control.pulseoptim as cpo
from qutip.operators import identity, sigmax, sigmaz, destroy
from qutip.qip.gates import expand_oper
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip import globalphase


class CircuitProcessor(object):
    """
    The base class for circuit processor, which is defined by the Hamiltonian available
    as dynamic generators. It can find the the corresponding driving pulses,
    either by analytical decomposition of by numerical method. 
    The processor can then
    calculate the state evolution under this defined dynamics

    Parameters
    ----------
    N : int
        The number of qubits in the system.
    T1 : list or float
        Characterize the decoherence of amplitude damping for
        each qubit.
    T2 : list of float
        Characterize the decoherence of dephase relaxation for
        each qubit.

    Attributes
    ----------
    tlist : array like
        A NumPy array specifies the time steps.
    amps : array like
        A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
        row corresponds to the control pulse sequence for
        one Hamiltonian.
    """
    def __init__(self, N, T1=None, T2=None):
        self.N = N
        self.tlist = np.empty(0)
        self.amps = np.empty((0,0))
        self.ctrls = []
        
        self.T1 = self._check_T_valid(T1, self.N)
        self.T2 = self._check_T_valid(T2, self.N)

    def _check_T_valid(self, T, N):
        if (isinstance(T, numbers.Real) and T>0) or T is None:
            return [T] * N
        elif isinstance(T, Iterable) and len(T)==N:
            if all([isinstance(t, numbers.Real) and t>0 for t in T]):
                return T
        else:
            raise ValueError("Invalid relaxation time T={}".format(T))

    def add_ctrl(self, ctrl, targets=None, expand_type=None):
        """
        Add a ctrl Hamiltonian to the processor

        Parameters
        ----------
        ctrl : Qobj
            A hermitian Qobj representation of the driving Hamiltonian
        targets : list of int
            The indices of qubits that are acted on.
        expand_type : string
            The tyoe of expansion
            None - only expand for the given target qubits
            "periodic" - the Hamiltonian is to be expanded for
                all cyclic permutation of target qubits
        """
        # Check validity of ctrl
        if not isinstance(ctrl, Qobj):
            raise TypeError("The Hamiltonian must be a qutip.Qobj.")
        if not ctrl.isherm:
            raise ValueError("The Hamiltonian must be Hermitian.")

        d = len(ctrl.dims[0])
        if targets is None:
            targets = list(range(d))

        if expand_type is None:
            if d == self.N:
                self.ctrls.append(ctrl)
            else:
                self.ctrls.append(expand_oper(ctrl, self.N, targets))
        elif expand_type == "periodic":
            for i in range(self.N):
                new_targets = np.mod(np.array(targets)+i, self.N)
                self.ctrls.append(
                    expand_oper(ctrl, self.N, new_targets))
        else:
            raise ValueError(
                "expand_type can only be None or 'periodic', "
                "not {}".format(expand_type))

    def remove_ctrl(self, indices):
        """
        Remove the ctrl Hamiltonian with given indices

        Parameters
        ----------
        indices : int or list of int
        """
        if not isinstance(indices, Iterable):
            indices = [indices]
        for ind in indices:
            if not isinstance(ind, numbers.Integral):
                raise TypeError("Index must in an integer")
            else:
                del self.ctrls[ind]

    def _is_time_amps_valid(self):
        amps_len = self.amps.shape[1]
        tlist_len = self.tlist.shape[0]
        if amps_len != tlist_len:
            raise ValueError(
                "tlist has length of {} while amps "
                "has {}".format(tlist_len, amps_len))

    def _is_ctrl_amps_valid(self):
        if self.amps.shape[0] != len(self.ctrls):
            raise ValueError(
                "The control amplitude matrix do not match the "
                "number of control Hamiltonians")

    def _is_amps_valid(self):
        self._is_time_amps_valid()
        self._is_ctrl_amps_valid()

    def save_amps(self, file_name, inctime=True):
        """
        Save a file with the current control amplitudes in each timeslot

        Parameters
        ----------
        file_name : string
            Name of the file
        inctime : boolean
            True if the time list in included in the first column
        """
        self._is_amps_valid()

        if inctime:
            shp = self.amps.T.shape
            data = np.empty([shp[0], shp[1] + 1], dtype=np.float)
            data[:, 0] = self.tlist
            data[:, 1:] = self.amps.T
        else:
            data = self.amps.T

        np.savetxt(file_name, data, delimiter='\t', fmt='%1.16f')

    def read_amps(self, file_name, inctime=True):
        """
        Read the pulse amplitude matrix save in a file by `save_amp`

        Parameters
        ----------
        file_name : string
            Name of the file
        inctime : boolean
            True if the time list in included in the first column
        """
        data = np.loadtxt(file_name, delimiter='\t')
        if not inctime:
            self.amps = data.T
        else:
            self.tlist = data[:, 0]
            self.amps = data[:, 1:].T
        try:
            self._is_amps_valid()
        except Exception as e:
            warnings.warn("{}".format(e))
        return self.tlist, self.amps

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
        raise NotImplementedError("Use the function in the sub-class")

    def load_circuit(self, qc):
        """
        Translates an abstract quantum circuit to its corresponding Hamiltonian
        for a specific model.

        Parameters
        ----------
        qc: QubitCircuit
            Takes the quantum circuit to be implemented.
        """
        raise NotImplementedError("Use the function in the sub-class")

    def get_ops_and_u(self):
        """
        Returns the Hamiltonian operators and corresponding values by stacking
        them together.
        """
        raise NotImplementedError("Use the function in the sub-class")

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

        for n in range(len(self.tlist)):
            H = sum([H_u[n, m] * H_ops[m] for m in range(len(H_ops))])
            U = (-1j * H * self.tlist[n]).expm()
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

        for n in range(len(self.tlist)):
            H = sum([H_u[n, m] * H_ops[m] for m in range(len(H_ops))])
            U = (-1j * H * self.tlist[n]).expm()
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

        t_tot = sum(self.tlist)
        n_t = int(np.ceil(t_tot / dt))
        n_ops = len(H_ops)

        t = np.linspace(0, t_tot, n_t)
        u = np.zeros((n_ops, n_t))

        t_start = 0
        for n in range(len(self.tlist)):

            t_idx_len = int(np.floor(self.tlist[n] / dt))

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
