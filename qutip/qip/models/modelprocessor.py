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
import numbers

import numpy as np

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.qip.gates import globalphase
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.circuitprocessor import CircuitProcessor


__all__ = ['ModelProcessor', 'GateDecomposer']


class ModelProcessor(CircuitProcessor):
    """
    The base class for a circuit processor based on physical hardwares,
    e.g cavityQED, spinchain.
    The available Hamiltonian of the system is predefined.
    The processor can simulate the evolution under the given
    control pulses either numerically or analytically.
    It cannot be used alone, please refer to the sub-classes.

    Parameters
    ----------
    N: int
        The number of component systems.

    correct_global_phase: boolean, optional
        If true, the analytical solution will track the global phase. It
        has no effect on the numerical solution.

    T1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    T2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.

    Attributes
    ----------
    N: int
        The number of component system

    ctrls: list
        A list of the control Hamiltonians driving the evolution.

    tlist: array-like
        A NumPy array specifies the time of each coefficient.

    coeffs: array-like
        A 2d NumPy array of the shape, the length is dependent on the
        spline type

    T1: list
        Characterize the decoherence of amplitude damping for
        each qubit.

    T2: list
        Characterize the decoherence of dephasing for
        each qubit.

    noise: :class:`qutip.qip.CircuitNoise`, optional
        A list of noise objects. They will be processed when creating the
        noisy :class:`qutip.QobjEvo` from the processor or run the simulation.

    dims: list
        The dimension of each component system.
        If not given, it will be a
        qutbis system of dim=[2,2,2,...,2]

    spline_kind: str
        Type of the coefficient interpolation. Default is "step_func".
        Note that they have different requirements for the shape of
        :attr:`qutip.qip.circuitprocessor.coeffs`.

    paras: dict
        A Python dictionary contains the name and the value of the parameters
        in the physical realization, such as laser frequency, detuning etc.

    correct_global_phase: float
        Save the global phase, the analytical solution
        will track the global phase.
        It has no effect on the numerical solution.
    """
    def __init__(self, N, correct_global_phase=True, T1=None, T2=None):
        super(ModelProcessor, self).__init__(N, T1=T1, T2=T2)
        self.correct_global_phase = correct_global_phase
        self.global_phase = 0.
        self._paras = {}

    def _para_list(self, para, N):
        """
        Transfer a parameter to list form and multiplied by 2*pi.
        """
        if isinstance(para, numbers.Real):
            return [para * 2 * np.pi] * N
        elif isinstance(para, Iterable):
            return [c * 2 * np.pi for c in para]

    def set_up_paras(self):
        """
        Save the parameters in the attribute `paras` and check the validity.
        (Defined in subclasses)

        Note
        ----
        All parameters will be multiplied by 2*pi for simplicity
        """
        raise NotImplementedError("Parameters should be defined in subclass.")

    @property
    def paras(self):
        return self._paras

    @paras.setter
    def paras(self, par):
        self.set_up_paras(**par)

    def get_ops_and_u(self):
        """
        Returns the Hamiltonian operators and corresponding values by stacking
        them together.

        Returns
        -------
        ctrls: list
            The list of Hamiltonians
        coeffs: array-like
            The transposed pulse matrix
        """
        return (self.ctrls, self.coeffs.T)

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

        diff_tlist = self.tlist[1:] - self.tlist[:-1]
        t_tot = sum(diff_tlist)
        n_t = int(np.ceil(t_tot / dt))
        n_ops = len(H_ops)

        t = np.linspace(0, t_tot, n_t)
        u = np.zeros((n_ops, n_t))

        t_start = 0
        for n in range(len(diff_tlist)):

            t_idx_len = int(np.floor(diff_tlist[n] / dt))

            mm = 0
            for m in range(len(H_ops)):
                u[mm, t_start:(t_start + t_idx_len)] = (np.ones(t_idx_len) *
                                                        H_u[n, m])
                mm += 1

            t_start += t_idx_len

        return t, u, self.get_ops_labels()

    def plot_pulses(self, title=None, noisy=None, figsize=(12, 6), dpi=None):
        """
        Maps the physical interaction between the circuit components for the
        desired physical system.

        Returns
        --------
        fig, ax: Figure
            Maps the physical interaction between the circuit components.
        """
        if noisy is not None:
            return super(ModelProcessor, self).plot_pulses(
                title=title, noisy=noisy)
        import matplotlib.pyplot as plt
        t, u, u_labels = self.pulse_matrix()
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        for n, uu in enumerate(u):
            ax.plot(t, u[n], label=u_labels[n])

        ax.axis('tight')
        ax.set_ylim(-1.5 * 2 * np.pi, 1.5 * 2 * np.pi)
        ax.legend(loc='center left',
                  bbox_to_anchor=(1, 0.5), ncol=(1 + len(u) // 16))
        ax.set_ylabel("Control pulse amplitude")
        ax.set_xlabel("Time")
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax


class GateDecomposer(object):
    """
    Decompose a :class:`qutip.QubitCircuit` into
    the pulse sequence for the processor.

    Parameters
    ----------
    N: int
        The number of the component systems.

    paras: dict
        A Python dictionary contains the name and the value of the parameters
        of the physical realization, such as laser frequency,detuning etc.

    num_ops: int
        Number of control Hamiltonians in the processor.

    Attributes
    ----------
    N: int
        The number of the component systems.

    paras: dict
        A Python dictionary contains the name and the value of the parameters
        of the physical realization, such as laser frequency,detuning etc.

    num_ops: int
        Number of control Hamiltonians in the processor.

    gate_decs: dict
        The Python dictionary in the form {gate_name: decompose_function}.
    """
    def __init__(self, N, paras, num_ops):
        self.gate_decs = {}
        self.N = N
        self.paras = paras
        self.num_ops = num_ops

    def decompose(self, gates):
        """
        Decompose the the elementary gates
        into control pulse sequence.

        Parameters
        ----------
        gates: list
            A list of elementary gates that can be implemented in this
            model. The gate names have to be in `gate_decs`.

        Returns
        -------
        tlist: array-like
            A NumPy array specifies the time of each coefficients

        coeffs: array-like
            A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
            row corresponds to the control pulse sequence for
            one Hamiltonian.

        global_phase: bool
            Recorded change of global phase.
        """
        # TODO further enhancement can be made here, e.g. merge the gate
        # acting on different qubits to make the pulse sequence shorter.
        self.dt_list = []
        self.coeff_list = []
        for gate in gates:
            if gate.name not in self.gate_decs:
                raise ValueError("Unsupported gate %s" % gate.name)
            self.gate_decs[gate.name](gate)
        coeffs = np.vstack(self.coeff_list).T

        tlist = np.empty(len(self.dt_list))
        t = 0
        for i in range(len(self.dt_list)):
            t += self.dt_list[i]
            tlist[i] = t
        return np.hstack([[0], tlist]), coeffs
