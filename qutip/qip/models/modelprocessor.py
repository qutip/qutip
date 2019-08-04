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
        The noise object, they will be processed when creating the
        noisy :class:`qutip.QobjEvo` or run the simulation.

    dims: list
        The dimension of each component system.
        If not given, it will be a
        qutbis system of dim=[2,2,2,...,2]

    spline_kind: str
        Type of the coefficient interpolation. Default is "step_func".
        Note that they have different requirement for the length of `coeffs`.

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
        coeffs: array like
            The transposed pulse matrix
        """
        return (self.ctrls, self.coeffs.T)

    def run(self, qc=None):
        """
        Calculate the propagator the evolution by matrix exponentiation.
        This method won't include the noise.

        Parameters
        ----------
        qc: :class:`qutip.qip.QubitCircuit`, optional
            Takes the quantum circuit to be implemented. If not given, use
            the quantum circuit saved in the processor by `load_circuit`.

        Returns
        --------
        U_list: list
            The propagator matrix obtained from the physical implementation.
        """
        if qc:
            self.load_circuit(qc)

        U_list = []
        tlist = self.tlist
        for n in range(len(tlist)-1):
            H = sum([self.coeffs[m, n] * self.ctrls[m]
                    for m in range(len(self.ctrls))])
            dt = tlist[n+1] - tlist[n]
            U = (-1j * H * dt).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        if self.correct_global_phase and self.global_phase != 0:
            U_list.append(globalphase(self.global_phase, N=self.N))

        return U_list

    def run_state(self, qc=None, rho0=None, states=None,
                  numerical=False, **kwargs):
        """
        Simulate the state evolution under the given `qutip.QubitCircuit`
        If `numerical` is False, it will calculate the propagator
        with matrix exponentiation and applies it to the initial state.
        This won't includes the noise.
        If `numerical` is True, it will use mesolve to
        calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as kwargs.

        Parameters
        ----------
        qc: :class:`qutip.qip.QubitCircuit`, optional
            Takes the quantum circuit to be implemented. If not given, use
            the quantum circuit saved in the processor by `load_circuit`.

        rho0: :class:`qutip.Qobj`, optional
            Initial state of the qubits in the register.
            If qutip solver is used to calculate the evolution numerically.
            Noise will only be considered if this is true.
    
        kwargs, optional
            Key word arguments for the `qutip.mesolve`.
    
        states: :class:`qutip.Qobj`, optional
            Old API, deprecated to be consistent with qutip solver.

        Returns
        --------
        evo_result: :class:`qutip.Result`
            If the numerical method is used, the result of the solver will
            be returned.
            An instance of the class :class:`qutip.Result`, which contains
            either an *array* `result.expect` of expectation values for
            the times specified by `tlist`, or an *array* `result.states`
            of state vectors or density matrices corresponding to
            the times in `tlist` [if `e_ops` is
            an empty list], or nothing if a callback function was
            given in place of
            operators for which to calculate the expectation values.

        U_list: list
            If the analytical method is used, the propagator is returned.
        """
        if rho0 is None and states is None:
            raise ValueError("Qubit state not defined.")
        elif rho0 is None:
            # just to keep the old prameters `states`, it is replaced by rho0
            rho0 = states
        if qc:
            self.load_circuit(qc)
            # TODO It noise exists, give warning
        if numerical:
            return super(ModelProcessor, self).run_state(rho0=rho0, **kwargs)

        U_list = [rho0]
        tlist = self.tlist
        for n in range(len(tlist)-1):
            H = sum([self.coeffs[m, n] * self.ctrls[m]
                    for m in range(len(self.ctrls))])
            dt = tlist[n+1] - tlist[n]
            U = (-1j * H * dt).expm()
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
        tlist: array like
            A NumPy array specifies the time of each coefficients

        coeffs: array like
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
