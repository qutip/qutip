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
import inspect

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
    The base class for a circuit processor,
    which is defined by the available Hamiltonian of the system
    and the decoherence time for each qubit.
    For a given pulse amplitude matrix, the processor can then
    calculate the state evolution under the given control pulses,
    either analytically or numerically.
    In the subclass, further methods are defined so that
    it can be used to find the corresponding driving pulses
    of a quantum circuit.

    Parameters
    ----------
    N : int
        The number of qubits in the system.
    T1 : list or float
        Characterize the decoherence of amplitude damping for
        each qubit.
    T2 : list of float
        Characterize the decoherence of dephasing relaxation for
        each qubit.

    Attributes
    ----------
    ctrls : list of :class:`Qobj`
        The Hamiltonians driving the evolution.
    tlist : array like
        A NumPy array specifies the time steps.
    amps : array like
        A 2d NumPy array of the shape (len(ctrls), len(tlist)). Each
        row corresponds to the control pulse sequence for
        one Hamiltonian in `ctrls`.
    """
    def __init__(self, N, T1=None, T2=None):
        self.N = N
        self.tlist = None
        self.amps = None
        self._hams = []
        self.T1 = self._check_T_valid(T1, self.N)
        self.T2 = self._check_T_valid(T2, self.N)

    @property
    def hams(self):
        return self._hams

    @hams.setter
    def hams(self, ctrl_list):
        self._hams = []
        for ctrl in ctrl_list:
            self.add_ctrl(ctrl)

    def _check_T_valid(self, T, N):
        if (isinstance(T, numbers.Real) and T > 0) or T is None:
            return [T] * N
        elif isinstance(T, Iterable) and len(T) == N:
            if all([isinstance(t, numbers.Real) and t > 0 for t in T]):
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
                self._hams.append(ctrl)
            else:
                self._hams.append(expand_oper(ctrl, self.N, targets))
        elif expand_type == "periodic":
            for i in range(self.N):
                new_targets = np.mod(np.array(targets)+i, self.N)
                self._hams.append(
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
                del self._hams[ind]

    def _is_time_amps_valid(self):
        if self.tlist is None and self.amps is None:
            pass
        elif self.tlist is None or self.amps is None:
            raise ValueError("`tlist` or `amps` is not given.")
        else:
            amps_len = self.amps.shape[1]
            tlist_len = self.tlist.shape[0]
            if amps_len != tlist_len:
                raise ValueError(
                    "`tlist` has the length of {} while amps "
                    "has {}".format(tlist_len, amps_len))

    def _is_ctrl_amps_valid(self):
        if self.amps is None and len(self._hams) != 0:
            raise ValueError(
                "The control amplitude is None while "
                "the number of ctrls is {}".format(len(self._hams)))
        if self.amps is not None:
            if self.amps.shape[0] != len(self._hams):
                raise ValueError(
                    "The control amplitude matrix do not match the "
                    "number of ctrls  "
                    "#ctrls = {}  "
                    "#amps = {}".format(len(self._hams), len(self.amps)))

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

    def run_state(self, rho0, **kwargs):
        """
        Use mesolve to calculate the time of the state evolution
        and return the result. Other arguments of mesolve can be
        given as kwargs.

        Parameters
        ----------
        rho0 : Qobj
            Initial density matrix or state vector (ket).
        **kwargs
            Key word arguments for `mesolve`

        Returns
        -------
        evo_result : :class:`qutip.Result`
            An instance of the class :class:`qutip.Result`, which contains
            either an *array* `result.expect` of expectation values for
            the times specified by `tlist`, or an *array* `result.states`
            of state vectors or density matrices corresponding to
            the times in `tlist` [if `e_ops` is
            an empty list], or nothing if a callback function was
            given in place of
            operators for which to calculate the expectation values.
        """
        if "H" in kwargs or "args" in kwargs:
            raise ValueError(
                "`H` and `args` are already specified by the processor "
                "and can not be given as a key word argument")

        # if no control pulse specified (drift evolution with c_ops)
        if self.tlist is None:
            if "tlist" in kwargs:
                tlist = kwargs["tlist"]
                # If first t is 0, remove it to match the define of self.tlist
                if tlist[0] == 0:
                    self.tlist = tlist[1:]
                del kwargs["tlist"]
            else:
                raise ValueError(
                    "`tlist` has to be given as a key word argument "
                    "since it's not defined.")
        elif self.tlist is not None and "tlist" in kwargs:
            raise ValueError(
                "`tlist` is already specified by the processor, "
                "thus can not be given as a key word argument")
        else:
            self.tlist = self.tlist
        if not self._hams:
            self._hams.append(tensor([identity(2)] * self.N))
        if self.amps is None:  # only drift/identity and no amps given
            self.amps = np.ones(len(self.tlist)).reshape((1, len(self.tlist)))

        # check validity
        self._is_amps_valid()

        tlist = np.hstack([[0], self.tlist])
        amps = self.amps

        # contruct time-dependent Hamiltonian
        # TODO modefy the structure so that
        # tlist does not have to be equaldistant
        def get_amp_td_func(t, args, row_ind):
            """
            This is the func as it is implemented in
            `qutip.rhs_generate._td_wrap_array_str`
            """
            times = args['times']
            amps = args['amps'][row_ind]
            n_t = len(times)
            t_f = times[-1]
            if t >= t_f:
                return 0.0
            else:
                return amps[int(np.floor((n_t-1)*t/t_f))]
        H = []
        for op_ind in range(len(self._hams)):
            # row_ind=op_ind cannot be deleted
            # see Late Binding Closures for detail
            H.append(
                [self._hams[op_ind],
                    lambda t, args, row_ind=op_ind:
                    get_amp_td_func(t, args, row_ind)])

        # add collapse for T1 & T2 decay
        sys_c_ops = []
        for qu_ind in range(self.N):
            T1 = self.T1[qu_ind]
            T2 = self.T2[qu_ind]
            if T1 is not None:
                sys_c_ops.append(
                    expand_oper(
                        1/np.sqrt(T1) * destroy(2), self.N, qu_ind))
            if T2 is not None:
                # Keep the total dephasing ~ exp(-t/T2)
                if T1 is not None:
                    if 2*T1 < T2:
                        raise ValueError(
                            "T1={}, T2={} does not fulfill "
                            "2*T1>T2".format(T1, T2))
                    T2_eff = 1./(1./T2-1./2./T1)
                else:
                    T2_eff = T2
                sys_c_ops.append(
                    expand_oper(
                        1/np.sqrt(2*T2_eff) * sigmaz(), self.N, qu_ind))
        if "c_ops" in kwargs:
            kwargs["c_ops"] += sys_c_ops
        else:
            kwargs["c_ops"] = sys_c_ops
        evo_result = mesolve(
            H=H, rho0=rho0, tlist=tlist,
            args={'times': tlist, 'amps': amps}, **kwargs)

        return evo_result

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

    def eliminate_auxillary_modes(self, U):
        return U

    def plot_pulses(self, amps=None, tlist=None, **kwargs):
        """
        Plot the pulse amplitude

        Parameters
        ----------
        **kwargs
            Key word arguments for figure

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `Figure` object for the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes for the plot.
        """
        if amps is None:
            amps = self.amps
        if tlist is None:
            tlist = self.tlist

        fig_keys = inspect.getfullargspec(plt.subplots)[0]
        fig, ax = plt.subplots(
            1, 1, **{key: kwargs[key] for key in kwargs if key in fig_keys})
        ax.set_ylabel("Control pulse amplitude")
        ax.set_xlabel("Time")
        amps = np.hstack([amps, amps[:, -1:]])
        plot_keys = inspect.getfullargspec(ax.step)[0]
        for i in range(amps.shape[0]):
            ax.step(np.hstack([[0], tlist]), amps[i], where='post',
                    **{key: kwargs[key] for key in kwargs if key in plot_keys})
        fig.tight_layout()
        return fig, ax


class ModelProcessor(CircuitProcessor):
    def __init__(self, N, correct_global_phase=True, T1=None, T2=None):
        super(ModelProcessor, self).__init__(N, T1=T1, T2=T2)
        self.correct_global_phase = correct_global_phase
        self.global_phase = 0.

    def get_ops_labels(self):
        """
        Returns the Hamiltonian operators and corresponding labels by stacking
        them together.
        """
        raise NotImplementedError("Use the function in the sub-class")

    def run(self, qc=None):
        """
        Generates the propagator matrix by running the Hamiltonian for the
        appropriate time duration for the desired physical system
        analytically with U = exp(-iHt).

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
        tlist = np.hstack([[0.], self.tlist])
        for n in range(len(tlist)-1):
            H = sum([self.amps[m, n] * self._hams[m] 
                    for m in range(len(self._hams))])
            dt = tlist[n+1] - tlist[n]
            U = (-1j * H * dt).expm()
            U = self.eliminate_auxillary_modes(U)
            U_list.append(U)

        if self.correct_global_phase and self.global_phase != 0:
            U_list.append(globalphase(self.global_phase, N=self.N))

        return U_list

    def run_state(self, qc=None, rho0=None, states=None):
        """
        Generates the propagator matrix by running the Hamiltonian for the
        appropriate time duration for the desired physical system with the
        given initial state of the qubit register.

        Parameters
        ----------
        qc: :class:`qutip.qip.QubitCircuit`
            Takes the quantum circuit to be implemented.

        states: :class:`qutip.Qobj`
            Initial state of the qubits in the register.

        Returns
        --------
        U_list: list
            The propagator matrix obtained from the physical implementation.
        """
        # TODO choice for numerical/anlytical
        # TODO Here this variable was called states
        # in the old circuitprocessor,
        # but there is actaully only one state,
        # change to state? or rho0 like in the solver?
        if rho0 is None or states is None:
            raise NotImplementedError("Qubit state not defined.")
        elif rho0 is None:
            rho0 = states  # just to keep the old prameters `states`
        if qc:
            self.load_circuit(qc)

        U_list = [states]
        tlist = np.hstack([[0.], self.tlist])
        for n in range(len(tlist)-1):
            H = sum([self.amps[m, n] * self._hams[m] 
                    for m in range(len(self._hams))])
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

        diff_tlist = self.tlist - np.hstack([[0], self.tlist[:-1]])
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
