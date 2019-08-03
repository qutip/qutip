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
from functools import reduce
import numbers
from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline

from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.operators import identity
from qutip.qip.gates import expand_operator, globalphase
from qutip.tensor import tensor
from qutip.mesolve import mesolve
from qutip.qip.circuit import QubitCircuit
from qutip.qip.models.circuitnoise import (
    CircuitNoise, RelaxationNoise, DecoherenceNoise,
    ControlAmpNoise, WhiteNoise, UserNoise)


__all__ = ['CircuitProcessor', 'ModelProcessor', 'GateDecomposer']


class CircuitProcessor(object):
    """
    The base class for a circuit processor, which is
    the simulator of quantum device using `qutip.mesolve`.
    It is defined by the available driving Hamiltonian and
    the decoherence time for each component systems.
    The processor can simulate the evolution under the given
    control pulses.

    Parameters
    ----------
    N: int
        The number of component systems

    T1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    T2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit.

    noise: :class:`qutip.qip.CircuitNoise`, optional
        The noise object, they will be processed when creating the
        noisy :class:`qutip.QobjEvo` or run the simulation.

    dims: list of int, optional
        The dimension of each component system.
        If not given, it will be a
        qutbis system of dim=[2,2,2,...,2]

    spline_kind: str, optional
        Type of the coefficient interpolation. Default is `step_func`
        Note that they have different requirement for the length of `coeffs`.

        -step_func: 
        The coefficient will be treated as a step function.
        E.g. tlist=[0,1,2] and coeffs=[3,2], means that the coefficient
        is 3 in t=[0,1) and 2 in t=[2,3). It requires
        len(coeffs)=len(tlist)-1 or len(coeffs)=len(tlist), but
        in the second case the last element has no effect.

        -cubic: Use cubic interpolation for the coefficient. It requires
        len(coeffs)=len(tlist)

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
    """
    def __init__(self, N, T1=None, T2=None, noise=None,
                 dims=None, spline_kind="step_func"):
        self.N = N
        self.tlist = None
        self.coeffs = None
        self.ctrls = []
        self.T1 = T1
        self.T2 = T2
        if noise is None:
            self.noise = []
        else:
            self.noise = noise
        if dims is None:
            self.dims = [2] * N
        else:
            self.dims = dims
        self.spline_kind = spline_kind

    def add_ctrl(self, ctrl, targets=None, cyclic_permutation=False):
        """
        Add a control Hamiltonian to the processor

        Parameters
        ----------
        ctrl: :class:`qutip.Qobj`
            The control Hamiltonian to be added

        targets: list or int, optional
            The indices of qubits that are acted on

        cyclic_permutation: boolean, optional
            If true, the Hamiltonian will be expanded for
                all cyclic permutation of target qubits
        """
        # Check validity of ctrl
        if not isinstance(ctrl, Qobj):
            raise TypeError("The control Hamiltonian must be a qutip.Qobj.")
        if not ctrl.isherm:
            raise ValueError("The control Hamiltonian must be Hermitian.")

        num_qubits = len(ctrl.dims[0])
        if targets is None:
            targets = list(range(num_qubits))

        if cyclic_permutation:
            self.ctrls += expand_operator(
                ctrl, self.N, targets, self.dims, cyclic_permutation=True)
        else:
            self.ctrls.append(
                expand_operator(ctrl, self.N, targets, self.dims))

    def remove_ctrl(self, indices):
        """
        Remove the control Hamiltonian with given indices.

        Parameters
        ----------
        indices: int or list of int
            The indices of the control Hamiltonians to be removed
        """
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices.sort(reverse=True)
        for ind in indices:
            del self.ctrls[ind]

    def _is_time_coeff_valid(self):
        """
        Check if the len(tlist) and coeffs.shape[1] are valid.

        Returns: boolean
            If they are valid or not
        """
        if self.coeffs is None:
            return True  # evolution with identity
        if self.tlist is None:
            raise ValueError("`tlist` is not given.")
        coeff_len = self.coeffs.shape[1]
        tlist_len = len(self.tlist)
        if self.spline_kind == "step_func":
            if coeff_len == tlist_len-1 or coeff_len == tlist_len:
                return True
            else:
                raise ValueError(
                    "The lenght of tlist and coeffs is not valid. "
                    "It's either len(tlist)=len(coeffs) or "
                    "len(tlist)-1=len(coeffs) for coefficients "
                    "as step function")
        if self.spline_kind == "cubic" and coeff_len == tlist_len:
            return True
        else:
            raise ValueError(
                "The lenght of tlist and coeffs is not valid. "
                "It sould be either len(tlist)=len(coeffs)")

    def _is_ctrl_coeff_valid(self):
        """
        Check if the number of control Hamiltonians
        and coeffs.shape[0] are the same.

        Returns: boolean
            If they are valid or not
        """
        if self.coeffs is None and len(self.ctrls) != 0:
            raise ValueError(
                "The control amplitude is None while "
                "the number of ctrls is {}.".format(len(self.ctrls)))
        if self.coeffs is not None:
            if self.coeffs.shape[0] != len(self.ctrls):
                raise ValueError(
                    "The control amplitude matrix do not match the "
                    "number of ctrls  "
                    "#ctrls = {}  "
                    "#coeffs = {}.".format(len(self.ctrls), len(self.coeffs)))
        return True

    def _is_coeff_valid(self):
        """
        Check if the attribute are in the corret shape.

        Returns: boolean
            If they are valid or not
        """
        return (self._is_time_coeff_valid() and
                self._is_ctrl_coeff_valid())

    def save_coeff(self, file_name, inctime=True):
        """
        Save a file with the current control amplitudes in each timeslot

        Parameters
        ----------
        file_name: string
            Name of the file.

        inctime: boolean, optional
            True if the time list in included in the first column.
        """
        self._is_coeff_valid()

        if inctime:
            shp = self.coeffs.T.shape
            data = np.empty([shp[0], shp[1] + 1], dtype=np.float)
            data[:, 0] = self.tlist[1:]
            data[:, 1:] = self.coeffs.T
        else:
            data = self.coeffs.T

        np.savetxt(file_name, data, delimiter='\t', fmt='%1.16f')

    def read_coeff(self, file_name, inctime=True):
        """
        Read the pulse amplitude matrix and time list
        saved in the file by `save_amp`

        Parameters
        ----------
        file_name: string
            Name of the file.

        inctime: boolean, optional
            True if the time list in included in the first column.

        Returns
        -------
        tlist: array like
            The time list read from the file.

        coeffs: array like
            The pulse matrix read from the file.
        """
        data = np.loadtxt(file_name, delimiter='\t')
        if not inctime:
            self.coeffs = data.T
        else:
            self.tlist = np.hstack([[0], data[:, 0]])
            self.coeffs = data[:, 1:].T
        return self.tlist, self.coeffs

    def get_unitary_qobjevo(self, args=None):
        """
        Create a :class:`qutip.QobjEvo` that can be given to
        the open system solver.

        Parameters
        ----------
        args: dict, optional
            Arguments for :class:`qutip.QobjEvo`

        Returns
        -------
        unitary_qobjevo: :class:`qutip.QobjEvo`
            The :class:`qutip.QobjEvo` representation of the unitary evolution.
        """
        # TODO add test
        # check validity
        self._is_coeff_valid()

        if args is None:
            args = {}
        else:
            args = args
        # set step function
        if self.coeffs is None:
            coeffs = np.empty((0, 0))
        elif self.spline_kind == "step_func":
            args.update({"_step_func_coeff": True})
            if self.coeffs.shape[1] == len(self.tlist) - 1:
                coeffs = np.hstack([self.coeffs, self.coeffs[:, -1:]])
            else:
                # TODO add
                coeffs = self.coeffs
        elif self.spline_kind == "cubic":
            # TODO add test
            args.update({"_step_func_coeff": False})
        else:
            raise ValueError("Wrong spline_kind.")

        H_list = []
        for op_ind in range(len(self.ctrls)):
            H_list.append(
                [self.ctrls[op_ind], coeffs[op_ind]])
        if not H_list:
            return _dummy_qobjevo(self.dims, tlist=self.tlist, args=args)
        else:
            return QobjEvo(H_list, tlist=self.tlist, args=args)

    def get_noisy_qobjevo(self, args=None):
        """
        Create a :class:`qutip.QobjEvo` with noise that can be given to
        the open system solver.

        Parameters
        ----------
        args: dict, optional
            Arguments for :class:`qutip.QobjEvo`

        Returns
        -------
        noisy_qobjevo: :class:`qutip.QobjEvo`
            The :class:`qutip.QobjEvo` representation of the noisy evolution.
        """
        # TODO add test for args
        unitary_qobjevo = self.get_unitary_qobjevo(args=args)
        ham_noise, c_ops = self.process_noise(unitary_qobjevo)
        noisy_qobjevo = self._compatible_coeff([unitary_qobjevo, ham_noise])
        return noisy_qobjevo, c_ops

    def get_noisy_coeff(self):
        """
        Create the array-like coefficients including the noise.

        Returns
        -------
        coeff_list: list
            A list of coefficient for each control Hamiltonian.

        tlist: np.array
            Time array for the coefficient.
        """
        # TODO add test
        noisy_qobjevo, c_ops = self.get_noisy_qobjevo()
        coeff_list = []
        H_list = noisy_qobjevo.to_list()
        for H in H_list:
            if isinstance(H, list):
                coeff_list.append(H[1])
        return coeff_list, noisy_qobjevo.tlist

    def run_state(self, rho0, **kwargs):
        """
        Use `qutip.mesolve` to calculate the time of the state evolution
        and return the result. Other arguments of the solver can be
        given as keyword arguments.

        Parameters
        ----------
        rho0: Qobj
            Initial density matrix or state vector (ket).

        kwargs
            Keyword arguments for the qutip solver.
            (currently `qutip.mesolve`)

        Returns
        -------
        evo_result: :class:`qutip.Result`
            An instance of the class :class:`qutip.Result`, which contains
            either an *array* `result.expect` of expectation values for
            the times specified by `tlist`, or an *array* `result.states`
            of state vectors or density matrices corresponding to
            the times in `tlist` [if `e_ops` is
            an empty list], or nothing if a callback function was
            given in place of
            operators for which to calculate the expectation values.
        """
        # kwargs can not contain H
        if "H" in kwargs or "tlist" in kwargs:
            raise ValueError(
                "`H` and `tlist` are already specified by the processor "
                "and can not be given as a key word argument")

        # construct qobjevo for unitary evolution
        if "args" in kwargs:
            noisy_qobjevo, sys_c_ops = self.get_noisy_qobjevo(
                                                        args=kwargs["args"])
        else:
            noisy_qobjevo, sys_c_ops = self.get_noisy_qobjevo()

        # add noise into kwargs
        if "c_ops" in kwargs:
            if isinstance(kwargs["c_ops"], (Qobj, QobjEvo)):
                kwargs["c_ops"] += [kwargs["c_ops"]] + sys_c_ops
            else:
                kwargs["c_ops"] += sys_c_ops
        else:
            kwargs["c_ops"] = sys_c_ops

        evo_result = mesolve(
            H=noisy_qobjevo, rho0=rho0, tlist=noisy_qobjevo.tlist, **kwargs)
        return evo_result

    def optimize_circuit(self, qc):
        """
        Function to take a quantum circuit/algorithm and convert it into the
        optimal form/basis for the desired physical system.
        (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def load_circuit(self, qc):
        """
        Translates an abstract quantum circuit to its
        corresponding Hamiltonians.(Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def get_ops_and_u(self):
        """
        Returns the Hamiltonian operators and the pulse matrix
        (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

    def eliminate_auxillary_modes(self, U):
        """
        Eliminate the auxillary modes like the cavity modes in cqed.
        (Defined in subclasses)
        """
        return U

    def plot_pulses(self, noisy=False, title=None):
        """
        Plot the pulse amplitude

        Parameters
        ----------
        noisy: boolean, optional
            If true, it will plot the noisy pulses.

        title: str
            Title for the plot.

        Returns
        -------
        fig: matplotlib.figure.Figure
            The `Figure` object for the plot.

        ax: matplotlib.axes._subplots.AxesSubplot
            The axes for the plot.


        Note
        ----
        It only works for array-like coefficients
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.set_ylabel("Control pulse amplitude")
        ax.set_xlabel("Time")
        if noisy:
            # TODO add test
            coeffs, tlist = self.get_noisy_coeff()    
        else:
            coeffs = [coeff for coeff in self.coeffs]
            tlist = self.tlist

        for i in range(len(coeffs)):
            if not isinstance(coeffs[i], (Iterable, np.ndarray)):
                raise ValueError(
                    "plot_pulse only accept array-like coefficients.")
            if self.spline_kind == "step_func":
                if len(coeffs[i]) == len(tlist) - 1:
                    coeffs[i] = np.hstack([self.coeffs[i], self.coeffs[i, -1:]])
                else:
                    coeffs[i][-1] = coeffs[i][-2]
                ax.step(tlist, coeffs[i], where='post')
            elif self.spline_kind == "cubic":
                sp = CubicSpline(tlist, coeffs[i])
                t_line = np.linspace(tlist[0], tlist[-1], 200)
                c_line = [sp(t) for t in t_line]
                ax.plot(t_line, c_line)
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    def add_noise(self, noise):
        """
        Add a noise object to the processor

        Parameters
        ----------
        noise: :class:`qutip.qip.CircuitNoise`
            The noise object defined outside the processor
        """
        if isinstance(noise, CircuitNoise):
            self.noise.append(noise)
        else:
            raise TypeError("Input is not a CircuitNoise object.")

    def process_noise(self, proc_qobjevo):
        """
        Call all the noise object saved in the processor and
        return noisy representation of the evolution

        Parameters
        ----------
        proc_qobjevo: :class:`qutip.qip.QobjEvo`
            The :class:`qutip.qip.QobjEvo` representing the unitary evolution in the noiseless processor.

        Returns
        -------
        H_noise: :class:`qutip.qip.QobjEvo`
            The :class:`qutip.qip.QobjEvo` representing the noisy
            part Hamiltonians.

        c_ops: list
            A list of :class:`qutip.qip.QobjEvo` or :class:`qutip.qip.Qobj`,
            representing the time-(in)dependent collapse operators.
        """
        c_ops = []
        evo_noise_list = []
        if (self.T1 is not None) or (self.T2 is not None):
            c_ops += RelaxationNoise(self.T1, self.T2).get_noise(
                N=self.N, dims=self.dims)
        for noise in self.noise:
            if isinstance(noise, (DecoherenceNoise, RelaxationNoise)):
                c_ops += noise.get_noise(
                    self.N, dims=self.dims)
            elif isinstance(noise, ControlAmpNoise):
                evo_noise_list.append(noise.get_noise(
                    self.N, proc_qobjevo, dims=self.dims))
            elif isinstance(noise, UserNoise):
                noise_qobjevo, new_c_ops = noise.get_noise(
                    self.N, proc_qobjevo, dims=self.dims)
                evo_noise_list.append(noise_qobjevo)
                c_ops += new_c_ops
            else:
                raise NotImplementedError(
                    "The noise type {} is not"
                    "implemented in the processor".format(
                        type(noise)))
        return self._compatible_coeff(evo_noise_list), c_ops

    def _compatible_coeff(self, qobjevo_list):
        """
        Combine a list of `:class:qutip.QobjEvo` into one,
        different tlist will be merged.
        """
        # TODO add test
        # no qobjevo
        if not qobjevo_list:
            return _dummy_qobjevo(self.dims)
        all_tlists = [qu.tlist for qu in qobjevo_list if qu.tlist is not None]
        # all tlists are None
        if not all_tlists:
            return sum(qobjevo_list, _dummy_qobjevo(self.dims))
        new_tlist = np.unique(np.sort(np.hstack(all_tlists)))
        for i, qobjevo in enumerate(qobjevo_list):
            H_list = qobjevo.to_list()
            for j, H in enumerate(H_list):
                # cte part or not array-like coeffs
                if isinstance(H, Qobj) or (not isinstance(H[1], np.ndarray)):
                    continue
                op, coeffs = H
                new_coeff = _fill_coeff(coeffs, qobjevo.tlist, new_tlist)
                H_list[j] = [op, new_coeff]
            # QobjEvo cannot handle [Qobj] as input
            if H_list and all(isinstance(H, Qobj) for H in H_list):
                H_list = sum(H_list, H_list[0] * 0.)
            # create a new qobjevo with the old arguments
            qobjevo_list[i] = QobjEvo(
                H_list, tlist=new_tlist, args=qobjevo.args)
        qobjevo = sum(qobjevo_list, _dummy_qobjevo(self.dims))
        qobjevo = _merge_id_evo(qobjevo)
        return qobjevo


def _fill_coeff(old_coeff, old_tlist, new_tlist):
    """
    Make a step function coefficients compatible with a longer `tlist` by
    filling the empty slot with the nearest left value.
    """
    new_n = len(new_tlist)
    old_ind = 0  # index for old coeffs and tlist
    new_coeff = np.zeros(new_n)
    for new_ind in range(new_n):
        t = new_tlist[new_ind]
        if t < old_tlist[0]:
            new_coeff[new_ind] = 0.
            continue
        if t > old_tlist[-1]:
            new_coeff[new_ind] = 0.
            continue
        if old_tlist[old_ind+1] == t:
            old_ind += 1
        new_coeff[new_ind] = old_coeff[old_ind]
    return new_coeff


def _merge_id_evo(qobjevo):
    """
    Merge identical Hamiltonians in the :class":`qutip.QobjEvo`.
    coeffs must all have the same length
    """
    H_list = qobjevo.to_list()
    new_H_list = []
    add_dict = {}
    op_list = []
    coeff_list = []
    for H in H_list:
        # cte part or not array-like coeffs
        if isinstance(H, Qobj) or (not isinstance(H[1], np.ndarray)):
            new_H_list.append(deepcopy(H))
            continue
        op, coeffs = H
        # Qobj is not hashable, so cannot be used as key in dict
        try:
            p = op_list.index(op)
            coeff_list[p] += coeffs
        except ValueError:
            op_list.append(op)
            coeff_list.append(coeffs)
    new_H_list += [[op_list[i], coeff_list[i]] for i in range(len(op_list))]
    if new_H_list and all(isinstance(H, Qobj) for H in new_H_list):
        new_H_list = sum(new_H_list, new_H_list[0] * 0.)
    return QobjEvo(new_H_list, tlist=qobjevo.tlist, args=qobjevo.args)


def _dummy_qobjevo(dims, **kwargs):
    """
    Create a dummy :class":`qutip.QobjEvo` with
    a constant zero Halmiltonian.
    """
    dummy = QobjEvo(tensor([identity(d) for d in dims]) * 0., **kwargs)
    return dummy


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

    def get_ops_labels(self):
        """
        Returns the Hamiltonian operators and corresponding labels by stacking
        them together. (Defined in subclasses)
        """
        raise NotImplementedError("Use the function in the sub-class")

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

        numerical: boolean, optional
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

    def plot_pulses(self):
        """
        Maps the physical interaction between the circuit components for the
        desired physical system.

        Returns
        --------
        fig, ax: Figure
            Maps the physical interaction between the circuit components.
        """
        # TODO merge it with plot pulse in circuitprocessor
        import matplotlib.pyplot as plt
        t, u, u_labels = self.pulse_matrix()
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for n, uu in enumerate(u):
            ax.plot(t, u[n], label=u_labels[n])

        ax.axis('tight')
        ax.set_ylim(-1.5 * 2 * np.pi, 1.5 * 2 * np.pi)
        ax.legend(loc='center left',
                  bbox_to_anchor=(1, 0.5), ncol=(1 + len(u) // 16))
        ax.set_ylabel("Control pulse amplitude")
        ax.set_xlabel("Time")
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
