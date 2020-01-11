import numbers
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from numpy.random import normal

from qutip.qobjevo import QobjEvo, EvoElement
from qutip.qip.gates import (
    expand_operator, _check_qubits_oper)
from qutip.qobj import Qobj
from qutip.operators import sigmaz, destroy, identity
from qutip.tensor import tensor
from qutip.qip.pulse import Pulse


__all__ = ["Noise", "DecoherenceNoise", "RelaxationNoise",
           "ControlAmpNoise", "RandomNoise", "UserNoise", "process_noise"]


def process_noise(ctrl_pulses, noise, N, dims, t1=None, t2=None):
    """
    Call all the noise object saved in the processor and
    return a noisy part of the evolution.

    Parameters
    ----------
    proc_qobjevo: :class:`qutip.qip.QobjEvo`
        The :class:`qutip.qip.QobjEvo` representing the unitary evolution
        in the noiseless processor.

    Returns
    -------
    noise: :class:`qutip.qip.QobjEvo`
        The :class:`qutip.qip.QobjEvo` representing the noisy
        part Hamiltonians.

    c_ops: list
        A list of :class:`qutip.qip.QobjEvo` or :class:`qutip.qip.Qobj`,
        representing the time-(in)dependent collapse operators.
    """
    ctrl_pulses = deepcopy(ctrl_pulses)
    noisy_dynamics = []
    c_ops = []

    if (t1 is not None) or (t2 is not None):
        noisy_dynamics += [RelaxationNoise(t1, t2).get_noisy_dynamics(
            N=N)]

    for noise in noise:
        if isinstance(noise, (DecoherenceNoise, RelaxationNoise)):
            noisy_dynamics += [noise.get_noisy_dynamics(N)]
        elif isinstance(noise, ControlAmpNoise):
            ctrl_pulses = noise.get_noisy_dynamics(N, ctrl_pulses)
        elif isinstance(noise, UserNoise):
            ctrl_pulses, new_c_ops = noise.get_noisy_dynamics(
                ctrl_pulses, N, dims)
            c_ops += new_c_ops
        else:
            raise NotImplementedError(
                "The noise type {} is not"
                "implemented in the processor".format(
                    type(noise)))
    # first the control pulse with noise,
    # then additional pulse independent noise.
    return ctrl_pulses + noisy_dynamics, c_ops


class Noise(object):
    """
    The base class representing noise in a processor.
    The noise object can be added to :class:`qutip.qip.Processor` and
    contributes to evolution.
    """
    def __init__(self):
        pass

    def _check_coeff_num(self, coeffs, ops_num):
        if len(coeffs) != ops_num:
            raise ValueError(
                "The length of coeffs is not {}".format(ops_num))


class DecoherenceNoise(Noise):
    """
    The decoherence noise in a processor. It generates a list of
    collapse operators.

    Parameters
    ----------
    c_ops: :class:`qutip.Qobj` or list
        The Hamiltonian representing the dynamics of the noise.
        len(ops)=len(coeffs) is required.

    targets: int or list, optional
        The indices of qubits that are acted on. Default is the first
        N qubits

    coeffs: list, optional
        A list of the coefficients for the control Hamiltonians.
        For available choice, see :class:`Qutip.QobjEvo`

    tlist: array_like, optional
        A NumPy array specifies the time of each coefficient.

    all_qubits: bool, optional
        If c_ops contains only single qubits collapse operator,
        all_qubits=True will allow it to be applied to all qubits.

    Attributes
    ----------
    c_ops: :class:`qutip.Qobj` or list
        The Hamiltonian representing the dynamics of the noise.

    targets: list
        The indices of qubits that are acted on.

    coeff: list
        A list of the coefficients for the control Hamiltonians.
        For available choice, see :class:`Qutip.QobjEvo`

    tlist: array_like
        A NumPy array specifies the time of each coefficient.

    all_qubits: bool
        If c_ops contains only single qubits collapse operator,
        all_qubits=True will allow it to be applied to all qubits.
    """
    def __init__(self, c_ops, targets=None, coeff=None, tlist=None,
                 all_qubits=False, spline_kind="step_func"):
        if isinstance(c_ops, Qobj):
            self.c_ops = [c_ops]
        else:
            self.c_ops = c_ops
        self.coeff = coeff
        self.tlist = tlist
        self.targets = targets
        if all_qubits:
            if not all([c_op.dims == [[2], [2]] for c_op in self.c_ops]):
                raise ValueError(
                    "The operator is not a single qubit operator, "
                    "thus cannot be applied to all qubits")
        self.all_qubits = all_qubits
        self.spline_kind = spline_kind

    def get_noisy_dynamics(self, N):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N: int
            The number of component systems.

        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.

        Returns
        -------
        lindblad_noise: list
            A list of :class:`qutip.Qobj` or :class:`qutip.QobjEvo`
            representing the decoherence noise.
        """
        # time-independent
        if (self.coeff is None) ^ (self.tlist is None):
            raise ValueError("Invalid input, coeffs and tlist are both required for time-dependent noise.")

        lindblad_noise = Pulse(None, None)
        for c_op in self.c_ops:
            if self.all_qubits:
                for targets in range(N):
                    lindblad_noise.add_lindblad_noise(c_op, targets, self.tlist, self.coeff)
            else:
                lindblad_noise.add_lindblad_noise(c_op, self.targets, self.tlist, self.coeff)
        return lindblad_noise


class RelaxationNoise(Noise):
    """
    The decoherence on each qubit characterized by two time scales t1 and t2.

    Parameters
    ----------
    t1: float or list, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: float or list, optional
        Characterize the decoherence of dephasing for
        each qubit.

    Attributes
    ----------
    t1: list
        Characterize the decoherence of amplitude damping for
        each qubit.

    t2: list
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, t1=None, t2=None):
        self.t1 = t1
        self.t2 = t2

    def _T_to_list(self, T, N):
        """
        Check if the relaxation time is valid

        Parameters
        ----------
        T: list of float
            The relaxation time

        N: int
            The number of component systems.

        Returns
        -------
        T: list
            The relaxation time in Python list form
        """
        if (isinstance(T, numbers.Real) and T > 0) or T is None:
            return [T] * N
        elif isinstance(T, Iterable) and len(T) == N:
            if all([isinstance(t, numbers.Real) and t > 0 for t in T]):
                return T
        else:
            raise ValueError(
                "Invalid relaxation time T={},"
                "either the length is not equal to the number of qubits, "
                "or T is not a positive number.".format(T))

    def get_noisy_dynamics(self, N):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N: int
            The number of component systems.

        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.

        Returns
        -------
        lindblad_noise: list
            A list of :class:`qutip.Qobj` or :class:`qutip.QobjEvo`
            representing the decoherence noise.
        """
        self.t1 = self._T_to_list(self.t1, N)
        self.t2 = self._T_to_list(self.t2, N)
        if len(self.t1) != N or len(self.t2) != N:
            raise ValueError(
                "Length of t1 or t2 does not match N, "
                "len(t1)={}, len(t2)={}".format(
                    len(self.t1), len(self.t2)))
        lindblad_noise = Pulse(None, None)
        for qu_ind in range(N):
            t1 = self.t1[qu_ind]
            t2 = self.t2[qu_ind]
            if t1 is not None:
                op = 1/np.sqrt(t1) * destroy(2)
                # lindblad_noise.append(
                #     expand_operator(
                #         1/np.sqrt(t1) * destroy(2), N, qu_ind, dims=dims))
                lindblad_noise.add_lindblad_noise(op, qu_ind)
            if t2 is not None:
                # Keep the total dephasing ~ exp(-t/t2)
                if t1 is not None:
                    if 2*t1 < t2:
                        raise ValueError(
                            "t1={}, t2={} does not fulfill "
                            "2*t1>t2".format(t1, t2))
                    T2_eff = 1./(1./t2-1./2./t1)
                else:
                    T2_eff = t2
                op = 1/np.sqrt(2*T2_eff) * sigmaz()
                # lindblad_noise.append(
                #     expand_operator(
                #         1/np.sqrt(2*T2_eff) * sigmaz(), N, qu_ind, dims=dims))
                lindblad_noise.add_lindblad_noise(op, qu_ind)
        return lindblad_noise


class ControlAmpNoise(Noise):
    """
    The noise in the amplitude of the control pulse.

    Parameters
    ----------
    coeffs: list
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`Qutip.QobjEvo`.

    tlist: array_like, optional
        A NumPy array specifies the time of each coefficient.

    ops: :class:`qutip.Qobj` or list
        The Hamiltonian representing the dynamics of the noise.
        len(ops)=len(coeffs) is required.

    targets: int or list, optional
        The indices of qubits that are acted on. Default is the first
        N qubits

    cyclic_permutation: boolean, optional
        If true, the Hamiltonian will be expanded for
        all cyclic permutation of the target qubits.

    Attributes
    ----------
    coeffs: list
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`Qutip.QobjEvo`.

    tlist: array_like
        A NumPy array specifies the time of each coefficient.

    ops: list
        The Hamiltonian representing the dynamics of the noise.

    targets: list
        The indices of qubits that are acted on.

    cyclic_permutation: boolean
        If true, the Hamiltonian will be expanded for
        all cyclic permutation of the target qubits.
    """
    def __init__(self, coeff, tlist=None):
        self.coeff = coeff
        self.tlist = tlist

    def get_noisy_dynamics(self, N, ctrl_pulses):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N: int
            The number of component systems.

        proc_qobjevo: :class:`qutip.QobjEvo`, optional
            If no operator is defined in the noise object, `proc_qobjevo`
            will be used as operators, otherwise the operators in the
            object is used.

        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.

        Returns
        -------
        noise_qobjevo: :class:`qutip.QobjEvo`
            A :class:`qutip.Qobj` representing the noise.
        """
        for i, pulse in enumerate(ctrl_pulses):
            if isinstance(self.coeff, (int, float)):
                coeff = pulse.coeff * self.coeff
            else:
                coeff = self.coeff
            if self.tlist is None:
                tlist = pulse.tlist
            else:
                tlist = self.tlist
            ctrl_pulses[i].add_coherent_noise(pulse.op, pulse.targets, tlist, coeff)

        return ctrl_pulses


class RandomNoise(ControlAmpNoise):
    """
    Random noise in the amplitude of the control pulse. The arguments for
    the random generator need to be given as key word arguments.

    Parameters
    ----------
    rand_gen: numpy.random, optional
        A random generator in numpy.random, it has to take a ``size``
        parameter.

    dt: float, optional
        The time interval between two random amplitude. The coefficients
        of the noise are the same within this time range.

    ops: list, optional
        The Hamiltonian representing the dynamics of the noise.

    targets: list or int, optional
        The indices of qubits that are acted on.

    cyclic_permutation: boolean, optional
        If true, the Hamiltonian will be expanded for
        all cyclic permutation of the target qubits.

    kwargs:
        Key word arguments for the random number generator.

    Attributes
    ----------
    ops: list
        The Hamiltonian representing the dynamics of the noise.

    coeffs: list
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`Qutip.QobjEvo`.

    targets: list
        The indices of qubits that are acted on.

    cyclic_permutation: boolean
        If true, the Hamiltonian will be expanded for
        all cyclic permutation of the target qubits.

    rand_gen: numpy.random
        A random generator in numpy.random, it has to take a ``size``
        parameter.

    kwargs: dict
        Key word arguments for the random number generator.
    """
    def __init__(
            self, dt, rand_gen, **kwargs):
        super(RandomNoise, self).__init__(coeff=None, tlist=None)
        if rand_gen is None:
            self.rand_gen = np.random.normal
        else:
            self.rand_gen = rand_gen
        self.kwargs = kwargs
        if "size" in kwargs:
            raise ValueError("size is preditermined inside the noise object.")
        self.dt = dt

    def get_noisy_dynamics(self, N, ctrl_pulses):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N: int
            The number of component systems.

        proc_qobjevo: :class:`qutip.QobjEvo`, optional
            If no operator is defined in the noise object, `proc_qobjevo`
            wil be used as operators, otherwise the operators in the
            object is used.

        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.

        Returns
        -------
        noise_qobjevo: :class:`qutip.QobjEvo`
            A :class:`qutip.Qobj` representing the decoherence noise.
        """
        t_max = -np.inf
        t_min = np.inf
        for pulse in ctrl_pulses:
            t_max = max(max(pulse.tlist), t_max)
            t_min = min(min(pulse.tlist), t_min)
        # create new tlist and random coeff
        num_rand = int(np.floor((t_max - t_min) / self.dt)) + 1
        tlist = (np.arange(0, self.dt*num_rand, self.dt)[:num_rand] + t_min)
        # [:num_rand] for round of error like 0.2*6=1.2000000000002

        for i, pulse in enumerate(ctrl_pulses):
            coeff = self.rand_gen(**self.kwargs, size=num_rand)
            ctrl_pulses[i].add_coherent_noise(pulse.op, pulse.targets, tlist, coeff)
        return ctrl_pulses


class UserNoise(Noise):
    """
    Template class for user defined noise. To define a noise object,
    please use this as a parent class. When simulating the noise, the
    mothod get_noisy_dynamics will be called with the input 
    proc_qobjevo, ctrl_hams and dims.
    """
    def __init__(self):
        pass

    def get_noisy_dynamics(self, ctrl_pulses, N, dims):
        """
        Template method. To define a noise object,
        one should over write this method and
        return the unitary evolution part as a :class: `qutip.QobjEvo`
        and a list of collapse operators in the form of either
        :class: `qutip.QobjEvo` or :class: `qutip.Qobj`.

        Parameters
        ----------
        N: int
            The number of component systems.

        proc_qobjevo: :class:`qutip.QobjEvo`
            The object representing the ideal evolution in the processor.

        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.

        Returns
        -------
        noise_qobjevo: :class:`qutip.QobjEvo`
            A :class:`qutip.Qobj` representing the decoherence noise.

        collapse_list: list
            A list of :class:`qutip.Qobj` or :class:`qutip.QobjEvo`
            representing the decoherence noise.
        """
        return ctrl_pulses, []
