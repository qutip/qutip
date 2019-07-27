import numbers
from collections.abc import Iterable
import numpy as np
from numpy.random import normal
from qutip.qobjevo import QobjEvo, EvoElement
from qutip.qip.gates import (
    expand_operator, _check_qubits_oper)
from qutip.qobj import Qobj
from qutip.operators import sigmaz, destroy, identity
from qutip.tensor import tensor


__all__ = ["CircuitNoise", "DecoherenceNoise", "RelaxationNoise",
           "ControlAmpNoise", "WhiteNoise", "UserNoise"]


def _dummy_qobjevo(dims, **kwargs):
    dummy = QobjEvo(tensor([identity(d) for d in dims]) * 0., **kwargs)
    return dummy


class CircuitNoise(object):
    """
    The base class representing noise in a circuit processor.
    The noise object can be added to `qutip.qip.CircuitProcessor` and
    contributes to the evolution.
    """
    def __init__(self):
        pass

    def _check_coeff_num(self, coeffs, ops_num):
        if len(coeffs) != ops_num:
            raise ValueError(
                "The length of coeffs is not {}".format(ops_num))


class DecoherenceNoise(CircuitNoise):
    """
    The decoherence noise in a circuit processor. It is defined the
    collapse operators.

    Parameters
    ----------
    c_ops : Qobj or list
        The collapse operators
    targets : int or list
        Index for the targets qubits
    coeffs : list
        A list of Hamiltonian coeffs.
        For available choice, see :class:`Qutip.QobjEvo`
    all_qubits : bool
        If c_ops contains only single qubits collapse operator,
        all_qubits=True will allow it to be applied to all qubits.
    """
    def __init__(self, c_ops, targets=None, coeffs=None, tlist=None,
                 all_qubits=False):
        if isinstance(c_ops, Qobj):
            self.c_ops = [c_ops]
        else:
            self.c_ops = c_ops
        self.coeffs = coeffs
        self.tlist = tlist
        self.targets = targets
        if all_qubits:
            if not all([c_op.dims == [[2], [2]] for c_op in self.c_ops]):
                raise ValueError(
                    "c_op is not a single qubit operator"
                    "and cannot be applied to all qubits")
        self.all_qubits = all_qubits

    def get_noise(self, N, dims=None):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N : int
            The number of qubtis in the system
        tlist : array like
            A NumPy array specifies at which time the next amplitude of
            a pulse is to be applied.

        Returns
        -------
        qobjevo_list : list
            A list of :class:`qutip.Qobj` or :class:`qutip.QobjEvo`
            representing the decoherence noise.
        """
        if dims is None:
            dims = [2] * N
        qobj_list = []
        for i, c_op in enumerate(self.c_ops):
            if self.all_qubits:
                qobj_list += expand_operator(
                    oper=c_op, N=N, targets=self.targets, dims=dims,
                    cyclic_permutation=True)
            else:
                qobj_list.append(
                    expand_operator(oper=c_op, N=N, targets=self.targets,
                                dims=dims))
        # time-independent
        if self.coeffs is None:
            return qobj_list
        # time-dependent
        if self.tlist is None:
            raise ValueError("tlist is not given if noise is time-dependent.")
        qobjevo_list = []
        for i, temp in enumerate(qobj_list):
            self._check_coeff_num(self.coeffs, len(qobj_list))
            qobjevo_list.append(QobjEvo(
                [qobj_list[i], self.coeffs[i]],
                tlist=self.tlist))
        return qobjevo_list


class RelaxationNoise(CircuitNoise):
    """
    The decoherence on each qubit characterized by two time scales T1 and T2.

    Parameters
    ----------
    T1 : list or float
        Characterize the decoherence of amplitude damping for
        each qubit.
    T2 : list of float
        Characterize the decoherence of dephasing relaxation for
        each qubit.
    """
    def __init__(self, T1, T2):
        self.T1 = T1
        self.T2 = T2

    def _T_to_list(self, T, N):
        """
        Check if the relaxation time is valid

        Parameters
        ----------
        T : list of float
            The relaxation time
        N : int
            The number of qubits in the system

        Returns
        -------
        T : list
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

    def get_noise(self, N, dims=None):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N : int
            The number of qubtis in the system

        Returns
        -------
        qobjevo_list : list
            A list of :class:`qutip.Qobj` or :class:`qutip.QobjEvo`
            representing the decoherence noise.
        """
        if dims is None:
            dims = [2] * N
        self.T1 = self._T_to_list(self.T1, N)
        self.T2 = self._T_to_list(self.T2, N)
        if len(self.T1) != N or len(self.T2) != N:
            raise ValueError(
                "Length of T1 or T2 does not match N, "
                "len(T1)={}, len(T2)={}".format(
                    len(self.T1), len(self.T2)))
        qobjevo_list = []
        for qu_ind in range(N):
            T1 = self.T1[qu_ind]
            T2 = self.T2[qu_ind]
            if T1 is not None:
                qobjevo_list.append(
                    expand_operator(1/np.sqrt(T1) * destroy(2),
                                N, qu_ind, dims=dims))
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
                qobjevo_list.append(
                    expand_operator(1/np.sqrt(2*T2_eff) * sigmaz(),
                                N, qu_ind, dims=dims))
        return qobjevo_list


class ControlAmpNoise(CircuitNoise):
    """
    The noise in the amplitude of the control pulse.

    Parameters
    ----------
    ops : :class:`qutip.Qobj`
        The Hamiltonian representing the dynamics of the noise
    coeffs : list
        A list of Hamiltonian coeffs.
        For available choice, see :class:`Qutip.QobjEvo`
    targets : list or int
        The indices of qubits that are acted on
    cyclic_permutation : boolean
        If true, the Hamiltonian will be expanded for
            all cyclic permutation of target qubits
    """
    def __init__(self, ops, coeffs, targets=None, cyclic_permutation=False):
        self.coeffs = coeffs
        if isinstance(ops, Qobj):
            self.ops = [ops]
        else:
            self.ops = ops
        self.targets = targets
        self.cyclic_permutation = cyclic_permutation

    def get_noise(self, N, proc_qobjevo=None, dims=None):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N : int
            The number of qubtis in the system
        tlist : array like
            A NumPy array specifies at which time the next amplitude of
            a pulse is to be applied.
        proc_qobjevo : :class:`qutip.QobjEvo`
            If no operator is saved in the noise object, `proc_qobjevo`
            wil be used
            as operators, otherwise it is ignored.

        Returns
        -------
        noise_qobjevo : :class:`qutip.QobjEvo`
            A :class:`qutip.Qobj` representing the decoherence noise.
        """
        if dims is None:
            dims = [2] * N
        tlist = proc_qobjevo.tlist
        # If new operators are given
        if self.ops is not None:
            if self.cyclic_permutation:
                ops = []
                for op in self.ops:
                    ops += expand_operator(
                        oper=op, N=N, targets=self.targets, dims=dims,
                        cyclic_permutation=True)
            else:
                ops = [
                    expand_operator(
                        oper=op, N=N, targets=self.targets, dims=dims)
                    for op in self.ops]

        # If no operators given, use operators in the processor
        elif proc_qobjevo is not None:
            # If there is a constant part
            if proc_qobjevo.cte.norm() > 1.e-15:
                ops = [proc_qobjevo.cte]
            else:
                ops = []
            ops += [ele.qobj for ele in proc_qobjevo.ops]
        else:
            raise ValueError(
                "No operators found.")

        noise_list = []
        for i, op in enumerate(ops):
            noise_list.append(
                QobjEvo([[op, self.coeffs[i]]], tlist=tlist))
        # print(noise_list[0].dummy_cte)
        # print(noise_list[0].cte)
        print(_dummy_qobjevo(dims).dummy_cte)
        # print(sum(noise_list, _dummy_qobjevo(dims)).dummy_cte)
        # print(sum(noise_list, _dummy_qobjevo(dims)).cte)
        return sum(noise_list, _dummy_qobjevo(dims))


class WhiteNoise(ControlAmpNoise):
    """
    White gaussian noise in the amplitude of the control pulse.

    Parameters
    ----------
    mean : float
        Mean of the noise
    std : float
        Standard deviation of the noise
    ops : :class:`qutip.Qobj`
        The Hamiltonian representing the dynamics of the noise
    targets : list or int
        The indices of qubits that are acted on
    cyclic_permutation : boolean
        If true, the Hamiltonian will be expanded for
            all cyclic permutation of target qubits
    """
    # TODO add background or gauss
    def __init__(
            self, mean, std, dt=None, ops=None, targets=None,
            cyclic_permutation=False):
        super(WhiteNoise, self).__init__(
            ops, coeffs=None, targets=targets,
            cyclic_permutation=cyclic_permutation)
        self.mean = mean
        self.std = std
        self.dt = dt

    def get_noise(self, N, proc_qobjevo=None, dims=None):
        """
        Return the quantum objects representing the noise.

        Parameters
        ----------
        N : int
            The number of qubtis in the system
        tlist : array like
            A NumPy array specifies at which time the next amplitude of
            a pulse is to be applied.
        proc_qobjevo : :class:`qutip.QobjEvo`
            If no operator is saved in the noise object, `proc_qobjevo`
            wil be used
            as operators, otherwise it is ignored.

        Returns
        -------
        noise_qobjevo : :class:`qutip.QobjEvo`
            A :class:`qutip.Qobj` representing the decoherence noise.
        """
        if dims is None:
            dims = [2] * N
        tlist = proc_qobjevo.tlist
        if self.ops is not None:
            if self.cyclic_permutation:
                ops_num = len(self.ops) * N
            else:
                ops_num = len(self.ops)
        elif proc_qobjevo is not None:
            # +1 for the constant part in QobjEvo,
            # if no cte part the last coeff will be ignored
            ops_num = len(proc_qobjevo.ops) + 1
        self.coeffs = normal(
            self.mean, self.std, (ops_num, len(tlist)))
        return super(WhiteNoise, self).get_noise(
            N, proc_qobjevo=proc_qobjevo, dims=dims)


class UserNoise(CircuitNoise):
    """
    Abstract class for user defined noise. To define a noise object,
    one could overwrite the constructor and the class method `get_noise`.
    """
    def get_noise(self, N, proc_qobjevo, dims=None):
        """
        Template method. To define a noise object,
        one should over write this method and
        return the unitary evolution part as a :class: `qutip.QobjEvo`
        and a list of collapse operators in the form of either
        :class: `qutip.QobjEvo` or :class: `qutip.Qobj`.

        Parameters
        ----------
        N : int
            The number of qubtis in the system
        tlist : array like
            A NumPy array specifies at which time the next amplitude of
            a pulse is to be applied.
        proc_qobjevo : :class:`qutip.QobjEvo`
            The object representing the ideal evolution in the processor

        Returns
        -------
        noise_qobjevo : :class:`qutip.QobjEvo`
            A :class:`qutip.Qobj` representing the decoherence noise.
        collapse_list : list
            A list of :class:`qutip.Qobj` or :class:`qutip.QobjEvo`
            representing the decoherence noise.
        """
        pass
