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


__all__ = ["Noise", "DecoherenceNoise", "RelaxationNoise",
           "ControlAmpNoise", "RandomNoise", "UserNoise"]


def _dummy_qobjevo(dims, **kwargs):
    """
    Create a dummy :class":`qutip.QobjEvo` with
    a constant zero Hamiltonian. This is used since empty QobjEvo
    is not yet supported.
    """
    dummy = QobjEvo(tensor([identity(d) for d in dims]) * 0., **kwargs)
    return dummy


class Noise(object):
    """
    The base class representing noise in a circuit processor.
    The noise object can be added to :class:`qutip.qip.CircuitProcessor` and
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
    The decoherence noise in a circuit processor. It will generate a list of
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

    coeffs: list
        A list of the coefficients for the control Hamiltonians.
        For available choice, see :class:`Qutip.QobjEvo`

    tlist: array_like
        A NumPy array specifies the time of each coefficient.

    all_qubits: bool
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
                    "The operator is not a single qubit operator, "
                    "thus cannot be applied to all qubits")
        self.all_qubits = all_qubits

    def get_noise(self, N, dims=None):
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
        qobjevo_list: list
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
                    expand_operator(
                        oper=c_op, N=N, targets=self.targets, dims=dims))
        # time-independent
        if self.coeffs is None:
            return qobj_list
        # time-dependent
        if self.tlist is None:
            raise ValueError("tlist is required for time-dependent noise.")
        qobjevo_list = []
        for i, temp in enumerate(qobj_list):
            self._check_coeff_num(self.coeffs, len(qobj_list))
            qobjevo_list.append(QobjEvo(
                [qobj_list[i], self.coeffs[i]],
                tlist=self.tlist))
        return qobjevo_list


class RelaxationNoise(Noise):
    """
    The decoherence on each qubit characterized by two time scales T1 and T2.

    Parameters
    ----------
    T1: float or list, optional
        Characterize the decoherence of amplitude damping for
        each qubit.

    T2: float or list, optional
        Characterize the decoherence of dephasing for
        each qubit.

    Attributes
    ----------
    T1: list
        Characterize the decoherence of amplitude damping for
        each qubit.

    T2: list
        Characterize the decoherence of dephasing for
        each qubit.
    """
    def __init__(self, T1=None, T2=None):
        self.T1 = T1
        self.T2 = T2

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

    def get_noise(self, N, dims=None):
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
        qobjevo_list: list
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
                    expand_operator(
                        1/np.sqrt(T1) * destroy(2), N, qu_ind, dims=dims))
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
                    expand_operator(
                        1/np.sqrt(2*T2_eff) * sigmaz(), N, qu_ind, dims=dims))
        return qobjevo_list


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
    def __init__(self, coeffs, tlist, ops=None, targets=None,
                 cyclic_permutation=False):
        self.coeffs = coeffs
        self.tlist = tlist
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
        if dims is None:
            dims = [2] * N

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

        if len(ops) > len(self.coeffs):
            raise ValueError("The number of coefficient has to be larger than"
                             "{}".format(len(ops)))
        return QobjEvo([[ops[i], self.coeffs[i]] for i in range(len(ops))],
                       tlist=self.tlist)


class RandomNoise(ControlAmpNoise):
    """
    Random noise in the amplitude of the control pulse.

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
            self, rand_gen=None, dt=None, ops=None, targets=None,
            cyclic_permutation=False, **kwargs):
        super(RandomNoise, self).__init__(
            coeffs=None, tlist=None, ops=ops, targets=targets,
            cyclic_permutation=cyclic_permutation)
        if rand_gen is None:
            self.rand_gen = np.random.normal
        else:
            self.rand_gen = rand_gen
        self.kwargs = kwargs
        if "size" in kwargs:
            raise ValueError("size is preditermined inside the noise object.")
        self.dt = dt

    def get_noise(self, N, proc_qobjevo=None, dims=None):
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
            # if no cte part the last coeffs will be ignored
            ops_num = len(proc_qobjevo.ops) + 1
        if self.dt is not None:
            # create new tlist and random coeffs
            num_rand = int(np.floor((tlist[-1]-tlist[0])/self.dt))+1
            self.coeffs = self.rand_gen(
                **self.kwargs, size=(ops_num, num_rand))
            tlist = (np.arange(0, self.dt*num_rand, self.dt)[:num_rand] +
                     tlist[0])
            # [:num_rand] for round of error like 0.2*6=1.2000000000002
        else:
            self.coeffs = self.rand_gen(
                **self.kwargs, size=(ops_num, len(tlist)))
        self.tlist = tlist
        return super(RandomNoise, self).get_noise(
            N, proc_qobjevo=proc_qobjevo, dims=dims)


class UserNoise(Noise):
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
        if dims is None:
            dims = [2] * N
        return _dummy_qobjevo(dims), []
