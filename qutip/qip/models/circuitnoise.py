import numbers
from collections.abc import Iterable
import numpy as np
from numpy.random import normal
from qutip.qobjevo import QobjEvo, EvoElement
from qutip.qip.gates import expand_oper
from qutip.qobj import Qobj
from qutip.operators import sigmaz, destroy


__all__ = ["CircuitNoise", "DecoherenceNoise", "RelaxationNoise",
           "ControlAmpNoise", "WhiteNoise"]


class CircuitNoise(object):
    """
    The base class representing noise in a circuit processor.
    The noise object can be added to `qutip.qip.CircuitProcessor` and
    contributes to the evolution.
    """
    def __init__(self):
        pass


class DecoherenceNoise(CircuitNoise):
    """
    The decoherence noise in a circuit processor. It is defined the
    collapse operators.
    """
    def __init__(self, c_ops, targets=None, coeffs=None, expand_type=None):
        if isinstance(c_ops, Qobj):
            self.c_ops = [c_ops]
        else:
            self.c_ops = c_ops
        if coeffs is None:  # time independent coeffs
            self.coeffs = None
        elif len(coeffs.shape) == 1:
            self.coeffs = coeffs.reshape((1, len(coeffs)))
        elif len(coeffs.shape) == 2:
            if coeffs.shape[0] != len(self.c_ops):
                raise ValueError(
                    "The row number of coeffs does not match"
                    "the number of collapse operators in c_ops.")
            self.coeffs = coeffs
        else:
            raise ValueError("`coeffs` is not a 2D-NumPy array.")
        self.targets = targets
        self.expand_type = expand_type

    def get_qobjlist(self, N, tlist):
        Q_objects = []
        for i, c_op in enumerate(self.c_ops):
            if self.coeffs is None:
                Q_objects.append(
                    expand_oper(oper=c_op, N=N, targets=self.targets))
            else:
                Q_objects.append(QobjEvo(
                    [expand_oper(oper=c_op, N=N, targets=self.targets),
                        self.coeffs[i]],
                    tlist=tlist))
        return Q_objects


class RelaxationNoise(CircuitNoise):
    def __init__(self, T1, T2):
        self.T1 = T1
        self.T2 = T2

    def _check_T_valid(self, T, N):
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

    def get_qobjlist(self, N):
        self.T1 = self._check_T_valid(self.T1, N)
        self.T2 = self._check_T_valid(self.T2, N)
        if len(self.T1) != N or len(self.T2) != N:
            raise ValueError(
                "Length of T1 or T2 does not match N, "
                "len(T1)={}, len(T2)={}".format(
                    len(self.T1), len(self.T2)))
        Q_objects = []
        for qu_ind in range(N):
            T1 = self.T1[qu_ind]
            T2 = self.T2[qu_ind]
            if T1 is not None:
                Q_objects.append(
                    expand_oper(1/np.sqrt(T1) * destroy(2), N, qu_ind))
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
                Q_objects.append(
                    expand_oper(1/np.sqrt(2*T2_eff) * sigmaz(), N, qu_ind))
        return Q_objects


class ControlAmpNoise(CircuitNoise):
    def __init__(self, oper, coeff=None, targets=None, expand_type=None):
        self.oper = oper
        self.coeff = coeff
        self.targets = targets

    def get_qobjevo(self, N, tlist):
        if self.coeff is None:
            noise_obj = expand_oper(
                oper=self.oper, N=N, targets=self.targets)
        else:
            noise_obj = QobjEvo(
                [[expand_oper(oper=self.oper, N=N, targets=self.targets),
                    self.coeff]],
                tlist=tlist)
        return noise_obj


class WhiteNoise(CircuitNoise):
    def __init__(
            self, mean, std, ops=None, targets=None, expand_type=None):
        self.mean = mean
        self.std = std
        if isinstance(ops, Qobj):
            self.ops = [ops]
        else:
            self.ops = ops
        self.targets = targets
        self.expand_type = expand_type

    def get_qobjevo(self, N, tlist, proc_qobjevo=None):
        # new Operators are given
        if self.ops is not None:
            ops = [
                expand_oper(oper=op, N=N, targets=self.targets)
                for op in self.ops]
            noise_coeffs = normal(
                self.mean, self.std, (len(ops), len(tlist)))
            noise_list = []
            for i, op in enumerate(ops):
                noise_list.append(
                    QobjEvo([[op, noise_coeffs[i]]], tlist=tlist))
        # If no operators given, use operators in the processor
        elif proc_qobjevo is not None:
            # If there is a constant part
            if proc_qobjevo.cte.norm() > 1.e-15:
                proc_ops = [proc_qobjevo.cte]
            else:
                proc_ops = []
            for evo_ele in proc_qobjevo.ops:
                proc_ops.append(evo_ele.qobj)
            noise_coeffs = normal(
                self.mean, self.std, (len(proc_ops), len(tlist)))
            noise_list = []
            for i, op in enumerate(proc_ops):
                noise_list.append(
                    QobjEvo([[op, noise_coeffs[i]]], tlist=tlist))
        else:
            raise ValueError(
                "No operators found.")
        return sum(noise_list)
