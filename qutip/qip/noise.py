import numbers
from collections.abc import Iterable
from copy import deepcopy
import numpy as np

from qutip.qobjevo import QobjEvo
from qutip.qip.operations import expand_operator
from qutip.qobj import Qobj
from qutip.operators import num, qeye, sigmaz, destroy
from qutip.qip.pulse import Pulse


__all__ = ["Noise", "DecoherenceNoise", "RelaxationNoise",
           "ControlAmpNoise", "RandomNoise", "UserNoise", "process_noise"]


def process_noise(pulses, noise_list, dims, t1=None, t2=None,
                  device_noise=False):
    """
    Apply noise to the input list of pulses. It does not modify the input
    pulse, but return a new one containing the noise.

    Parameters
    ----------
    pulses: list of :class:`qutip.qip.Pulse`
        The input pulses, on which the noise object will be applied.
    noise_list: list of :class:`qutip.qip.noise`
        A list of noise objects.
    dims: int or list
        Dimension of the system.
        If int, we assume it is the number of qubits in the system.
        If list, it is the dimension of the component systems.
    t1: list or float, optional
        Characterize the decoherence of amplitude damping for
        each qubit. A list of size `N` or a float for all qubits.
    t2: list of float, optional
        Characterize the decoherence of dephasing for
        each qubit. A list of size `N` or a float for all qubits.
    device_noise: bool
        If pulse independent noise such as relaxation are included.
        Default is False.

    Returns
    -------
    noisy_pulses: list of :class:`qutip.qip.Pulse`
        The noisy pulses, including the system noise.
    """
    if device_noise:  # first pulse independent noise
        systematic_noise = Pulse(None, None, label="systematic_noise")
        if (t1 is not None) or (t2 is not None):
            RelaxationNoise(t1, t2).get_noisy_dynamics(dims, systematic_noise)

        for noise in noise_list:
            if isinstance(noise, (DecoherenceNoise, RelaxationNoise)):
                noise.get_noisy_dynamics(dims, systematic_noise)

    # then pulse-dependent noise
    noisy_pulses = deepcopy(pulses)
    for noise in noise_list:
        if isinstance(noise, ControlAmpNoise):
            noise.get_noisy_dynamics(noisy_pulses)
        elif isinstance(noise, UserNoise):
            noise.get_noisy_dynamics(noisy_pulses, systematic_noise, dims)

    if device_noise:
        return noisy_pulses + [systematic_noise]
    else:
        return noisy_pulses


class Noise(object):
    """
    The base class representing noise in a processor.
    The noise object can be added to :class:`qutip.qip.device.Processor` and
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
    The decoherence noise in a processor. It generates lindblad noise
    according to the given collapse operator `c_ops`.

    Parameters
    ----------
    c_ops: :class:`qutip.Qobj` or list
        The Hamiltonian representing the dynamics of the noise.
    targets: int or list, optional
        The indices of qubits that are acted on. Default is on all
        qubits
    coeff: list, optional
        A list of the coefficients for the control Hamiltonians.
    tlist: array_like, optional
        A NumPy array specifies the time of each coefficient.
    all_qubits: bool, optional
        If `c_ops` contains only single qubits collapse operator,
        ``all_qubits=True`` will allow it to be applied to all qubits.

    Attributes
    ----------
    c_ops: :class:`qutip.Qobj` or list
        The Hamiltonian representing the dynamics of the noise.
    targets: int or list
        The indices of qubits that are acted on.
    coeff: list
        A list of the coefficients for the control Hamiltonians.
    tlist: array_like
        A NumPy array specifies the time of each coefficient.
    all_qubits: bool
        If `c_ops` contains only single qubits collapse operator,
        ``all_qubits=True`` will allow it to be applied to all qubits.
    """
    def __init__(self, c_ops, targets=None, coeff=None, tlist=None,
                 all_qubits=False):
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

    def get_noisy_dynamics(self, dims, systematic_noise=None):
        """
        Return a list of Pulse object with only trivial ideal pulse (H=0) but
        non-trivial lindblad noise.

        Parameters
        ----------
        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.

        systematic_noise: :class:`qutip.qip.pulse.Pulse`
            The dummy pulse representing systematic noise.
        """
        if systematic_noise is None:
            systematic_noise = Pulse(None, None, label="system")
        if isinstance(dims, list):
            N = len(dims)
        else:
            N = dims
        # time-independent
        if (self.coeff is None) and (self.tlist is None):
            self.coeff = True

        for c_op in self.c_ops:
            if self.all_qubits:
                for targets in range(N):
                    systematic_noise.add_lindblad_noise(
                        c_op, targets, self.tlist, self.coeff)
            else:
                systematic_noise.add_lindblad_noise(
                    c_op, self.targets, self.tlist, self.coeff)


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
    targets: int or list, optional
        The indices of qubits that are acted on. Default is on all
        qubits

    Attributes
    ----------
    t1: float or list
        Characterize the decoherence of amplitude damping for
        each qubit.
    t2: float or list
        Characterize the decoherence of dephasing for
        each qubit.
    targets: int or list
        The indices of qubits that are acted on.
    """
    def __init__(self, t1=None, t2=None, targets=None):
        self.t1 = t1
        self.t2 = t2
        self.targets = targets

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

    def get_noisy_dynamics(self, dims, systematic_noise=None):
        """
        Return a list of Pulse object with only trivial ideal pulse (H=0) but
        non-trivial relaxation noise.

        Parameters
        ----------
        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.
        systematic_noise: :class:`qutip.qip.pulse.Pulse`
            The dummy pulse representing systematic noise.
        """
        if systematic_noise is None:
            systematic_noise = Pulse(None, None, label="system")
        if isinstance(dims, list):
            N = len(dims)
        else:
            N = dims

        self.t1 = self._T_to_list(self.t1, N)
        self.t2 = self._T_to_list(self.t2, N)
        if len(self.t1) != N or len(self.t2) != N:
            raise ValueError(
                "Length of t1 or t2 does not match N, "
                "len(t1)={}, len(t2)={}".format(
                    len(self.t1), len(self.t2)))

        if self.targets is None:
            targets = range(N)
        else:
            targets = self.targets
        for qu_ind in targets:
            t1 = self.t1[qu_ind]
            t2 = self.t2[qu_ind]
            if t1 is not None:
                op = 1/np.sqrt(t1) * destroy(dims[qu_ind])
                systematic_noise.add_lindblad_noise(op, qu_ind, coeff=True)
            if t2 is not None:
                # Keep the total dephasing ~ exp(-t/t2)
                if t1 is not None:
                    if 2*t1 < t2:
                        raise ValueError(
                            "t1={}, t2={} does not fulfill "
                            "2*t1>t2".format(t1, t2))
                    elif 2*t1==t2:
                        op=0*qeye(dims[qu_ind])
                    else:
                        T2_eff = 1./(1./t2-1./2./t1)
                        op = 1/np.sqrt(2*T2_eff) * 2*num(dims[qu_ind])
                else:
                    T2_eff = t2
                    op = 1/np.sqrt(2*T2_eff) * 2*num(dims[qu_ind])
                systematic_noise.add_lindblad_noise(op, qu_ind, coeff=True)


class ControlAmpNoise(Noise):
    """
    The noise in the amplitude of the control pulse.

    Parameters
    ----------
    coeff: list
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`qutip.QobjEvo`.
    tlist: array_like, optional
        A NumPy array specifies the time of each coefficient.
    indices: list of int, optional
        The indices of target pulse in the list of pulses.
    Attributes
    ----------
    coeff: list
        A list of the coefficients for the control Hamiltonians.
        For available choices, see :class:`qutip.QobjEvo`.
    tlist: array_like
        A NumPy array specifies the time of each coefficient.
    indices: list of int
        The indices of target pulse in the list of pulses.
    """
    def __init__(self, coeff, tlist=None, indices=None):
        self.coeff = coeff
        self.tlist = tlist
        self.indices = indices

    def get_noisy_dynamics(self, pulses):
        """
        Return a list of pulses containing the input pulse with additional
        coherent noise.

        Parameters
        ----------
        pulses: list of :class:`qutip.qip.Pulse`
            The input pulses, on which the noise object will be applied.
        systematic_noise: :class:`qutip.qip.pulse.Pulse`
            The dummy pulse representing systematic noise.
        """
        if self.indices is None:
            indices = range(len(pulses))
        else:
            indices = self.indices
        for i in indices:
            pulse = pulses[i]
            if isinstance(self.coeff, (int, float)):
                coeff = pulse.coeff * self.coeff
            else:
                coeff = self.coeff
            if self.tlist is None:
                tlist = pulse.tlist
            else:
                tlist = self.tlist
            pulses[i].add_control_noise(
                pulse.qobj, pulse.targets, tlist, coeff)


class RandomNoise(ControlAmpNoise):
    """
    Random noise in the amplitude of the control pulse. The arguments for
    the random generator need to be given as key word arguments.

    Parameters
    ----------
    dt: float, optional
        The time interval between two random amplitude. The coefficients
        of the noise are the same within this time range.
    rand_gen: numpy.random, optional
        A random generator in numpy.random, it has to take a ``size``
        parameter as the size of random numbers in the output array.
    indices: list of int, optional
        The indices of target pulse in the list of pulses.
    kwargs:
        Key word arguments for the random number generator.

    Attributes
    ----------
    dt: float, optional
        The time interval between two random amplitude. The coefficients
        of the noise are the same within this time range.
    rand_gen: numpy.random, optional
        A random generator in numpy.random, it has to take a ``size``
        parameter.
    indices: list of int
        The indices of target pulse in the list of pulses.
    kwargs:
        Key word arguments for the random number generator.

    Examples
    --------
    >>> gaussnoise = RandomNoise( \
            dt=0.1, rand_gen=np.random.normal, loc=mean, scale=std) \
            # doctest: +SKIP
    """
    def __init__(
            self, dt, rand_gen, indices=None, **kwargs):
        super(RandomNoise, self).__init__(coeff=None, tlist=None)
        self.rand_gen = rand_gen
        self.kwargs = kwargs
        if "size" in kwargs:
            raise ValueError("size is preditermined inside the noise object.")
        self.dt = dt
        self.indices = indices

    def get_noisy_dynamics(self, pulses):
        """
        Return a list of pulses containing the input pulse with additional
        coherent noise.

        Parameters
        ----------
        pulses: list of :class:`qutip.qip.Pulse`
            The input pulses, on which the noise object will be applied.
        systematic_noise: :class:`qutip.qip.pulse.Pulse`
            The dummy pulse representing systematic noise.
        """
        if self.indices is None:
            indices = range(len(pulses))
        else:
            indices = self.indices
        t_max = -np.inf
        t_min = np.inf
        for pulse in pulses:
            t_max = max(max(pulse.tlist), t_max)
            t_min = min(min(pulse.tlist), t_min)
        # create new tlist and random coeff
        num_rand = int(np.floor((t_max - t_min) / self.dt)) + 1
        tlist = (np.arange(0, self.dt*num_rand, self.dt)[:num_rand] + t_min)
        # [:num_rand] for round of error like 0.2*6=1.2000000000002

        for i in indices:
            pulse = pulses[i]
            coeff = self.rand_gen(**self.kwargs, size=num_rand)
            pulses[i].add_control_noise(
                pulse.qobj, pulse.targets, tlist, coeff)


class UserNoise(Noise):
    """
    Template class for user defined noise. It is classified as a pulse
    dependent noise. By calling the method `get_noisy_dynamics`,
    it should return the input pulses with additional
    coherent and/or lindblad noise. If there are
    pulse independent noise, a dummy Pulse can be created by
    ``Pulse(None, None)``. The modified input pulses should always
    be at the begining of the list, followed by the dummy pulses.
    """
    def __init__(self):
        pass

    def get_noisy_dynamics(self, pulses, systematic_noise, dims):
        """
        Template method.
        It should return a list of pulses with noise.

        Parameters
        ----------
        pulses: list of :class:`qutip.qip.Pulse`
            The input pulses, on which the noise object will be applied.
        systematic_noise: :class:`qutip.qip.pulse.Pulse`
            The dummy pulse representing systematic noise.
        dims: list, optional
            The dimension of the components system, the default value is
            [2,2...,2] for qubits system.
        """
        pass
