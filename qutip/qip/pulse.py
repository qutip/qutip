from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline

from qutip.qobjevo import QobjEvo
from qutip.qobj import Qobj
from qutip.qip.operations import expand_operator
from qutip.operators import identity


__all__ = ["Pulse", "Drift"]


class _EvoElement():
    """
    The class object saving the information of one evolution element.
    Each dynamic element is characterized by four variables:
    `qobj`, `targets`, `tlist` and `coeff`.

    For documentation and use instruction of the attributes, please
    refer to :class:`qutip.qip.Pulse`.
    """
    def __init__(self, qobj, targets, tlist=None, coeff=None):
        self.qobj = qobj
        self.targets = targets
        self.tlist = tlist
        self.coeff = coeff

    def get_qobj(self, dims):
        """
        Get the `Qobj` representation of the element. If `qobj` is None,
        a zero :class:`qutip.Qobj` with the corresponding dimension is
        returned.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        qobj: :class:`qutip.Qobj`
            The operator of this element.
        """
        if isinstance(dims, (int, np.integer)):
            dims = [2] * dims
        if self.qobj is None:
            qobj = identity(dims[0]) * 0.
            targets = 0
        else:
            qobj = self.qobj
            targets = self.targets
        return expand_operator(qobj, len(dims), targets, dims)

    def _get_qobjevo_helper(self, spline_kind, dims):
        """
        Please refer to `_Evoelement.get_qobjevo` for documentation.
        """
        mat = self.get_qobj(dims)
        if self.tlist is None and self.coeff is None:
            qu = QobjEvo(mat) * 0.
        elif isinstance(self.coeff, bool):
            if self.coeff:
                if self.tlist is None:
                    qu = QobjEvo(mat, tlist=self.tlist)
                else:
                    qu = QobjEvo([mat, np.ones(len(self.tlist))],
                                 tlist=self.tlist)
            else:
                qu = QobjEvo(mat * 0., tlist=self.tlist)
        else:
            if spline_kind == "step_func":
                args = {"_step_func_coeff": True}
                if len(self.coeff) == len(self.tlist) - 1:
                    self.coeff = np.concatenate([self.coeff, [0.]])
            elif spline_kind == "cubic":
                args = {"_step_func_coeff": False}
            else:
                # The spline will follow other pulses or
                # use the default value of QobjEvo
                args = {}
            qu = QobjEvo([mat, self.coeff], tlist=self.tlist, args=args)
        return qu

    def get_qobjevo(self, spline_kind, dims):
        """
        Get the `QobjEvo` representation of the evolution element.
        If both `tlist` and `coeff` are None, treated as zero matrix.
        If ``coeff=True`` and ``tlist=None``,
        treated as time-independent operator.

        Parameters
        ----------
        spline_kind: str
            Type of the coefficient interpolation.
            "step_func" or "cubic"

            -"step_func":
            The coefficient will be treated as a step function.
            E.g. ``tlist=[0,1,2]`` and ``coeff=[3,2]``, means that the
            coefficient is 3 in t=[0,1) and 2 in t=[2,3). It requires
            ``len(coeff)=len(tlist)-1`` or ``len(coeff)=len(tlist)``, but
            in the second case the last element of `coeff` has no effect.

            -"cubic": Use cubic interpolation for the coefficient. It requires
            ``len(coeff)=len(tlist)``
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        qobjevo: :class:`qutip.QobjEvo`
            The `QobjEvo` representation of the evolution element.
        """
        try:
            return self._get_qobjevo_helper(spline_kind, dims=dims)
        except Exception as err:
            print(
                "The Evolution element went wrong was\n {}".format(str(self)))
            raise(err)

    def __str__(self):
        return str({"qobj": self.qobj,
                    "targets": self.targets,
                    "tlist": self.tlist,
                    "coeff": self.coeff
                    })


class Pulse():
    """
    Representation of a control pulse and the pulse dependent noise.
    The pulse is characterized by the ideal control pulse, the coherent
    noise and the lindblad noise. The later two are lists of
    noisy evolution dynamics.
    Each dynamic element is characterized by four variables:
    `qobj`, `targets`, `tlist` and `coeff`.

    See examples for different construction behavior.

    Parameters
    ----------
    qobj: :class:'qutip.Qobj'
        The Hamiltonian of the ideal pulse.
    targets: list
        target qubits of the ideal pulse
        (or subquantum system of other dimensions).
    tlist: array-like, optional
        `tlist` of the ideal pulse.
        A list of time at which the time-dependent coefficients are applied.
        `tlist` does not have to be equidistant, but must have the same length
        or one element shorter compared to `coeff`. See documentation for
        the parameter `spline_kind`.
    coeff: array-like or bool, optional
        Time-dependent coefficients of the ideal control pulse.
        If an array, the length
        must be the same or one element longer compared to `tlist`.
        See documentation for the parameter `spline_kind`.
        If a bool, the coefficient is a constant 1 or 0.
    spline_kind: str, optional
        Type of the coefficient interpolation:
        "step_func" or "cubic".

        -"step_func":
        The coefficient will be treated as a step function.
        E.g. ``tlist=[0,1,2]`` and ``coeff=[3,2]``, means that the coefficient
        is 3 in t=[0,1) and 2 in t=[2,3). It requires
        ``len(coeff)=len(tlist)-1`` or ``len(coeff)=len(tlist)``, but
        in the second case the last element of `coeff` has no effect.

        -"cubic":
        Use cubic interpolation for the coefficient. It requires
        ``len(coeff)=len(tlist)``
    label: str
        The label (name) of the pulse.

    Attributes
    ----------
    ideal_pulse: :class:`qutip.qip.pulse._EvoElement`
        The ideal dynamic of the control pulse.
    coherent_noise: list of :class:`qutip.qip.pulse._EvoElement`
        The coherent noise caused by the control pulse. Each dynamic element is
        still characterized by a time-dependent Hamiltonian.
    lindblad_noise: list of :class:`qutip.qip.pulse._EvoElement`
        The dissipative noise of the control pulse. Each dynamic element
        will be treated as a (time-dependent) lindblad operator in the
        master equation.
    spline_kind: str
        See parameter `spline_kind`.
    label: str
        See parameter `label`.

    Examples
    --------
    Create a pulse that is turned off

    >>> Pulse(sigmaz(), 0) # doctest: +SKIP
    >>> Pulse(sigmaz(), 0, None, None) # doctest: +SKIP

    Create a time dependent pulse

    >>> tlist = np.array([0., 1., 2., 4.]) # doctest: +SKIP
    >>> coeff = np.array([0.5, 1.2, 0.8]) # doctest: +SKIP
    >>> spline_kind = "step_func" # doctest: +SKIP
    >>> Pulse(sigmaz(), 0, tlist=tlist, coeff=coeff, spline_kind="step_func") # doctest: +SKIP

    Create a time independent pulse

    >>> Pulse(sigmaz(), 0, coeff=True) # doctest: +SKIP

    Create a constant pulse with time range

    >>> Pulse(sigmaz(), 0, tlist=tlist, coeff=True) # doctest: +SKIP

    Create an dummy Pulse (H=0)

    >>> Pulse(None, None) # doctest: +SKIP

    """
    def __init__(self, qobj, targets, tlist=None, coeff=None,
                 spline_kind=None, label=""):
        self.spline_kind = spline_kind
        self.ideal_pulse = _EvoElement(qobj, targets, tlist, coeff)
        self.coherent_noise = []
        self.lindblad_noise = []
        self.label = label

    @property
    def qobj(self):
        """
        See parameter `qobj`.
        """
        return self.ideal_pulse.qobj

    @qobj.setter
    def qobj(self, x):
        self.ideal_pulse.qobj = x

    @property
    def targets(self):
        """
        See parameter `targets`.
        """
        return self.ideal_pulse.targets

    @targets.setter
    def targets(self, x):
        self.ideal_pulse.targets = x

    @property
    def tlist(self):
        """
        See parameter `tlist`
        """
        return self.ideal_pulse.tlist

    @tlist.setter
    def tlist(self, x):
        self.ideal_pulse.tlist = x

    @property
    def coeff(self):
        """
        See parameter `coeff`.
        """
        return self.ideal_pulse.coeff

    @coeff.setter
    def coeff(self, x):
        self.ideal_pulse.coeff = x

    def add_coherent_noise(self, qobj, targets, tlist=None, coeff=None):
        """
        Add a new (time-dependent) Hamiltonian to the coherent noise.

        Parameters
        ----------
        qobj: :class:'qutip.Qobj'
            The Hamiltonian of the pulse.
        targets: list
            target qubits of the pulse
            (or subquantum system of other dimensions).
        tlist: array-like, optional
            A list of time at which the time-dependent coefficients are
            applied.
            `tlist` does not have to be equidistant, but must have the same
            length
            or one element shorter compared to `coeff`. See documentation for
            the parameter `spline_kind` of :class:`qutip.qip.Pulse`.
        coeff: array-like or bool, optional
            Time-dependent coefficients of the pulse noise.
            If an array, the length
            must be the same or one element longer compared to `tlist`.
            See documentation for
            the parameter `spline_kind` of :class:`qutip.qip.Pulse`.
            If a bool, the coefficient is a constant 1 or 0.
        """
        self.coherent_noise.append(_EvoElement(qobj, targets, tlist, coeff))

    def add_lindblad_noise(self, qobj, targets, tlist=None, coeff=None):
        """
        Add a new (time-dependent) lindblad noise to the coherent noise.

        Parameters
        ----------
        qobj: :class:'qutip.Qobj'
            The collapse operator of the lindblad noise.
        targets: list
            target qubits of the collapse operator
            (or subquantum system of other dimensions).
        tlist: array-like, optional
            A list of time at which the time-dependent coefficients are
            applied.
            `tlist` does not have to be equidistant, but must have the same
            length
            or one element shorter compared to `coeff`.
            See documentation for
            the parameter `spline_kind` of :class:`qutip.qip.Pulse`.
        coeff: array-like or bool, optional
            Time-dependent coefficients of the pulse noise.
            If an array, the length
            must be the same or one element longer compared to `tlist`.
            See documentation for
            the parameter `spline_kind` of :class:`qutip.qip.Pulse`.
            If a bool, the coefficient is a constant 1 or 0.
        """
        self.lindblad_noise.append(_EvoElement(qobj, targets, tlist, coeff))

    def get_ideal_qobj(self, dims):
        """
        Get the Hamiltonian of the ideal pulse.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        qobj: :class:`qutip.Qobj`
            The Hamiltonian of the ideal pulse.
        """
        return self.ideal_pulse.get_qobj(dims)

    def get_ideal_qobjevo(self, dims):
        """
        Get a `QobjEvo` representation of the ideal evolution.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        ideal_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the ideal evolution.
        """
        return self.ideal_pulse.get_qobjevo(self.spline_kind, dims)

    def get_noisy_qobjevo(self, dims):
        """
        Get the `QobjEvo` representation of the noisy evolution. The result
        can be used directly as input for the qutip solvers.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        noisy_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the ideal evolution and coherent noise.
        c_ops: list of :class:`qutip.QobjEvo`
            A list of (time-dependent) lindbald operators.
        """
        ideal_qu = self.get_ideal_qobjevo(dims)
        noise_qu_list = [noise.get_qobjevo(self.spline_kind, dims)
                         for noise in self.coherent_noise]
        qu = _merge_qobjevo([ideal_qu] + noise_qu_list)
        c_ops = [noise.get_qobjevo(self.spline_kind, dims)
                 for noise in self.lindblad_noise]
        full_tlist = self.get_full_tlist()
        qu = _merge_qobjevo([qu], full_tlist)
        for i, c_op in enumerate(c_ops):
            c_ops[i] = _merge_qobjevo([c_op], full_tlist)
        return qu, c_ops

    def get_full_tlist(self):
        """
        Return the full tlist of the pulses and noise.
        It means that if different `tlist`s are present, they will be merged
        to one with all time points stored in a sorted array.

        Returns
        -------
        full_tlist: array-like 1d
            The full time sequence for the nosiy evolution.
        """
        # TODO add test
        all_tlists = []
        all_tlists.append(self.ideal_pulse.tlist)
        for pulse in self.coherent_noise:
            all_tlists.append(pulse.tlist)
        for c_op in self.lindblad_noise:
            all_tlists.append(c_op.tlist)
        all_tlists = [tlist for tlist in all_tlists if tlist is not None]
        if not all_tlists:
            return None
        full_tlist = np.unique(np.sort(np.hstack(all_tlists)))
        return full_tlist

    def print_info(self):
        """
        Print the information of the pulse, including the ideal dynamics,
        the coherent noise and the lindblad noise.
        """
        print("-----------------------------------"
              "-----------------------------------")
        if self.label is not None:
            print("Pulse label:", self.label)
        print("The pulse contains: {} coherent noise elements and {} "
              "Lindblad noise elements.".format(
                  len(self.coherent_noise), len(self.lindblad_noise)))
        print()
        print("Ideal pulse:")
        print(self.ideal_pulse)
        if self.coherent_noise:
            print()
            print("Coherent noise:")
            for ele in self.coherent_noise:
                print(ele)
        if self.lindblad_noise:
            print()
            print("Lindblad noise:")
            for ele in self.lindblad_noise:
                print(ele)
        print("-----------------------------------"
              "-----------------------------------")


class Drift():
    """
    The time independent drift Hamiltonian. Usually its the intrinsic
    evolution of the quantum system that can not be tuned.

    Parameters
    ----------
    qobj: :class:`qutip.Qobj` or list of :class:`qutip.Qobj`, optional
        The drift Hamiltonians.

    Attributes
    ----------
    qobj: list of :class:`qutip.Qobj`
        A list of the the drift Hamiltonians.
    """
    def __init__(self, qobj=None):
        if qobj is None:
            self.drift_hamiltonians = []
        elif isinstance(qobj, list):
            self.drift_hamiltonians = qobj
        else:
            self.drift_hamiltonians = [qobj]

    def add_drift(self, qobj, targets):
        """
        Add a Hamiltonian to the drift.

        Parameters
        ----------
        qobj: :class:'qutip.Qobj'
            The collapse operator of the lindblad noise.
        targets: list
            target qubits of the collapse operator
            (or subquantum system of other dimensions).
        """
        self.drift_hamiltonians.append(_EvoElement(qobj, targets))

    def get_ideal_qobjevo(self, dims):
        """
        Get the QobjEvo representation of the drift Hamiltonian.

        Parameters
        ----------
        dims: int or list
            Dimension of the system.
            If int, we assume it is the number of qubits in the system.
            If list, it is the dimension of the component systems.

        Returns
        -------
        ideal_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the drift evolution.
        """
        if not self.drift_hamiltonians:
            self.drift_hamiltonians = [_EvoElement(None, None)]
        qu_list = [QobjEvo(evo.get_qobj(dims)) for evo in self.drift_hamiltonians]
        return _merge_qobjevo(qu_list)

    def get_noisy_qobjevo(self, dims):
        """
        Same as the `get_ideal_qobjevo` method. There is no additional noise
        for the drift evolution.

        Returns
        -------
        noisy_evo: :class:`qutip.QobjEvo`
            A `QobjEvo` representing the ideal evolution and coherent noise.
        c_ops: list of :class:`qutip.QobjEvo`
            Always an empty list for Drift
        """
        return self.get_ideal_qobjevo(dims), []


def _find_common_tlist(qobjevo_list):
    """
    Find the common `tlist` of a list of :class:`qutip.QobjEvo`.
    """
    all_tlists = [qu.tlist for qu in qobjevo_list
                  if isinstance(qu, QobjEvo) and qu.tlist is not None]
    if not all_tlists:
        return None
    full_tlist = np.unique(np.sort(np.hstack(all_tlists)))
    return full_tlist

########################################################################
# These functions are moved here from qutip.qip.device.processor.py
########################################################################


def _merge_qobjevo(qobjevo_list, full_tlist=None):
    """
    Combine a list of `:class:qutip.QobjEvo` into one,
    different tlist will be merged.
    """
    # TODO This method can be eventually integrated into QobjEvo, for
    # which a more thorough test is required

    # no qobjevo
    if not qobjevo_list:
        raise ValueError("qobjevo_list is empty.")

    if full_tlist is None:
        full_tlist = _find_common_tlist(qobjevo_list)
    spline_types_num = set()
    args = {}
    for qu in qobjevo_list:
        if isinstance(qu, QobjEvo):
            try:
                spline_types_num.add(qu.args["_step_func_coeff"])
            except Exception:
                pass
            args.update(qu.args)
    if len(spline_types_num) > 1:
        raise ValueError("Cannot merge Qobjevo with different spline kinds.")

    for i, qobjevo in enumerate(qobjevo_list):
        if isinstance(qobjevo, Qobj):
            qobjevo_list[i] = QobjEvo(qobjevo)
            qobjevo = qobjevo_list[i]
        for j, ele in enumerate(qobjevo.ops):
            if isinstance(ele.coeff, np.ndarray):
                new_coeff = _fill_coeff(
                    ele.coeff, qobjevo.tlist, full_tlist, args)
                qobjevo_list[i].ops[j].coeff = new_coeff
        qobjevo_list[i].tlist = full_tlist

    qobjevo = sum(qobjevo_list)
    return qobjevo


def _fill_coeff(old_coeffs, old_tlist, full_tlist, args=None):
    """
    Make a step function coefficients compatible with a longer `tlist` by
    filling the empty slot with the nearest left value.

    The returned `coeff` always have the same size as the `tlist`.
    If `step_func`, the last element is 0.
    """
    if args is None:
        args = {}
    if "_step_func_coeff" in args and args["_step_func_coeff"]:
        if len(old_coeffs) == len(old_tlist) - 1:
            old_coeffs = np.concatenate([old_coeffs, [0]])
        new_n = len(full_tlist)
        old_ind = 0  # index for old coeffs and tlist
        new_coeff = np.zeros(new_n)
        for new_ind in range(new_n):
            t = full_tlist[new_ind]
            if t < old_tlist[0]:
                new_coeff[new_ind] = 0.
                continue
            if t > old_tlist[-1]:
                new_coeff[new_ind] = 0.
                continue
            if old_tlist[old_ind+1] <= t:
                old_ind += 1
            new_coeff[new_ind] = old_coeffs[old_ind]
    else:
        sp = CubicSpline(old_tlist, old_coeffs)
        new_coeff = sp(full_tlist)
        new_coeff *= (full_tlist <= old_tlist[-1])
        new_coeff *= (full_tlist >= old_tlist[0])
    return new_coeff
