from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline

from qutip.qobjevo import QobjEvo
from qutip.qobj import Qobj
from qutip.qip.gates import expand_operator
from qutip.operators import identity


def _get_qobjevo_help(pulse_ele, N, spline_kind, dims):
    if dims is not None:
        # Sometimes N is the number of qubits exposed, but there is additional 
        # hidden dimension such as the resonator. len(dims) can be diffrent 
        # from N in this case.
        N = len(dims)
    if pulse_ele.op is None:
        if dims is not None:
            d = dims[0]
        else:
            d = 2
        return QobjEvo(expand_operator(identity(d), N, 0, dims))

    if pulse_ele.tlist is not None and pulse_ele.coeff is None:
        pulse_ele.coeff = np.ones(len(pulse_ele.tlist))
    mat = expand_operator(pulse_ele.op, N, pulse_ele.targets, dims)

    if pulse_ele.tlist is None:
        qu = QobjEvo(mat)
    else:
        if spline_kind == "step_func":
            args = {"_step_func_coeff": True}
            if len(pulse_ele.coeff) == len(pulse_ele.tlist) - 1:
                pulse_ele.coeff = np.concatenate([pulse_ele.coeff, [0.]])
        elif spline_kind == "cubic":
            args = {"_step_func_coeff": False}
        else:
            args = {}
        qu = QobjEvo([mat, pulse_ele.coeff], tlist=pulse_ele.tlist, args=args)
    return qu


def _get_qobjevo(pulse_ele, N, spline_kind, dims=None):
    try:
        return _get_qobjevo_help(pulse_ele, N, spline_kind, dims=dims)
    except Exception as err:
        print("The Evolution element went wrong was\n {}".format(str(pulse_ele)))
        raise(err)


class _EvoElement():
    def __init__(self, op, targets, tlist=None, coeff=None):
        self.op = op
        self.targets = targets
        self.tlist = tlist
        self.coeff = coeff

    def __str__(self):
        return str({"op": self.op,
                    "targets": self.targets,
                    "tlist": self.tlist,
                    "coeff": self.coeff
                    })


class Pulse():
    def __init__(self, op, targets, tlist=None, coeff=None, spline_kind=None):
        self.spline_kind = spline_kind
        self.ideal_pulse = _EvoElement(op, targets, tlist, coeff)
        self.coherent_noise = []
        self.lindblad_noise = []

    @property
    def op(self):
        return self.ideal_pulse.op

    @op.setter
    def op(self, x):
        self.ideal_pulse.op = x

    @property
    def targets(self):
        return self.ideal_pulse.targets

    @targets.setter
    def targets(self, x):
        self.ideal_pulse.targets = x

    @property
    def tlist(self):
        return self.ideal_pulse.tlist

    @tlist.setter
    def tlist(self, x):
        self.ideal_pulse.tlist = x

    @property
    def coeff(self):
        return self.ideal_pulse.coeff

    @coeff.setter
    def coeff(self, x):
        self.ideal_pulse.coeff = x

    def add_coherent_noise(self, op, targets, tlist=None, coeff=None):
        self.coherent_noise.append(_EvoElement(op, targets, tlist, coeff))

    def add_lindblad_noise(self, op, targets, tlist=None, coeff=None):
        self.lindblad_noise.append(_EvoElement(op, targets, tlist, coeff))

    def get_ideal_evo(self, N, dims=None):
        return _get_qobjevo(self.ideal_pulse, N, self.spline_kind, dims)
    
    def get_full_evo(self, N, dims=None):
        ideal_qu = self.get_ideal_evo(N, dims)
        noise_qu_list = [_get_qobjevo(noise, N, self.spline_kind, dims) for noise in self.coherent_noise]
        qu = _merge_qobjevo([ideal_qu] + noise_qu_list)
        c_ops = [_get_qobjevo(noise, N, self.spline_kind, dims) for noise in self.lindblad_noise]
        return qu, c_ops

    def get_full_ham(self, N, dims=None):
        return expand_operator(self.ideal_pulse.op, N, self.ideal_pulse.targets, dims)

    def print_info(self):
        print("The pulse contains: {} coherent noise and {} "
              "Lindblad noise.".format(
                  len(self.coherent_noise), len(self.lindblad_noise)))
        print()
        print("Pulse Element:")
        print("Ideal pulse:")
        print(self.ideal_pulse)
        print()
        print("Coherent noise:")
        for ele in self.coherent_noise:
            print(ele)
        print()
        print("Lindblad noise:")
        for ele in self.lindblad_noise:
            print(ele)
        print()
        

class Drift():
    def __init__(self):
        self.drift_hams = []

    def add_ham(self, op, targets):
        self.drift_hams.append(_EvoElement(op, targets))

    def get_ideal_evo(self, N, dims=None):
        if not self.drift_hams:
            self.drift_hams = [Pulse(None, None)]
        qu_list = [_get_qobjevo(evo, N, None, dims) for evo in self.drift_hams]
        return _merge_qobjevo(qu_list)

    def get_full_evo(self, N, dims=None):
        return self.get_ideal_evo(N, dims), []


def _merge_qobjevo(qobjevo_list):
    """
    Combine a list of `:class:qutip.QobjEvo` into one,
    different tlist will be merged.
    """
    # TODO This method can be eventually integrated into QobjEvo, for
    # which a more thorough test is required

    # no qobjevo
    if not qobjevo_list:
        raise ValueError("qobjevo_list is empty.")

    spline_types_num = set()
    args = {}
    for qu in qobjevo_list:
        if isinstance(qu, QobjEvo):
            try:
                spline_types_num.add(qu.args["_step_func_coeff"])
            except:
                pass
            args.update(qu.args)
    if len(spline_types_num) > 1:
        raise ValueError("Cannot merge Qobjevo with different spline kinds.")

    all_tlists = [qu.tlist for qu in qobjevo_list if qu.tlist is not None]
    # all tlists are None
    if not all_tlists:
        return sum(qobjevo_list)
    new_tlist = np.unique(np.sort(np.hstack(all_tlists)))
    for i, qobjevo in enumerate(qobjevo_list):
        H_list = qobjevo.to_list()
        for j, H in enumerate(H_list):
            # cte part or not array_like coeffs
            if isinstance(H, Qobj) or (not isinstance(H[1], np.ndarray)):
                continue
            op, coeffs = H
            new_coeff = _fill_coeff(
                coeffs, qobjevo.tlist, new_tlist, args)
            H_list[j] = [op, new_coeff]
        # create a new qobjevo with the old arguments
        qobjevo_list[i] = QobjEvo(
            H_list, tlist=new_tlist, args=args)

    qobjevo = sum(qobjevo_list)
    qobjevo = _merge_id_evo(qobjevo)
    return qobjevo


def _fill_coeff(old_coeffs, old_tlist, new_tlist, args=None):
    """
    Make a step function coefficients compatible with a longer `tlist` by
    filling the empty slot with the nearest left value.
    """
    if args is None:
        args = {}
    if "_step_func_coeff" in args and args["_step_func_coeff"]:
        if len(old_coeffs) == len(old_tlist) - 1:
            old_coeffs = np.concatenate([old_coeffs, [0]])
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
            new_coeff[new_ind] = old_coeffs[old_ind]
    else:
        sp = CubicSpline(old_tlist, old_coeffs)
        new_coeff = sp(new_tlist)
        new_coeff *= (new_tlist <= old_tlist[-1]) * (new_tlist >= old_tlist[0])
    return new_coeff


def _merge_id_evo(qobjevo):
    """
    Merge identical Hamiltonians in the :class":`qutip.QobjEvo`.
    coeffs must all have the same length
    """
    H_list = qobjevo.to_list()
    new_H_list = []
    op_list = []
    coeff_list = []
    for H in H_list:  # H = [op, coeff]
        # cte part or not array_like coeffs
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
    return QobjEvo(new_H_list, tlist=qobjevo.tlist, args=qobjevo.args)
