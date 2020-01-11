from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline

from qutip.qobjevo import QobjEvo
from qutip.qobj import Qobj
from qutip.qip.gates import expand_operator
from qutip.operators import identity



class _EvoElement():
    def __init__(self, op, targets, tlist=None, coeff=None):
        self.op = op
        self.targets = targets
        self.tlist = tlist
        self.coeff = coeff

    def get_qobj(self, dims):
        if isinstance(dims, (int, np.integer)):
            dims = [2] * dims
        if self.op is None:
            op = identity(dims[0]) * 0.
            targets = 0
        else: 
            op = self.op
            targets = self.targets
        return expand_operator(op, len(dims), targets, dims)

    def get_qobjevo_help(self, spline_kind, dims):
        mat = self.get_qobj(dims)
        if self.tlist is None:
            qu = QobjEvo(mat)
        elif self.tlist is not None and self.coeff is None:
            qu = QobjEvo(mat, tlist=self.tlist)
        else:
            if spline_kind == "step_func":
                args = {"_step_func_coeff": True}
                if len(self.coeff) == len(self.tlist) - 1:
                    self.coeff = np.concatenate([self.coeff, [0.]])
            elif spline_kind == "cubic":
                args = {"_step_func_coeff": False}
            else:
                args = {}
            qu = QobjEvo([mat, self.coeff], tlist=self.tlist, args=args)
        return qu

    def get_qobjevo(self, spline_kind, dims):
        try:
            return self.get_qobjevo_help(spline_kind, dims=dims)
        except Exception as err:
            print("The Evolution element went wrong was\n {}".format(str(self)))
            raise(err)

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

    def get_ideal_evo(self, dims):
        return self.ideal_pulse.get_qobjevo(self.spline_kind, dims)

    def get_full_evo(self, dims):
        ideal_qu = self.get_ideal_evo(dims)
        noise_qu_list = [noise.get_qobjevo(self.spline_kind, dims) for noise in self.coherent_noise]
        qu = _merge_qobjevo([ideal_qu] + noise_qu_list)
        c_ops = [noise.get_qobjevo(self.spline_kind, dims) for noise in self.lindblad_noise]
        full_tlist = _find_common_tlist(c_ops + [qu])
        qu = _merge_qobjevo([qu], full_tlist)
        for i, c_op in enumerate(c_ops):
            c_ops[i] = _merge_qobjevo([c_op], full_tlist)
        return qu, c_ops

    def get_ideal_qobj(self, dims):
        return self.ideal_pulse.get_qobj(dims)

    def print_info(self):
        print("-----------------------------------"
              "-----------------------------------")
        print("The pulse contains: {} coherent noise elements and {} "
              "Lindblad noise elements.".format(
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
        print("-----------------------------------"
              "-----------------------------------")


class Drift():
    def __init__(self):
        self.drift_hams = []

    def add_ham(self, op, targets):
        self.drift_hams.append(_EvoElement(op, targets))

    def get_ideal_evo(self, dims):
        if not self.drift_hams:
            self.drift_hams = [_EvoElement(None, None)]
        qu_list = [evo.get_qobjevo(None, dims) for evo in self.drift_hams]
        return _merge_qobjevo(qu_list)

    def get_full_evo(self, dims):
        return self.get_ideal_evo(dims), []


def _find_common_tlist(qobjevo_list):
    all_tlists = [qu.tlist for qu in qobjevo_list if isinstance(qu, QobjEvo) and qu.tlist is not None]
    if not all_tlists:
        return None
    full_tlist = np.unique(np.sort(np.hstack(all_tlists)))
    return full_tlist


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
            except:
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
                new_coeff = _fill_coeff(ele.coeff, qobjevo.tlist, full_tlist, args)
                qobjevo_list[i].ops[j].coeff = new_coeff
        qobjevo_list[i].tlist = full_tlist

    qobjevo = sum(qobjevo_list)
    return qobjevo


def _fill_coeff(old_coeffs, old_tlist, full_tlist, args=None):
    """
    Make a step function coefficients compatible with a longer `tlist` by
    filling the empty slot with the nearest left value.
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
            if old_tlist[old_ind+1] == t:
                old_ind += 1
            new_coeff[new_ind] = old_coeffs[old_ind]
    else:
        sp = CubicSpline(old_tlist, old_coeffs)
        new_coeff = sp(full_tlist)
        new_coeff *= (full_tlist <= old_tlist[-1]) * (full_tlist >= old_tlist[0])
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
