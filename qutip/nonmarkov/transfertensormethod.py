"""
This module contains an implementation of the non-Markovian transfer tensor
method (TTM), introduced in [1].

[1] Javier Cerrillo and Jianshu Cao, Phys. Rev. Lett 112, 110401 (2014)
"""

import numpy as np


from qutip import Options, spre, vector_to_operator
from qutip.solver import Result
from qutip.expect import expect_rho_vec


class TTMSolverOptions:
    """Class of options for

    Attributes
    ----------
    """

    def __init__(self, E=None, times=[], learningtimes=[],
                 thres=0.0, options=None):

        if options is None:
            options = Options()

        self.E = E
        self.times = times
        self.learningtimes = learningtimes
        self.thres = thres
        self.store_states = options.store_states


def ttmsolve(E, rho0, times, e_ops=[], learningtimes=None, tensors=None, 
             diff=None, **kwargs):
    """
    Parameters
    ----------

    times : *list* / *array*
        list of times :math:`t_n` at which to compute :math:`\rho(t_n)`.

    learningtimes : *list* / *array*
        list of times :math:`t_k` for which we have knowledge of the dynamical
        maps :math:`E(t_k)`.

    Returns
    -------
    """

    opt = TTMSolverOptions(E=E, times=times, learningtimes=learningtimes,
                           **kwargs)

    output = Result()
    e_sops_data = []

    if callable(e_ops):
        n_expt_op = 0
        expt_callback = True

    else:
        try:
            tmp = e_ops[:]
            del tmp

            n_expt_op = len(e_ops)
            expt_callback = False

            if n_expt_op == 0:
                # fall back on storing states
                opt.store_states = True

            for op in e_ops:
                e_sops_data.append(spre(op).data)
                if op.isherm and rho0.isherm:
                    output.expect.append(np.zeros(len(times)))
                else:
                    output.expect.append(np.zeros(len(times), dtype=complex))
        except TypeError:
            raise TypeError("Argument 'e_ops' should be a callable or" +
                            "list-like.")

    if tensors is None:
        tensors, diff = _generatetensors(E, learningtimes, opt=opt)

    K = len(tensors)
    states = [rho0]
    for n in range(1, len(times)):
        states.append(None)
        for k in range(n):
            if n-k < K:
                states[-1] += tensors[n-k]*states[k]
    for i, r in enumerate(states):
        if opt.store_states or expt_callback:
            if r.type == 'operator-ket':
                states[i] = vector_to_operator(r)
            else:
                states[i] = r
            if expt_callback:
                # use callback method
                e_ops(times[i], states[i])
        for m in range(n_expt_op):
            if output.expect[m].dtype == complex:
                output.expect[m][i] = expect_rho_vec(e_sops_data[m],r, 0)
            else:
                output.expect[m][i] = expect_rho_vec(e_sops_data[m],r, 1)


    output.solver = "ttmsolve"
    output.times = times

    output.ttmconvergence = diff

    if opt.store_states:
        output.states = states

    return output


def _generatetensors(E, learningtimes=None, **kwargs):
    """
    Generate the tensors :math:`T_1,\dots,T_K` from the dynamical maps
    :math:`E(t_k)`.

    A stationary process is assumed, i.e., :math:`T_{n,k} = T_{n-k}`.

    Parameters
    ----------

    E : list of :class:`qutip.Qobj`
        List of precomputed dynamical maps (superoperators) at the times
        specified in `learningtimes`, or a callback function that returns the
        superoperator at a given time.

    learningtimes : *list* / *array*
        list of times :math:`t_k` to use if argument `E` is a callback
        function.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.nonmarkov.ttm.TTMSolverOptions`.

    Returns
    -------

    Tlist: list of :class:`qutip.Qobj.`
        A list of transfer tensors :math:`T_1,\dots,T_K`
    """

    # Determine if E is callable or list-like
    if callable(E):
        if learningtimes is None:
            raise TypeError("Argument 'learnintimes' required when 'E' is a" +
                            "callback function.")

        def Efunc(n): return E(learningtimes[n])
        Kmax = len(learningtimes)
    else:
        try:
            tmp = E[:]
            del tmp

            def Efunc(n): return E[n]
            Kmax = len(E)
        except TypeError:
            raise TypeError("Argument 'E' should be a callable or list-like.")

    if "opt" not in kwargs:
        opt = TTMSolverOptions(E=E, learningtimes=learningtimes, **kwargs)
    else:
        opt = kwargs['opt']

    Tlist = []
    diff = [0.0]
    for n in range(Kmax):
        T = Efunc(n)
        for m in range(1, n):
            T -= Tlist[n-m]*Efunc(m)
        Tlist.append(T)
        if n > 1:
            diff.append((Tlist[-1]-Tlist[-2]).norm())
            if diff[-1]  < opt.thres:
                # Below threshold for truncation
                print('breaking', (Tlist[-1]-Tlist[-2]).norm(), n)
                break
    return Tlist,diff
