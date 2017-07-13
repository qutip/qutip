# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2015 and later, Arne L. Grimsmo
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

# @author: Arne L. Grimsmo
# @email1: arne.grimsmo@gmail.com
# @organization: University of Sherbrooke

"""
This module contains an implementation of the non-Markovian transfer tensor
method (TTM), introduced in [1].

[1] Javier Cerrillo and Jianshu Cao, Phys. Rev. Lett 112, 110401 (2014)
"""

import numpy as np


from qutip import (Options, spre, vector_to_operator, operator_to_vector,
                   ket2dm, isket)
from qutip.solver import Result
from qutip.expect import expect_rho_vec


class TTMSolverOptions:
    """Class of options for the Transfer Tensor Method solver.

    Attributes
    ----------
    dynmaps : list of :class:`qutip.Qobj`
        List of precomputed dynamical maps (superoperators),
        or a callback function that returns the
        superoperator at a given time.

    times : array_like
        List of times :math:`t_n` at which to calculate :math:`\\rho(t_n)`

    learningtimes : array_like
        List of times :math:`t_k` to use as learning times if argument
        `dynmaps` is a callback function.

    thres : float
        Threshold for halting. Halts if  :math:`||T_{n}-T_{n-1}||` is below
        treshold.

    options : :class:`qutip.solver.Options`
        Generic solver options.
    """

    def __init__(self, dynmaps=None, times=[], learningtimes=[],
                 thres=0.0, options=None):

        if options is None:
            options = Options()

        self.dynmaps = dynmaps
        self.times = times
        self.learningtimes = learningtimes
        self.thres = thres
        self.store_states = options.store_states


def ttmsolve(dynmaps, rho0, times, e_ops=[], learningtimes=None, tensors=None,
             **kwargs):
    """
    Solve time-evolution using the Transfer Tensor Method, based on a set of
    precomputed dynamical maps.

    Parameters
    ----------
    dynmaps : list of :class:`qutip.Qobj`
        List of precomputed dynamical maps (superoperators),
        or a callback function that returns the
        superoperator at a given time.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : array_like
        list of times :math:`t_n` at which to compute :math:`\\rho(t_n)`.
        Must be uniformily spaced.

    e_ops : list of :class:`qutip.Qobj` / callback function
        single operator or list of operators for which to evaluate
        expectation values.

    learningtimes : array_like
        list of times :math:`t_k` for which we have knowledge of the dynamical
        maps :math:`E(t_k)`.

    tensors : array_like
        optional list of precomputed tensors :math:`T_k`

    kwargs : dictionary
        Optional keyword arguments. See
        :class:`qutip.nonmarkov.ttm.TTMSolverOptions`.

    Returns
    -------
    output: :class:`qutip.solver.Result`
        An instance of the class :class:`qutip.solver.Result`.
    """

    opt = TTMSolverOptions(dynmaps=dynmaps, times=times,
                           learningtimes=learningtimes, **kwargs)

    diff = None

    if isket(rho0):
        rho0 = ket2dm(rho0)

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
        tensors, diff = _generatetensors(dynmaps, learningtimes, opt=opt)

    if rho0.isoper:
        # vectorize density matrix
        rho0vec = operator_to_vector(rho0)
    else:
        # rho0 might be a super in which case we should not vectorize
        rho0vec = rho0

    K = len(tensors)
    states = [rho0vec]
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
                output.expect[m][i] = expect_rho_vec(e_sops_data[m], r, 0)
            else:
                output.expect[m][i] = expect_rho_vec(e_sops_data[m], r, 1)

    output.solver = "ttmsolve"
    output.times = times

    output.ttmconvergence = diff

    if opt.store_states:
        output.states = states

    return output


def _generatetensors(dynmaps, learningtimes=None, **kwargs):
    """
    Generate the tensors :math:`T_1,\dots,T_K` from the dynamical maps
    :math:`E(t_k)`.

    A stationary process is assumed, i.e., :math:`T_{n,k} = T_{n-k}`.

    Parameters
    ----------
    dynmaps : list of :class:`qutip.Qobj`
        List of precomputed dynamical maps (superoperators) at the times
        specified in `learningtimes`, or a callback function that returns the
        superoperator at a given time.

    learningtimes : array_like
        list of times :math:`t_k` to use if argument `dynmaps` is a callback
        function.

    kwargs : dictionary
        Optional keyword arguments. See
        :class:`qutip.nonmarkov.ttm.TTMSolverOptions`.

    Returns
    -------
    Tlist: list of :class:`qutip.Qobj.`
        A list of transfer tensors :math:`T_1,\dots,T_K`
    """

    # Determine if dynmaps is callable or list-like
    if callable(dynmaps):
        if learningtimes is None:
            raise TypeError("Argument 'learnintimes' required when 'dynmaps'" +
                            "is a callback function.")

        def dynmapfunc(n): return dynmaps(learningtimes[n])
        Kmax = len(learningtimes)
    else:
        try:
            tmp = dynmaps[:]
            del tmp

            def dynmapfunc(n): return dynmaps[n]
            Kmax = len(dynmaps)
        except TypeError:
            raise TypeError("Argument 'dynmaps' should be a callable or" +
                            "list-like.")

    if "opt" not in kwargs:
        opt = TTMSolverOptions(dynmaps=dynmaps, learningtimes=learningtimes,
                               **kwargs)
    else:
        opt = kwargs['opt']

    Tlist = []
    diff = [0.0]
    for n in range(Kmax):
        T = dynmapfunc(n)
        for m in range(1, n):
            T -= Tlist[n-m]*dynmapfunc(m)
        Tlist.append(T)
        if n > 1:
            diff.append((Tlist[-1]-Tlist[-2]).norm())
            if diff[-1] < opt.thres:
                # Below threshold for truncation
                print('breaking', (Tlist[-1]-Tlist[-2]).norm(), n)
                break
    return Tlist, diff
