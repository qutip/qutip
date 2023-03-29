# -*- coding: utf-8 -*-
# @author: Arne L. Grimsmo
# @email1: arne.grimsmo@gmail.com
# @organization: University of Sherbrooke

"""
This module contains an implementation of the non-Markovian transfer tensor
method (TTM), introduced in [1].

[1] Javier Cerrillo and Jianshu Cao, Phys. Rev. Lett 112, 110401 (2014)
"""

import numpy as np


from qutip import spre, vector_to_operator, operator_to_vector
from qutip.core import data as _data
from ..solver import Result, SolverOptions


def ttmsolve(dynmaps, state0, times, e_ops=[], learningtimes=None,
             options=None):
    """
    Solve time-evolution using the Transfer Tensor Method, based on a set of
    precomputed dynamical maps.

    Parameters
    ----------
    dynmaps : list of :class:`qutip.Qobj`
        List of precomputed dynamical maps (superoperators),
        or a callback function that returns the
        superoperator at a given time.

    state0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : array_like
        list of times :math:`t_n` at which to compute :math:`\\rho(t_n)`.
        Must be uniformily spaced.

    e_ops : list of :class:`qutip.Qobj` / callback function
        single operator or list of operators for which to evaluate
        expectation values.

    options : dictionary
        Dictionary of options for the solver.

        - store_final_state : bool
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, None
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - normalize_output : bool
          Normalize output state to hide ODE numerical errors.

    Returns
    -------
    output: :class:`qutip.solver.Result`
        An instance of the class :class:`qutip.solver.Result`.
    """

    options = {
        "store_final_state": False,
        "store_states": None,
        "normalize_output": "ket",
        "threshold": 0.0,
        "num_learning": 0,
    }.update(options)

    if TEST times

    if callable(dynmaps):
        if not options["num_learning"]:
            raise ValueError(...)
        dynmaps = [dynmaps(t) for t in times[:options["num_learning"]]]

    start = time.time()
    tensors, diff = _generatetensors(dynmaps, options["threshold"])
    end = time.time()

    stats = {
        "preparation time": end - start,
        "run time": 0.0,
        "ttmconvergence": diff,
        "num_tensor": len(tensor)
    }

    if state0.isoper:
        # vectorize density matrix
        rho0vec = operator_to_vector(state0)
        restore = vector_to_operator
    else:
        # state0 might be a super in which case we should not vectorize
        rho0vec = state0
        restore = lambda state: state

    K = len(tensors)
    start = time.time()
    results = Result(e_ops, options, solver="ttmsolve", stats=stats)
    states = [rho0vec]
    results.add(times[0], state0)
    for n in range(1, len(times)):
        # Set current state
        state = 0
        for j in range(1, min(K, n + 1)):
            tmp = tensors[j] * states[n - j]
            state = tmp + state
        # Append state to all states
        states.append(state)
        results.add(times[n], restore(state))
    end = time.time()
    stats["run time"] = end - start

    return results


def _generatetensors(dynmaps, threshold):
    r"""
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
        :class:`qutip.nonmarkov.transfertensor.TTMSolverOptions`.

    Returns
    -------
    Tlist: list of :class:`qutip.Qobj.`
        A list of transfer tensors :math:`T_1,\dots,T_K`
    """
    Tlist = []
    diff = [0.0]
    for n in range(Kmax):
        T = dynmaps[n]
        for m in range(1, n):
            T -= Tlist[n - m] * dynmaps[m]
        Tlist.append(T)
        if n > 1:
            diff.append((Tlist[-1] - Tlist[-2]).norm())
            if diff[-1] < threshold:
                # Below threshold for truncation
                break
    return Tlist, diff
