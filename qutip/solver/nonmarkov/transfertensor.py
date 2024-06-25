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
import time
from qutip import spre, vector_to_operator, operator_to_vector, Result


def ttmsolve(dynmaps, state0, times, e_ops=(), num_learning=0, options=None):
    """
    Expand time-evolution using the Transfer Tensor Method [1]_, based on a set
    of precomputed dynamical maps.

    Parameters
    ----------
    dynmaps : list of :class:`.Qobj`, callable
        List of precomputed dynamical maps (superoperators) for the first times
        of ``times`` or a callback function that returns the superoperator at a
        given time.

    state0 : :class:`.Qobj`
        Initial density matrix or state vector (ket).

    times : array_like
        List of times :math:`t_n` at which to compute results.
        Must be uniformily spaced.

    e_ops : :class:`.Qobj`, callable, or list, optional
        Single operator or list of operators for which to evaluate
        expectation values or callable or list of callable.
        Callable signature must be, `f(t: float, state: Qobj)`.
        See :func:`expect` for more detail of operator expectation.

    num_learning : int, default: 0
        Number of times used to construct the dynmaps operators when
        ``dynmaps`` is a callable.

    options : dictionary, optional
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
        - threshold : float
          Threshold for halting. Halts if  :math:`||T_{n}-T_{n-1}||` is below
          treshold.

    Returns
    -------
    output: :class:`.Result`
        An instance of the class :class:`.Result`.

    .. [1] Javier Cerrillo and Jianshu Cao, Phys. Rev. Lett 112, 110401 (2014)
    """

    opt = {
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        "threshold": 0.0,
        "num_learning": 0,
    }
    if options:
        opt.update(options)

    if not np.allclose(np.diff(times), times[1] - times[0]):
        raise ValueError("The time should be uniformily distributed.")

    if callable(dynmaps):
        if num_learning <= 0:
            raise ValueError(
                "When dynmaps is a callable, options['num_learning'] must be "
                "the number of dynamical maps to compute."
            )
        dynmaps = [dynmaps(t) for t in times[:num_learning]]

    if (
        not dynmaps
        or not dynmaps[0].issuper
        or not all(dmap.dims == dynmaps[0].dims for dmap in dynmaps)
    ):
        raise ValueError("`dynmaps` entries must be super operators.")

    start = time.time()
    tensors, diff = _generatetensors(dynmaps, opt["threshold"])
    end = time.time()

    stats = {
        "preparation time": end - start,
        "run time": 0.0,
        "ttmconvergence": diff,
        "num_tensor": len(tensors),
    }
    if state0.isket:
        state0 = state0.proj()

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
    results = Result(e_ops, opt, solver="ttmsolve", stats=stats)
    states = [rho0vec]
    results.add(times[0], state0)

    for n in range(1, len(times)):
        # Set current state
        state = 0
        for j in range(1, min(K, n + 1)):
            tmp = tensors[j] @ states[n - j]
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
    dynmaps : list of :class:`.Qobj`
        List of precomputed dynamical maps (superoperators) at the times
        specified in `learningtimes`.

    threshold : float
        Threshold for halting. Halts if  :math:`||T_{n}-T_{n-1}||` is below
        treshold.

    Returns
    -------
    Tensors, diffs: list of :class:`.Qobj.`
        A list of transfer tensors :math:`T_1,\dots,T_K`
    """
    Tensors = []
    diff = [0.0]
    for n in range(len(dynmaps)):
        T = dynmaps[n]
        for m in range(1, n):
            T -= Tensors[n - m] @ dynmaps[m]
        Tensors.append(T)
        if n > 1:
            diff.append((Tensors[-1] - Tensors[-2]).norm())
            if diff[-1] < threshold:
                # Below threshold for truncation
                break
    return Tensors, diff
