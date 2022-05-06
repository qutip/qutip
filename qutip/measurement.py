"""
Module for measuring quantum objects.
"""

import numpy as np

from . import Qobj, expect, identity, tensor


def _check_qubits_oper(oper, dims=None, targets=None):
    """
    Check if the given operator is valid.
    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        The quantum object to be checked.
    dims : list, optional
        A list of integer for the dimension of each composite system.
        e.g ``[2, 2, 2, 2, 2]`` for 5 qubits system. If None, qubits system
        will be the default.
    targets : int or list of int, optional
        The indices of qubits that are acted on.
    """
    # if operator matches N
    if not isinstance(oper, Qobj) or oper.dims[0] != oper.dims[1]:
        raise ValueError(
            "The operator is not an "
            "Qobj with the same input and output dimensions.")
    # if operator dims matches the target dims
    if dims is not None and targets is not None:
        targ_dims = [dims[t] for t in targets]
        if oper.dims[0] != targ_dims:
            raise ValueError(
                "The operator dims {} do not match "
                "the target dims {}.".format(
                    oper.dims[0], targ_dims))


def _targets_to_list(targets, oper=None, N=None):
    """
    transform targets to a list and check validity.
    Parameters
    ----------
    targets : int or list of int
        The indices of qubits that are acted on.
    oper : :class:`qutip.Qobj`, optional
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    N : int, optional
        The number of qubits in the system.
    """
    # if targets is a list of integer
    if targets is None:
        targets = list(range(len(oper.dims[0])))
    if not hasattr(targets, '__iter__'):
        targets = [targets]
    if not all([isinstance(t, int) for t in targets]):
        raise TypeError(
            "targets should be "
            "an integer or a list of integer")
    # if targets has correct length
    if oper is not None:
        req_num = len(oper.dims[0])
        if len(targets) != req_num:
            raise ValueError(
                "The given operator needs {} "
                "target qutbis, "
                "but {} given.".format(
                    req_num, len(targets)))
    # if targets is smaller than N
    if N is not None:
        if not all([t < N for t in targets]):
            raise ValueError("Targets must be smaller than N={}.".format(N))
    return targets


def _expand_operator(oper, N, targets):
    """
    Expand a qubits operator to one that acts on a N-qubit system.
    Parameters
    ----------
    oper : :class:`qutip.Qobj`
        An operator acts on qubits, the type of the :class:`qutip.Qobj`
        has to be an operator
        and the dimension matches the tensored qubit Hilbert space
        e.g. dims = ``[[2, 2, 2], [2, 2, 2]]``
    N : int
        The number of qubits in the system.
    targets : int or list of int
        The indices of qubits that are acted on.
    Returns
    -------
    expanded_oper : :class:`qutip.Qobj`
        The expanded qubits operator acting on a system with N qubits.
    """
    dims = [2] * N
    targets = _targets_to_list(targets, oper=oper, N=N)
    _check_qubits_oper(oper, dims=dims, targets=targets)

    # Generate the correct order for qubits permutation,
    # eg. if N = 5, targets = [3,0], the order is [1,2,3,0,4].
    # If the operator is cnot,
    # this order means that the 3rd qubit controls the 0th qubit.
    new_order = [0] * N
    for i, t in enumerate(targets):
        new_order[t] = i
    # allocate the rest qutbits (not targets) to the empty
    # position in new_order
    rest_pos = [q for q in list(range(N)) if q not in targets]
    rest_qubits = list(range(len(targets), N))
    for i, ind in enumerate(rest_pos):
        new_order[ind] = rest_qubits[i]
    id_list = [identity(dims[i]) for i in rest_pos]
    return tensor([oper] + id_list).permute(new_order)


def _verify_input(op, state):
    if not isinstance(op, Qobj):
        raise TypeError("op must be a Qobj")
    if not op.isoper:
        raise ValueError("op must be all operators or all kets")
    if not isinstance(state, Qobj):
        raise TypeError("state must be a Qobj")
    if state.isket:
        if op.dims[-1] != state.dims[0]:
            raise ValueError(
                "op and state dims should be compatible when state is a ket")
    elif state.isoper:
        if op.dims != state.dims:
            raise ValueError(
                "op and state dims should match"
                " when state is a density matrix")
    else:
        raise ValueError("state must be a ket or a density matrix")


def _measurement_statistics_povm_ket(state, ops):
    r"""
    Returns measurement statistics (resultant states and probabilities)
    for a measurements specified by a set of positive operator valued
    measurements on a specified ket.

    Parameters
    ----------
    state : :class:`.Qobj` (ket)
            The ket specifying the state to measure.

    ops : list of :class:`.Qobj`
      List of measurement operators :math:`M_i` (specifying a POVM such that
      :math:`E_i = M_i^\dagger M_i`).


    Returns
    -------
    collapsed_states : list of :class:`.Qobj` (kets)
        The collapsed states (kets) obtained after measuring the qubits and
        obtaining the qubit specified by the target in the state specified by
        the index.

    probabilities : list of floats
        The probability of measuring a state in a the state specified by the
        index.
    """
    probabilities = []
    collapsed_states = []

    for i, op in enumerate(ops):
        p = np.absolute((state.dag() * op.dag() * op * state))
        probabilities.append(p)
        if p != 0:
            collapsed_states.append((op * state) / np.sqrt(p))
        else:
            collapsed_states.append(None)

    return collapsed_states, probabilities


def _measurement_statistics_povm_dm(density_mat, ops):
    r"""
    Returns measurement statistics (resultant states and probabilities)
    for a measurements specified by a set of positive operator valued
    measurements on a specified ket or density matrix.

    Parameters
    ----------
    state : :class:`.Qobj` (density matrix)
        The ket or density matrix specifying the state to measure.

    ops : list of :class:`.Qobj`
        List of measurement operators :math:`M_i` (specifying a POVM s.t.
        :mathm:`E_i = M_i^\dagger M_i`)

    Returns
    -------
    collapsed_states : list of :class:`.Qobj`
        The collapsed states (density matrices) obtained after measuring the
        qubits and obtaining the qubit specified by the target in the state
        specified by the index.

    probabilities : list of float
        The probability of measuring a state in a the state specified by the
        index.
    """
    probabilities = []
    collapsed_states = []

    for i, op in enumerate(ops):
        st = op * density_mat * op.dag()
        p = st.tr()
        probabilities.append(p)
        if p != 0:
            collapsed_states.append(st/p)
        else:
            collapsed_states.append(None)

    return collapsed_states, probabilities


def measurement_statistics_povm(state, ops, targets=None):
    r"""
    Returns measurement statistics (resultant states and probabilities) for a
    measurement specified by a set of positive operator valued measurements on
    a specified ket or density matrix.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    ops : list of :class:`.Qobj`
        List of measurement operators :math:`M_i` or kets.  Either:

        1. specifying a POVM s.t. :math:`E_i = M_i^\dagger M_i`
        2. projection operators if ops correspond to
           projectors (s.t. :math:`E_i = M_i^\dagger = M_i`)
        3. kets (transformed to projectors)

    targets : list of ints, optional
              Specifies a list of target "qubit" indices on which to apply
              the measurement.

    Returns
    -------
    collapsed_states : list of :class:`.Qobj`
        The collapsed states obtained after measuring the qubits and obtaining
        the qubit specified by the target in the state specified by the index.

    probabilities : list of floats
        The probability of measuring a state in a the state specified by the
        index.
    """
    if all(map(lambda x: x.isket, ops)):
        ops = [op * op.dag() for op in ops]

    if targets:
        N = int(np.log2(state.shape[0]))
        ops = [_expand_operator(op, N=N, targets=targets) for op in ops]

    for op in ops:
        _verify_input(op, state)

    E = [op.dag() * op for op in ops]

    is_ID = sum(E)
    if not is_ID == identity(is_ID.dims[0]):
        raise ValueError("measurement operators must sum to identity")

    if state.isket:
        return _measurement_statistics_povm_ket(state, ops)
    else:
        return _measurement_statistics_povm_dm(state, ops)


def measurement_statistics_observable(state, op, targets=None):
    """
    Return the measurement eigenvalues, eigenstates (or projectors) and
    measurement probabilities for the given state and measurement operator.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    op : :class:`.Qobj`
        The measurement operator.

    targets : list of ints, optional
        Specifies a list of targets "qubit" indices on which to apply the
        measurement.

    Returns
    -------
    eigenvalues: list of float
        The list of eigenvalues of the measurement operator.

    eigenstates_or_projectors: list of :class:`.Qobj`
        If the state was a ket, return the eigenstates of the measurement
        operator. Otherwise return the projectors onto the eigenstates.

    probabilities: list of float
        The probability of measuring the state as being in the corresponding
        eigenstate (and the measurement result being the corresponding
        eigenvalue).
    """
    if targets:
        N = int(np.log2(state.shape[0]))
        op = _expand_operator(op, N=N, targets=targets)

    _verify_input(op, state)

    eigenvalues, eigenstates = op.eigenstates()
    if state.isket:
        probabilities = [abs(e.overlap(state))**2 for e in eigenstates]
        return eigenvalues, eigenstates, probabilities
    else:
        projectors = [e.proj() for e in eigenstates]
        probabilities = [expect(v, state) for v in projectors]
        return eigenvalues, projectors, probabilities


def measure_observable(state, op, targets=None):
    """
    Perform a measurement specified by an operator on the given state.

    This function simulates the classic quantum measurement described in many
    introductory texts on quantum mechanics. The measurement collapses the
    state to one of the eigenstates of the given operator and the result of the
    measurement is the corresponding eigenvalue.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    op : :class:`.Qobj`
        The measurement operator.

    targets : list of ints, optional
        Specifies a list of target "qubit" indices on which to apply the
        measurement.

    Returns
    -------
    measured_value : float
        The result of the measurement (one of the eigenvalues of op).

    state : :class:`.Qobj`
        The new state (a ket if a ket was given, otherwise a density matrix).

    Examples
    --------

    Measure the z-component of the spin of the spin-up basis state:

    >>> measure_observable(basis(2, 0), sigmaz())
    (1.0, Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[-1.]
     [ 0.]])

    Since the spin-up basis is an eigenstate of sigmaz, this measurement always
    returns 1 as the measurement result (the eigenvalue of the spin-up basis)
    and the original state (up to a global phase).

    Measure the x-component of the spin of the spin-down basis state:

    >>> measure_observable(basis(2, 1), sigmax())
    (-1.0, Quantum object: dims = [[2], [1]], shape = (2, 1), type = ket
    Qobj data =
    [[-0.70710678]
     [ 0.70710678]])

    This measurement returns 1 fifty percent of the time and -1 the other fifty
    percent of the time. The new state returned is the corresponding eigenstate
    of sigmax.

    One may also perform a measurement on a density matrix. Below we perform
    the same measurement as above, but on the density matrix representing the
    pure spin-down state:

    >>> measure_observable(ket2dm(basis(2, 1)), sigmax())
    (-1.0, Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper
    Qobj data =
    [[ 0.5 -0.5]
     [-0.5  0.5]])

    The measurement result is the same, but the new state is returned as a
    density matrix.
    """
    eigenvalues, eigenstates_or_projectors, probabilities = (
        measurement_statistics_observable(state, op, targets))
    i = np.random.choice(range(len(eigenvalues)), p=probabilities)
    if state.isket:
        eigenstates = eigenstates_or_projectors
        state = eigenstates[i]
    else:
        projectors = eigenstates_or_projectors
        state = (projectors[i] * state * projectors[i]) / probabilities[i]
    return eigenvalues[i], state


def measure_povm(state, ops, targets=None):
    r"""
    Perform a measurement specified by list of POVMs.

    This function simulates a POVM measurement. The measurement collapses the
    state to one of the resultant states of the measurement and returns the
    index of the operator corresponding to the collapsed state as well as the
    collapsed state.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    ops : list of :class:`.Qobj`
        List of measurement operators :math:`M_i` or kets.  Either:

        1. specifying a POVM s.t. :math:`E_i = M_i^\dagger M_i`
        2. projection operators if ops correspond to projectors (s.t.
           :math:`E_i = M_i^\dagger = M_i`)
        3. kets (transformed to projectors)

    targets : list of ints, optional
        Specifies a list of target "qubit" indices on which to apply
        the measurement.

    Returns
    -------
    index : float
        The resultant index of the measurement.

    state : :class:`.Qobj`
        The new state (a ket if a ket was given, otherwise a density matrix).
    """
    collapsed_states, probabilities = measurement_statistics_povm(state,
                                                                  ops, targets)
    index = np.random.choice(range(len(collapsed_states)), p=probabilities)
    state = collapsed_states[index]
    return index, state


def measurement_statistics(state, ops, targets=None):
    r"""
    A dispatch method that provides measurement statistics handling both
    observable style measurements and projector style measurements(POVMs and
    PVMs).

    For return signatures, please check:

    - :func:`~measurement_statistics_observable` for observable measurements.
    - :func:`~measurement_statistics_povm` for POVM measurements.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    ops : :class:`.Qobj` or list of :class:`.Qobj`
        - measurement observable (:class:.Qobj); or
        - list of measurement operators :math:`M_i` or kets (list of
          :class:`.Qobj`) Either:

          1. specifying a POVM s.t. :math:`E_i = M_i^\dagger * M_i`
          2. projection operators if ops correspond to projectors (s.t.
             :math:`E_i = M_i^\dagger = M_i`)
          3. kets (transformed to projectors)

    targets : list of ints, optional
        Specifies a list of target "qubit" indices on which to apply the
        measurement.
    """
    if isinstance(ops, list):
        return measurement_statistics_povm(state, ops, targets)
    else:
        return measurement_statistics_observable(state, ops, targets)


def measure(state, ops, targets=None):
    r"""
    A dispatch method that provides measurement results handling both
    observable style measurements and projector style measurements (POVMs and
    PVMs).

    For return signatures, please check:

    - :func:`~measure_observable` for observable measurements.
    - :func:`~measure_povm` for POVM measurements.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    ops : :class:`.Qobj` or list of :class:`.Qobj`
        - measurement observable (:class:`.Qobj`); or
        - list of measurement operators :math:`M_i` or kets (list of
          :class:`.Qobj`) Either:

          1. specifying a POVM s.t. :math:`E_i = M_i^\dagger M_i`
          2. projection operators if ops correspond to projectors (s.t.
             :math:`E_i = M_i^\dagger = M_i`)
          3. kets (transformed to projectors)

    targets : list of ints, optional
        Specifies a list of target "qubit" indices on which to apply the
        measurement.
    """
    if isinstance(ops, list):
        return measure_povm(state, ops, targets)
    else:
        return measure_observable(state, ops, targets)
