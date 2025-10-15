from .qobj import Qobj
from .data.local_matmul import target_mode_matmul, target_mode_matmul_super


__all__ = ["local_matmul"]


def local_matmul(
    operator: Qobj,
    state: Qobj,
    modes: int | list[int],
    dual: bool = False
) -> Qobj:
    """
    Applies an operator to specific modes of a quantum state or operator.

    This function performs a tensor product operation where a smaller
    `operator` acts only on the specified `modes` of a larger `state`.

    Super operator are applied on the same mode on both side of a density
    matrix at once. The state must be a density matrix
    (operator-ket are not supported).

    Parameters
    ----------
    operator: Qobj
        The operator to apply. Its Hilbert space dimensions must correspond to
        the dimensions of the state on the targeted modes.

    state: Qobj
        The quantum state to be acted upon.

    modes: int | list[int]
        The indices specifying the modes on which the operator should act.

    dual: bool, default: False
        If True, the operator is applied from the right (state @ operator).
        Ignored for superoperators.

    Returns
    -------
    Qobj
        Updated state as a Qobj.

    Examples
    --------
    Applying a CNOT gate to a multi-qubit state:
    >>> local_matmul(cnot(), state, modes=[2, 4])

    An operator's dimension must match the number of modes. To apply the same
    gate to multiple modes, it must be tensored first:
    >>> # This will fail:
    >>> # local_matmul(sigmax(), state, modes=[0, 1])
    >>> # This is correct:
    >>> op = tensor(sigmax(), sigmax())
    >>> local_matmul(op, state, modes=[0, 1])

    Applying a superoperator to a density matrix:
    >>> super_operator = sprepost(a, a.dag()) - ...
    >>> local_matmul(super_operator, rho, modes=[0])
    """
    if operator.issuper:
        return _local_super_apply(operator, state, modes)
    if not dual:
        hilbert_state = state.dims[0]
        hilbert_oper_in = operator.dims[1]
        hilbert_oper_out = operator.dims[0]
    else:
        hilbert_state = state.dims[1]
        hilbert_oper_in = operator.dims[0]
        hilbert_oper_out = operator.dims[1]

    N = len(hilbert_state)
    if not isinstance(modes, list):
        modes = [modes]

    #TODO: Exotic dims could be supported if shared
    operator._dims._require_pure_dims("local_matmul")
    state._dims._require_pure_dims("local_matmul")
    if not all(mode < N for mode in modes):
        raise ValueError(
            "Target mode index is out of bounds for the state's dimensions."
        )
    if not len(operator.dims[1]) == len(modes):
        raise ValueError("Number of modes do not match operator dimensions.")
    if not all(
        hilbert_oper_in[i] == hilbert_state[mode]
        for i, mode in enumerate(modes)
    ):
        raise TypeError(
            "Operator's input dimensions do not match the state's "
            "dimensions on the specified modes."
        )
    new_hilbert = list(hilbert_state)
    for i, mode in enumerate(modes):
        new_hilbert[mode] = hilbert_oper_out[i]

    out_data = target_mode_matmul(
        operator.data,
        state.data,
        modes,
        hilbert_state,
        new_hilbert,
        dual,
    )
    out_dims = (
        [new_hilbert, state.dims[1]] if not dual
        else [state.dims[0], new_hilbert]
    )

    return Qobj(out_data, dims=out_dims, copy=False)


def _local_super_apply(super_operator, state, modes):
    """
    Apply a local super operator on a state.
    """
    if not isinstance(modes, list):
        modes = [modes]
    hilbert_left_state = state.dims[0]
    hilbert_right_state = state.dims[1]
    N = len(hilbert_left_state)
    super_operator._dims._require_pure_dims("local_matmul")
    state._dims._require_pure_dims("local_matmul")
    if not N == len(hilbert_right_state):
        raise TypeError(
            "The state must have the same number of subsystems "
            "on its left and right dimensions."
        )
    if not all(mode < N for mode in modes):
        raise ValueError("Target mode index is out of bounds for the state.")
    if not super_operator._dims.issuper:
        raise TypeError("The operator must be a superoperator.")
    if not len(super_operator.dims[1][0]) == len(modes):
        raise ValueError("Number of modes do not match operator dimensions.")
    if not all(
        (
            super_operator.dims[1][0][i] == state.dims[0][mode] and
            super_operator.dims[1][1][i] == state.dims[1][mode]
        )
        for i, mode in enumerate(modes)
    ):
        raise TypeError(
            "Superoperator and state dimensions do not "
            "match on the target modes."
        )

    hilbert_left_state_out = list(hilbert_left_state)
    for i, mode in enumerate(modes):
        hilbert_left_state_out[mode] = super_operator.dims[0][0][i]

    hilbert_right_state_out = list(hilbert_right_state)
    for i, mode in enumerate(modes):
        hilbert_right_state_out[mode] = super_operator.dims[0][1][i]

    out_data = target_mode_matmul_super(
        super_operator.data,
        state.data,
        modes,
        hilbert_left_state, hilbert_right_state,
        hilbert_left_state_out, hilbert_right_state_out,
    )
    dims_out = [hilbert_left_state_out, hilbert_right_state_out]

    return Qobj(out_data, dims_out, copy=False)
