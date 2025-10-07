from .qobj import Qobj
from .core.local_matmul import target_mode_matmul, target_mode_matmul_super


__all__ = ["local_matmul"]


def local_matmul(
    operator: Qobj,
    state: Qobj,
    modes: int | list[int],
    dual: bool = False
) -> Qobj:
    """
    Apply the operator on the state at the desired mode(s).

    For example applying a cnot gate on qubits 2 and 4 is:

        local_matmul(cnot, state, [2, 4])

    To apply a single operator on multiple modes require multiple calls or
    extending the operator:

        local_matmul(sigmax, state, [2, 4, 5])  # Error

        local_matmul(tensor([sigmax]*3), state, [2, 4, 5])  # Ok

    For super operator, each mode is applyed on both side as one and the state
    must be a density matrix (operator-ket are not supported).
    For example:

        local_matmul(sprepost(sigmax, sigmax), state, [0])

    will apply sigmax on the first qubit on both side.

    Parameters
    ----------
    operator: Qobj
        Operator action on a smaller hilbert space than the state's.

    state: Qobj
        State to be acted on.

    modes: int | list[int]
        Modes to act on.

        The hilbert space sizes must match at these modes:

            operator.dims[1][i] == state.dims[0][modes[i]] for all i

    dual: bool, default: False
        When true, the operator is apply from the right.
        Ignored for super operator.

    Returns
    -------
    Updated state as a Qobj.
    """
    if operator.issuper:
        _local_super_apply(operator, state, modes)
    if not dual:
        hilbert_state = state.dims[0]
        hilbert_oper_in = operator.dims[1]
        hilbert_oper_out = operator.dims[0]
    else:
        hilbert_state = state.dims[1]
        hilbert_oper_in = operator.dims[0]
        hilbert_oper_out = operator.dims[1]

    N = len(hilbert_state)
    if not isintance(modes, list):
        modes = [modes]

    #TODO: Exotic dims could be supported if shared
    assert operator._dims._require_pure_dims("local_matmul")
    assert state._dims._require_pure_dims("local_matmul")
    assert all(mode < N for mode in modes)
    assert all(
        hilbert_oper_in[i] == hilbert_state[mode]
        for i, mode in enumerate(modes)
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
    hilbert_left_state = state.dims[0]
    hilbert_right_state = state.dims[1]
    N = len(hilbert_left_state)
    assert oper._dims._require_pure_dims("local_matmul")
    assert state._dims._require_pure_dims("local_matmul")
    assert N == len(hilbert_right_state)
    assert all(mode < N for mode in modes)
    assert oper._dims.issuper
    assert len(oper.dims[1][0]) == len(modes)
    assert all(
        (
            oper.dims[1][0][i] == state.dims[0][mode] and
            oper.dims[1][1][i] == state.dims[1][mode]
        )
        for i, mode in enumerate(modes)
    )

    hilbert_left_state_out = [
        size if i not in modes else oper.dims[0][0][modes.index(i)]
        for i, size in enumerate(hilbert_left_state)
    ]

    hilbert_right_state_out = [
        size if i not in modes else oper.dims[0][1][modes.index(i)]
        for i, size in enumerate(hilbert_right_state)
    ]

    out_data = target_mode_matmul_super(
        oper.data,
        state.data,
        modes,
        hilbert_left_state, hilbert_right_state,
        hilbert_left_state_out, hilbert_right_state_out,
    )
    dims_out = [hilbert_left_state_out, hilbert_right_state_out]

    return qt.Qobj(out_data, dims_out, copy=False)
