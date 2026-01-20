import itertools
import math
import numpy as np

from . import Data, Dense, CSR, Dia
from qutip.core.data._local_matmul import (
    one_mode_matmul_data_dense,
    one_mode_matmul_dual_dense_data,
    n_mode_matmul_data_dense,
    _flatten_view_dense, _unflatten_view_dense
)


__all__ = ["target_mode_matmul", "target_mode_matmul_super"]


def target_mode_matmul_data_dense(
    operator: Data,
    state: Dense,
    /,
    modes: list[int],
    hilbert: list[int],
    new_hilbert: list[int],
    dual: bool = False,
    transpose: bool = False,
    conj: bool = False,
) -> Dense:
    """
    Apply the operator to the target modes.

    Roughtly equivalent to:

        expend_operator(operator, hilbert, modes) @ state

    Parameters
    ----------
    operator: Data
        Operator action on a smaller hilbert space than the state's.

    state: Dense
        State to be acted on.

    hilbert: list[int]
        Hilbert space of the state on the contracted side.

    new_hilbert: list[int]
        Hilbert space of the output state on the contracted side.

    modes: list[int]
        Modes to act on.

    dual: bool, default: False
        When true, the operator is apply from the right.

    transpose: bool, default: False
        Transpose the operator before applying it.

    conj: bool, default: False
        Conjugate the operator before applying it.
    """

    hilbert, modes, new_hilbert = _contract_hilbert_space(
        hilbert, modes, new_hilbert
    )

    if len(modes) == 1 and not dual and not transpose and not conj:
        # one_mode_matmul_... is not always faster than n_mode_matmul_...
        out = one_mode_matmul_data_dense(operator, state, hilbert, modes[0])
    elif len(modes) == 1 and dual and not transpose and not conj:
        # one_mode_matmul_... is not always faster than n_mode_matmul_...
        out = one_mode_matmul_dual_dense_data(state, operator, hilbert, modes[0])
    elif not dual:
        out = n_mode_matmul_data_dense(
            operator, state, hilbert, modes, new_hilbert, transpose, conj
        )
    else:
        # For product from the right, we take the expend the hilbert space to
        # includ the first index as an extra space.
        # dims = [[2, 3], [5, 7]] -> hilbert [6, 5, 7] (C order)
        # then contract with the operator transposed.
        shape = (state.shape[0], math.prod(new_hilbert))
        if state.fortran:
            hilbert = hilbert + [state.shape[0]]
            new_hilbert = new_hilbert + [state.shape[0]]
        else:
            hilbert = [state.shape[0]] + hilbert
            new_hilbert = [state.shape[0]] + new_hilbert
            modes = [mode + 1 for mode in modes]

        flat = _flatten_view_dense(state)
        new = n_mode_matmul_data_dense(
            operator, flat, hilbert, modes, new_hilbert, not transpose, conj
        )
        out = _unflatten_view_dense(new, shape)

    return out


def target_mode_matmul_super_data_dense(
    operator: Data,
    state: Dense,
    /,
    modes: list[int],
    hilbert_left_state: list[int],
    hilbert_right_state: list[int],
    hilbert_left_state_out: list[int],
    hilbert_right_state_out: list[int],
) -> Dense:
    """
    Apply the super-operator to the target modes.

    If operator is ``sprepost(A, B)``, it's roughtly equivalent to

    expend_operator(A, hilbert_left_state, modes)
    @ state
    @ expend_operator(B, hilbert_right_state, modes)
    """
    if not state.fortran:
        N = len(hilbert_left_state)
        modes = [mode + N for mode in modes] + modes
        hilbert = hilbert_left_state + hilbert_right_state
        hilbert_out = hilbert_left_state_out + hilbert_right_state_out

    else:
        N = len(hilbert_right_state)
        modes = modes + [mode + N for mode in modes]
        hilbert = hilbert_right_state + hilbert_left_state
        hilbert_out = hilbert_right_state_out + hilbert_left_state_out

    shape = (np.prod(hilbert_left_state_out), np.prod(hilbert_right_state_out))

    hilbert, modes, hilbert_out = _contract_hilbert_space(
        hilbert, modes, hilbert_out
    )

    state_flat = _flatten_view_dense(state)
    out_flat = n_mode_matmul_data_dense(
        operator, state_flat, hilbert, modes, hilbert_out
    )
    return _unflatten_view_dense(out_flat, shape)


def _contract_hilbert_space(
    hilbert: list[int],
    modes: list[int],
    hilbert_out: list[int]
) -> tuple[list[int], list[int], list[int]]:
    """
    Contracts a Hilbert space by grouping adjacent modes.

    - Modes that are adjacent in the Hilbert space and are both "inactive" are
      grouped.
    - Modes that are adjacent in the Hilbert space and are both "active" are
      only grouped if they are also adjacent in the input `modes` list.

    Parameters
    ----------
    hilbert: list of int
        A list of integers representing the dimensions of each mode.

    modes:
        A list of unique integers for the active mode indices.

    hilbert_out:
        The output Hilbert space dimensions.

    Return
    ------
        A tuple containing the new Hilbert space, the new modes, and the new
        output Hilbert space.
    """

    if len(hilbert) != len(hilbert_out):
        raise ValueError(
            "Input 'hilbert' and 'hilbert_out' must have the same length."
        )

    if len(modes) != len(set(modes)):
        raise ValueError("Repeated values in 'modes' are not supported.")

    group_ids = [0]
    current_group_id = 0

    # Group hilbert spaces
    # Consecutive not in mode with will have the same group
    # Consicutive mode's space will have the same group
    for i in range(1, len(hilbert)):
        curr_in_modes = i in modes
        prev_in_modes = i-1 in modes
        if curr_in_modes != prev_in_modes:
            current_group_id += 1
        elif curr_in_modes:
            pos_curr = modes.index(i)
            pos_prev = modes.index(i-1)
            if pos_prev + 1 != pos_curr:
                current_group_id += 1

        group_ids.append(current_group_id)

    new_hilbert = []
    new_modes = []
    new_hilbert_out = []

    # Group the original hilbert indices by their newly assigned group ID
    for _, group_indices_iter in itertools.groupby(
        range(len(hilbert)),
        key=lambda i: group_ids[i]
    ):
        group_indices = list(group_indices_iter)

        contracted_dim = math.prod(hilbert[i] for i in group_indices)
        new_hilbert.append(contracted_dim)

        contracted_out_dim = math.prod(hilbert_out[i] for i in group_indices)
        new_hilbert_out.append(contracted_out_dim)

    for mode in modes:
        new_pos = group_ids[mode]
        if not new_modes or new_pos != new_modes[-1]:
            new_modes.append(new_pos)

    return new_hilbert, new_modes, new_hilbert_out


from .dispatch import Dispatcher as _Dispatcher

target_mode_matmul = _Dispatcher(
    target_mode_matmul_data_dense, name='target_mode_matmul',
    inputs=('operator', 'state',), out=True
)
target_mode_matmul.add_specialisations([
    (CSR, Dense, Dense, target_mode_matmul_data_dense),
    (Dia, Dense, Dense, target_mode_matmul_data_dense),
    (Dense, Dense, Dense, target_mode_matmul_data_dense),
], _defer=True)

target_mode_matmul_super = _Dispatcher(
    target_mode_matmul_super_data_dense, name='target_mode_matmul_super',
    inputs=('operator', 'state',), out=True
)
target_mode_matmul_super.add_specialisations([
    (CSR, Dense, Dense, target_mode_matmul_super_data_dense),
    (Dia, Dense, Dense, target_mode_matmul_super_data_dense),
    (Dense, Dense, Dense, target_mode_matmul_super_data_dense),
], _defer=True)

del _Dispatcher
