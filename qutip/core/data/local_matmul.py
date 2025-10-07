#cython: language_level=3

import numpy as np
import qutip as qt

import itertools
import math

from qutip.core.data.local_matmul_mix import N_mode_data_dense
from qutip.core.data.local_matmul_one import one_mode_matmul_data_dense

#TODO:
# - One head function
# - Reusable interne functions + factory
# - What about jax etc?
# - merge left and right?

def _flatten_view_dense(state):
    return Dense(
        state.as_ndarray().ravel("A"),
        shape=(state.shape[0]*state.shape[1], 1),
        copy=False
    )


def _unflatten_view_dense(out, shape, order):
    return Dense(
        out.as_ndarray().reshape(*shape, order=order),
        shape=shape,
        copy=False
    )


def local_super_apply(oper, state, modes):
    hilbert_left_state = state.dims[0]
    hilbert_right_state = state.dims[1]
    N = len(hilbert_left_state)
    assert N == len(hilbert_right_state)
    assert all(mode < N for mode in modes)
    assert oper._dims.issuper
    assert len(oper.dims[1][0]) == len(modes)

    hilbert_left_state_out = [
        size if i not in modes else oper.dims[0][0][modes.index(i)]
        for i, size in enumerate(hilbert_left_state)
    ]

    hilbert_right_state_out = [
        size if i not in modes else oper.dims[0][1][modes.index(i)]
        for i, size in enumerate(hilbert_right_state)
    ]

    if not state.data.fortran:
        order="C"
        modes = [mode + N for mode in modes] + modes
        hilbert = hilbert_left_state + hilbert_right_state
        hilbert_out = hilbert_left_state_out + hilbert_right_state_out
        dims_out = [hilbert_out[:N], hilbert_out[N:]]

    else:
        order="F"
        modes = modes + [mode + N for mode in modes]
        hilbert = hilbert_right_state + hilbert_left_state
        hilbert_out = hilbert_right_state_out + hilbert_left_state_out
        dims_out = [hilbert_out[N:], hilbert_out[:N]]

    shape = (np.prod(dims_out[0]), np.prod(dims_out[1]))

    state_flat = _flatten_view_dense(state.data)
    out_flat = N_mode_data_dense(oper.data, state_flat, hilbert, modes, hilbert_out)
    out_data = _unflatten_view_dense(out_flat, shape, order)

    return qt.Qobj(out_data, dims_out, copy=False)


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
            current_group_id =+ 1
        elif curr_in_modes:
            pos_curr = modes.index(i)
            pos_prev = modes.index(i-1)
            if pos_prev + 1 != pos_curr:
                current_group_id =+ 1

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
