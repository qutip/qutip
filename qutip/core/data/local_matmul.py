#cython: language_level=3

import numpy as np
import qutip as qt

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
    from qutip.core.data.local_matmul_mix import N_mode_data_dense

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


def ref_super(pres, posts, state, modes):
    hilbert = state.dims[0]
    if 0 not in modes:
        pre_oper = qt.qeye(hilbert[0])
        post_oper = qt.qeye(hilbert[0])
    else:
        for mode, pre, post in zip(modes, pres, posts):
            if mode == 0:
                pre_oper = pre
                post_oper = post
                break

    for i in range(1, len(hilbert)):
        if i not in modes:
            pre_oper = pre_oper & qt.qeye(hilbert[i])
            post_oper = post_oper & qt.qeye(hilbert[i])
        else:
            for mode, pre, post in zip(modes, pres, posts):
                if mode == i:
                    pre_oper = pre_oper & pre
                    post_oper = post_oper & post
                    break

    return pre_oper @ state @ post_oper


def wrap_super(pres, posts, state, modes):
    pre = qt.tensor(pres)
    post = qt.tensor(posts)
    return local_super_apply(qt.sprepost(pre, post), state, modes)

#
def ref_one_mode(oper, state, mode):
    hilbert = state.dims[0]
    if hilbert[:mode]:
        oper = qt.qeye(hilbert[:mode]) & oper
    if hilbert[mode+1:]:
        oper = oper & qt.qeye(hilbert[mode+1:])
    return oper @ state

def wrap_one_mode(oper, state, mode):
    hilbert = state.dims[0]
    new_hilbert = hilbert.copy()
    new_hilbert[mode] = oper.dims[0][0]
    from qutip.core.data.local_matmul_one import one_mode_matmul_data_dense

    return qt.Qobj(
        one_mode_matmul_data_dense(oper.data, state.data, hilbert, mode),
        dims=[new_hilbert, state.dims[1]]
    )

def ref_N_mode(opers, state, modes):
    hilbert = state.dims[0]
    if 0 not in modes:
        out_oper = qt.qeye(hilbert[0])
    else:
        for mode, oper in zip(modes, opers):
            if mode == 0:
                out_oper = oper
                break

    for i in range(1, len(hilbert)):
        if i not in modes:
            out_oper = out_oper & qt.qeye(hilbert[i])
        else:
            for mode, oper in zip(modes, opers):
                if mode == i:
                    out_oper = out_oper & oper
                    break

    return out_oper @ state

def wrap_N_mode(opers, state, modes):
    hilbert = state.dims[0]
    new_hilbert = hilbert.copy()
    for mode, oper in zip(modes, opers):
        new_hilbert[mode] = oper.dims[0][0]
    from qutip.core.data.local_matmul_mix import N_mode_data_dense

    oper = qt.tensor(opers)
    oper = oper.to(opers[0].dtype)
    return qt.Qobj(
        N_mode_data_dense(oper.data, state.data, hilbert, modes, new_hilbert),
        dims=[new_hilbert, state.dims[1]]
    )


def wrap_N_inverse_mode(opers, state, modes):
    hilbert = state.dims[1]
    new_hilbert = hilbert.copy()
    for mode, oper in zip(modes, opers):
        new_hilbert[mode] = oper.dims[1][0]
    from qutip.core.data.local_matmul_mix import N_mode_data_dense

    oper = qt.tensor(opers)
    oper = oper.to(opers[0].dtype)
    shape = (state.shape[0], np.prod(new_hilbert))
    dims_out = [state.dims[0], new_hilbert]

    if state.data.fortran:
        hilbert = hilbert + state.dims[0]
        new_hilbert = new_hilbert  + state.dims[0]

    else:
        hilbert = state.dims[0] + hilbert
        new_hilbert = state.dims[0] + new_hilbert
        N = len(state.dims[0])
        modes = [mode + N for mode in modes]

    flat = _flatten_view_dense(state.data)
    new = N_mode_data_dense(oper.data, flat, hilbert, modes, new_hilbert, "T")
    restored = _unflatten_view_dense(new, shape, "F" if state.data.fortran else "C")

    return qt.Qobj(restored, dims=dims_out)


def ref_N_inverse_mode(opers, state, modes):
    hilbert = state.dims[0]
    if 0 not in modes:
        out_oper = qt.qeye(hilbert[0])
    else:
        for mode, oper in zip(modes, opers):
            if mode == 0:
                out_oper = oper
                break

    for i in range(1, len(hilbert)):
        if i not in modes:
            out_oper = out_oper & qt.qeye(hilbert[i])
        else:
            for mode, oper in zip(modes, opers):
                if mode == i:
                    out_oper = out_oper & oper
                    break

    return state @ out_oper


def clean_hilbert(hilbert, modes, hilbert_out):
    sets = [0]
    idx_prev = -2

    for i in range(1, len(hilbert)):
        if i in modes:
            idx = modes.index(i)
            if idx_prev + 1 == idx:
                sets.append(sets[-1])
            else:
                sets.append(sets[-1] + 1)
            idx_prev = idx

        elif i-1 not in modes:
            sets.append(sets[-1])
            idx_prev = -2
        else:
            sets.append(sets[-1] + 1)
            idx_prev = -2

    hilbert_new = [1]
    hilbert_out_new = [1]
    modes_new = []
    i = 0
    prev_group = 0
    prev_mode = 0
    while i < len(sets):
        if sets[i] == prev_group:
            hilbert_new[-1] *= hilbert[i]
            hilbert_out_new[-1] *= hilbert_out[i]
            if i in modes:
                prev_mode += 1
        else:
            hilbert_new.append( hilbert[i] )
            hilbert_out_new.append( hilbert_out[i] )

            if i in modes:
                modes_new.append( sets[modes[prev_mode]] )
                prev_mode += 1

        prev_group = sets[i]
        i += 1

    return hilbert_new, modes_new, hilbert_out_new
