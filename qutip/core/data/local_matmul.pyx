#cython: language_level=3

from qutip.core.data cimport Data, Dense, dense, Dia
from qutip.core.data.matmul cimport imatmul_data_dense, matmul_dense
from qutip.core.data.matmul import matmul, matmul_dense_dia_dense
from qutip.core.data.add cimport iadd_dense
import numpy as np


#TODO:
# - One head function
# - Reusable interne functions + factory
# - What about jax etc?
# - merge left and right?

def _flatten_view_dense(state):
    return qt.data.Dense(
        state.as_ndarray().ravel(),
        shape=(state.shape[0]*state.shape[1], 1),
        copy=False
    )

def _unflatten_view_dense(out, order):
    N = int( (out.shape[0] * out.shape[1])**0.5 )
    return qt.data.Dense(
        out.as_ndarray().reshape(N, N, order=order),
        shape=(N, N),
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
        # modes = [mode + N for mode in modes] + modes
        hilbert = hilbert_right_state + hilbert_left_state
        hilbert_out = hilbert_right_state_out + hilbert_left_state_out
        dims_out = [hilbert_out[N:], hilbert_out[:N]]

    shape = (np.prod(dims_out[0]), np.prod(dims_out[1]))

    print(order)
    print(modes)
    print(hilbert)
    print(hilbert_out)
    print(dims_out)
    state_flat = _flatten_view_dense(state.data)
    out_flat = N_mode_data_dense(oper.data, state_flat, hilbert, modes, hilbert_out)
    out_data = _unflatten_view_dense(out_flat, shape, "F")

    return qt.Qobj(out_data, dims_out, copy=False)


#
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
