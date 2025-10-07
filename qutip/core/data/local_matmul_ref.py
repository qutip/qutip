# Function to uses in tests (TODO: move later)


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

def ref_one_mode(oper, state, mode):
    hilbert = state.dims[0]
    if hilbert[:mode]:
        oper = qt.qeye(hilbert[:mode]) & oper
    if hilbert[mode+1:]:
        oper = oper & qt.qeye(hilbert[mode+1:])
    return oper @ state


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


#

def wrap_super(pres, posts, state, modes):
    pre = qt.tensor(pres)
    post = qt.tensor(posts)
    return local_super_apply(qt.sprepost(pre, post), state, modes)


def wrap_one_mode(oper, state, mode):
    hilbert = state.dims[0]
    new_hilbert = hilbert.copy()
    new_hilbert[mode] = oper.dims[0][0]

    return qt.Qobj(
        one_mode_matmul_data_dense(oper.data, state.data, hilbert, mode),
        dims=[new_hilbert, state.dims[1]]
    )


def wrap_N_mode(opers, state, modes):
    hilbert = state.dims[0]
    new_hilbert = hilbert.copy()
    for mode, oper in zip(modes, opers):
        new_hilbert[mode] = oper.dims[0][0]

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
