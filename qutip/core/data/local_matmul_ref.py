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
