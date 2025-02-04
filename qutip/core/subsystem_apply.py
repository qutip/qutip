#
# This code was contributed by Ben Criger. Resemblance to
# partial_transpose is intentional, and meant to enhance legibility.
#

__all__ = ['subsystem_apply']

import itertools

import numpy as np

from . import Qobj, qeye, to_kraus, tensor
from . import data as _data


def subsystem_apply(
    state: Qobj,
    channel: Qobj,
    mask: list[bool],
    reference: bool=False
)-> Qobj:
    """
    Returns the result of applying the propagator `channel` to the
    subsystems indicated in `mask`, which comprise the density operator
    `state`.

    Parameters
    ----------
    state : :class:`.Qobj`
        A density matrix or ket.

    channel : :class:`.Qobj`
        A propagator, either an `oper` or `super`.

    mask : *list* / *array*
        A mask that selects which subsystems should be subjected to the
        channel.

    reference : bool
        Decides whether explicit Kraus map should be used to evaluate action
        of channel.

    Returns
    -------
    rho_out: :class:`.Qobj`
        A density matrix with the selected subsystems transformed
        according to the specified channel.
    """
    # TODO: Include sparse/dense methods a la partial_transpose.
    if not (state.isket or state.isoper):
        raise ValueError("input state must be a ket or oper")
    if not (channel.issuper or channel.isoper):
        raise ValueError("input channel must be a super or oper")
    # Since there's only one channel, all affected subsystems must have
    # the same dimensions:
    aff_subs_dim_ar = np.transpose(np.array(state.dims))[np.array(mask)]
    if any((aff_subs_dim_ar[j] != aff_subs_dim_ar[0]).any()
           for j in range(len(aff_subs_dim_ar))):
        raise ValueError("affected subsystems must have the same dimension")
    # If the map is on the Hilbert space, it must have the same dimension
    # as the affected subsystem. If it is on the Liouville space, it must
    # exist on a space as large as the square of the Hilbert dimension.
    required_shape = ([x**2 for x in aff_subs_dim_ar[0]] if channel.issuper
                      else aff_subs_dim_ar[0])
    if tuple(required_shape) != channel.shape:
        raise ValueError(
            "superoperator dimension must be the subsystem dimension squared"
        )
    mask = np.asarray(mask)
    if reference:
        return _subsystem_apply_reference(state, channel, mask)
    if state.isoper:
        return _subsystem_apply_dm(state, channel, mask)
    return _subsystem_apply_ket(state, channel, mask)


def _subsystem_apply_ket(state, channel, mask):
    # TODO Write more efficient code for single-matrix map on pure states
    # TODO Write more efficient code for single-subsystem map . . .
    return _subsystem_apply_dm(state.proj(), channel, mask)


def _subsystem_apply_dm(state, channel, mask):
    """
    Applies a channel to every subsystem indicated by a mask, by
    repeatedly applying the channel to each affected subsystem.
    """
    # Initialize Output Matrix
    rho_out = state
    # checked affected subsystems print arange(len(state.dims[0]))[mask]
    for subsystem in np.arange(len(state.dims[0]))[mask]:
        rho_out = _one_subsystem_apply(rho_out, channel, subsystem)
    return rho_out


def _one_subsystem_apply(state, channel, idx):
    """
    Applies a channel to a state on one subsystem, by breaking it into
    blocks and applying a reduced channel to each block.
    """
    subs_dim_ar = np.array(state.dims).T

    # Calculate number of blocks
    n_blks = 1
    for mul_idx in range(idx):
        n_blks = n_blks * subs_dim_ar[mul_idx][0]
    blk_sz = state.shape[0] // n_blks

    # Apply channel to top subsystem of each block in matrix
    full_data_matrix = state.full()
    for blk_r in range(n_blks):
        for blk_c in range(n_blks):
            # Apply channel to high-level blocks of matrix:
            blk_rx = blk_r * blk_sz
            blk_cx = blk_c * blk_sz

            full_data_matrix[blk_rx : blk_rx+blk_sz, blk_cx : blk_cx+blk_sz] =\
                _block_apply(full_data_matrix[blk_rx : blk_rx+blk_sz,
                                              blk_cx : blk_cx+blk_sz],
                             channel)
    return Qobj(full_data_matrix, dims=state.dims)


def _block_apply(block, channel):
    return (_top_apply_U(block, channel) if channel.isoper
            else _top_apply_S(block, channel))


def _top_apply_U(block, channel):
    """
    Uses scalar-matrix multiplication to efficiently apply a channel to
    the leftmost register in the tensor product, given a unitary matrix
    for a channel.
    """
    split_mat = _block_split(block, *channel.shape)
    temp_split_mat = np.zeros(np.shape(split_mat)).astype(complex)
    for dm_row_idx in range(channel.shape[0]):
        for dm_col_idx in range(channel.shape[1]):
            for op_row_idx in range(channel.shape[0]):
                for op_col_idx in range(channel.shape[1]):
                    temp_split_mat[dm_row_idx][dm_col_idx] += (
                        channel[dm_row_idx, op_col_idx]
                        * channel[dm_col_idx, op_row_idx].conjugate()
                        * split_mat[op_col_idx][op_row_idx]
                    )
    return _block_join(temp_split_mat)


def _top_apply_S(block, channel):
    # If the channel is a super-operator, perform second block decomposition;
    # block-size matches Hilbert space of affected subsystem:
    # FIXME use state shape?
    n_v = int(np.sqrt(channel.shape[0]))
    n_h = int(np.sqrt(channel.shape[1]))
    column = _block_col(block, n_v, n_h)
    chan_mat = channel.full()
    temp_col = np.zeros(np.shape(column)).astype(complex)
    for i, row in enumerate(chan_mat):
        temp_col[i] = sum(s * mat for s, mat in zip(row.T, column))
    return _block_stack(temp_col, n_v, n_h)


def _block_split(mat_in, n_v, n_h):
    """
    Returns a 4D array of matrices, splitting mat_in into
    n_v * n_h square sub-arrays.
    """
    return [np.hsplit(x, n_h) for x in np.vsplit(mat_in, n_v)]


def _block_join(mat_in):
    return np.hstack(np.hstack(mat_in))


def _block_col(mat_in, n_v, n_h):
    """
    Returns a 3D array of matrices, splitting mat_in into
    n_v * n_h square sub-arrays.
    """
    rows, cols = np.shape(mat_in)
    return np.reshape(
        np.array(_block_split(mat_in, n_v, n_h)).transpose(1, 0, 2, 3),
        (n_v * n_h, rows // n_v, cols // n_h)
    )


def _block_stack(arr_in, n_v, n_h):
    """
    Inverse of _block_split
    """
    rs, cs = np.shape(arr_in)[-2:]
    temp = list(map(np.transpose, arr_in))
    # print shape(arr_in)
    temp = np.reshape(temp, (n_v, n_h, rs, cs))
    return np.hstack(np.hstack(temp)).T


def _subsystem_apply_reference(state, channel, mask):
    state = state.proj() if state.isket else state
    if channel.isoper:
        full_oper = tensor([channel if mask_ else qeye(size)
                            for size, mask_ in zip(state.dims[0], mask)])
        return full_oper @ state @ full_oper.dag()
    kraus_list = to_kraus(channel)
    # Kraus operators to be padded with identities:
    k_qubit_kraus_list = itertools.product(kraus_list, repeat=np.sum(mask))
    rho_out = Qobj(_data.csr.zeros(*state.shape), dims=state.dims)
    for operator_iter in k_qubit_kraus_list:
        operator_iter = iter(operator_iter)
        op_iter_list = [next(operator_iter) if mask[j]
                        else qeye(state.dims[0][j])
                        for j in range(len(state.dims[0]))]
        full_oper = tensor(list(map(Qobj, op_iter_list)))
        rho_out = rho_out + full_oper * state * full_oper.dag()
    return rho_out
