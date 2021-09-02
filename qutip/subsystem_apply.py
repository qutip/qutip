#
# This code was contributed by Ben Criger. Resemblance to
# partial_transpose is intentional, and meant to enhance legibility.
#

__all__ = ['subsystem_apply']

from itertools import product

from numpy import transpose, sqrt, arange, array, isreal, zeros, shape
from numpy import hstack, vsplit, hsplit, reshape
from scipy.linalg import eig

from qutip.qobj import Qobj, issuper, isket, isoper
from qutip.states import ket2dm
from qutip.operators import qeye
from qutip.superoperator import vec2mat
from qutip.superop_reps import super_to_choi
from qutip.tensor import tensor


def subsystem_apply(state, channel, mask, reference=False):
    """
    Returns the result of applying the propagator `channel` to the
    subsystems indicated in `mask`, which comprise the density operator
    `state`.

    Parameters
    ----------

    state : :class:`qutip.qobj`
        A density matrix or ket.

    channel : :class:`qutip.qobj`
        A propagator, either an `oper` or `super`.

    mask : *list* / *array*
        A mask that selects which subsystems should be subjected to the
        channel.

    reference : bool
        Decides whether explicit Kraus map should be used to evaluate action
        of channel.

    Returns
    -------

    rho_out: :class:`qutip.qobj`

        A density matrix with the selected subsystems transformed
        according to the specified channel.

    """

    # TODO: Include sparse/dense methods a la partial_transpose.

    # ---Sanity Checks---#

    # state must be a ket or density matrix, channel must be a propagator.
    assert state.type == 'ket' or state.type == 'oper', "Input state must be\
    a ket or oper, given: " + repr(state.type)
    assert channel.type == 'super' or channel.type == 'oper', "Input channel \
    must be a super or oper, given: " + repr(channel.type)

    # Since there's only one channel, all affected subsystems must have
    # the same dimensions:
    aff_subs_dim_ar = transpose(array(state.dims))[array(mask)]

    assert all([(aff_subs_dim_ar[j] == aff_subs_dim_ar[0]).all()
                for j in range(len(aff_subs_dim_ar))]), \
        "Affected subsystems must have the same dimension. Given:" +\
        repr(aff_subs_dim_ar)

    # If the map is on the Hilbert space, it must have the same dimension
    # as the affected subsystem. If it is on the Liouville space, it must
    # exist on a space as large as the square of the Hilbert dimension.
    if issuper(channel):
        required_shape = tuple(map(lambda x: x ** 2, aff_subs_dim_ar[0]))
    else:
        required_shape = aff_subs_dim_ar[0]

    assert array([channel.shape == required_shape]).all(), \
        "Superoperator dimension must be the " + \
        "subsystem dimension squared, given: " \
        + repr(channel.shape)

    # Ensure mask is an array:
    mask = array(mask)

    if reference:
        return _subsystem_apply_reference(state, channel, mask)

    if state.type == 'oper':
        return _subsystem_apply_dm(state, channel, mask)
    else:
        return _subsystem_apply_ket(state, channel, mask)


def _subsystem_apply_ket(state, channel, mask):
    # TODO Write more efficient code for single-matrix map on pure states
    # TODO Write more efficient code for single-subsystem map . . .
    return _subsystem_apply_dm(ket2dm(state), channel, mask)


def _subsystem_apply_dm(state, channel, mask):
    """
    Applies a channel to every subsystem indicated by a mask, by
    repeatedly applying the channel to each affected subsystem.
    """
    # Initialize Output Matrix
    rho_out = state
    # checked affected subsystems print arange(len(state.dims[0]))[mask]
    for subsystem in arange(len(state.dims[0]))[mask]:
        rho_out = _one_subsystem_apply(rho_out, channel, subsystem)
    return rho_out


def _one_subsystem_apply(state, channel, idx):
    """
    Applies a channel to a state on one subsystem, by breaking it into
    blocks and applying a reduced channel to each block.
    """
    subs_dim_ar = array(state.dims).T
    # Calculate number of blocks
    n_blks = 1
    for mul_idx in range(idx):
        # print mul_idx
        # print subs_dim_ar[mul_idx]
        n_blks = n_blks * subs_dim_ar[mul_idx][0]

    blk_sz = state.shape[0] // n_blks
    # Apply channel to top subsystem of each block in matrix
    full_data_matrix = state.full()

    if isreal(full_data_matrix).all():
        full_data_matrix = full_data_matrix.astype(complex)

    for blk_r in range(n_blks):
        for blk_c in range(n_blks):
            # Apply channel to high-level blocks of matrix:
            blk_rx = blk_r * blk_sz
            blk_cx = blk_c * blk_sz

            full_data_matrix[blk_rx:blk_rx + blk_sz, blk_cx:blk_cx + blk_sz] =\
                _block_apply(
                    full_data_matrix[
                        blk_rx:blk_rx + blk_sz, blk_cx:blk_cx + blk_sz],
                    channel)

    return Qobj(dims=state.dims, inpt=full_data_matrix)


def _block_apply(block, channel):
    if isoper(channel):
        block = _top_apply_U(block, channel)
    elif issuper(channel):
        block = _top_apply_S(block, channel)
    return block


def _top_apply_U(block, channel):
    """
    Uses scalar-matrix multiplication to efficiently apply a channel to
    the leftmost register in the tensor product, given a unitary matrix
    for a channel.
    """
    if isreal(block).all():
        block = block.astype(complex)

    split_mat = _block_split(block, *channel.shape)
    temp_split_mat = zeros(shape(split_mat)).astype(complex)

    for dm_row_idx in range(channel.shape[0]):
        for dm_col_idx in range(channel.shape[1]):
            for op_row_idx in range(channel.shape[0]):
                for op_col_idx in range(channel.shape[1]):
                    temp_split_mat[dm_row_idx][dm_col_idx] =\
                        temp_split_mat[dm_row_idx][dm_col_idx] +\
                        channel[dm_row_idx, op_col_idx] *\
                        channel[dm_col_idx, op_row_idx].conjugate() *\
                        split_mat[op_col_idx][op_row_idx]
    return _block_join(temp_split_mat)


def _top_apply_S(block, channel):
    # If the channel is a super-operator,
    # perform second block decomposition; block-size
    # matches Hilbert space of affected subsystem:
    # FIXME use state shape?
    n_v =  int(sqrt(channel.shape[0]))
    n_h =  int(sqrt(channel.shape[1]))
    column = _block_col(block, n_v, n_h)
    chan_mat = channel.full()
    temp_col = zeros(shape(column)).astype(complex)
    for row_idx in range(len(chan_mat)):
        row = chan_mat[row_idx]
        temp_col[row_idx] = sum([s * mat
                                 for s, mat in zip(row, column)])
    return _block_stack(temp_col, n_v, n_h)


def _block_split(mat_in, n_v, n_h):
    """
    Returns a 4D array of matrices, splitting mat_in into
    n_v * n_h square sub-arrays.
    """
    return list(map(lambda x: hsplit(x, n_h), vsplit(mat_in, n_v)))


def _block_join(mat_in):
    return hstack(hstack(mat_in))


def _block_col(mat_in, n_v, n_h):
    """
    Returns a 3D array of matrices, splitting mat_in into
    n_v * n_h square sub-arrays.
    """
    rows, cols = shape(mat_in)
    return reshape(
        array(_block_split(mat_in, n_v, n_h)).transpose(1, 0, 2, 3),
        (n_v * n_h, rows // n_v, cols // n_h))


def _block_stack(arr_in, n_v, n_h):
    """
    Inverse of _block_split
    """
    rs, cs = shape(arr_in)[-2:]
    temp = list(map(transpose, arr_in))
    # print shape(arr_in)
    temp = reshape(temp, (n_v, n_h, rs, cs))
    return hstack(hstack(temp)).T


def _subsystem_apply_reference(state, channel, mask):
    if isket(state):
        state = ket2dm(state)

    if isoper(channel):
        full_oper = tensor([channel if mask[j]
                            else qeye(state.dims[0][j])
                            for j in range(len(state.dims[0]))])
        return full_oper * state * full_oper.dag()
    else:
        # Go to Choi, then Kraus
        choi_matrix = super_to_choi(channel)
        vals, vecs = eig(choi_matrix.full())
        vecs = list(map(array, zip(*vecs)))
        kraus_list = [sqrt(vals[j]) * vec2mat(vecs[j])
                      for j in range(len(vals))]
        # Kraus operators to be padded with identities:
        k_qubit_kraus_list = product(kraus_list, repeat=sum(mask))
        rho_out = Qobj(inpt=zeros(state.shape), dims=state.dims)
        for operator_iter in k_qubit_kraus_list:
            operator_iter = iter(operator_iter)
            op_iter_list = [next(operator_iter) if mask[j]
                            else qeye(state.dims[0][j])
                            for j in range(len(state.dims[0]))]
            full_oper = tensor(list(map(Qobj, op_iter_list)))
            rho_out = rho_out + full_oper * state * full_oper.dag()
        return Qobj(rho_out)
