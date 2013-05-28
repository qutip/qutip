# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
# This code was contributed by Ben Criger. Resemblance to
# partial_transpose is intentional, and meant to enhance legibility.
#
#

from numpy.linalg import norm
import numpy as np
from qutip import *
from operator import mul, add
from itertools import imap, product
from super_to_choi import super_to_choi


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
    aff_subs_dim_ar = np.transpose(array(state.dims))[array(mask)]

    assert np.all([aff_subs_dim_ar[j] == aff_subs_dim_ar[0]
                   for j in range(len(aff_subs_dim_ar))]), \
        "Affected subsystems must have the same dimension. Given:" +\
        repr(aff_subs_dim_ar)

    # If the map is on the Hilbert space, it must have the same dimension
    # as the affected subsystem. If it is on the Liouville space, it must
    # exist on a space as large as the square of the Hilbert dimension.
    if issuper(channel):
        required_shape = map(lambda x: x ** 2, aff_subs_dim_ar[0])
    else:
        required_shape = aff_subs_dim_ar[0]
    assert np.all(channel.shape == required_shape),\
    "Superoperator dimension must be the subsystem dimension squared, given: "\
    + repr(channel.shape)

    # Ensure mask is an array:
    mask = np.array(mask)

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
    # checked affected subsystems print np.arange(len(state.dims[0]))[mask]
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
        # print mul_idx
        # print subs_dim_ar[mul_idx]
        n_blks = n_blks * subs_dim_ar[mul_idx][0]

    blk_sz = state.shape[0] / n_blks
    # Apply channel to top subsystem of each block in matrix
    full_data_matrix = state.data.todense()

    if np.all(np.isreal(full_data_matrix)):
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
    if np.all(np.isreal(block)):
        block = block.astype(complex)
    split_mat = _block_split(block, *channel.shape)
    # print split_mat
    temp_split_mat = np.zeros(np.shape(split_mat)).astype(complex)
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
    column = _block_col(block, *map(
        sqrt, channel.shape))  # FIXME use state shape?
    chan_mat = channel.data.todense()
    temp_col = np.zeros(np.shape(column)).astype(complex)
    # print chan_mat.shape
    for row_idx in range(len(chan_mat)):
        row = chan_mat[row_idx]
        # print [scal[0,0]*mat for (scal,mat) in zip(transpose(row),column)]
        temp_col[row_idx] = reduce(add, [scal[0, 0] * mat for (
            scal, mat) in zip(transpose(row), column)])
    return _block_stack(temp_col, *map(sqrt, channel.shape))


def _block_split(mat_in, n_v, n_h):
    """
    Returns a 3D array of matrices, splitting mat_in into
    n_v * n_h square sub-arrays.
    """
    return map(lambda x: np.hsplit(x, n_h), np.vsplit(mat_in, n_v))


def _block_join(mat_in):
    return np.hstack(np.hstack(mat_in))


def _block_col(mat_in, n_v, n_h):
    """
    Returns a 3D array of matrices, splitting mat_in into
    n_v * n_h square sub-arrays.
    """
    rows, cols = np.shape(mat_in)
    return np.reshape(
           array(_block_split(mat_in, n_v, n_h)).transpose(1, 0, 2, 3),
           (n_v * n_h, rows / n_v, cols / n_h))


def _block_stack(arr_in, n_v, n_h):
    """
    Inverse of _block_split
    """
    rs, cs = np.shape(arr_in)[-2:]
    temp = map(np.transpose, arr_in)
    # print np.shape(arr_in)
    temp = np.reshape(temp, (n_v, n_h, rs, cs))
    return np.hstack(np.hstack(temp)).T


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
        chan_mat = np.array(channel.data.todense())
        choi_matrix = super_to_choi(chan_mat)
        vals, vecs = np.linalg.eig(choi_matrix)
        vecs = map(np.array, zip(*vecs))
        kraus_list = [np.sqrt(vals[j]) * vec2mat(vecs[j])
                      for j in range(len(vals))]
        # Kraus operators to be padded with identities:
        k_qubit_kraus_list = product(kraus_list, repeat=sum(mask))
        rho_out = Qobj(inpt=zeros(state.shape), dims=state.dims)
        for operator_iter in k_qubit_kraus_list:
            operator_iter = iter(operator_iter)
            op_iter_list = [operator_iter.next() if mask[j]
                            else qeye(state.dims[0][j])
                            for j in range(len(state.dims[0]))]
            full_oper = tensor(map(Qobj, op_iter_list))
            rho_out = rho_out + full_oper * state * full_oper.dag()
        return Qobj(rho_out)

if __name__ == '__main__':
    # Test, apply lowering operator to 2 out of 5 subsystems, on
    # multi-dimensional state.
    np.set_printoptions(precision=3)

    print "Applying Single Operator to Two Subsystems Out Of Five"
    print "------------------------------------------------------"
    state = tensor(
        [coherent_dm(2, 0.2), coherent_dm(3, 0.3), coherent_dm(5, 0.5),
         coherent_dm(3, 0.5), coherent_dm(7, 0.7j)])
    op_1 = create(3) + destroy(3)
    analytic_output = tensor([coherent_dm(2, 0.2),
                              op_1 * coherent_dm(3, 0.3) * op_1.dag(),
                              coherent_dm(5, 0.5),
                              op_1 * coherent_dm(3, 0.5) * op_1.dag(),
                              coherent_dm(7, 0.7j)])
    print "Difference between analytic result and Kraus map: \n"
    test_output = subsystem_apply(
        state, op_1, [False, True, False, True, False],
        reference=True)
    print norm(test_output.data.todense() - analytic_output.data.todense())
    print "Difference between analytic result and efficient numerics: \n"
    efficient_output = subsystem_apply(
        state, op_1, [False, True, False, True, False])
    print norm(efficient_output.data.todense() -
               analytic_output.data.todense())

    print "Applying Superoperator to Two Subsystems Out Of Five"
    print "------------------------------------------------------"
    state = tensor(
        [coherent_dm(2, 0.2), coherent_dm(3, 0.3), coherent_dm(5, 0.5),
         coherent_dm(3, 0.5), coherent_dm(7, 0.7j)])
    superop_1 = propagator(Qobj(np.zeros([3, 3])), 0.2, [
                           destroy(3), create(3), jmat(1., 'z')])
    inpt_5 = vec2mat(superop_1.data.todense() * mat2vec(
        coherent_dm(3, 0.5).data.todense()))
    inpt_3 = vec2mat(superop_1.data.todense() * mat2vec(
        coherent_dm(3, 0.3).data.todense()))
    dm_out_5 = Qobj(inpt=inpt_5, type='oper')
    dm_out_3 = Qobj(inpt=inpt_3, type='oper')
    analytic_output = tensor(
        [coherent_dm(2, 0.2), dm_out_3, coherent_dm(5, 0.5),
         dm_out_5, coherent_dm(7, 0.7j)])
    print "Difference between analytic result and Kraus map: \n"
    test_output = subsystem_apply(
        state, superop_1, [False, True, False, True, False],
        reference=True)
    print norm(test_output.data.todense() - analytic_output.data.todense())
    print "Difference between analytic result and efficient numerics: \n"
    efficient_output = subsystem_apply(
        state, superop_1, [False, True, False, True, False])
    print norm(efficient_output.data.todense() -
               analytic_output.data.todense())

    """
    print "Applying Superoperator to One Subsystem Out Of Two"
    print "------------------------------------------------------"
    state=tensor([coherent_dm(2,0.2),coherent_dm(2,0.7j)])
    superop_1 = propagator(Qobj(np.zeros([2,2])),0.2,
                           [destroy(2),create(2),jmat(1/2.,'z')])
    inpt_2=vec2mat(superop_1.data.todense() *
                   mat2vec(coherent_dm(2,0.2).data.todense()))
    #inpt_3=vec2mat(superop_1.data.todense() *
                    mat2vec(coherent_dm(3,0.3).data.todense()))
    dm_out_2=Qobj(inpt=inpt_2, type='oper')
    #dm_out_3=Qobj(inpt=inpt_3, type='oper')
    analytic_output=tensor([dm_out_2,coherent_dm(2,0.7j)])
    print "Difference between analytic result and Kraus map: \n"
    test_output=subsystem_apply(state,superop_1,[True,False],\
    reference=True)
    print norm(test_output.data.todense()-analytic_output.data.todense())
    print "Difference between analytic result and efficient numerics: \n"
    efficient_output=subsystem_apply(state,superop_1,[True,False])
    print norm(efficient_output.data.todense()-analytic_output.data.todense())
    print analytic_output
    print test_output
    print efficient_output
    """

    """
    print "Single Operator, two subsystems out of four:"
    state=tensor([coherent_dm(2,0.2),
                  coherent_dm(3,0.3),
                  coherent_dm(2,0.5j),
                  coherent_dm(3,0.3j)])
    op_1=create(3)
    analytic_output=tensor([coherent_dm(2,0.2),
                            op_1*coherent_dm(3,0.3)*op_1.dag(),
                            coherent_dm(2,0.5j),
                            op_1*coherent_dm(3,0.3j)*op_1.dag()])
    print("Explicit Kraus Map Output")
    print("-----------------------------------------------------")
    test_output=subsystem_apply(state,op_1,[False,True,False,True],
                                reference=True)
    print "Difference between analytic result and Kraus map: \n"
    print norm(test_output.data.todense()-analytic_output.data.todense())
    print("Fast Application Output")
    print("-----------------------------------------------------")
    efficient_output=subsystem_apply(state,op_1,[False,True,False,True])
    print "Difference between analytic result and efficient numerics: \n"
    print norm(efficient_output.data.todense()-analytic_output.data.todense())

    print "Qubit Qutrit Qubit Test"
    state=tensor([coherent_dm(2,0.4j),coherent_dm(3,0.6j),coherent_dm(2,0.3)])
    op_1=create(3)+destroy(3)
    analytic_output=tensor([coherent_dm(2,0.4j),
                            op_1*coherent_dm(3,0.6j)*op_1.dag(),
                            coherent_dm(2,0.3)])
    print("Analytic Output")
    print("-----------------------------------------------------")
    print(analytic_output.data.todense())
    print("Explicit Kraus Map Output")
    print("-----------------------------------------------------")
    test_output=subsystem_apply(state,op_1,[False,True,False], reference=True)
    print(norm((test_output.data.todense()-analytic_output.data.todense())))
    print("Fast Application Output")
    print("-----------------------------------------------------")
    efficient_output=subsystem_apply(state,op_1,[False,True,False])
    print(norm((efficient_output.data.todense() -
                analytic_output.data.todense())))
    """
