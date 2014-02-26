# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:35:42 2013

@author: dcriger
"""

from operator import add

from numpy.core.multiarray import array
from numpy.core.shape_base import hstack
from numpy.matrixlib.defmatrix import matrix
from numpy import sqrt
from scipy.linalg import eig

from qutip.superoperator import vec2mat
from qutip.qobj import Qobj


def _dep_super(pe):
    """
    Returns the superoperator corresponding to qubit depolarization for a
    given parameter pe.

    # TODO if this is going into production (hopefully it isn't) then check
    CPTP, expand to arbitrary dimensional systems, etc.
    """
    return Qobj(dims=[[2, 2], [2, 2]],
                inpt=array([[1. - pe / 2., 0., 0., pe / 2.],
                            [0., 1. - pe, 0., 0.],
                            [0., 0., 1. - pe, 0.],
                            [pe / 2., 0., 0., 1. - pe / 2.]]))


def _dep_choi(pe):
    """
    Returns the choi matrix corresponding to qubit depolarization for a
    given parameter pe.

    # TODO if this is going into production (hopefully it isn't) then check
    CPTP, expand to arbitrary dimensional systems, etc.
    """
    return Qobj(dims=[[2, 2], [2, 2]],
                inpt=array([[1. - pe / 2., 0., 0., 1. - pe],
                            [0., pe / 2., 0., 0.],
                            [0., 0., pe / 2., 0.],
                            [1. - pe, 0., 0., 1. - pe / 2.]]))


def super_to_choi(q_oper):
    """
    Takes a superoperator to a Choi matrix
    # TODO Sanitize input, incorporate as method on Qobj if type=='super'
    """
    data = q_oper.data.toarray()
    sqrt_shape = sqrt(data.shape[0])
    return Qobj(dims=q_oper.dims,

                inpt=data.reshape([sqrt_shape] * 4).
                transpose(3, 1, 2, 0).reshape(q_oper.data.shape))


def choi_to_super(q_oper):
    """
    Takes a Choi matrix to a superoperator
    # TODO Sanitize input, Abstract-ify application of channels to states
    """
    return super_to_choi(q_oper)


def choi_to_kraus(q_oper):
    """
    Takes a Choi matrix and returns a list of Kraus operators.
    # TODO Create a new class structure for quantum channels, perhaps as a
    strict sub-class of Qobj.
    """
    vals, vecs = eig(q_oper.data.todense())
    vecs = list(map(array, zip(*vecs)))
    return list(map(lambda x: Qobj(inpt=x),
                    [sqrt(vals[j]) * vec2mat(vecs[j])
                     for j in range(len(vals))]))


def kraus_to_choi(kraus_list):
    """
    Takes a list of Kraus operators and returns the Choi matrix for the channel
    represented by the Kraus operators in `kraus_list`
    """
    kraus_mat_list = list(map(lambda x: matrix(x.data.todense()), kraus_list))
    op_len = len(kraus_mat_list[0])
    op_rng = range(op_len)

    choi_blocks = array([[sum([op[:, c_ix] * array([op.H[r_ix, :]])
                               for op in kraus_mat_list])
                          for r_ix in op_rng]
                         for c_ix in op_rng])
    return Qobj(inpt=hstack(hstack(choi_blocks)),
                dims=[kraus_list[0].dims, kraus_list[0].dims])


def kraus_to_super(kraus_list):
    return choi_to_super(kraus_to_choi(kraus_list))
