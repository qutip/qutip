# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:35:42 2013

@author: dcriger
"""

import numpy as np
from qutip import vec2mat


def _dep_super(pe):
    """
    Returns the superoperator corresponding to qubit depolarization for a
    given parameter pe.

    #TODO if this is going into production (hopefully it isn't) then check
    CPTP, expand to arbitrary dimensional systems, etc.
    """
    return np.array([[1 - pe / 2, 0, 0, pe / 2],
                     [0, 1 - pe, 0, 0],
                     [0, 0, 1 - pe, 0],
                     [pe / 2, 0, 0, 1 - pe / 2]])


def _dep_choi(pe):
    """
    Returns the choi matrix corresponding to qubit depolarization for a
    given parameter pe.

    #TODO if this is going into production (hopefully it isn't) then check
    CPTP, expand to arbitrary dimensional systems, etc.
    """
    return np.array([[1 - pe / 2, 0, 0, 1 - pe],
                     [0, pe / 2, 0, 0],
                     [0, 0, pe / 2, 0],
                     [1 - pe, 0, 0, 1 - pe / 2]])


def super_to_choi(q_oper):
    """
    Takes a superoperator to a Choi matrix
    #TODO Sanitize input, incorporate as method on Qobj if type=='super'
    """
    sqrt_shape = np.sqrt(q_oper.shape[0])
    return q_oper.reshape([sqrt_shape] * 4).transpose(0, 2, 1, 3).reshape(q_oper.shape)


def choi_to_super(q_oper):
    """
    Takes a Choi matrix to a superoperator
    #TODO Sanitize input, Abstract-ify application of channels to states
    """
    sqrt_shape = np.sqrt(q_oper.shape[0])
    return q_oper.reshape([sqrt_shape] * 4).transpose(0, 2, 1, 3).reshape(q_oper.shape)


def choi_to_kraus(q_oper):
    """
    Takes a Choi matrix and returns a list of Kraus operators.
    #TODO Create a new class structure for quantum channels, perhaps as a
    strict sub-class of Qobj.
    """
    vals, vecs = np.linalg.eig(q_oper)
    print vals
    print vecs
    vecs = map(np.array, zip(*vecs))
    return [np.sqrt(vals[j]) * vec2mat(vecs[j]) for j in range(len(vals))]
