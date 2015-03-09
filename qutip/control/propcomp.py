# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Propagator Computer
Classes used to calculate the propagators,
and also the propagator gradient when exact gradient methods are used

Note the methods in the _Diag class was inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
"""

# import os
import numpy as np
import scipy.linalg as la
# QuTiP logging
import qutip.logging as logging
logger = logging.get_logger()
# QuTiP control modules
from qutip.control import errors


class PropagatorComputer:
    """
    Base for all  Propagator Computer classes
    that are used to calculate the propagators,
    and also the propagator gradient when exact gradient methods are used
    Note: they must be instantiated with a Dynamics object, that is the
    container for the data that the functions operate on
    This base class cannot be used directly. See subclass descriptions
    and choose the appropriate one for the application

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN
        Note value should be set using set_log_level

    grad_exact : boolean
        indicates whether the computer class instance is capable
        of computing propagator gradients. It is used to determine
        whether to create the Dynamics prop_grad array
    """
    def __init__(self, dynamics):
        self.parent = dynamics
        self.reset()

    def reset(self):
        """
        reset any configuration data
        """
        self.id_text = 'PROP_COMP_BASE'
        self.set_log_level(self.parent.log_level)
        self.grad_exact = False

    def set_log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        self.log_level = lvl
        logger.setLevel(lvl)

    def compute_propagator(self, k):
        """
        calculate the progator between X(k) and X(k+1)
        Uses matrix expm of the dyn_gen at that point (in time)
        Assumes that the dyn_gen have been been calculated,
        i.e. drift and ctrls combined
        Return the propagator
        """
        raise errors.UsageError("Not implemented in the baseclass."
                                " Choose a subclass")

    def compute_diff_prop(self, k, j, epsilon):
        """
        Calculate the propagator from the current point to a trial point
        a distance 'epsilon' (change in amplitude)
        in the direction the given control j in timeslot k
        Returns the propagator
        """
        raise errors.UsageError("Not implemented in the baseclass."
                                " Choose a subclass")

    def compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot.
        """
        raise errors.UsageError("Not implemented in the baseclass."
                                " Choose a subclass")


class PropCompApproxGrad(PropagatorComputer):
    """
    This subclass can be used when the propagator is calculated simply
    by expm of the dynamics generator, i.e. when gradients will be calculated
    using approximate methods.
    """
    def __init__(self, dynamics):
        self.parent = dynamics
        self.reset()

    def reset(self):
        """
        reset any configuration data
        """
        PropagatorComputer.reset(self)
        self.id_text = 'APPROX'
        self.grad_exact = False

    def compute_propagator(self, k):
        """
        calculate the progator between X(k) and X(k+1)
        Uses matrix expm of the dyn_gen at that point (in time)
        Assumes that the dyn_gen have been been calculated,
        i.e. drift and ctrls combined
        Return the propagator
        """
        dyn = self.parent
        dgt = dyn.get_dyn_gen(k)*dyn.tau[k]
        prop = la.expm(dgt)
        return prop

    def compute_diff_prop(self, k, j, epsilon):
        """
        Calculate the propagator from the current point to a trial point
        a distance 'epsilon' (change in amplitude)
        in the direction the given control j in timeslot k
        Returns the propagator
        """
        dyn = self.parent
        dgt_eps = np.asarray(dyn.get_dyn_gen(k) +
                             epsilon*dyn.get_ctrl_dyn_gen(j))
        prop_eps = la.expm(dgt_eps*dyn.tau[k])
        return prop_eps


class PropCompDiag(PropagatorComputer):
    """
    Coumputes the propagator exponentiation using diagonalisation of
    of the dynamics generator
    """
    def reset(self):
        """
        reset any configuration data
        """
        PropagatorComputer.reset(self)
        self.id_text = 'DIAG'
        self.grad_exact = True

    def compute_propagator(self, k):
        """
        Calculates the exponentiation of the dynamics generator (H)
        As part of the calc the the eigen decomposition is required, which
        is reused in the propagator gradient calculation
        """
        dyn = self.parent
        dyn.ensure_decomp_curr(k)

        eig_vec = dyn.dyn_gen_eigenvectors[k]
        prop_eig_diag = np.diagflat(dyn.prop_eigen[k])
        prop = eig_vec.dot(prop_eig_diag).dot(eig_vec.conj().T)
        return prop

    def compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot.

        Returns:
            [prop], prop_grad
        """
        dyn = self.parent
        dyn.ensure_decomp_curr(k)

        if compute_prop:
            prop = self.compute_propagator(k)

        eig_vec = dyn.dyn_gen_eigenvectors[k]
        eig_vec_adj = eig_vec.conj().T

        # compute ctrl dyn gen in diagonalised basis
        # i.e. the basis of the full dyn gen for this timeslot
        dg_diag = \
            dyn.tau[k]*eig_vec_adj.dot(dyn.get_ctrl_dyn_gen(j)).dot(eig_vec)

        # multiply by factor matrix
        factors = dyn.dyn_gen_factormatrix[k]
        # note have to use multiply method as .dot returns matrix
        # and hence * implies inner product i.e. dot
        dg_diag_fact = np.multiply(dg_diag, factors)
        # Return to canonical basis
        prop_grad = eig_vec.dot(dg_diag_fact).dot(eig_vec_adj)

        if compute_prop:
            return prop, prop_grad
        else:
            return prop_grad


class PropCompAugMat(PropagatorComputer):
    """
    Augmented Matrix (deprecated - see _Frechet)

    It should work for all systems, e.g. open, symplectic
    There will be other PropagatorComputer subclasses that are more efficient
    The _Frechet class should provide exactly the same functionality
    more efficiently.

    Note the propagator gradient calculation using the augmented matrix
    is taken from:
    'Robust quantum gates for open systems via optimal control:
    Markovian versus non-Markovian dynamics'
    Frederik F Floether, Pierre de Fouquieres, and Sophie G Schirmer
    """
    def reset(self):
        PropagatorComputer.reset(self)
        self.id_text = 'AUG_MAT'
        self.grad_exact = True

    def get_aug_mat(self, k, j):
        """
        Generate the matrix [[A, E], [0, A]] where
            A is the overall dynamics generator
            E is the control dynamics generator
        for a given timeslot and control
        returns this augmented matrix
        """
        dyn = self.parent
        A = dyn.get_dyn_gen(k)*dyn.tau[k]
        E = dyn.get_ctrl_dyn_gen(j)*dyn.tau[k]

        l = np.concatenate((A, np.zeros(A.shape)))
        r = np.concatenate((E, A))
        aug = np.concatenate((l, r), 1)

        return aug

    def compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot using the exponentiation of the the augmented
        matrix.
        The propagtor is calculated for 'free' in this method
        and hence it is returned if compute_prop==True
        Returns:
            [prop], prop_grad
        """
        dyn = self.parent
        dyn_gen_shp = dyn.get_dyn_gen(k).shape
        aug = self.get_aug_mat(k, j)
        aug_exp = la.expm(aug)
        prop_grad = aug_exp[:dyn_gen_shp[0], dyn_gen_shp[1]:]
        if compute_prop:
            prop = aug_exp[:dyn_gen_shp[0], :dyn_gen_shp[1]]
            return prop, prop_grad
        else:
            return prop_grad


class PropCompFrechet(PropagatorComputer):
    """
    Frechet method for calculating the propagator:
        exponentiating the combined dynamics generator
    and the propagator gradient
    It should work for all systems, e.g. unitary, open, symplectic
    There are other PropagatorComputer subclasses that may be more efficient
    """
    def reset(self):
        PropagatorComputer.reset(self)
        self.id_text = 'FRECHET'
        self.grad_exact = True

    def compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot using the expm_frechet method
        The propagtor is calculated (almost) for 'free' in this method
        and hence it is returned if compute_prop==True
        Returns:
            [prop], prop_grad
        """
        dyn = self.parent
        A = dyn.get_dyn_gen(k)*dyn.tau[k]
        E = dyn.get_ctrl_dyn_gen(j)*dyn.tau[k]

        if compute_prop:
            prop, propGrad = la.expm_frechet(A, E)
            return prop, propGrad
        else:
            propGrad = la.expm_frechet(A, E, compute_expm=False)
            return propGrad
