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
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
# QuTiP
from qutip import Qobj
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
from qutip.control import errors

def _func_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

class PropagatorComputer(object):
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
        Options are attributes of qutip_utils.logging,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    grad_exact : boolean
        indicates whether the computer class instance is capable
        of computing propagator gradients. It is used to determine
        whether to create the Dynamics prop_grad array
    """
    def __init__(self, dynamics, params=None):
        self.parent = dynamics
        self.params = params
        self.reset()

    def reset(self):
        """
        reset any configuration data
        """
        self.id_text = 'PROP_COMP_BASE'
        self.log_level = self.parent.log_level
        self._grad_exact = False

    def apply_params(self, params=None):
        """
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        """
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    def grad_exact(self):
        return self._grad_exact

    def compute_propagator(self, k):
        _func_deprecation("'compute_propagator' has been replaced "
                        "by '_compute_propagator'")
        return self._compute_propagator(k)
                               
    def _compute_propagator(self, k):
        """
        calculate the progator between X(k) and X(k+1)
        Uses matrix expm of the dyn_gen at that point (in time)
        Assumes that the dyn_gen have been been calculated,
        i.e. drift and ctrls combined
        Return the propagator
        """
        dyn = self.parent
        dgt = dyn._get_phased_dyn_gen(k)*dyn.tau[k]
        if dyn.oper_dtype == Qobj:
            prop = dgt.expm()
        else:
            prop = la.expm(dgt)
        return prop

    def compute_diff_prop(self, k, j, epsilon):
        _func_deprecation("'compute_diff_prop' has been replaced "
                        "by '_compute_diff_prop'")
        return self._compute_diff_prop( k, j, epsilon)

    def _compute_diff_prop(self, k, j, epsilon):
        """
        Calculate the propagator from the current point to a trial point
        a distance 'epsilon' (change in amplitude)
        in the direction the given control j in timeslot k
        Returns the propagator
        """
        raise errors.UsageError("Not implemented in the baseclass."
                                " Choose a subclass")

    def compute_prop_grad(self, k, j, compute_prop=True):
        _func_deprecation("'compute_prop_grad' has been replaced "
                        "by '_compute_prop_grad'")
        return self._compute_prop_grad(self, k, j, compute_prop=compute_prop)

    def _compute_prop_grad(self, k, j, compute_prop=True):
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

    def reset(self):
        """
        reset any configuration data
        """
        PropagatorComputer.reset(self)
        self.id_text = 'APPROX'
        self.grad_exact = False
        self.apply_params()

    def _compute_diff_prop(self, k, j, epsilon):
        """
        Calculate the propagator from the current point to a trial point
        a distance 'epsilon' (change in amplitude)
        in the direction the given control j in timeslot k
        Returns the propagator
        """
        dyn = self.parent
        dgt_eps = (dyn._get_phased_dyn_gen(k) +
                epsilon*dyn._get_phased_ctrl_dyn_gen(j))*dyn.tau[k]

        if dyn.oper_dtype == Qobj:
            prop_eps = dgt_eps.expm()
        else:
            prop_eps = la.expm(dgt_eps)

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
        self.apply_params()

    def _compute_propagator(self, k):
        """
        Calculates the exponentiation of the dynamics generator (H)
        As part of the calc the the eigen decomposition is required, which
        is reused in the propagator gradient calculation
        """
        dyn = self.parent
        dyn._ensure_decomp_curr(k)

        if dyn.oper_dtype == Qobj:

            prop = (dyn._dyn_gen_eigenvectors[k]*dyn._prop_eigen[k]*
                                dyn._get_dyn_gen_eigenvectors_adj(k))
        else:
            prop = dyn._dyn_gen_eigenvectors[k].dot(
                                    dyn._prop_eigen[k]).dot(
                                dyn._get_dyn_gen_eigenvectors_adj(k))

        return prop

    def _compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot.

        Returns:
            [prop], prop_grad
        """
        dyn = self.parent
        dyn._ensure_decomp_curr(k)

        if compute_prop:
            prop = self._compute_propagator(k)

        if dyn.oper_dtype == Qobj:
            # put control dyn_gen in combined dg diagonal basis
            cdg =  (dyn._get_dyn_gen_eigenvectors_adj(k)*
                        dyn._get_phased_ctrl_dyn_gen(j)*
                        dyn._dyn_gen_eigenvectors[k])
            # multiply (elementwise) by timeslice and factor matrix
            cdg = Qobj(np.multiply(cdg.full()*dyn.tau[k],
                        dyn._dyn_gen_factormatrix[k]), dims=dyn.dyn_dims)
            # Return to canonical basis
            prop_grad = (dyn._dyn_gen_eigenvectors[k]*cdg*
                        dyn._get_dyn_gen_eigenvectors_adj(k))
        else:
            # put control dyn_gen in combined dg diagonal basis
            cdg =  dyn._get_dyn_gen_eigenvectors_adj(k).dot(
                        dyn._get_phased_ctrl_dyn_gen(j)).dot(
                        dyn._dyn_gen_eigenvectors[k])
            # multiply (elementwise) by timeslice and factor matrix
            cdg = np.multiply(cdg*dyn.tau[k], dyn._dyn_gen_factormatrix[k])
            # Return to canonical basis
            prop_grad = dyn._dyn_gen_eigenvectors[k].dot(cdg).dot(
                        dyn._get_dyn_gen_eigenvectors_adj(k))

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
        self.apply_params()

    def _get_aug_mat(self, k, j):
        """
        Generate the matrix [[A, E], [0, A]] where
            A is the overall dynamics generator
            E is the control dynamics generator
        for a given timeslot and control
        returns this augmented matrix
        """
        dyn = self.parent
        dg = dyn._get_phased_dyn_gen(k)

        if dyn.oper_dtype == Qobj:
            A = dg.data*dyn.tau[k]
            E = dyn._get_phased_ctrl_dyn_gen(j).data*dyn.tau[k]
            Z = sp.csr_matrix(dg.data.shape)
            aug = Qobj(sp.vstack([sp.hstack([A, E]), sp.hstack([Z, A])]))
        elif dyn.oper_dtype == np.ndarray:
            A = dg*dyn.tau[k]
            E = dyn._get_phased_ctrl_dyn_gen(j)*dyn.tau[k]
            Z = np.zeros(dg.shape)
            aug = np.vstack([np.hstack([A, E]), np.hstack([Z, A])])
        else:
            A = dg*dyn.tau[k]
            E = dyn._get_phased_ctrl_dyn_gen(j)*dyn.tau[k]
            Z = dg*0.0
            aug = sp.vstack([sp.hstack([A, E]), sp.hstack([Z, A])])
        return aug

    def _compute_prop_grad(self, k, j, compute_prop=True):
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
        dg = dyn._get_phased_dyn_gen(k)
        aug = self._get_aug_mat(k, j)


        if dyn.oper_dtype == Qobj:
            aug_exp = aug.expm()
            prop_grad = Qobj(aug_exp.data[:dg.shape[0], dg.shape[1]:],
                         dims=dyn.dyn_dims)
            if compute_prop:
                prop = Qobj(aug_exp.data[:dg.shape[0], :dg.shape[1]],
                            dims=dyn.dyn_dims)
        else:
            aug_exp = la.expm(aug)
            prop_grad = aug_exp[:dg.shape[0], dg.shape[1]:]
            if compute_prop:
                prop = aug_exp[:dg.shape[0], :dg.shape[1]]

        if compute_prop:
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
        self.apply_params()

    def _compute_prop_grad(self, k, j, compute_prop=True):
        """
        Calculate the gradient of propagator wrt the control amplitude
        in the timeslot using the expm_frechet method
        The propagtor is calculated (almost) for 'free' in this method
        and hence it is returned if compute_prop==True
        Returns:
            [prop], prop_grad
        """
        dyn = self.parent

        if dyn.oper_dtype == Qobj:
            A = dyn._get_phased_dyn_gen(k).full()*dyn.tau[k]
            E = dyn._get_phased_ctrl_dyn_gen(j).full()*dyn.tau[k]
            if compute_prop:
                prop_dense, prop_grad_dense = la.expm_frechet(A, E)
                prop = Qobj(prop_dense, dims=dyn.dyn_dims)
                prop_grad = Qobj(prop_grad_dense,
                                            dims=dyn.dyn_dims)
            else:
                prop_grad_dense = la.expm_frechet(A, E, compute_expm=False)
                prop_grad = Qobj(prop_grad_dense,
                                            dims=dyn.dyn_dims)
        elif dyn.oper_dtype == np.ndarray:
            A = dyn._get_phased_dyn_gen(k)*dyn.tau[k]
            E = dyn._get_phased_ctrl_dyn_gen(j)*dyn.tau[k]
            if compute_prop:
                prop, prop_grad = la.expm_frechet(A, E)
            else:
                prop_grad = la.expm_frechet(A, E,
                                                    compute_expm=False)
        else:
            # Assuming some sparse matrix
            spcls = dyn._dyn_gen[k].__class__
            A = (dyn._get_phased_dyn_gen(k)*dyn.tau[k]).toarray()
            E = (dyn._get_phased_ctrl_dyn_gen(j)*dyn.tau[k]).toarray()
            if compute_prop:
                prop_dense, prop_grad_dense = la.expm_frechet(A, E)
                prop = spcls(prop_dense)
                prop_grad = spcls(prop_grad_dense)
            else:
                prop_grad_dense = la.expm_frechet(A, E, compute_expm=False)
                prop_grad = spcls(prop_grad_dense)

        if compute_prop:
            return prop, prop_grad
        else:
            return prop_grad
