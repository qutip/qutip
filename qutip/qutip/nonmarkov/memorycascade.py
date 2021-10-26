# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2015 and later, Arne L. Grimsmo
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

# @author: Arne L. Grimsmo
# @email1: arne.grimsmo@gmail.com
# @organization: University of Sherbrooke

"""
This module is an implementation of the method introduced in [1], for
solving open quantum systems subject to coherent feedback with a single
discrete time-delay. This method is referred to as the ``memory cascade''
method in qutip.

[1] Arne L. Grimsmo, Phys. Rev. Lett 115, 060402 (2015)
"""

import numpy as np
import warnings

import qutip as qt


class MemoryCascade:
    """Class for running memory cascade simulations of open quantum systems
    with time-delayed coherent feedback.

    Attributes
    ----------
    H_S : :class:`qutip.Qobj`
        System Hamiltonian (can also be a Liouvillian)

    L1 : :class:`qutip.Qobj` / list of :class:`qutip.Qobj`
        System operators coupling into the feedback loop. Can be a single
        operator or a list of operators.

    L2 : :class:`qutip.Qobj` / list of :class:`qutip.Qobj`
        System operators coupling out of the feedback loop. Can be a single
        operator or a list of operators. L2 must have the same length as L1.

    S_matrix: *array*
        S matrix describing which operators in L1 are coupled to which
        operators in L2 by the feedback channel. Defaults to an n by n identity
        matrix where n is the number of elements in L1/L2.

    c_ops_markov : :class:`qutip.Qobj` / list of :class:`qutip.Qobj`
        Decay operators describing conventional Markovian decay channels.
        Can be a single operator or a list of operators.

    integrator : str {'propagator', 'mesolve'}
        Integrator method to use. Defaults to 'propagator' which tends to be 
        faster for long times (i.e., large Hilbert space).

    parallel : bool
        Run integrator in parallel if True. Only implemented for 'propagator'
        as the integrator method.

    options : :class:`qutip.solver.Options`
        Generic solver options.
    """

    def __init__(self, H_S, L1, L2, S_matrix=None, c_ops_markov=None,
                 integrator='propagator', parallel=False, options=None):

        if options is None:
            self.options = qt.Options()
        else:
            self.options = options

        self.H_S = H_S
        self.sysdims = H_S.dims
        if isinstance(L1, qt.Qobj):
            self.L1 = [L1]
        else:
            self.L1 = L1
        if isinstance(L2, qt.Qobj):
            self.L2 = [L2]
        else:
            self.L2 = L2
        if not len(self.L1) == len(self.L2):
            raise ValueError('L1 and L2 has to be of equal length.')
        if isinstance(c_ops_markov, qt.Qobj):
            self.c_ops_markov = [c_ops_markov]
        else:
            self.c_ops_markov = c_ops_markov

        if S_matrix is None:
            self.S_matrix = np.identity(len(self.L1))
        else:
            self.S_matrix = S_matrix
        # create system identity superoperator
        self.Id = qt.qeye(H_S.shape[0])
        self.Id.dims = self.sysdims
        self.Id = qt.sprepost(self.Id, self.Id)
        self.store_states = self.options.store_states
        self.integrator = integrator
        self.parallel = parallel

    def propagator(self, t, tau, notrace=False):
        """
        Compute propagator for time t and time-delay tau

        Parameters
        ----------
        t : *float*
            current time

        tau : *float*
            time-delay

        notrace : *bool* {False}
            If this optional is set to True, a propagator is returned for a
            cascade of k systems, where :math:`(k-1) tau < t < k tau`.
            If set to False (default), a generalized partial trace is performed
            and a propagator for a single system is returned.
        Returns
        -------
        : :class:`qutip.Qobj`
            time-propagator for reduced system dynamics
        """
        k = int(t/tau)+1
        s = t-(k-1)*tau
        G1, E0 = _generator(k, self.H_S, self.L1, self.L2, self.S_matrix,
                            self.c_ops_markov)
        E = _integrate(G1, E0, 0., s, integrator=self.integrator,
                       parallel=self.parallel, opt=self.options)
        if k > 1:
            G2, null = _generator(k-1, self.H_S, self.L1, self.L2,
                                  self.S_matrix, self.c_ops_markov)
            G2 = qt.composite(G2, self.Id)
            E = _integrate(G2, E, s, tau, integrator=self.integrator, 
                    parallel=self.parallel, opt=self.options)
        E.dims = E0.dims
        if not notrace:
            E = _genptrace(E, k)
        return E

    def outfieldpropagator(self, blist, tlist, tau, c1=None, c2=None,
                           notrace=False):
        """
        Compute propagator for computing output field expectation values
        <O_n(tn)...O_2(t2)O_1(t1)> for times t1,t2,... and
        O_i = I, b_out, b_out^\dagger, b_loop, b_loop^\dagger

        Parameters
        ----------
        blist : array_like
            List of integers specifying the field operators:
            0: I (nothing)
            1: b_out
            2: b_out^\dagger
            3: b_loop
            4: b_loop^\dagger

        tlist : array_like
            list of corresponding times t1,..,tn at which to evaluate the field
            operators

        tau : float
            time-delay

        c1 : :class:`qutip.Qobj`
            system collapse operator that couples to the in-loop field in
            question (only needs to be specified if self.L1 has more than one
            element)

        c2 : :class:`qutip.Qobj`
            system collapse operator that couples to the output field in
            question (only needs to be specified if self.L2 has more than one
            element)

        notrace : bool {False}
            If this optional is set to True, a propagator is returned for a
            cascade of k systems, where :math:`(k-1) tau < t < k tau`.
            If set to False (default), a generalized partial trace is performed
            and a propagator for a single system is returned.

        Returns
        -------
        : :class:`qutip.Qobj`
            time-propagator for computing field correlation function
        """
        if c1 is None and len(self.L1) == 1:
            c1 = self.L1[0]
        else:
            raise ValueError('Argument c1 has to be specified when more than' +
                             'one collapse operator couples to the feedback' +
                             'loop.')
        if c2 is None and len(self.L2) == 1:
            c2 = self.L2[0]
        else:
            raise ValueError('Argument c1 has to be specified when more than' +
                             'one collapse operator couples to the feedback' +
                             'loop.')
        klist = []
        slist = []
        for t in tlist:
            klist.append(int(t/tau)+1)
            slist.append(t-(klist[-1]-1)*tau)
        kmax = max(klist)
        zipped = sorted(zip(slist, klist, blist))
        slist = [s for (s, k, b) in zipped]
        klist = [k for (s, k, b) in zipped]
        blist = [b for (s, k, b) in zipped]

        G1, E0 = _generator(kmax, self.H_S, self.L1, self.L2, self.S_matrix,
                            self.c_ops_markov)
        sprev = 0.
        E = E0
        for i, s in enumerate(slist):
            E = _integrate(G1, E, sprev, s, integrator=self.integrator,
                    parallel=self.parallel, opt=self.options)
            if klist[i] == 1:
                l1 = 0.*qt.Qobj()
            else:
                l1 = _localop(c1, klist[i]-1, kmax)
            l2 = _localop(c2, klist[i], kmax)
            if blist[i] == 0:
                superop = self.Id
            elif blist[i] == 1:
                superop = qt.spre(l1+l2)
            elif blist[i] == 2:
                superop = qt.spost(l1.dag()+l2.dag())
            elif blist[i] == 3:
                superop = qt.spre(l1)
            elif blist[i] == 4:
                superop = qt.spost(l1.dag())
            else:
                raise ValueError('Allowed values in blist are 0, 1, 2, 3 ' +
                                 'and 4.')
            superop.dims = E.dims
            E = superop*E
            sprev = s
        E = _integrate(G1, E, slist[-1], tau, integrator=self.integrator,
                parallel=self.parallel, opt=self.options)

        E.dims = E0.dims
        if not notrace:
            E = _genptrace(E, kmax)
        return E

    def rhot(self, rho0, t, tau):
        """
        Compute the reduced system density matrix :math:`\\rho(t)`

        Parameters
        ----------
        rho0 : :class:`qutip.Qobj`
            initial density matrix or state vector (ket)

        t : float
            current time

        tau : float
            time-delay

        Returns
        -------
        : :class:`qutip.Qobj`
            density matrix at time :math:`t`
        """
        if qt.isket(rho0):
            rho0 = qt.ket2dm(rho0)

        E = self.propagator(t, tau)
        rhovec = qt.operator_to_vector(rho0)
        return qt.vector_to_operator(E*rhovec)

    def outfieldcorr(self, rho0, blist, tlist, tau, c1=None, c2=None):
        """
        Compute output field expectation value
        <O_n(tn)...O_2(t2)O_1(t1)> for times t1,t2,... and
        O_i = I, b_out, b_out^\dagger, b_loop, b_loop^\dagger


        Parameters
        ----------
        rho0 : :class:`qutip.Qobj`
            initial density matrix or state vector (ket).

        blist : array_like
            List of integers specifying the field operators:
            0: I (nothing)
            1: b_out
            2: b_out^\dagger
            3: b_loop
            4: b_loop^\dagger

        tlist : array_like
            list of corresponding times t1,..,tn at which to evaluate the field
            operators

        tau : float
            time-delay

        c1 : :class:`qutip.Qobj`
            system collapse operator that couples to the in-loop field in
            question (only needs to be specified if self.L1 has more than one
            element)

        c2 : :class:`qutip.Qobj`
            system collapse operator that couples to the output field in
            question (only needs to be specified if self.L2 has more than one
            element)

        Returns
        -------
        : complex
            expectation value of field correlation function
        """
        E = self.outfieldpropagator(blist, tlist, tau)
        rhovec = qt.operator_to_vector(rho0)
        return (qt.vector_to_operator(E*rhovec)).tr()


def _localop(op, l, k):
    """
    Create a local operator on the l'th system by tensoring
    with identity operators on all the other k-1 systems
    """
    if l < 1 or l > k:
        raise IndexError('index l out of range')
    h = op
    I = qt.qeye(op.shape[0])
    I.dims = op.dims
    for i in range(1, l):
        h = qt.tensor(I, h)
    for i in range(l+1, k+1):
        h = qt.tensor(h, I)
    return h


def _genptrace(E, k):
    """
    Perform a gneralized partial trace on a superoperator E, tracing out all
    subsystems but one.
    """
    for l in range(k-1):
        nsys = len(E.dims[0][0])
        E = qt.tensor_contract(E, (0, 2*nsys+1), (nsys, 3*nsys+1))
    return E


def _generator(k, H, L1, L2, S=None, c_ops_markov=None):
    """
    Create a Liouvillian for a cascaded chain of k system copies
    """
    id = qt.qeye(H.dims[0][0])
    Id = qt.sprepost(id, id)
    if S is None:
        S = np.identity(len(L1))
    # create Lindbladian
    L = qt.Qobj()
    E0 = Id
    # first system
    L += qt.liouvillian(None, [_localop(c, 1, k) for c in L2])
    for l in range(1, k):
        # Identiy superoperator
        E0 = qt.composite(E0, Id)
        # Bare Hamiltonian
        Hl = _localop(H, l, k)
        L += qt.liouvillian(Hl, [])
        # Markovian Decay channels
        if c_ops_markov is not None:
            for c in c_ops_markov:
                cl = _localop(c, l, k)
                L += qt.liouvillian(None, [cl])
        # Cascade coupling
        c1 = np.array([_localop(c, l, k) for c in L1])
        c2 = np.array([_localop(c, l+1, k) for c in L2])
        c2dag = np.array([c.dag() for c in c2])
        Hcasc = -0.5j*np.dot(c2dag, np.dot(S, c1))
        Hcasc += Hcasc.dag()
        Lvec = c2 + np.dot(S, c1)
        L += qt.liouvillian(Hcasc, [c for c in Lvec])
    # last system
    L += qt.liouvillian(_localop(H, k, k), [_localop(c, k, k) for c in L1])
    if c_ops_markov is not None:
        for c in c_ops_markov:
            cl = _localop(c, k, k)
            L += qt.liouvillian(None, [cl])
    E0.dims = L.dims
    # return generator and identity superop E0
    return L, E0


def _integrate(L, E0, ti, tf, integrator='propagator', parallel=False,
        opt=qt.Options()):
    """
    Basic ode integrator
    """
    if tf > ti:
        if integrator == 'mesolve':
            if parallel:
                warnings.warn('parallelization not implemented for "mesolve"')
            opt.store_final_state = True
            sol = qt.mesolve(L, E0, [ti, tf], [], [], options=opt)
            return sol.final_state
        elif integrator == 'propagator':
            return qt.propagator(L, (tf-ti), [], [], parallel=parallel,
                                 options=opt)*E0
        else:
            raise ValueError('integrator keyword must be either "propagator"' +
                              'or "mesolve"')
    else:
        return E0
