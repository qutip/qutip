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
This module contains

-

"""

import qutip as qt


class MemoryCascade:
    """Class of options for

    Attributes
    ----------
    """

    def __init__(self, H_S, L1, L2, times=[], options=None):

        if options is None:
            self.options = qt.Options()
        else:
            self.options = options

        self.H_S = H_S
        self.sysdims = H_S.dims
        self.L1 = L1
        self.L2 = L2
        self.times = times
        self.store_states = self.options.store_states
        # create system identity superoperator
        self.Id = qt.qeye(H_S.shape[0])
        self.Id.dims = self.sysdims
        self.Id = qt.sprepost(self.Id, self.Id)

    def propagator(self, t, tau, notrace=False):
        """
        Compute propagator for time t
        """
        k = int(t/tau)+1
        s = t-(k-1)*tau
        G1, E0 = _generator(k, self.H_S, self.L1, self.L2)
        E = _integrate(G1, E0, 0., s, opt=self.options)
        if k > 1:
            G2, null = _generator(k-1, self.H_S, self.L1, self.L2)
            G2 = qt.composite(G2, self.Id)
            E = _integrate(G2, E, s, tau, opt=self.options)
        E.dims = E0.dims
        if not notrace:
            E = _genptrace(E, k)
        return qt.Qobj(E)

    def outfieldpropagator(self, blist, tlist, tau, notrace=False):
        """
        Compute propagator for computing field expectation values
        <O_n(tn)...O_2(t2)O_1(t1)> for times t1,t2,... and
        O_i = I, b_out, b_out^\dagger, b_loop, b_loop^\dagger
        tlist: list of times t1,..,tn
        blist: corresponding list of operators:
            0: I (nothing)
            1: b_out
            2: b_out^\dagger
            3: b_loop
            4: b_loop^\dagger
        """
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

        G1, E0 = _generator(kmax, self.H_S, self.L1, self.L2)
        sprev = 0.
        E = E0
        for i, s in enumerate(slist):
            E = _integrate(G1, E, sprev, s, opt=self.options)
            if klist[i] == 1:
                l1 = 0.*qt.Qobj()
            else:
                l1 = _localop(self.L1, klist[i]-1, kmax)
            l2 = _localop(self.L2, klist[i], kmax)
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
        E = _integrate(G1, E, slist[-1], tau, opt=self.options)

        E.dims = E0.dims
        if not notrace:
            E = _genptrace(E, kmax)
        return qt.Qobj(E)

    def rhot(self, rho0, t, tau):
        """
        Compute rho(t)
        """
        E = self.propagator(t, tau)
        rhovec = qt.operator_to_vector(rho0)
        return qt.vector_to_operator(E*rhovec)

    def outfieldcorr(self, rho0, blist, tlist, tau):
        """
        Compute <O_n(tn)...O_2(t2)O_1(t1)> for times t1,t2,... and
        O_i = b or b^\dagger are output field annihilation/creation operators
        times: list of times t1,..,tn
        blist: corresponding list of 0 for "b" and 1 for "b^\dagger"
        tau: time-delay
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
    for l in range(k-1):
        nsys = len(E.dims[0][0])
        E = qt.tensor_contract(E, (0, 2*nsys+1), (nsys, 3*nsys+1))
    return E


def _generator(k, H, L1, L2):
    """
    Create the generator for the cascaded chain of k system copies
    """
    id = qt.qeye(H.dims[0][0])
    Id = qt.sprepost(id, id)
    # create Lindbladian
    L = qt.Qobj()
    # first system
    H0 = 0.5*_localop(H, 1, k)
    L0 = _localop(L2, 1, k)
    L += qt.liouvillian(H0, [L0])
    E0 = Id
    # coupling systems 1 to k
    for l in range(1, k):
        E0 = qt.composite(E0, Id)
        h = _localop(H, l, k)
        hp = _localop(H, l+1, k)
        l1 = _localop(L1, l, k)
        l2p = _localop(L2, l+1, k)
        Hl = 0.5*(h + hp + 1j*(l1.dag()*l2p - l2p.dag()*l1))
        Ll = l1 + l2p
        L += qt.liouvillian(Hl, [Ll])
    # last system
    Hk = 0.5*_localop(H, k, k)
    Lk = _localop(L1, k, k)
    L += qt.liouvillian(Hk, [Lk])
    E0.dims = L.dims
    # return generator and identity superop E0
    return L, E0


def _integrate(L, E0, ti, tf, opt=qt.Options()):
    """
    Basic ode integrator
    """
    opt.store_final_state = True
    if tf > ti:
        sol = qt.mesolve(L, E0, [ti, tf], [], [], options=opt)
        return sol.final_state
    else:
        return E0
