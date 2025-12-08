# -*- coding: utf-8 -*-

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

from qutip import (
    sprepost,
    Qobj,
    spre,
    spost,
    liouvillian,
    qeye,
    mesolve,
    propagator,
    composite,
    isket,
    ket2dm,
    tensor_contract,
)


__all__ = ["MemoryCascade"]


class MemoryCascade:
    """Class for running memory cascade simulations of open quantum systems
    with time-delayed coherent feedback.

    Attributes
    ----------
    H_S : :class:`.Qobj`
        System Hamiltonian (can also be a Liouvillian)

    L1 : :class:`.Qobj` / list of :class:`.Qobj`
        System operators coupling into the feedback loop. Can be a single
        operator or a list of operators.

    L2 : :class:`.Qobj` / list of :class:`.Qobj`
        System operators coupling out of the feedback loop. Can be a single
        operator or a list of operators. L2 must have the same length as L1.

    S_matrix: *array*
        S matrix describing which operators in L1 are coupled to which
        operators in L2 by the feedback channel. Defaults to an n by n identity
        matrix where n is the number of elements in L1/L2.

    c_ops_markov : :class:`.Qobj` / list of :class:`.Qobj`
        Decay operators describing conventional Markovian decay channels.
        Can be a single operator or a list of operators.

    integrator : str {'propagator', 'mesolve'}
        Integrator method to use. Defaults to 'propagator' which tends to be
        faster for long times (i.e., large Hilbert space).

    options : dict
        Generic solver options.
    """

    def __init__(
        self,
        H_S,
        L1,
        L2,
        S_matrix=None,
        c_ops_markov=None,
        integrator="propagator",
        options=None,
    ):

        if options is None:
            self.options = {"progress_bar": False}
        else:
            self.options = options

        self.H_S = H_S
        self.sysdims = H_S.dims
        if isinstance(L1, Qobj):
            self.L1 = [L1]
        else:
            self.L1 = L1
        if isinstance(L2, Qobj):
            self.L2 = [L2]
        else:
            self.L2 = L2
        if not len(self.L1) == len(self.L2):
            raise ValueError("L1 and L2 has to be of equal length.")
        if isinstance(c_ops_markov, Qobj):
            self.c_ops_markov = [c_ops_markov]
        else:
            self.c_ops_markov = c_ops_markov

        if S_matrix is None:
            self.S_matrix = np.identity(len(self.L1))
        else:
            self.S_matrix = S_matrix
        # create system identity superoperator
        self.Id = qeye(H_S.shape[0])
        self.Id.dims = self.sysdims
        self.Id = sprepost(self.Id, self.Id)
        self.store_states = self.options.get("store_states", False)
        self.integrator = integrator
        self._generators = {}

    def generator(self, k):
        if k not in self._generators:
            self._generators[k] = _generator(
                k, self.H_S, self.L1, self.L2, self.S_matrix, self.c_ops_markov
            )
        return self._generators[k]

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
        : :class:`.Qobj`
            time-propagator for reduced system dynamics
        """
        k = int(t / tau) + 1
        s = t - (k - 1) * tau
        G1 = self.generator(k)
        E0 = qeye(G1.dims[0])
        E = _integrate(
            G1, E0, 0.0, s, integrator=self.integrator, opt=self.options
        )
        if k > 1:
            G2 = self.generator(k - 1)
            G2 = composite(G2, self.Id)
            E = _integrate(
                G2, E, s, tau, integrator=self.integrator, opt=self.options
            )

        if not notrace:
            E = _genptrace(E, k)
        return E

    def outfieldpropagator(
        self, blist, tlist, tau, c1=None, c2=None, notrace=False
    ):
        r"""
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

        c1 : :class:`.Qobj`
            system collapse operator that couples to the in-loop field in
            question (only needs to be specified if self.L1 has more than one
            element)

        c2 : :class:`.Qobj`
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
        : :class:`.Qobj`
            time-propagator for computing field correlation function
        """
        if c1 is None and len(self.L1) == 1:
            c1 = self.L1[0]
        else:
            raise ValueError(
                "Argument c1 has to be specified when more than"
                + "one collapse operator couples to the feedback"
                + "loop."
            )
        if c2 is None and len(self.L2) == 1:
            c2 = self.L2[0]
        else:
            raise ValueError(
                "Argument c1 has to be specified when more than"
                + "one collapse operator couples to the feedback"
                + "loop."
            )
        klist = []
        slist = []
        for t in tlist:
            klist.append(int(t / tau) + 1)
            slist.append(t - (klist[-1] - 1) * tau)
        kmax = max(klist)
        zipped = sorted(zip(slist, klist, blist))
        slist = [s for (s, k, b) in zipped]
        klist = [k for (s, k, b) in zipped]
        blist = [b for (s, k, b) in zipped]

        G1 = self.generator(kmax)
        sprev = 0.0
        E0 = qeye(G1.dims[0])
        E = E0
        for i, s in enumerate(slist):
            E = _integrate(
                G1, E, sprev, s, integrator=self.integrator, opt=self.options
            )
            l2 = _localop(c2, klist[i], kmax)
            if klist[i] == 1:
                l1 = l2 * 0.0
            else:
                l1 = _localop(c1, klist[i] - 1, kmax)
            if blist[i] == 0:
                superop = self.Id
            elif blist[i] == 1:
                superop = spre(l1 + l2)
            elif blist[i] == 2:
                superop = spost(l1.dag() + l2.dag())
            elif blist[i] == 3:
                superop = spre(l1)
            elif blist[i] == 4:
                superop = spost(l1.dag())
            else:
                raise ValueError(
                    "Allowed values in blist are 0, 1, 2, 3 and 4."
                )

            E = superop @ E
            sprev = s
        E = _integrate(
            G1, E, slist[-1], tau, integrator=self.integrator, opt=self.options
        )

        if not notrace:
            E = _genptrace(E, kmax)
        return E

    def rhot(self, rho0, t, tau):
        """
        Compute the reduced system density matrix :math:`\\rho(t)`

        Parameters
        ----------
        rho0 : :class:`.Qobj`
            initial density matrix or state vector (ket)

        t : float
            current time

        tau : float
            time-delay

        Returns
        -------
        : :class:`.Qobj`
            density matrix at time :math:`t`
        """
        if isket(rho0):
            rho0 = ket2dm(rho0)

        E = self.propagator(t, tau)
        return E(rho0)

    def outfieldcorr(self, rho0, blist, tlist, tau, c1=None, c2=None):
        r"""
        Compute output field expectation value
        <O_n(tn)...O_2(t2)O_1(t1)> for times t1,t2,... and
        O_i = I, b_out, b_out^\dagger, b_loop, b_loop^\dagger


        Parameters
        ----------
        rho0 : :class:`.Qobj`
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

        c1 : :class:`.Qobj`
            system collapse operator that couples to the in-loop field in
            question (only needs to be specified if self.L1 has more than one
            element)

        c2 : :class:`.Qobj`
            system collapse operator that couples to the output field in
            question (only needs to be specified if self.L2 has more than one
            element)

        Returns
        -------
        : complex
            expectation value of field correlation function
        """
        if isket(rho0):
            rho0 = ket2dm(rho0)

        E = self.outfieldpropagator(blist, tlist, tau)
        return (E(rho0)).tr()


def _localop(op, l, k):
    """
    Create a local operator on the l'th system by tensoring
    with identity operators on all the other k-1 systems
    """
    if l < 1 or l > k:
        raise IndexError("index l out of range")
    out = op
    if l > 1:
        out = qeye(op.dims[0] * (l - 1)) & out
    if l < k:
        out = out & qeye(op.dims[0] * (k - l))

    return out


def _genptrace(E, k):
    """
    Perform a gneralized partial trace on a superoperator E, tracing out all
    subsystems but one.
    """
    for l in range(k - 1):
        nsys = len(E.dims[0][0])
        E = tensor_contract(E, (0, 2 * nsys + 1), (nsys, 3 * nsys + 1))
    return E


def _generator(k, H, L1, L2, S=None, c_ops_markov=None):
    """
    Create a Liouvillian for a cascaded chain of k system copies
    """
    id = qeye(H.dims[0][0])
    Id = sprepost(id, id)
    if S is None:
        S = np.identity(len(L1))
    # create Lindbladian

    # first system
    L = liouvillian(None, [_localop(c, 1, k) for c in L2])
    for l in range(1, k):
        # Bare Hamiltonian
        Hl = _localop(H, l, k)
        L += liouvillian(Hl, [])
        # Markovian Decay channels
        if c_ops_markov is not None:
            for c in c_ops_markov:
                cl = _localop(c, l, k)
                L += liouvillian(None, [cl])
        # Cascade coupling
        c1 = np.array([_localop(c, l, k) for c in L1])
        c2 = np.array([_localop(c, l + 1, k) for c in L2])
        c2dag = np.array([c.dag() for c in c2])
        Hcasc = -0.5j * np.dot(c2dag, np.dot(S, c1))
        Hcasc += Hcasc.dag()
        Lvec = c2 + np.dot(S, c1)
        L += liouvillian(Hcasc, [c for c in Lvec])
    # last system
    L += liouvillian(_localop(H, k, k), [_localop(c, k, k) for c in L1])
    if c_ops_markov is not None:
        for c in c_ops_markov:
            cl = _localop(c, k, k)
            L += liouvillian(None, [cl])
    # return generator
    return L


def _integrate(L, E0, ti, tf, integrator="propagator", opt=None):
    """
    Basic ode integrator
    """
    opt = opt or {}
    if tf > ti:
        if integrator == "mesolve":
            opt["store_final_state"] = True
            sol = mesolve(L, E0, [ti, tf], options=opt)
            return sol.final_state
        elif integrator == "propagator":
            return propagator(L, (tf - ti), options=opt) @ E0
        else:
            raise ValueError(
                'integrator keyword must be either "propagator" or "mesolve"'
            )
    else:
        return E0
