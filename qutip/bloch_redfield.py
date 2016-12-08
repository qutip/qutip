# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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

__all__ = ['brmesolve', 'bloch_redfield_solve', 'bloch_redfield_tensor']

import numpy as np
import scipy.integrate
import scipy.sparse as sp
from qutip.qobj import Qobj, isket
from qutip.superoperator import spre, spost, vec2mat, mat2vec, vec2mat_index
from qutip.expect import expect
from qutip.solver import Options, _solver_safety_check
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.cy.sparse_utils import dense2D_to_fastcsr_fmode
from qutip.solver import Result
from qutip.superoperator import liouvillian


# -----------------------------------------------------------------------------
# Solve the Bloch-Redfield master equation
#
def brmesolve(H, psi0, tlist, a_ops, e_ops=[], spectra_cb=[], c_ops=[],
              args={}, options=Options(),
              _safe_mode=True):
    """
    Solve the dynamics for a system using the Bloch-Redfield master equation.

    .. note::

        This solver does not currently support time-dependent Hamiltonians.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 / psi0: :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    a_ops : list of :class:`qutip.qobj`
        List of system operators that couple to bath degrees of freedom.

    e_ops : list of :class:`qutip.qobj` / callback function
        List of operators for which to evaluate expectation values.

    c_ops : list of :class:`qutip.qobj`
        List of system collapse operators.

    args : *dictionary*
        Placeholder for future implementation, kept for API consistency.

    options : :class:`qutip.solver.Options`
        Options for the solver.

    Returns
    -------

    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by `tlist`.
    """

    if _safe_mode:
        _solver_safety_check(H, psi0, a_ops, e_ops, args)
    
    if not spectra_cb:
        # default to infinite temperature white noise
        spectra_cb = [lambda w: 1.0 for _ in a_ops]

    R, ekets = bloch_redfield_tensor(H, a_ops, spectra_cb, c_ops)

    output = Result()
    output.solver = "brmesolve"
    output.times = tlist

    results = bloch_redfield_solve(R, ekets, psi0, tlist, e_ops, options)

    if e_ops:
        output.expect = results
    else:
        output.states = results

    return output


# -----------------------------------------------------------------------------
# Evolution of the Bloch-Redfield master equation given the Bloch-Redfield
# tensor.
#
def bloch_redfield_solve(R, ekets, rho0, tlist, e_ops=[], options=None):
    """
    Evolve the ODEs defined by Bloch-Redfield master equation. The
    Bloch-Redfield tensor can be calculated by the function
    :func:`bloch_redfield_tensor`.

    Parameters
    ----------

    R : :class:`qutip.qobj`
        Bloch-Redfield tensor.

    ekets : array of :class:`qutip.qobj`
        Array of kets that make up a basis tranformation for the eigenbasis.

    rho0 : :class:`qutip.qobj`
        Initial density matrix.

    tlist : *list* / *array*
        List of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        List of operators for which to evaluate expectation values.

    options : :class:`qutip.Qdeoptions`
        Options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`.

    """

    if options is None:
        options = Options()

    if options.tidy:
        R.tidyup()

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    dt = tlist[1] - tlist[0]
    result_list = []

    #
    # transform the initial density matrix and the e_ops opterators to the
    # eigenbasis
    #
    rho_eb = rho0.transform(ekets)
    e_eb_ops = [e.transform(ekets) for e in e_ops]

    for e_eb in e_eb_ops:
        if e_eb.isherm:
            result_list.append(np.zeros(n_tsteps, dtype=float))
        else:
            result_list.append(np.zeros(n_tsteps, dtype=complex))

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho_eb.full())
    r = scipy.integrate.ode(cy_ode_rhs)
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step, max_step=options.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    dt = np.diff(tlist)
    for t_idx, _ in enumerate(tlist):

        if not r.successful():
            break

        rho_eb.data = dense2D_to_fastcsr_fmode(vec2mat(r.y), rho0.shape[0], rho0.shape[1])

        # calculate all the expectation values, or output rho_eb if no
        # expectation value operators are given
        if e_ops:
            rho_eb_tmp = Qobj(rho_eb)
            for m, e in enumerate(e_eb_ops):
                result_list[m][t_idx] = expect(e, rho_eb_tmp)
        else:
            result_list.append(rho_eb.transform(ekets, True))

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    return result_list


# -----------------------------------------------------------------------------
# Functions for calculating the Bloch-Redfield tensor for a time-independent
# system.
#
def bloch_redfield_tensor(H, a_ops, spectra_cb, c_ops=[], use_secular=True):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment.

    .. note::

        This tensor generation requires a time-independent Hamiltonian.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        System Hamiltonian.

    a_ops : list of :class:`qutip.qobj`
        List of system operators that couple to the environment.

    spectra_cb : list of callback functions
        List of callback functions that evaluate the noise power spectrum
        at a given frequency.

    c_ops : list of :class:`qutip.qobj`
        List of system collapse operators.

    use_secular : bool
        Flag (True of False) that indicates if the secular approximation should
        be used.

    Returns
    -------

    R, kets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

        R is the Bloch-Redfield tensor and kets is a list eigenstates of the
        Hamiltonian.

    """

    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")

    for a in a_ops:
        if not isinstance(a, Qobj) or not a.isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")

    # default spectrum
    if not spectra_cb:
        spectra_cb = [lambda w: 1.0 for _ in a_ops]

    if c_ops is None:
        c_ops = []

    # use the eigenbasis
    evals, ekets = H.eigenstates()

    N = len(evals)
    K = len(a_ops)
    
    #only Lindblad collapse terms
    if K==0:
        Heb = H.transform(ekets)
        L = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])
        return L, ekets
    
    
    A = np.array([a_ops[k].transform(ekets).full() for k in range(K)])
    Jw = np.zeros((K, N, N), dtype=complex)

    # pre-calculate matrix elements and spectral densities
    # W[m,n] = real(evals[m] - evals[n])
    W = np.real(evals[:,np.newaxis] - evals[np.newaxis,:])

    for k in range(K):
        # do explicit loops here in case spectra_cb[k] can not deal with array arguments
        for n in range(N):
            for m in range(N):
                Jw[k, n, m] = spectra_cb[k](W[n, m])

    dw_min = np.abs(W[W.nonzero()]).min()

    # pre-calculate mapping between global index I and system indices a,b
    Iabs = np.empty((N*N,3),dtype=int)
    for I, Iab in enumerate(Iabs):
        # important: use [:] to change array values, instead of creating new variable Iab
        Iab[0]  = I
        Iab[1:] = vec2mat_index(N, I)

    # unitary part + dissipation from c_ops (if given):
    Heb = H.transform(ekets)
    L = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])
    
    # dissipative part:
    rows = []
    cols = []
    data = []
    for I, a, b in Iabs:
        # only check use_secular once per I
        if use_secular:
            # only loop over those indices J which actually contribute
            Jcds = Iabs[np.where(np.abs(W[a, b] - W[Iabs[:,1], Iabs[:,2]]) < dw_min / 10.0)]
        else:
            Jcds = Iabs
        for J, c, d in Jcds:
            elem = 0+0j
            # summed over k, i.e., each operator coupling the system to the environment
            elem += 0.5 * np.sum(A[:, a, c] * A[:, d, b] * (Jw[:, c, a] + Jw[:, d, b]))
            if b==d:
                #                  sum_{k,n} A[k, a, n] * A[k, n, c] * Jw[k, c, n])
                elem -= 0.5 * np.sum(A[:, a, :] * A[:, :, c] * Jw[:, c, :])
            if a==c:
                #                  sum_{k,n} A[k, d, n] * A[k, n, b] * Jw[k, d, n])
                elem -= 0.5 * np.sum(A[:, d, :] * A[:, :, b] * Jw[:, d, :])
            if elem != 0:
                rows.append(I)
                cols.append(J)
                data.append(elem)

    R = sp.coo_matrix((np.array(data),(np.array(rows),np.array(cols))),
            shape=(N**2,N**2),dtype=complex).tocsr()
    
    L.data = L.data + R
    
    return L, ekets
