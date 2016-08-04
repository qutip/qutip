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

__all__ = ['essolve', 'ode2es']

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from qutip.qobj import Qobj, issuper, isket, isoper
from qutip.eseries import eseries, estidy, esval
from qutip.expect import expect
from qutip.superoperator import liouvillian, mat2vec, vec2mat
from qutip.solver import Result
from qutip.operators import qzero


# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
def essolve(H, rho0, tlist, c_op_list, e_ops):
    """
    Evolution of a state vector or density matrix (`rho0`) for a given
    Hamiltonian (`H`) and set of collapse operators (`c_op_list`), by
    expressing the ODE as an exponential series. The output is either
    the state vector at arbitrary points in time (`tlist`), or the
    expectation values of the supplied operators (`e_ops`).

    Parameters
    ----------
    H : qobj/function_type
        System Hamiltonian.

    rho0 : :class:`qutip.qobj`
        Initial state density matrix.

    tlist : list/array
        ``list`` of times for :math:`t`.

    c_op_list : list of :class:`qutip.qobj`
        ``list`` of :class:`qutip.qobj` collapse operators.

    e_ops : list of :class:`qutip.qobj`
        ``list`` of :class:`qutip.qobj` operators for which to evaluate
        expectation values.


    Returns
    -------
     expt_array : array
        Expectation values of wavefunctions/density matrices for the
        times specified in ``tlist``.


    .. note:: This solver does not support time-dependent Hamiltonians.

    """
    n_expt_op = len(e_ops)
    n_tsteps = len(tlist)

    # Calculate the Liouvillian
    if (c_op_list is None or len(c_op_list) == 0) and isket(rho0):
        L = H
    else:
        L = liouvillian(H, c_op_list)

    es = ode2es(L, rho0)

    # evaluate the expectation values
    if n_expt_op == 0:
        results = [Qobj()] * n_tsteps
    else:
        results = np.zeros([n_expt_op, n_tsteps], dtype=complex)

    for n, e in enumerate(e_ops):
        results[n, :] = expect(e, esval(es, tlist))

    data = Result()
    data.solver = "essolve"
    data.times = tlist
    data.expect = [np.real(results[n, :]) if e.isherm else results[n, :]
                   for n, e in enumerate(e_ops)]

    return data


# -----------------------------------------------------------------------------
#
#
def ode2es(L, rho0):
    """Creates an exponential series that describes the time evolution for the
    initial density matrix (or state vector) `rho0`, given the Liouvillian
    (or Hamiltonian) `L`.

    Parameters
    ----------
    L : qobj
        Liouvillian of the system.

    rho0 : qobj
        Initial state vector or density matrix.

    Returns
    -------
    eseries : :class:`qutip.eseries`
        ``eseries`` represention of the system dynamics.

    """

    if issuper(L):

        # check initial state
        if isket(rho0):
            # Got a wave function as initial state: convert to density matrix.
            rho0 = rho0 * rho0.dag()

        # check if state is below error threshold
        if abs(rho0.full().sum()) < 1e-10 + 1e-24:
            # enforce zero operator
            return eseries(qzero(rho0.dims[0]))

        w, v = L.eigenstates()
        v = np.hstack([ket.full() for ket in v])
        # w[i]   = eigenvalue i
        # v[:,i] = eigenvector i

        rlen = np.prod(rho0.shape)
        r0 = mat2vec(rho0.full())
        v0 = la.solve(v, r0)
        vv = v * sp.spdiags(v0.T, 0, rlen, rlen)

        out = None
        for i in range(rlen):
            qo = Qobj(vec2mat(vv[:, i]), dims=rho0.dims, shape=rho0.shape)
            if out:
                out += eseries(qo, w[i])
            else:
                out = eseries(qo, w[i])

    elif isoper(L):

        if not isket(rho0):
            raise TypeError('Second argument must be a ket if first' +
                            'is a Hamiltonian.')

        # check if state is below error threshold
        if abs(rho0.full().sum()) < 1e-5 + 1e-20:
            # enforce zero operator
            dims = rho0.dims
            return eseries(Qobj(sp.csr_matrix((dims[0][0], dims[1][0]),
                                              dtype=complex)))

        w, v = L.eigenstates()
        v = np.hstack([ket.full() for ket in v])
        # w[i]   = eigenvalue i
        # v[:,i] = eigenvector i

        rlen = np.prod(rho0.shape)
        r0 = rho0.full()
        v0 = la.solve(v, r0)
        vv = v * sp.spdiags(v0.T, 0, rlen, rlen)

        out = None
        for i in range(rlen):
            qo = Qobj(np.matrix(vv[:, i]).T, dims=rho0.dims, shape=rho0.shape)
            if out:
                out += eseries(qo, -1.0j * w[i])
            else:
                out = eseries(qo, -1.0j * w[i])

    else:
        raise TypeError('First argument must be a Hamiltonian or Liouvillian.')

    return estidy(out)
