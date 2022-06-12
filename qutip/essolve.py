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

# Only used for deprecation warnings.
import functools
import warnings


def _deprecate(alternative):
    def decorated(f):
        message = (
            f"{f.__name__} is to be removed in QuTiP 5.0"
            f", consider swapping to {alternative}."
        )

        @functools.wraps(f)
        def out(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return f(*args, **kwargs)
        return out
    return decorated


# -----------------------------------------------------------------------------
# pass on to wavefunction solver or master equation solver depending on whether
# any collapse operators were given.
#
@_deprecate("mesolve")
def essolve(H, rho0, tlist, c_op_list, e_ops):
    """
    Evolution of a state vector or density matrix (`rho0`) for a given
    Hamiltonian (`H`) and set of collapse operators (`c_op_list`), by
    expressing the ODE as an exponential series. The output is either
    the state vector at arbitrary points in time (`tlist`), or the
    expectation values of the supplied operators (`e_ops`).

    .. deprecated:: 4.6.0
        :obj:`~essolve` will be removed in QuTiP 5.  Please use
        :obj:`~qutip.sesolve` or :obj:`~qutip.mesolve` for general-purpose
        integration of the Schroedinger/Lindblad master equation.
        This will likely be faster than :obj:`~essolve` for you.

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
@_deprecate("direct eigenstate and -value calculation")
def ode2es(L, rho0):
    """Creates an exponential series that describes the time evolution for the
    initial density matrix (or state vector) `rho0`, given the Liouvillian
    (or Hamiltonian) `L`.

    .. deprecated:: 4.6.0
        :obj:`~ode2es` will be removed in QuTiP 5.  Please use
        :obj:`qutip.Qobj.eigenstates` to get the eigenstates and -values,
        and use :obj:`~qutip.QobjEvo` for general time-dependence.

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
        if abs(rho0.full()).sum() < 1e-10 + 1e-24:
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
        if abs(rho0.full()).sum() < 1e-5 + 1e-20:
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
            qo = Qobj(np.array(vv[:, i]).T, dims=rho0.dims, shape=rho0.shape)
            if out:
                out += eseries(qo, -1.0j * w[i])
            else:
                out = eseries(qo, -1.0j * w[i])

    else:
        raise TypeError('First argument must be a Hamiltonian or Liouvillian.')

    return estidy(out)
