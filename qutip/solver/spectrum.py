__all__ = ['spectrum', 'spectrum_correlation_fft']

import numpy as np
import scipy.fftpack

from .steadystate import steadystate
from ..core import (
    qeye, Qobj, liouvillian, spre, unstack_columns, stack_columns,
    tensor, qzero, expect
)


def spectrum(H, wlist, c_ops, a_op, b_op, solver="es", use_pinv=False):
    r"""
    Calculate the spectrum of the correlation function
    :math:`\lim_{t \to \infty} \left<A(t+\tau)B(t)\right>`,
    i.e., the Fourier transform of the correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty}
        \lim_{t \to \infty} \left<A(t+\tau)B(t)\right>
        e^{-i\omega\tau} d\tau.

    using the solver indicated by the `solver` parameter. Note: this spectrum
    is only defined for stationary statistics (uses steady state rho0)

    Parameters
    ----------
    H : :class:`qutip.qobj`
        system Hamiltonian.
    wlist : array_like
        List of frequencies for :math:`\omega`.
    c_ops : list
        List of collapse operators.
    a_op : Qobj
        Operator A.
    b_op : Qobj
        Operator B.
    solver : str
        Choice of solver (`es` for exponential series and
        `pi` for psuedo-inverse).
    use_pinv : bool
        For use with the `pi` solver: if `True` use numpy's pinv method,
        otherwise use a generic solver.

    Returns
    -------
    spectrum : array
        An array with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """
    if solver == "es":
        return _spectrum_es(H, wlist, c_ops, a_op, b_op)
    elif solver == "pi":
        return _spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv)
    raise ValueError("Unrecognized choice of solver {} (use es or pi)."
                     .format(solver))


def spectrum_correlation_fft(tlist, y, inverse=False):
    """
    Calculate the power spectrum corresponding to a two-time correlation
    function using FFT.

    Parameters
    ----------
    tlist : array_like
        list/array of times :math:`t` which the correlation function is given.
    y : array_like
        list/array of correlations corresponding to time delays :math:`t`.
    inverse: boolean
        boolean parameter for using a positive exponent in the Fourier
        Transform instead. Default is False.

    Returns
    -------
    w, S : tuple
        Returns an array of angular frequencies 'w' and the corresponding
        two-sided power spectrum 'S(w)'.

    """
    tlist = np.asarray(tlist)
    N = tlist.shape[0]
    dt = tlist[1] - tlist[0]
    if not np.allclose(np.diff(tlist), dt * np.ones(N - 1, dtype=float)):
        raise ValueError('tlist must be equally spaced for FFT.')
    F = (N * scipy.fftpack.ifft(y)) if inverse else scipy.fftpack.fft(y)
    # calculate the frequencies for the components in F
    f = scipy.fftpack.fftfreq(N, dt)
    # re-order frequencies from most negative to most positive (centre on 0)
    idx = np.array([], dtype='int')
    idx = np.append(idx, np.where(f < 0.0))
    idx = np.append(idx, np.where(f >= 0.0))
    return 2 * np.pi * f[idx], 2 * dt * np.real(F[idx])


def _spectrum_es(H, wlist, c_ops, a_op, b_op):
    r"""
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\tau)B(0)\right>`.
    """
    # construct the Liouvillian
    L = liouvillian(H, c_ops)
    # find the steady state density matrix and a_op and b_op expecation values
    rho0 = steadystate(L)
    a_op_ss = expect(a_op, rho0)
    b_op_ss = expect(b_op, rho0)
    # eseries solution for (b * rho0)(t)
    states, rates = _diagonal_evolution(L, b_op * rho0)
    # correlation
    ampls = expect(a_op, states)
    # make covariance
    ampls = np.concatenate([ampls, [-a_op_ss * b_op_ss]])
    rates = np.concatenate([rates, [0]])
    # Tidy up similar rates.
    uniques = {}
    for r, a in zip(rates, ampls):
        for r_ in uniques:
            if np.abs(r - r_) < 1e-10:
                uniques[r_] += a
                break
        else:
            uniques[r] = a
    ampls, rates = [], []
    for r, a in uniques.items():
        if np.abs(a) > 1e-10:
            ampls.append(a)
            rates.append(r)
    ampls, rates = np.array(ampls), np.array(rates)
    return np.array([2 * np.dot(ampls, 1 / (1j * w - rates)).real
                     for w in wlist])


#
# pseudo-inverse solvers
def _spectrum_pi(H, wlist, c_ops, a_op, b_op, use_pinv=False):
    r"""
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\tau)B(0)\right>`.
    """
    L = H if H.issuper else liouvillian(H, c_ops)
    tr_mat = tensor([qeye(n) for n in L.dims[0][0]])
    N = np.prod(L.dims[0][0])
    A = L.full()
    b = spre(b_op).full()
    a = spre(a_op).full()

    tr_vec = np.transpose(stack_columns(tr_mat.full()))

    rho_ss = steadystate(L)
    rho = np.transpose(stack_columns(rho_ss.full()))

    I = np.identity(N * N)
    P = np.kron(np.transpose(rho), tr_vec)
    Q = I - P

    spectrum = np.zeros(len(wlist))
    for idx, w in enumerate(wlist):
        if use_pinv:
            MMR = np.linalg.pinv(-1.0j * w * I + A)
        else:
            MMR = np.dot(Q, np.linalg.solve(-1.0j * w * I + A, Q))

        s = np.dot(tr_vec,
                   np.dot(a, np.dot(MMR, np.dot(b, np.transpose(rho)))))
        spectrum[idx] = -2 * np.real(s[0, 0])
    return spectrum


def _diagonal_evolution(L, rho0):
    """
    Diagonalise the evolution of density matrix rho0 under a constant
    Liouvillian L.  Returns a list of `states` and an array of the eigenvalues
    such that the time evolution of rho0 is represented by
        sum_k states[k] * exp(evals[k] * t)
    This is effectively the same as the legacy QuTiP function ode2es, but does
    not use the removed eseries class.  It exists here because ode2es and
    essolve were removed.
    """
    rho0_full = rho0.full()
    if np.abs(rho0_full).sum() < 1e-10 + 1e-24:
        return qzero(rho0.dims[0]), np.array([0])
    evals, evecs = L.eigenstates()
    evecs = np.vstack([ket.full()[:, 0] for ket in evecs]).T
    # evals[i]   = eigenvalue i
    # evecs[:, i] = eigenvector i
    size = rho0.shape[0] * rho0.shape[1]
    r0 = stack_columns(rho0_full)[:, 0]
    v0 = scipy.linalg.solve(evecs, r0)
    vv = evecs * v0[None, :]  # product equivalent to `evecs @ np.diag(v0)`
    states = [Qobj(unstack_columns(vv[:, i]), dims=rho0.dims, type='oper')
              for i in range(size)]
    # We don't use QobjEvo because it isn't designed to be efficient when
    # calculating
    return states, evals
