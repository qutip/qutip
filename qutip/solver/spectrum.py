__all__ = ['spectrum', 'spectrum_correlation_fft']

import numpy as np
import scipy.fftpack

from .steadystate import steadystate
from ..core import liouvillian, spre, expect
from ..core import data as _data
from qutip.settings import settings

def spectrum(H, wlist, c_ops, a_op, b_op, solver="es"):
    r"""
    Calculate the spectrum of the correlation function
    :math:`\lim_{t \to \infty} \left<A(t+\tau)B(t)\right>`,
    i.e., the Fourier transform of the correlation function:

    .. math::

        S(\omega) = \int_{-\infty}^{\infty}
        \lim_{t \to \infty} \left<A(t+\tau)B(t)\right>
        e^{-i\omega\tau} d\tau.

    using the solver indicated by the ``solver`` parameter. Note: this spectrum
    is only defined for stationary statistics (uses steady state rho0)

    Parameters
    ----------
    H : :class:`.qobj`
        system Hamiltonian.
    wlist : array_like
        List of frequencies for :math:`\omega`.
    c_ops : list
        List of collapse operators.
    a_op : :class:`.Qobj`
        Operator A.
    b_op : :class:`.Qobj`
        Operator B.
    solver : str, {'es', 'pi', 'solve'}, default: 'es'
        Choice of solver, ``es`` for exponential series and
        ``pi`` for psuedo-inverse, ``solve`` for generic solver.

    Returns
    -------
    spectrum : array
        An array with spectrum :math:`S(\omega)` for the frequencies
        specified in `wlist`.

    """
    if not H.issuper:
        L = liouvillian(H, c_ops)
    else:
        L = H + sum([lindblad_dissipator(c) for c in c_ops])
    if solver == "es":
        return _spectrum_es(L, wlist, a_op, b_op)
    elif solver in ["pi", "solve"]:
        return _spectrum_pi(L, wlist, a_op, b_op, use_pinv=solver=="pi")
    raise ValueError(
        f"Unrecognized choice of solver {solver} (use 'es', 'pi' or 'solve')."
    )


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
    inverse: bool, default: False
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


def _spectrum_es(L, wlist, a_op, b_op):
    r"""
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\tau)B(0)\right>`.
    """
    # find the steady state density matrix and a_op and b_op expecation values
    rho0 = steadystate(L)
    a_op_ss = expect(a_op, rho0)
    b_op_ss = expect(b_op, rho0)
    # eseries solution for (b * rho0)(t)
    states, rates = _diagonal_evolution(L, b_op * rho0)
    # correlation
    ampls = [_data.expect(a_op.data, state) for state in states]
    # make covariance
    ampls += [-a_op_ss * b_op_ss]
    rates += [0]
    # Tidy up similar rates.
    order = np.argsort(rates)
    clean_rates = []
    clean_ampls = []
    prev_rate = np.nan
    for idx in order:
        if np.abs(rates[idx] - prev_rate) < settings.core["atol"]:
            clean_ampls[-1] += ampls[idx]
        else:
            clean_rates.append(rates[idx])
            clean_ampls.append(ampls[idx])
            prev_rate = rates[idx]
    # Remove 0 amplitude
    rates, ampls = zip(*[
        (rate, ampl)
        for rate, ampl in zip(clean_rates, clean_ampls)
        if np.abs(ampl) > settings.core["atol"]
    ])
    ampls, rates = np.array(ampls), np.array(rates)
    LW = np.subtract.outer(1j * np.array(wlist), rates).T
    return (ampls @ (2 / LW)).real


#
# pseudo-inverse solvers
def _spectrum_pi(L, wlist, a_op, b_op, use_pinv=False):
    r"""
    Internal function for calculating the spectrum of the correlation function
    :math:`\left<A(\tau)B(0)\right>`.
    """
    dtype = type(L.data)
    rho_ss = steadystate(L)
    tr_mat = _data.identity[dtype](rho_ss.shape[0])
    tr_vec = _data.column_stack(tr_mat).transpose()
    rho = _data.column_stack(rho_ss.data)

    A = L.data
    ket = spre(b_op).data @ rho
    bra = tr_vec @ spre(a_op).data

    I = _data.identity[dtype](L.shape[0])
    P = _data.kron(rho, tr_vec)
    Q = I - P

    spectrum = np.zeros(len(wlist))
    for idx, w in enumerate(wlist):
        if use_pinv and np.abs(w) > settings.core["atol"]:
            # At w == 0., "L - iw" is singular
            MMR = _data.inv(-1.0j * w * I + A)
        else:
            MMR = Q @ _data.solve(-1.0j * w * I + A, Q)

        spectrum[idx] = -2 * _data.inner_op(bra, MMR, ket).real
    return spectrum


def _diagonal_evolution(L, rho0, sparse=False):
    if rho0.norm() < settings.core["atol"]:
        return [_data.zeros["CSR"](*rho0.shape)], [0]
    if isinstance(L.data, _data.CSR) and not sparse:
        L = L.to(_data.Dense)
    evals, evecs = _data.eigs(L.data)
    size = rho0.shape[0] * rho0.shape[1]
    r0 = _data.column_stack(rho0.data)
    v0 = _data.solve(evecs, r0)
    vv = evecs @ _data.diag(v0.to_array().flatten(), [0])
    states = []
    rates = []
    for ket, rate in zip(_data.split_columns(vv), evals):
        if _data.norm.l2(ket) < settings.core["atol"]:
            continue
        states.append(_data.column_unstack(ket, rho0.shape[0]))
        rates.append(rate)
    return states, rates
