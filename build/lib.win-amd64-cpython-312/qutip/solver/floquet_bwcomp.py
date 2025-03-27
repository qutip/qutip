""" Floquet solver compatibility functions that behave like the corresponding
functions from QuTiP 4.7.

These functions are indented to be used when porting code from QuTiP 4.7 to
QuTiP 5. They are deprecated and will be removed in QuTiP 5.1.
"""

__all__ = [
    "floquet_modes",
    "floquet_modes_t",
    "floquet_modes_table",
    "floquet_modes_t_lookup",
    "floquet_states",
    "floquet_states_t",
    "floquet_wavefunction",
    "floquet_wavefunction_t",
    "floquet_state_decomposition",
    "floquet_master_equation_rates",
]

from .floquet import *
import numpy as np
import warnings


def floquet_modes(H, T, args=None, sort=False, U=None, options=None):
    """
    Calculate the initial Floquet modes Phi_alpha(0) for a driven system with
    period T.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options, sort=sort)
    f_mode_0 = fbasis.mode(0)
    f_energies = fbasis.e_quasi
    """
    warnings.warn(FutureWarning(
        "`floquet_modes` is deprecated. Use `FloquetBasis.mode` instead."
    ))
    fbasis = FloquetBasis(H, T, args=args, options=options, sort=sort)
    f_mode_0 = fbasis.mode(0)
    f_energies = fbasis.e_quasi
    return f_mode_0, f_energies


def floquet_modes_t(f_modes_0, f_energies, t, H, T, args=None, options=None):
    """
    Calculate the Floquet modes at times tlist Phi_alpha(tlist) propagting the
    initial Floquet modes Phi_alpha(0).

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options)
    f_mode_t = fbasis.mode(t)
    """
    warnings.warn(FutureWarning(
        "`floquet_modes_t` is deprecated. Use `FloquetBasis.mode` instead."
    ))
    fbasis = FloquetBasis(H, T, args=args, options=options)
    return fbasis.mode(t)


def floquet_modes_table(
    f_modes_0, f_energies, tlist, H, T, args=None, options=None
):
    """
    Pre-calculate the Floquet modes for a range of times spanning the floquet
    period. Can later be used as a table to look up the floquet modes for
    any time.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options, precompute=tlist)
    """
    raise NotImplementedError("`floquet_modes_table` is deprecated.")


def floquet_modes_t_lookup(f_modes_table_t, t, T):
    """
    Lookup the floquet mode at time t in the pre-calculated table of floquet
    modes in the first period of the time-dependence.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    f_modes_table_t = fbasis = FloquetBasis(...)
    f_mode_t = f_modes_table_t.mode(t)
    """
    raise NotImplementedError(
        "`floquet_modes_t_lookup` is no longer provided. "
        "Use `FloquetBasis` instead."
    )


def floquet_states(f_modes_t, f_energies, t):
    """
    Evaluate the floquet states at time t given the Floquet modes at that time.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options)
    f_state_t = fbasis.state(t)
    """
    warnings.warn(FutureWarning(
        "`floquet_states` is deprecated. "
        "Use `FloquetBasis.state` instead."
    ))
    return [
        (f_modes_t[i] * np.exp(-1j * f_energies[i] * t))
        for i in np.arange(len(f_energies))
    ]


def floquet_states_t(f_modes_0, f_energies, t, H, T, args=None, options=None):
    """
    Evaluate the floquet states at time t given the initial Floquet modes.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options)
    f_state_t = fbasis.state(t)
    """
    warnings.warn(FutureWarning(
        "`floquet_states_t` is deprecated. "
        "Use `FloquetBasis.state` instead."
    ))
    fbasis = FloquetBasis(H, T, args=args, options=options)
    return fbasis.state(t)


def floquet_wavefunction(f_modes_t, f_energies, f_coeff, t):
    """
    Evaluate the wavefunction for a time t using the Floquet state
    decompositon, given the Floquet modes at time `t`.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options)
    psi_t = fbasis.from_floquet_basis(f_coeff, t)
    """
    raise NotImplementedError(
        "`floquet_wavefunction` is not longer provided. "
        "Use `FloquetBasis.from_floquet_basis` instead."
    )


def floquet_wavefunction_t(
    f_modes_0, f_energies, f_coeff, t, H, T, args=None, options=None
):
    """
    Evaluate the wavefunction for a time t using the Floquet state
    decompositon, given the initial Floquet modes.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options)
    psi_t = fbasis.from_floquet_basis(f_coeff, t)
    """
    warnings.warn(FutureWarning(
        "`floquet_wavefunction_t` is deprecated. "
        "Use `FloquetBasis.from_floquet_basis` instead."
    ))
    fbasis = FloquetBasis(H, T, args=args, options=options)
    return fbasis.from_floquet_basis(f_coeff, t)


def floquet_state_decomposition(f_states, f_energies, psi):
    r"""
    Decompose the wavefunction `psi` (typically an initial state) in terms of
    the Floquet states, :math:`\psi = \sum_\alpha c_\alpha \psi_\alpha(0)`.

    Deprecated from qutip v5. Use :class:`.FloquetBasis` instead:

    fbasis = FloquetBasis(H, T, args=args, options=options)
    f_coeff = fbasis.to_floquet_basis(psi)
    """
    raise NotImplementedError(
        "`floquet_state_decomposition` is deprecated. "
        "Use `FloquetBasis.to_floquet_basis` instead."
    )


def floquet_master_equation_rates(
    f_modes_0,
    f_energies,
    c_op,
    H,
    T,
    args,
    J_cb,
    w_th,
    kmax=5,
    f_modes_table_t=None,
):
    """
    Calculate the rates and matrix elements for the Floquet-Markov master
    equation.

    .. note ::

        Deprecated. For the Floquet-Markov master equation's tensor, use
        :func:`floquet_tensor`. For the rates matrices, use
        :func:`floquet_delta_tensor`, :func:`floquet_X_matrices`,
        :func:`floquet_gamma_matrices` and/or s:func:`floquet_A_matrix`.

    Parameters
    ----------
    f_modes_0 : Any
        No longer used.
    f_energies : Any
        No longer used.
    c_op : :class:`.Qobj`
        The collapse operators describing the dissipation.
    H : :class:`.Qobj`
        System Hamiltonian, time-dependent with period `T`.
    T : float
        The period of the time-dependence of the hamiltonian.
    args : dictionary
        Dictionary with variables required to evaluate H.
    J_cb : callback functions
        A callback function that computes the noise power spectrum, as
        a function of frequency, associated with the collapse operator `c_op`.
    w_th : float
        The temperature in units of frequency.
    kmax : int, default=5
        The truncation of the number of sidebands.
    f_modes_table_t : Any
        No longer used.

    Returns
    -------
    output : list
        A list (Delta, X, Gamma, A) containing the matrices Delta, X, Gamma
        and A used in the construction of the Floquet-Markov master equation.
    """
    warnings.warn(
        FutureWarning("`floquet_master_equation_rates` is deprecated.")
    )
    floquet_basis = FloquetBasis(H, T, args=args)
    energy = floquet_basis.e_quasi
    delta = floquet_delta_tensor(energy, kmax, T)
    x = floquet_X_matrices(floquet_basis, [c_op], kmax, nT)
    gamma = floquet_gamma_matrices(x, delta, [J_cb])
    a = floquet_A_matrix(delta, gamma, w_th)
    return delta, x[0], gamma, a
