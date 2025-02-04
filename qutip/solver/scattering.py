"""
Photon scattering in quantum optical systems

This module includes a collection of functions for numerically computing photon
scattering in driven arbitrary systems coupled to some configuration of output
waveguides. The implementation of these functions closely follows the
mathematical treatment given in K.A. Fischer, et. al., Scattering of Coherent
Pulses from Quantum Optical Systems (2017, arXiv:1710.02875).
"""
# Author:  Ben Bartlett
# Contact: benbartlett@stanford.edu

import numpy as np
from scipy.integrate import trapezoid
from itertools import product, combinations_with_replacement
from ..core import basis, tensor, zero_ket, Qobj, QobjEvo
from .propagator import propagator, Propagator

__all__ = ['temporal_basis_vector',
           'temporal_scattered_state',
           'scattering_probability']


def set_partition(collection, num_sets):
    """
    Enumerate all ways of partitioning collection into num_sets different
    lists, e.g. :
    list(set_partition([1,2], 2))
    >>> [[[1, 2], []], [[1], [2]], [[2], [1]], [[], [1, 2]]]

    Parameters
    ----------
    collection : iterable
        Collection to generate a set partition of.
    num_sets : int
        Number of sets to partition collection into.

    Returns
    -------
    partition : iterable
        The partitioning of collection into num_sets sets.
    """
    for partitioning in product(range(num_sets), repeat=len(collection)):
        partition = [[] for _ in range(num_sets)]
        for i, set_index in enumerate(partitioning):
            partition[set_index].append(collection[i])
        yield tuple(tuple(indices) for indices in partition)


def photon_scattering_amplitude(propagator, c_ops, tlist, taus, psi, psit):
    """
    Compute the scattering amplitude for a system emitting into multiple
    waveguides.

    Parameters
    ----------
    propagator : :class:`.Propagator`
        Propagator
    c_ops : list
        list of collapse operators for each waveguide; these are assumed to
        include spontaneous decay rates, e.g.
        :math:`\\sigma = \\sqrt \\gamma \\cdot a`
    tlist : array_like
        List of times starting at the beginning and ending at the end of the
        evolution.
    taus : list-like
        List of (list of emission times) for each waveguide.
    psi : Qobj
        State at the start of the evolution
    psit : Qobj
        State at the end of the evolution.
    """
    # Extract the full list of taus
    tau_collapse = []
    for i, tau_wg in enumerate(taus):
        for t in tau_wg:
            tau_collapse.append((t, i))
    tau_collapse.sort(key=lambda tup: tup[0])  # sort tau_collapse by time

    tq = tlist[0]
    # Compute Prod Ueff(tq, tq-1)
    for tau in tau_collapse:
        tprev = tq
        tq, q = tau
        psi = c_ops[q] * propagator(tq, tprev) * psi

    psi = propagator(tlist[-1], tq) * psi
    return psit.overlap(psi)


def _temporal_basis_idx(waveguide_emission_indices, n_time_bins):
    """
    Generate a the global index for the excitation.
    """
    idx = []
    for i, wg_indices in enumerate(waveguide_emission_indices):
        for index in wg_indices:
            idx += [i * n_time_bins + index]
    idx = idx or [0]
    return tuple(idx)


def _temporal_basis_dims(waveguide_emission_indices, n_time_bins,
                         n_emissions=None):
    """
    Return the dims of the ``temporal_basis_vector``.
    """
    # TODO: Review n_emissions: change the number of dims but the equivalent
    # does not exist in _temporal_basis_idx
    num_col = len(waveguide_emission_indices)
    if n_emissions is None:
        n_emissions = sum(
            [len(waveguide_indices)
             for waveguide_indices in waveguide_emission_indices])
    n_emissions = n_emissions or 1
    return [num_col * n_time_bins] * n_emissions


def temporal_basis_vector(waveguide_emission_indices, n_time_bins):
    """
    Generate a temporal basis vector for emissions at specified time bins into
    specified waveguides.

    Parameters
    ----------
    waveguide_emission_indices : list or tuple
        List of indices where photon emission occurs for each waveguide,
        e.g. [[t1_wg1], [t1_wg2, t2_wg2], [], [t1_wg4, t2_wg4, t3_wg4]].
    n_time_bins : int
        Number of time bins; the range over which each index can vary.

    Returns
    -------
    temporal_basis_vector : :class:`.Qobj`
        A basis vector representing photon scattering at the specified indices.
        If there are W waveguides, T times, and N photon emissions, then the
        basis vector has dimensionality (W*T)^N.
    """
    idx = _temporal_basis_idx(waveguide_emission_indices, n_time_bins)
    dims = _temporal_basis_dims(waveguide_emission_indices, n_time_bins, None)
    return basis(dims, list(idx))


def _temporal_scattered_matrix(H, psi0, n_emissions, c_ops, tlist,
                               system_zero_state=None,
                               construct_effective_hamiltonian=True):
    """
    Compute the scattered n-photon state as an ndarray.
    """
    T = len(tlist)
    W = len(c_ops)
    em_dims = max(n_emissions, 1)
    phi_n = np.zeros([W * T] * em_dims, dtype=complex)

    if construct_effective_hamiltonian:
        # Construct an effective Hamiltonian from system hamiltonian and c_ops
        Heff = QobjEvo(H) - 1j / 2 * sum([op.dag() * op for op in c_ops])
    else:
        Heff = H

    evolver = Propagator(Heff, memoize=len(tlist))

    all_emission_indices = combinations_with_replacement(range(T), n_emissions)

    if system_zero_state is None:
        system_zero_state = psi0

    # Compute <omega_tau> for all combinations of tau
    for emission_indices in all_emission_indices:
        # Consider unique partitionings of emission times into waveguides
        partition = tuple(set(set_partition(emission_indices, W)))
        # Consider all possible partitionings of time bins by waveguide
        for indices in partition:
            taus = [[tlist[i] for i in wg_indices] for wg_indices in indices]
            phi_n_amp = photon_scattering_amplitude(
                evolver, c_ops, tlist,
                taus, psi0, system_zero_state
            )
            # Add scatter amplitude times temporal basis to overall state
            idx = _temporal_basis_idx(indices, T)
            phi_n[idx] = phi_n_amp
    return phi_n


def temporal_scattered_state(H, psi0, n_emissions, c_ops, tlist,
                             system_zero_state=None,
                             construct_effective_hamiltonian=True):
    """
    Compute the scattered n-photon state projected onto the temporal basis.

    Parameters
    ----------
    H : :class:`.Qobj` or list
        System-waveguide(s) Hamiltonian or effective Hamiltonian in Qobj or
        list-callback format. If construct_effective_hamiltonian is not
        specified, an effective Hamiltonian is constructed from `H` and
        `c_ops`.
    psi0 : :class:`.Qobj`
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`.
    n_emissions : int
        Number of photon emissions to calculate.
    c_ops : list
        List of collapse operators for each waveguide; these are assumed to
        include spontaneous decay rates, e.g.
        :math:`\\sigma = \\sqrt \\gamma \\cdot a`
    tlist : array_like
        List of times for :math:`\\tau_i`. tlist should contain 0 and exceed
        the pulse duration / temporal region of interest.
    system_zero_state : :class:`.Qobj`, optional
        State representing zero excitations in the system. Defaults to
        :math:`\\psi(t_0)`
    construct_effective_hamiltonian : bool, default: True
        Whether an effective Hamiltonian should be constructed from H and
        c_ops:
        :math:`H_{eff} = H - \\frac{i}{2} \\sum_n \\sigma_n^\\dagger \\sigma_n`
        Default: True.

    Returns
    -------
    phi_n : :class:`.Qobj`
        The scattered bath state projected onto the temporal basis given by
        tlist. If there are W waveguides, T times, and N photon emissions, then
        the state is a tensor product state with dimensionality T^(W*N).
    """
    T = len(tlist)
    W = len(c_ops)
    em_dims = max(n_emissions, 1)
    phi_n = _temporal_scattered_matrix(
        H, psi0, n_emissions, c_ops, tlist,
        system_zero_state, construct_effective_hamiltonian
    )
    return Qobj(phi_n.ravel(), dims=[[W * T] * em_dims, [1] * em_dims])


def scattering_probability(H, psi0, n_emissions, c_ops, tlist,
                           system_zero_state=None,
                           construct_effective_hamiltonian=True):
    """
    Compute the integrated probability of scattering n photons in an arbitrary
    system. This function accepts a nonlinearly spaced array of times.

    Parameters
    ----------
    H : :class:`.Qobj` or list
        System-waveguide(s) Hamiltonian or effective Hamiltonian in Qobj or
        list-callback format. If construct_effective_hamiltonian is not
        specified, an effective Hamiltonian is constructed from H and
        `c_ops`.
    psi0 : :class:`.Qobj`
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`.
    n_emissions : int
        Number of photons emitted by the system (into any combination of
        waveguides).
    c_ops : list
        List of collapse operators for each waveguide; these are assumed to
        include spontaneous decay rates, e.g.
        :math:`\\sigma = \\sqrt \\gamma \\cdot a`.
    tlist : array_like
        List of times for :math:`\\tau_i`. tlist should contain 0 and exceed
        the pulse duration / temporal region of interest; tlist need not be
        linearly spaced.
    system_zero_state : :class:`.Qobj`, optional
        State representing zero excitations in the system. Defaults to
        `basis(systemDims, 0)`.
    construct_effective_hamiltonian : bool, default: True
        Whether an effective Hamiltonian should be constructed from H and
        c_ops:
        :math:`H_{eff} = H - \\frac{i}{2} \\sum_n \\sigma_n^\\dagger \\sigma_n`
        Default: True.

    Returns
    -------
    scattering_prob : float
        The probability of scattering n photons from the system over the time
        range specified.
    """
    phi_n = _temporal_scattered_matrix(H, psi0, n_emissions, c_ops, tlist,
                                       system_zero_state,
                                       construct_effective_hamiltonian)
    T = len(tlist)
    W = len(c_ops)

    # Compute <omega_tau> for all combinations of tau
    all_emission_indices = combinations_with_replacement(range(T), n_emissions)
    probs = np.zeros([T] * n_emissions)

    # Project scattered state onto temporal basis
    for emit_indices in all_emission_indices:
        # Consider unique emission time partitionings
        partition = tuple(set(set_partition(emit_indices, W)))
        # wg_indices_list = list(set_partition(indices, W))
        for wg_indices in partition:
            idx = _temporal_basis_idx(wg_indices, T)
            amplitude = phi_n[idx]
            probs[emit_indices] += np.real(amplitude.conjugate() * amplitude)

    # Iteratively integrate to obtain single value
    while probs.shape != ():
        probs = trapezoid(probs, x=tlist)
    return np.abs(probs)
