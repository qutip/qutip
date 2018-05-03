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
from itertools import product, combinations_with_replacement
from qutip import propagator, Options, basis, tensor, zero_ket, Qobj

__all__ = ['temporal_basis_vector',
           'temporal_scattered_state',
           'scattering_probability']


class Evolver:
    """
    A caching class which takes a Hamiltonian and a list of times to calculate
    and memoize propagators for the system between any two times as demanded.

    Parameters
    ----------
    H : :class: qutip.Qobj or list
        System-waveguide(s) Hamiltonian or effective Hamiltonian in `Qobj` or
        list-callback format. If construct_effective_hamiltonian is not
        specified, an effective Hamiltonian is constructed from H and c_ops.
    times : list-like
        List of times to evaluate propagators over.
    options : :class: qutip.Options
        Solver options to use when computing propagators.

    Attributes
    ----------
    H : :class: qutip.Qobj or list
        System-waveguide(s) Hamiltonian, may be time-dependent.
    tlist : list-like
        List of times to evaluate propagators over.
    propagators : (dict of float: (dict of float: :class: qutip.Qobj))
        Dictionary of dictionaries of propagator objects with keys of
        evaluation times, e.g. propagators[t2][t1] returns U[t2,t1].
    """
    def __init__(self, H, tlist, options=None):
        self.H = H
        self.tlist = tlist
        if options is None:
            self.options = Options(nsteps=10000, normalize_output=False)
        else:
            self.options = options
        # Make a blank nested dictionary to store propagators
        self.propagators = dict.fromkeys(tlist)
        for t in tlist:
            self.propagators[t] = dict.fromkeys(tlist)

    def prop(self, tf, ti):
        """Compute U[t2,t1] where t2 > t1 or return the cached operator.

        Parameters
        ----------
        tf : float
            Final time to compute the propagator U[tf, ti].
        ti : float
            Initial time to compute the propagator U[tf,ti].

        Returns
        -------
        propagator : :class: qutip.Qobj
            The propagation operator.
        """
        left, right = np.searchsorted(self.tlist, [ti, tf], side='left')
        t1, t2 = self.tlist[left], self.tlist[right]
        if self.propagators[t2][t1] is None:
            self.propagators[t2][t1] = propagator(self.H, [t1, t2],
                                                  options=self.options,
                                                  unitary_mode='single')
            # Something is still broken about batch unitary mode (see #807)
        return self.propagators[t2][t1]


def set_partition(collection, num_sets):
    """
    Enumerate all ways of partitioning collection into num_sets different lists,
    e.g. list(set_partition([1,2], 2)) = [[[1, 2], []], [[1], [2]], [[2], [1]],
    [[], [1, 2]]].

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


def photon_scattering_operator(evolver, c_ops, taus_list):
    """
    Compute the scattering operator for a system emitting into multiple
    waveguides.

    Parameters
    ----------
    evolver : :class: qutip.scattering.Evolver
        Evolver-wrapped Hamiltonian describing the system.
    c_ops : list
        list of collapse operators for each waveguide; these are assumed to
        include spontaneous decay rates, e.g.
        :math:`\\sigma = \\sqrt \\gamma \\cdot a`
    taus_list : list-like
        List of (list of emission times) for each waveguide.

    Returns
    -------
    omega : :class: qutip.Qobj
        The temporal scattering operator with dimensionality equal to the
        system state.
    """
    omega = 1

    # Extract the full list of taus
    taus = [(0.0, -1)]  # temporal "ground state" for arbitrary waveguide
    for i, tau_wg in enumerate(taus_list):
        for tau in tau_wg:
            taus.append((tau, i))
    taus.sort(key = lambda tup: tup[0])  # sort taus by time

    # Compute Prod Ueff(tq, tq-1)
    for i in range(1, len(taus)):
        tq, q = taus[i]
        tprev, _ = taus[i - 1]
        omega = c_ops[q] * evolver.prop(tq, tprev) * omega

    # Add the <0|Uff(TP, tm)|0> term
    tmax = evolver.tlist[-1]
    taumax, _ = taus[-1]
    # if taus[-1] < tmax:
    omega = evolver.prop(tmax, taumax) * omega

    return omega


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
    temporal_basis_vector : :class: qutip.Qobj
        A basis vector representing photon scattering at the specified indices.
        If there are W waveguides, T times, and N photon emissions, then the
        basis vector has dimensionality (W*T)^N.
    """
    # Cast waveguide_emission_indices to list for mutability
    waveguide_emission_indices = [list(i) for i in waveguide_emission_indices]

    # Calculate total number of waveguides
    W = len(waveguide_emission_indices)

    # Calculate total number of emissions
    num_emissions = sum([len(waveguide_indices) for waveguide_indices in
                         waveguide_emission_indices])
    if num_emissions == 0:
        return basis(W * n_time_bins, 0)

    # Pad the emission indices with zeros
    offset_indices = []
    for i, wg_indices in enumerate(waveguide_emission_indices):
        offset_indices += [index + (i * n_time_bins) for index in wg_indices]

    # Return an appropriate tensor product state
    return tensor([basis(n_time_bins * W, i) for i in offset_indices])


def temporal_scattered_state(H, psi0, n_emissions, c_ops, tlist,
                             system_zero_state=None,
                             construct_effective_hamiltonian=True):
    """
    Compute the scattered n-photon state projected onto the temporal basis.

    Parameters
    ----------
    H : :class: qutip.Qobj or list
        System-waveguide(s) Hamiltonian or effective Hamiltonian in Qobj or
        list-callback format. If construct_effective_hamiltonian is not
        specified, an effective Hamiltonian is constructed from `H` and
        `c_ops`.
    psi0 : :class: qutip.Qobj
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
    system_zero_state : :class: qutip.Qobj
        State representing zero excitations in the system. Defaults to
        :math:`\\psi(t_0)`
    construct_effective_hamiltonian : bool
        Whether an effective Hamiltonian should be constructed from H and c_ops:
        :math:`H_{eff} = H - \\frac{i}{2} \\sum_n \\sigma_n^\\dagger \\sigma_n`
        Default: True.

    Returns
    -------
    phi_n : :class: qutip.Qobj
        The scattered bath state projected onto the temporal basis given by
        tlist. If there are W waveguides, T times, and N photon emissions, then
        the state is a tensor product state with dimensionality T^(W*N).
    """
    T = len(tlist)
    W = len(c_ops)

    if n_emissions == 0:
        phi_n = zero_ket(W * T)
    else:
        phi_n = tensor([zero_ket(W * T)] * n_emissions)

    if construct_effective_hamiltonian:
        # Construct an effective Hamiltonian from system hamiltonian and c_ops
        if type(H) is Qobj:
            Heff = H - 1j / 2 * sum([op.dag() * op for op in c_ops])
        elif type(H) is list:
            Heff = H + [-1j / 2 * sum([op.dag() * op for op in c_ops])]
        else:
            raise TypeError("Hamiltonian must be Qobj or list-callback format")
    else:
        Heff = H

    evolver = Evolver(Heff, tlist)

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
            omega = photon_scattering_operator(evolver, c_ops, taus)
            phi_n_amp = system_zero_state.dag() * omega * psi0
            # Add scatter amplitude times temporal basis to overall state
            phi_n += phi_n_amp * temporal_basis_vector(indices, T)

    return phi_n


def scattering_probability(H, psi0, n_emissions, c_ops, tlist,
                           system_zero_state=None,
                           construct_effective_hamiltonian=True):
    """
    Compute the integrated probability of scattering n photons in an arbitrary
    system. This function accepts a nonlinearly spaced array of times.

    Parameters
    ----------
    H : :class: qutip.Qobj or list
        System-waveguide(s) Hamiltonian or effective Hamiltonian in Qobj or
        list-callback format. If construct_effective_hamiltonian is not
        specified, an effective Hamiltonian is constructed from H and
        `c_ops`.
    psi0 : :class: qutip.Qobj
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
    system_zero_state : :class: qutip.Qobj
        State representing zero excitations in the system. Defaults to
        `basis(systemDims, 0)`.
    construct_effective_hamiltonian : bool
        Whether an effective Hamiltonian should be constructed from H and c_ops:
        :math:`H_{eff} = H - \\frac{i}{2} \\sum_n \\sigma_n^\\dagger \\sigma_n`
        Default: True.

    Returns
    -------
    scattering_prob : float
        The probability of scattering n photons from the system over the time
        range specified.
    """
    phi_n = temporal_scattered_state(H, psi0, n_emissions, c_ops, tlist,
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
            projector = temporal_basis_vector(wg_indices, T)
            amplitude = (projector.dag() * phi_n).full().item()
            probs[emit_indices] += np.real(amplitude.conjugate() * amplitude)

    # Iteratively integrate to obtain single value
    while probs.shape != ():
        probs = np.trapz(probs, x = tlist)
    return np.abs(probs)
