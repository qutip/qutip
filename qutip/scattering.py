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

__all__ = ['photon_scattering_operator',
           'temporal_basis_vector',
           'temporal_scattered_state',
           'scattering_probability']

import numpy as np
import itertools

from qutip import propagator, Options, basis, tensor, zero_ket

class Evolver:
    """
    A caching class which takes a Hamiltonian and a list of times to calculate
    and memoize propagators for the system between any two times as demanded.

    Parameters
    ----------
    H: qutip.Qobj
        system-waveguide(s) Hamiltonian, may be time-dependent
    times: list-like
        list of times to evaluate propagators over

    Attributes
    ----------
    H: qutip.Qobj
        system-waveguide(s) Hamiltonian, may be time-dependent
    times: list-like
        list of times to evaluate propagators over
    propagators: (dict of float: (dict of float: qutip.Qobj))
        dictionary of dictionaries of propagator objects with keys of
        evaluation times (e.g. propagators[t2][t1] returns U[t2,t1]

    """

    def __init__(self, H, times):
        self.H = H
        self.times = times
        # Make a blank nested dictionary to store propagators
        self.propagators = dict.fromkeys(times)
        for t in times:
            self.propagators[t] = dict.fromkeys(times)

    def prop(self, tf, ti):
        """Compute U[t2,t1] where t2 > t1 or return the cached operator

        Parameters
        ----------
        tf: float
            final time to compute the propagator U[tf,ti] for; the propagator
            is computed to the nearest time in self.times
        ti: float
            initial time to compute the propagator U[tf,ti] for; the propagator
            is computed to the nearest time in self.times

        Returns
        -------
        propagator: qutip.Qobj
            the propagation operator
        """
        left, right = np.searchsorted(self.times, [ti, tf], side = 'left')
        t1, t2 = self.times[left], self.times[right]
        if self.propagators[t2][t1] is None:
            self.propagators[t2][t1] = propagator(
                    self.H, [t1, t2],
                    options = Options(nsteps = 10000, normalize_output = False),
                    unitary_mode = 'single')
        return self.propagators[t2][t1]

    def countFilled(self):
        """Count the number of currently computed propagators in
        self.propagators.

        Returns
        -------
        count: int
            the number of propagators which have been computed so far
        """
        count = 0
        for _, dic in self.propagators.items():
            for _, prop in dic.items():
                if prop is not None:
                    count += 1
        return count


def set_partition(collection, num_sets):
    """
    Enumerate all ways of partitioning collection into num_sets different lists,
    e.g. list(set_partition([1,2], 2)) = [[[1, 2], []], [[1], [2]], [[2], [1]],
    [[], [1, 2]]]
    """
    for partitioning in itertools.product(range(num_sets),
                                          repeat = len(collection)):
        partition = [[] for _ in range(num_sets)]
        for i, set_index in enumerate(partitioning):
            partition[set_index].append(collection[i])
        yield partition


def photon_scattering_operator(evolver, taus_list, c_ops, gammalist = None):
    """
    Compute the scattering operator for a system emitting into multiple
    waveguides.

    Parameters
    ----------
    evolver : Evolver
        Evolver-wrapped Hamiltonian describing the system
    taus_list : list-like
        list of (list of emission times) for each waveguide
    c_ops : list
        list of collapse operators for each waveguide
    gammalist : list
        list of spontaneous decay rates for each waveguide. If not provided,
        :math:`\\sigma` as provided in c_ops will be treated as
        :math:`\\gamma \\cdot \\sigma`.

    Returns
    -------
    Omega : Qobj
        The temporal scattering operator with dimensionality equal to the
        system state
    """

    Omega = 1

    # Extract the full list of taus
    taus = [(0.0, None)]
    for i, tauWG in enumerate(taus_list):
        for tau in tauWG:
            taus.append((tau, i))
    taus.sort(key = lambda tup: tup[0])  # sort taus by time

    if gammalist == None:
        gammalist = [1.0] * len(c_ops)

    # Compute Prod Ueff(tq, tq-1)
    for i in range(1, len(taus)):
        tq, q = taus[i]
        tprev, _ = taus[i - 1]
        gamma = gammalist[i]
        Omega = np.sqrt(gamma) * c_ops[q] * evolver.prop(tq, tprev) * Omega

    # Add the <0|Uff(TP, tm)|0> term
    tmax = evolver.times[-1]
    taumax = taus[-1][0]
    # if taus[-1] < tmax:
    Omega = evolver.prop(tmax, taumax) * Omega

    return Omega


def temporal_basis_vector(waveguide_emission_indices, n_time_bins):
    """
    Generate a temporal basis vector for emissions at specified time bins into
    specified waveguides

    Parameters
    ----------
    waveguide_emission_indices : list
        list of indices where photon emission occurs for each waveguide,
        e.g. [[t_1]_wg1, [t_1, t_2]_wg2, []_wg3, [t_1, t_2, t_3]_wg4]
    n_time_bins : int
        number of time bins; the range over which each index can vary

    Returns
    -------
    temporal_basis_vector : Qobj
        A basis vector representing photon scattering at the specified indices.
        If there are W waveguides, T times, and N photon emissions, then the
        state is a tensor product state with dimensionality T^(W*N).
    """
    # Calculate total number of emissions
    num_emissions = sum([len(waveguide_indices) for waveguide_indices in
                         waveguide_emission_indices])
    if num_emissions == 0:
        return basis(n_time_bins, 0)

    # Pad the emission indices with zeros
    for i, waveguide_indices in enumerate(waveguide_emission_indices):
        waveguide_emission_indices[i] \
            = [0] * (num_emissions - len(waveguide_indices)) + waveguide_indices

    # Return an appropriate tensor product state
    return tensor([tensor([basis(n_time_bins, i) for i in waveguide_indices]) \
                   for waveguide_indices in waveguide_emission_indices])


def temporal_scattered_state(H, n_emissions, psi0, tlist, c_ops,
                             gammalist = None, system_zero_state = None):
    """
    Compute the scattered n-photon state projected onto the temporal basis.

    Parameters
    ----------
    H : Qobj
        system-waveguide(s) Hamiltonian, may be time-dependent
    n_emissions : int
        number of photon emissions to calculate
    psi0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`.
    c_ops : list
        list of collapse operators, one for each waveguide
    tlist : array_like
        list of times for :math:`\\tau`. taulist must be positive and `0` will
        be added to the list if not present.
    gammalist : list
        list of spontaneous decay rates for each waveguide. If not provided,
        :math:`\\sigma` as provided in c_ops will be treated as
        :math:`\\gamma \\cdot \\sigma`.
    system_zero_state : Qobj
        State representing zero excitations in the system. Defaults to
        :math:`\\psi(t_0)`

    Returns
    -------
    phi_n : Qobj
        The scattered bath state projected onto the temporal basis given by
        tlist. If there are W waveguides, T times, and N photon emissions, then
        the state is a tensor product state with dimensionality T^(W*N)
    """

    T = len(tlist)
    W = len(c_ops)
    if n_emissions == 0:
        phi_n = zero_ket(T)
    else:
        phi_n = tensor([zero_ket(T)] * (W * n_emissions))
    evolver = Evolver(H, tlist)
    indicesList = itertools.product(range(T), repeat = n_emissions)

    if system_zero_state is None:
        system_zero_state = psi0

    # Compute <omega_tau> for all combinations of tau
    for indices in indicesList:
        # time indices where a photon is scattered into some waveguide
        taus = [tlist[i] for i in indices]
        # Consider all possible partitionings of time bins by waveguide
        for partitioned_taus, partitioned_indices in \
                zip(set_partition(taus, W), set_partition(indices, W)):
            omega = photon_scattering_operator(evolver, partitioned_taus,
                                               c_ops, gammalist)
            phi_n_amp = system_zero_state.dag() * omega * psi0
            # Add scatter amplitude times temporal basis to overall state
            phi_n += phi_n_amp * temporal_basis_vector(partitioned_indices, T)

    return phi_n


def scattering_probability(H, n_emissions, psi0, tlist, c_ops, gammalist = None,
                           system_zero_state = None):
    """
    Compute the integrated probability of scattering n photons in an arbitrary
    system. This function accepts a nonlinearly spaced array of times.

    Parameters
    ----------
    H : Qobj
        system-waveguide(s) Hamiltonian, may be time-dependent
    n_emissions : int
        number of photons emitted by the system (into any combination of
        waveguides)
    psi0 : Qobj
        Initial state density matrix :math:`\\rho(t_0)` or state vector
        :math:`\\psi(t_0)`.
    tlist : array_like
        list of times for :math:`\\tau`. taulist must be positive and `0` will
        be added to the list if not present. This list can be nonlinearly
        spaced.
    c_ops : list
        list of collapse operators, one for each waveguide
    gammalist : list
        list of spontaneous decay rates for each waveguide. If not provided,
        :math:`\\sigma` as provided in c_ops will be treated as
        :math:`\\gamma \\cdot \\sigma`.
    system_zero_state : Qobj
        State representing zero excitations in the system. Defaults to
        basis(systemDims, 0)

    Returns
    -------
    scattering_prob : float
        The probability of scattering n photons from the system over the time
        range specified.
    """

    phi_n = temporal_scattered_state(H, n_emissions, psi0, tlist, c_ops,
                                     gammalist, system_zero_state)
    T = len(tlist)
    W = len(c_ops)
    probs = np.zeros([len(tlist)] * n_emissions, dtype = np.complex64)

    # Compute <omega_tau> for all combinations of tau
    indicesList = itertools.combinations(range(T), n_emissions)

    # Project scattered state onto temporal basis
    for indices in indicesList:
        sum_wg_projectors = sum([temporal_basis_vector(wg_indices, T)
                                 for wg_indices in set_partition(indices, W)])
        probs[indices] = (sum_wg_projectors.dag() * phi_n).full().item()

    # Conjugate amplitudes to get probability
    probs = probs.conj() * probs

    # Iteratively integrate to obtain single value
    while probs.shape != ():
        probs = np.trapz(probs, x = tlist)
    return np.real(probs)
