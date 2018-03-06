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
"""Permutational Invariant Quantum Solver (PIQS)

This module calculates the Liouvillian for the dynamics of ensembles of
identical two-level systems (TLS) in the presence of local and collective
processes by exploiting permutational symmetry and using the Dicke basis.
"""

# Authors: Nathan Shammah, Shahnawaz Ahmed
# Contact: shahnawaz.ahmed95@gmail.com

from math import factorial
from decimal import Decimal

import numpy as np

from scipy import constants
from scipy.sparse import dok_matrix, block_diag

from qutip import Qobj, spre, spost, tensor, identity, ket2dm
from qutip import sigmax, sigmay, sigmaz, sigmap, sigmam
from qutip.cy.piqs import Dicke as _Dicke
from qutip.cy.piqs import (jmm1_dictionary, _num_dicke_states,
                           _num_dicke_ladders, get_blocks)


# Functions necessary to generate the Lindbladian/Liouvillian
def num_dicke_states(N):
    """Calculate the number of Dicke states.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    nds: int
        The number of Dicke states.
    """
    return _num_dicke_states(N)

def num_dicke_ladders(N):
    """Calculate the total number of ladders in the Dicke space.

    For a collection of N two-level systems it counts how many different
    "j" exist or the number of blocks in the block-diagonal matrix.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    Nj: int
        The number of Dicke ladders.
    """
    return _num_dicke_ladders(N)

def num_tls(nds):
    """Calculate the number of two-level systems.

    Parameters
    ----------
    nds: int
         The number of Dicke states.

    Returns
    -------
    N: int
        The number of two-level systems.
    """
    if np.sqrt(nds).is_integer():
        # N is even
        N = 2*(np.sqrt(nds)-1)
    else:
        # N is odd
        N = 2*(np.sqrt(nds + 1/4)-1)
    return int(N)


class Dicke(object):
    """The Dicke class which builds the Lindbladian and Liouvillian matrix.

    Example
    -------
    >> from qutip.models.piqs import Dicke, j_algebra
    >> N = 2
    >> jx, jy, jz, jp, jm = j_algebra(N)
    >> ensemble = Dicke(N, emission=1.)
    >> L = ensemble.liouvillian()

    Attributes
    ----------
    nds: int
        The number of Dicke states.

    dshape: tuple
        The tuple (nds, nds)

    blocks : list
        A list which gets the number of cumulative elements at each block
        boundary.
    """
    def __init__(self, N, hamiltonian=None,
                 emission=0., dephasing=0., pumping=0.,
                 collective_emission=0., collective_dephasing=0.,
                 collective_pumping=0.):
        """
        The ensemble initialization for N two-level systems.

        Parameters
        ----------
        N: int
            The number of two-level systems.

        hamiltonian: :class: qutip.Qobj
            A Hamiltonian in the reduced Dicke basis set.

            The matrix dimensions are (nds, nds), with nds being the number of
            dicke states. The hamiltonian can be built with the operators given
            by the `j_algebra` function in the "dicke" basis.

        emission: float
            Incoherent emission coefficient (also nonradiative emission).
            default: 0.0

        dephasing: float
            Local dephasing coefficient.
            default: 0.0

        pumping: float
            Incoherent pumping coefficient.
            default: 0.0

        collective_emission: float
            Collective (superradiant) emmission coefficient.
            default: 0.0

        collective_pumping: float
            Collective pumping coefficient.
            default: 0.0

        collective_dephasing: float
            Collective dephasing coefficient.
            default: 0.0
        """
        self.N = N
        self.hamiltonian = hamiltonian
        self.emission = emission
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_emission = collective_emission
        self.collective_dephasing = collective_dephasing
        self.collective_pumping = collective_pumping
        self.nds = num_dicke_states(self.N)
        self.dshape = (num_dicke_states(self.N), num_dicke_states(self.N))

    def __repr__(self):
        """Print the current parameters of the system."""
        string = []
        string.append("N = {}".format(self.N))
        string.append("Hilbert space dim = {}".format(self.dshape))
        string.append("Number of dicke states = {}".format(self.nds))
        string.append("Liouvillian space dim = {}".format((self.nds**2,
                                                           self.nds**2)))
        nonzero = np.sum(get_blocks(self.N))
        sparsity = nonzero/self.N**4
        string.append("Sparsity of Liouvillian = {}".format(sparsity))
        if self.emission != 0:
            string.append("emission = {}".format(self.emission))
        if self.dephasing != 0:
            string.append("dephasing = {}".format(self.dephasing))
        if self.pumping != 0:
            string.append("pumping = {}".format(self.pumping))
        if self.collective_emission != 0:
            string.append(
                "collective_emission = {}".format(self.collective_emission))
        if self.collective_dephasing != 0:
            string.append(
                "collective_dephasing = {}".format(self.collective_dephasing))
        if self.collective_pumping != 0:
            string.append(
                "collective_pumping = {}".format(self.collective_pumping))
        return "\n".join(string)

    def lindbladian(self):
        """Build the Lindbladian superoperator of the dissipative dynamics.

        Returns
        -------
        lindbladian: :class: qutip.Qobj
            The Lindbladian matrix as a `qutip.Qobj`.
        """
        cythonized_dicke = _Dicke(int(self.N),
                                  float(self.emission),
                                  float(self.dephasing),
                                  float(self.pumping),
                                  float(self.collective_emission),
                                  float(self.collective_dephasing),
                                  float(self.collective_pumping))
        return cythonized_dicke.lindbladian()

    def liouvillian(self):
        """Build the total Liouvillian in the Dicke basis.

        Returns
        -------
        liouv: :class: qutip.Qobj
            The Liouvillian matrix for the system.
        """
        lindblad = self.lindbladian()
        if self.hamiltonian is None:
            liouv = lindblad

        else:
            hamiltonian = self.hamiltonian
            hamiltonian_superoperator = - 1j * \
                spre(hamiltonian) + 1j * spost(hamiltonian)
            liouv = lindblad + hamiltonian_superoperator
        return liouv

    def eigenstates(self, liouvillian):
        """Calculate the eigenvalues and eigenvectors of the Liouvillian.

        It removes the spurious eigenvectors with corresponding
        non-hermitian density matrices.

        Parameters
        ----------
        liouvillian: :class: qutip.Qobj
            The Liouvillian of which to calculate the spectrum.

        Returns
        -------
        eigen_states: list
            The list of eigenvalues and correspondent eigenstates.
        """
        unpruned_eigenstates = liouvillian.eigenstates()
        eigen_states = self.prune_eigenstates(unpruned_eigenstates)
        return eigen_states

    def prune_eigenstates(self, liouvillian_eigenstates):
        """Remove spurious eigenvalues and eigenvectors of the Liouvillian.

        Spurious means that the given eigenvector has elements outside of the
        block-diagonal matrix.

        Parameters
        ----------
        liouvillian_eigenstates: list
            A list with the eigenvalues and eigenvectors of the Liouvillian
            including spurious ones.

        Returns
        -------
        correct_eigenstates: list
            The list with the correct eigenvalues and eigenvectors of the
            Liouvillian.
        """
        N = self.N
        block_mat = block_matrix(N)
        nnz_tuple_bm = [(i, j) for i, j in zip(*block_mat.nonzero())]

        # 0. Create  a copy of the eigenvalues to approximate values
        eig_val, eig_vec = liouvillian_eigenstates
        tol = 10
        eig_val_round = np.round(eig_val, tol)

        # 2. Use 'block_matrix(N)' to remove eigenvectors with matrix
        # elements
        # outside of the block matrix.
        forbidden_eig_index = []
        for k in range(0, len(eig_vec)):
            dm = vector_to_operator(eig_vec[k])
            nnz_tuple = [(i, j) for i, j in zip(*dm.data.nonzero())]
            for i in nnz_tuple:
                if i not in nnz_tuple_bm:
                    if np.round(dm[i], tol) != 0:
                        forbidden_eig_index.append(k)

        forbidden_eig_index = np.array(list(set(forbidden_eig_index)))
        # 3. Remove the forbidden eigenvalues and eigenvectors.
        correct_eig_val = np.delete(eig_val, forbidden_eig_index)
        correct_eig_vec = np.delete(eig_vec, forbidden_eig_index)
        correct_eigenstates = correct_eig_val, correct_eig_vec
        return correct_eigenstates


# Utility functions for operators in the Dicke basis
def energy_degeneracy(N, m):
    """Calculate the number of Dicke states with same energy.

    The use of the `Decimals` class allows to explore N > 1000,
    unlike the built-in function `scipy.special.binom`

    Parameters
    ----------
    N: int
        The number of two-level systems.

    m: float
        Total spin z-axis projection eigenvalue.
        This is proportional to the total energy

    Returns
    -------
    degeneracy: int
        The energy degeneracy
    """
    numerator = Decimal(factorial(N))
    d1 = Decimal(factorial(N/2 + m))
    d2 = Decimal(factorial(N/2 - m))
    degeneracy = numerator/(d1 * d2)
    return int(degeneracy)

def state_degeneracy(N, j):
    """Calculate the degeneracy of the Dicke state.

    Each state |j, m> includes D(N,j) irreducible representations |j, m,alpha>
    Uses Decimals to calculate higher numerator and denominators numbers.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    j: float
        Total spin eigenvalue (cooperativity)

    Returns
    -------
    degeneracy: int
        The state degeneracy
    """
    numerator = Decimal(factorial(N)) * Decimal(2*j + 1)
    denominator_1 = Decimal(factorial(N/2 + j + 1))
    denominator_2 = Decimal(factorial(N/2 - j))
    degeneracy = numerator/(denominator_1 * denominator_2)
    degeneracy = int(np.round(float(degeneracy)))
    if degeneracy < 0:
        raise ValueError("m-degeneracy must be >=0")
    return degeneracy

def m_degeneracy(N, m):
    """Calculate the number of Dicke states |j, m> with same energy.

    Parameters
    ----------
    N: int
        The number of two-level systems

    m: float
        Total spin z-axis projection eigenvalue (proportional to the total
        energy)

    Returns
    -------
    degeneracy: int
        The m-degeneracy
    """
    degeneracy = N/2 + 1 - abs(m)
    if degeneracy % 1 != 0 or degeneracy <= 0:
        e = "m-degeneracy must be integer >=0, "
        e.append("but degeneracy = {}".format(degeneracy))
        raise ValueError(e)
    return int(degeneracy)

def ap(j, m):
    """Calculate the operator `ap` used later.

    The action of ap is given by: J_{+}|j, m> = A_{+}(jm)|j, m+1>

    Parameters
    ----------
    j, m: float
        The value for j and m in the dicke basis |j,m>

    Returns
    -------
    a_plus: float
        The value of `a_plus`
    """
    a_plus = np.sqrt((j-m) * (j+m+1))
    return a_plus

def am(j, m):
    """Calculate the operator `am` used later.

    The action of ap is given by: J_{-}|j, m> = A_{-}(jm)|j, m-1>

    Parameters
    ----------
    j, m: float
        The value for j and m in the dicke basis |j, m>.

    Returns
    -------
    a_minus: float
        The value of `a_minus`.
    """
    a_minus = np.sqrt((j+m) * (j-m+1))
    return a_minus

def spin_algebra(N, op=None):
    """Create the list [sx, sy, sz, sp, sm] with the spin operators.

    The operators are constructed for a collection of N two-level systems
    (TLSs). Each element of the list, i.e., sx, is a vector of `qutip.Qobj`
    objects (spin matrices), as it cointains the list of the SU(2) Pauli
    matrices for the N TLSs. Each TLS operator sx[i], with i = 0, ..., (N-1),
    is placed in a 2^N-dimensional Hilbert space.

    Notes
    -----
    sx[i] is igmax()/2 in the composite Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    Returns
    -------
    spin_operators: list or :class: qutip.Qobj
        A list of `qutip.Qobj` operators - [sx, sy, sz, sp, sm] or the
        requested operator.
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)
    sx = [0 for i in range(N)]
    sy = [0 for i in range(N)]
    sz = [0 for i in range(N)]
    sp = [0 for i in range(N)]
    sm = [0 for i in range(N)]

    sx[0] = 0.5 * sigmax()
    sy[0] = 0.5 * sigmay()
    sz[0] = 0.5 * sigmaz()
    sp[0] = sigmap()
    sm[0] = sigmam()

    # 2. Place operators in total Hilbert space
    for k in range(N - 1):
        sx[0] = tensor(sx[0], identity(2))
        sy[0] = tensor(sy[0], identity(2))
        sz[0] = tensor(sz[0], identity(2))
        sp[0] = tensor(sp[0], identity(2))
        sm[0] = tensor(sm[0], identity(2))

    # 3. Cyclic sequence to create all N operators
    a = [i for i in range(N)]
    b = [[a[i - i2] for i in range(N)] for i2 in range(N)]

    # 4. Create N operators
    for i in range(1, N):
        sx[i] = sx[0].permute(b[i])
        sy[i] = sy[0].permute(b[i])
        sz[i] = sz[0].permute(b[i])
        sp[i] = sp[0].permute(b[i])
        sm[i] = sm[0].permute(b[i])

    spin_operators = [sx, sy, sz, sp, sm]

    if not op:
        return spin_operators
    elif op == 'x':
        return sx
    elif op == 'y':
        return sy
    elif op == 'z':
        return sz
    elif op == '+':
        return sp
    elif op == '-':
        return sm
    else:
        raise TypeError('Invalid type')

def _j_algebra_uncoupled(N, op=None):
    """Construct the the collective spin algebra in the uncoupled basis.

    jx, jy, jz, jp, jm are constructed in the uncoupled basis of the
    two-level system (TLS). Each collective operator is placed in a
    Hilbert space of dimension 2^N.

    Parameters
    ----------
    N: int
        The number of two-level systems

    op: str
        The operator to return 'x','y','z','+','-'.
        If no operator given, then output is the list of operators
        for ['x','y','z', '+', '-']

    Returns
    -------
    collective_operators: list or :class: qutip.Qobj
        A list of `qutip.Qobj` representing all the operators in
        uncoupled" basis or a single operator requested.
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)

    si_TLS = spin_algebra(N)

    sx = si_TLS[0]
    sy = si_TLS[1]
    sz = si_TLS[2]
    sp = si_TLS[3]
    sm = si_TLS[4]

    jx = sum(sx)
    jy = sum(sy)
    jz = sum(sz)
    jp = sum(sp)
    jm = sum(sm)

    collective_operators = [jx, jy, jz, jp, jm]

    if not op:
        return collective_operators
    elif op == 'x':
        return jx
    elif op == 'y':
        return jy
    elif op == 'z':
        return jz
    elif op == '+':
        return jp
    elif op == '-':
        return jm
    else:
        raise TypeError('Invalid type')

def j_algebra(N, op=None, basis="dicke"):
    """
    Calculate the list of collective operators of the total algebra.

    The Dicke basis |j,m><j,m'| is used by default. Otherwise with "uncoupled"
    the operators are in a 2^N space.

    Parameters
    ----------
    N: int
        Number of two-level systems

    op: str
        The operator to return 'x','y','z','+','-'.
        If no operator given, then output is the list of operators
        for ['x','y','z', '+', '-']

    basis: str
        The basis of the operators - "dicke" or "uncoupled"
        default: "dicke"

    Returns
    -------
    j_alg: list or :class: qutip.Qobj
        A list of `qutip.Qobj` representing all the operators in
        the "dicke" or "uncoupled" basis or a single operator requested.
    """
    if basis == "uncoupled":
        return _j_algebra_uncoupled(N, op)

    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    jz_operator = dok_matrix((nds, nds))
    jp_operator = dok_matrix((nds, nds))
    jm_operator = dok_matrix((nds, nds))
    s = 0

    for k in range(0, num_ladders):
        j = 0.5 * N - k
        mmax = int(2*j + 1)
        for i in range(0, mmax):
            m = j-i
            jz_operator[s, s] = m
            if (s+1) in range(0, nds):
                jp_operator[s, s+1] = ap(j, m-1)
            if (s-1) in range(0, nds):
                jm_operator[s, s-1] = am(j, m+1)
            s = s+1

    jx_operator = 1/2 * (jp_operator+jm_operator)
    jy_operator = 1j/2 * (jm_operator-jp_operator)
    jx = Qobj(jx_operator)
    jy = Qobj(jy_operator)
    jz = Qobj(jz_operator)
    jp = Qobj(jp_operator)
    jm = Qobj(jm_operator)

    if not op:
        return [jx, jy, jz, jp, jm]
    if op == '+':
        return jp
    elif op == '-':
        return jm
    elif op == 'x':
        return jx
    elif op == 'y':
        return jy
    elif op == 'z':
        return jz
    else:
        raise TypeError('Invalid type')

def c_ops_tls(N, emission=0., dephasing=0., pumping=0.,
              collective_emission=0., collective_dephasing=0.,
              collective_pumping=0.):
    """
    Create the collapse operators (c_ops) of the Lindblad master equation.

    These operators are in the uncoupled basis of the two-level system
    (TLS) SU(2) Pauli matrices.

    Notes
    -----
    The collapse operator list can be given to `qutip.mesolve`.
    Notice that the operators are placed in a Hilbert space of dimension 2^N.
    Thus the method is suitable only for small N (upto 10).

    Parameters
    ----------
    N: int
        The number of two-level systems.

    emission: float
        Incoherent emission coefficient (also nonradiative emission).
        default: 0.0

    dephasing: float
        Local dephasing coefficient.
        default: 0.0

    pumping: float
        Incoherent pumping coefficient.
        default: 0.0

    collective_emission: float
        Collective (superradiant) emmission coefficient.
        default: 0.0

    collective_pumping: float
        Collective pumping coefficient.
        default: 0.0

    collective_dephasing: float
        Collective dephasing coefficient.
        default: 0.0

    Returns
    -------
    c_ops: list
        The list of collapse operators as `qutip.Qobj` for the system.
    """
    N = int(N)

    if N > 10:
        msg = "N > 10. dim(H) = 2^N. "
        msg += "Better use `qutip.models.piqs` to reduce Hilbert space "
        msg += "dimension and exploit permutational symmetry"
        raise Warning(msg)

    [sx, sy, sz, sp, sm] = spin_algebra(N)
    [jx, jy, jz, jp, jm] = _j_algebra_uncoupled(N)

    c_ops = []

    if emission != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(emission) * sm[i])

    if dephasing != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(dephasing) * sz[i])

    if pumping != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(pumping) * sp[i])

    if collective_emission != 0:
        c_ops.append(np.sqrt(collective_emission) * jm)

    if collective_dephasing != 0:
        c_ops.append(np.sqrt(collective_dephasing) * jz)

    if collective_pumping != 0:
        c_ops.append(np.sqrt(collective_pumping) * jp)

    return c_ops

# State definitions in the Dicke basis with an option for basis transformation
def dicke_basis(N, jmm1=None):
    """
    Initialize the density matrix of a Dicke state for several (j, m, m1).

    This function can be used to build arbitrary states in the Dicke basis
    |j, m><j, m1|. We create coefficients for each (j, m, m1) value in the
    dictionary jmm1. For instance, if we start from the most excited state for
    N = 2, we have the following state represented as a density matrix of size
    (nds, nds) or
    (4, 4).

    1 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 0 0

    The mapping for the (i, k) index of the density matrix to the |j, m>
    values is given by the cythonized function `jmm1_dictionary`.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    jmm1: dict
        A dictionary of {(j, m, m1): p} that gives a density p for the
        (j, m, m1) matrix element.

    Returns
    -------
    rho: :class: qutip.Qobj
        The density matrix in the Dicke basis.
    """
    if jmm1 is None:
        msg = "Please specify the jmm1 values as a dictionary"
        msg += "or use the `excited(N)` function to create an"
        msg += "excited state where jmm1 = {(N/2, N/2, N/2): 1}"
        raise AttributeError(msg)

    nds = _num_dicke_states(N)
    rho = np.zeros((nds, nds))
    jmm1_dict = jmm1_dictionary(N)[1]
    for key in jmm1:
        i, k = jmm1_dict[key]
        rho[i, k] = jmm1[key]
    return Qobj(rho)

def dicke(N, j, m):
    """
    Generate a Dicke state as a pure density matrix in the Dicke basis.

    For instance, if the superradiant state is given |j, m> = |1, 0> for N = 2,
    the state is represented as a density matrix of size (nds, nds) or (4, 4),

    0 0 0 0
    0 1 0 0
    0 0 0 0
    0 0 0 0

    Parameters
    ----------
    N: int
        The number of two-level systems

    j: float
        The eigenvalue j of the Dicke state |j, m>.

    m: float
        The eigenvalue m of the Dicke state |j, m>.

    Returns
    -------
    rho: :class: qutip.Qobj
        The density matrix.
    """
    nds = num_dicke_states(N)
    rho = np.zeros((nds, nds))

    jmm1_dict = jmm1_dictionary(N)[1]

    i, k = jmm1_dict[(j, m, m)]
    rho[i, k] = 1.
    return Qobj(rho)

# Uncoupled states in the full Hilbert space. These are returned with the
# choice of the keyword argument "dicke" in the states
def _uncoupled_excited(N):
    """
    Generate the density matrix of the excited Dicke state in the full
    2^N dimensional Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    Returns
    -------
    psi0: :class: qutip.Qobj
        The density matrix in the uncoupled basis.
    """
    N = int(N)
    jz = _collective_algebra_uncoupled(N)[2]
    en, vn = jz.eigenstates()
    psi0 = vn[2**N - 1]
    return ket2dm(psi0)

def _uncoupled_superradiant(N):
    """
    Generate the density matrix of a superradiant state in the full 2^N
    dimensional Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    Returns
    -------
    psi0: :class: qutip.Qobj
    """
    N = int(N)
    jz = _j_algebra_uncoupled(N, "z")
    en, vn = jz.eigenstates()
    psi0 = vn[2**N - (N+1)]
    return ket2dm(psi0)

def _uncoupled_ground(N):
    """
    Generate the density matrix of the ground state in the full 2^N
    dimensional Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    Returns
    -------
    psi0: :class: qutip.Qobj
    """
    N = int(N)
    jz = _j_algebra_uncoupled(N, "z")
    en, vn = jz.eigenstates()
    psi0 = vn[0]
    return ket2dm(psi0)

def _uncoupled_ghz(N):
    """
    Generate the density matrix of the GHZ state in the full 2^N
    dimensional Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    Returns
    -------
    ghz: :class: qutip.Qobj
        The density matrix for the GHZ state in the full Hilbert space.
    """
    N = int(N)

    rho = np.zeros((2**N, 2**N))
    rho[0, 0] = 1/2
    rho[2**N - 1, 0] = 1/2
    rho[0, 2**N - 1] = 1/2
    rho[2**N - 1, 2**N - 1] = 1/2

    spin_dim = [2 for i in range(0, N)]
    spins_dims = list((spin_dim, spin_dim))
    rho = Qobj(rho, dims=spins_dims)
    return rho

def _uncoupled_css(N, a, b):
    """
    Generate the density matrix of the CSS state in the full 2^N
    dimensional Hilbert space.

    The CSS states are non-entangled states given by
    |a, b> =  \Prod_i (a|1>_i + b|0>_i).

    Parameters
    ----------
    N: int
        The number of two-level systems

    a: complex
        The coefficient of the |1_i> state

    b: complex
        The coefficient of the |0_i> state

    Returns
    -------
    css: :class: qutip.Qobj
        The density matrix for the CSS state in the full Hilbert space.
    """
    N = int(N)
    # 1. Define i_th factorized density matrix in the uncoupled basis
    rho_i = np.zeros((2, 2), dtype=complex)
    rho_i[0, 0] = a * np.conj(a)
    rho_i[1, 1] = b * np.conj(b)
    rho_i[0, 1] = a * np.conj(a)
    rho_i[1, 0] = b * np.conj(b)
    rho_i = Qobj(rho_i)
    rho = [0 for i in range(N)]
    rho[0] = rho_i
    # 2. Place single-two-level-system density matrices in total Hilbert space
    for k in range(N - 1):
        rho[0] = tensor(rho[0], identity(2))
    # 3. Cyclic sequence to create all N factorized density matrices
    # |CSS>_i<CSS|_i
    a = [i for i in range(N)]
    b = [[a[i - i2] for i in range(N)] for i2 in range(N)]
    # 4. Create all other N-1 factorized density matrices
    # |+><+| = Prod_(i=1)^N |CSS>_i<CSS|_i
    for i in range(1, N):
        rho[i] = rho[0].permute(b[i])
    identity_i = Qobj(np.eye(2**N), dims=rho[0].dims, shape=rho[0].shape)
    rho_tot = identity_i
    for i in range(0, N):
        rho_tot = rho_tot * rho[i]
    return rho_tot

def excited(N, basis="dicke"):
    """
    Generate the density matrix for the excited state.

    This state is given by |N/2, N/2> in the default Dicke basis. If the
    argument `basis` is "uncoupled" then it generates the state in a
    2**N dim Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    basis: str
        The basis to use. Either "dicke" or "uncoupled"

    Returns
    -------
    state: :class: qutip.Qobj
        The excited state density matrix in the requested basis.
    """
    if basis == "uncoupled":
        state = _uncoupled_excited(N)
        return state

    jmm1 = {(N/2, N/2, N/2): 1}
    return dicke_basis(N, jmm1)

def superradiant(N, basis="dicke"):
    """
    Generate the density matrix of the superradiant state.

    This state is given by |N/2, 0> or |N/2, 0.5> in the Dicke basis.
    If the argument `basis` is "uncoupled" then it generates the state
    in a 2**N dim Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    basis: str
        The basis to use. Either "dicke" or "uncoupled".

    Returns
    -------
    state: :class: qutip.Qobj
        The superradiant state density matrix in the requested basis.
    """
    if basis == "uncoupled":
        state = _uncoupled_superradiant(N)
        return ket2dm(state)

    if N % 2 == 0:
        jmm1 = {(N/2, 0, 0): 1.}
        return dicke_basis(N, jmm1)
    else:
        jmm1 = {(N/2, 0.5, 0.5): 1.}
    return dicke_basis(N, jmm1)

def css(N, x=1/np.sqrt(2), y=1/np.sqrt(2),
        basis="dicke", coordinates="cartesian"):
    """
    Generate the density matrix of the Coherent Spin State (CSS).

    It can be defined as |CSS>= Prod_i^N(a|1>_i + b|0>_i)
    with a = sin(theta/2), b = exp(1j*phi) * cos(theta/2).
    The default basis is that of Dicke space |j, m> < j, m'|.
    The default state is the symmetric CSS, |CSS> = |+>.

    Parameters
    ----------
    N: int
        The number of two-level systems

    x, y: float
        The coefficients of the CSS state

    basis: str
        The basis to use. Either "dicke" or "uncoupled"

    coordinates: str
        Either "cartesian" or "polar". If polar then the coefficients
        are constructed as sin(x/2), cos(x/2)e^(iy)

    Returns
    -------
    rho: :class: qutip.Qobj
        The CSS state density matrix
    """
    if coordinates == "polar":
        a = np.cos(0.5 * x) * np.exp(1j * y)
        b = np.sin(0.5 * x)
    else:
        a = x
        b = y
    if basis == "uncoupled":
        return _uncoupled_css(N, a, b)
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    rho = dok_matrix((nds, nds))

    # loop in the allowed matrix elements
    jmm1_dict = jmm1_dictionary(N)[1]

    j = 0.5*N
    mmax = int(2*j + 1)
    for i in range(0, mmax):
        m = j-i
        psi_m = np.sqrt(float(energy_degeneracy(N, m))) * \
            a**(N*0.5 + m) * b**(N*0.5 - m)
        for i1 in range(0, mmax):
            m1 = j - i1
            row_column = jmm1_dict[(j, m, m1)]
            psi_m1 = np.sqrt(float(energy_degeneracy(N, m1))) * \
                np.conj(a)**(N*0.5 + m1) * np.conj(b)**(N*0.5 - m1)
            rho[row_column] = psi_m*psi_m1
    return Qobj(rho)

def ghz(N, basis="dicke"):
    """
    Generate the density matrix of the GHZ state.

    If the argument `basis` is "uncoupled" then it generates the state
    in a 2**N dim Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    basis: str
        The basis to use. Either "dicke" or "uncoupled"

    Returns
    -------
    state: :class: qutip.Qobj
        The GHZ state density matrix in the requested basis.
    """
    if basis == "uncoupled":
        return _uncoupled_ghz(N)
    nds = _num_dicke_states(N)
    rho = dok_matrix((nds, nds))
    rho[0, 0] = 1/2
    rho[N, N] = 1/2
    rho[N, 0] = 1/2
    rho[0, N] = 1/2
    return Qobj(rho)

def ground(N, basis="dicke"):
    """
    Generate the density matrix of the ground state.

    This state is given by |N/2, -N/2> in the Dicke basis. If the argument `basis`
    is "uncoupled" then it generates the state in a 2**N dim Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems

    basis: str
        The basis to use. Either "dicke" or "uncoupled"

    Returns
    -------
    state: :class: qutip.Qobj
        The ground state density matrix in the requested basis.
    """
    if basis == "uncoupled":
        state = _uncoupled_ground(N)
        return ket2dm(state)
    nds = _num_dicke_states(N)
    rho = dok_matrix((nds, nds))
    rho[N, N] = 1
    return Qobj(rho)

def identity_uncoupled(N):
    """
    Generate the identity in a 2**N dimensional Hilbert space.

    The identity matrix is formed from the tensor product of N TLSs.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    identity: :class: qutip.Qobj
        The identity matrix
    """
    N = int(N)
    rho = np.zeros((2**N, 2**N))
    for i in range(0, 2**N):
        rho[i, i] = 1
    spin_dim = [2 for i in range(0, N)]
    spins_dims = list((spin_dim, spin_dim))
    identity = Qobj(rho, dims=spins_dims)
    return identity

def block_matrix(N):
    """Construct the block-diagonal matrix for the Dicke basis.

    Parameters
    ----------
    N: int
        Number of two-level systems

    Returns
    -------
    block_matr: ndarray
        A 2D block-diagonal matrix of ones with dimension (nds,nds),
        where nds is the number of Dicke states for N two-level
        systems.
    """
    nds = _num_dicke_states(N)
    # create a list with the sizes of the blocks, in order
    blocks_dimensions = int(N/2 + 1 - 0.5*(N % 2))
    blocks_list = [(2 * (i+1 * (N % 2)) + 1*((N+1) % 2))
                   for i in range(blocks_dimensions)]
    blocks_list = np.flip(blocks_list, 0)
    # create a list with each block matrix as element
    square_blocks = []
    k = 0
    for i in blocks_list:
        square_blocks.append(np.ones((i, i)))
        k = k + 1
    return block_diag(square_blocks)
