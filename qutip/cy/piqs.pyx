#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
Cythonized code for permutationally invariant Lindbladian generation
"""
import numpy as np

from scipy.sparse import csr_matrix, dok_matrix
from qutip import Qobj

cimport numpy as cnp
cimport cython


def _num_dicke_states(N):
    """
    Calculate the number of Dicke states.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    nds: int
        The number of Dicke states.
    """
    if (not float(N).is_integer()):
        raise ValueError("Number of TLS should be an integer")

    if (N < 1):
        raise ValueError("Number of TLS should be non-negative")

    nds = (N/2 + 1)**2 - (N % 2)/4
    return int(nds)

def _num_dicke_ladders(N):
    """
    Calculate the total number of Dicke ladders in the Dicke space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    Nj: int
        The number of Dicke ladders.
    """
    Nj = (N+1) * 0.5 + (1-np.mod(N, 2)) * 0.5
    return int(Nj)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list get_blocks(int N):
    """
    Calculate the number of cumulative elements at each block boundary.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    blocks: np.ndarray
        An array with the number of cumulative elements at the boundary of
        each block.
    """
    cdef int num_blocks = _num_dicke_ladders(N)
    cdef list blocks
    blocks = [i * (N+2-i) for i in range(1, num_blocks+1)]
    return blocks

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float j_min(N):
    """
    Calculate the minimum value of j for given N.

    Parameters
    ----------
    N: int
        Number of two-level systems.

    Returns
    -------
    jmin: float
        The minimum value of j for odd or even number of two
        level systems.
    """
    if N % 2 == 0:
        return 0
    else:
        return 0.5

def j_vals(N):
    """
    Get the valid values of j for given N.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    jvals: np.ndarray
        The j values for given N as a 1D array.
    """
    j = np.arange(j_min(N), N/2 + 1, 1)
    return j

def m_vals(j):
    """
    Get all the possible values of m or m1 for given j.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    mvals: np.ndarray
        The m values for given j as a 1D array.
    """
    return np.arange(-j, j+1, 1)

def get_index(N, j, m, m1, blocks):
    """
    Get the index in the density matrix for this j, m, m1 value.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    j, m, m1: float
        The j, m, m1 values.

    blocks: np.ndarray
        An 1D array with the number of cumulative elements at the boundary of
        each block.

    Returns
    -------
    mvals: array
        The m values for given j.
    """
    _k = int(j-m1)
    _k_prime = int(j-m)
    block_number = int(N/2 - j)
    offset = 0
    if block_number > 0:
        offset = blocks[block_number-1]
    i = _k_prime + offset
    k = _k + offset
    return (i, k)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list jmm1_dictionary(int N):
    """
    Get the index in the density matrix for this j, m, m1 value.

    The (j, m, m1) values are mapped to the (i, k) index of a block
    diagonal matrix which has the structure to capture the permutationally
    symmetric part of the density matrix. For each (j, m, m1) value, first
    we get the block by using the "j" value and then the addition in the
    row/column due to the m and m1 is determined. Four dictionaries are
    returned giving a map from the (j, m, m1) values to (i, k), the inverse
    map, a flattened map and the inverse of the flattened map.
    """
    cdef long i
    cdef long k
    cdef dict jmm1_dict = {}
    cdef dict jmm1_inv = {}
    cdef dict jmm1_flat = {}
    cdef dict jmm1_flat_inv = {}
    cdef int l
    cdef int nds = _num_dicke_states(N)
    cdef list blocks = get_blocks(N)

    jvalues = j_vals(N)
    for j in jvalues:
        mvalues = m_vals(j)
        for m in mvalues:
            for m1 in mvalues:
                i, k = get_index(N, j, m, m1, blocks)
                jmm1_dict[(i, k)] = (j, m, m1)
                jmm1_inv[(j, m, m1)] = (i, k)
                l = nds * i+k
                jmm1_flat[l] = (j, m, m1)
                jmm1_flat_inv[(j, m, m1)] = l
    return [jmm1_dict, jmm1_inv, jmm1_flat, jmm1_flat_inv]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Dicke(object):
    """
    A faster Cythonized Dicke state class to build the Lindbladian.

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

    Attributes
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
    """
    cdef int N
    cdef float emission, dephasing, pumping
    cdef float collective_emission, collective_dephasing, collective_pumping

    def __init__(self, int N, float emission=0., float dephasing=0.,
                 float pumping=0., float collective_emission=0.,
                 collective_dephasing=0., collective_pumping=0.):
        self.N = N
        self.emission = emission
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_emission = collective_emission
        self.collective_dephasing = collective_dephasing
        self.collective_pumping = collective_pumping

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object lindbladian(self):
        """
        Build the Lindbladian superoperator of the dissipative dynamics as a
        sparse matrix.

        Returns
        ----------
        lindblad_qobj: :class: qutip.Qobj
            The matrix size is (nds**2, nds**2) where nds is the number of
            Dicke states.
        """
        N = self.N
        cdef int nds = _num_dicke_states(N)
        cdef int num_ladders = _num_dicke_ladders(N)

        cdef list lindblad_row = []
        cdef list lindblad_col = []
        cdef list lindblad_data = []
        cdef tuple jmm1_1
        cdef tuple jmm1_2
        cdef tuple jmm1_3
        cdef tuple jmm1_4
        cdef tuple jmm1_5
        cdef tuple jmm1_6
        cdef tuple jmm1_7
        cdef tuple jmm1_8
        cdef tuple jmm1_9

        _1, _2, jmm1_row, jmm1_inv = jmm1_dictionary(N)

        # perform loop in each row of matrix
        for r in jmm1_row:
            j, m, m1 = jmm1_row[r]
            jmm1_1 = (j, m, m1)
            jmm1_2 = (j, m+1, m1+1)
            jmm1_3 = (j+1, m+1, m1+1)
            jmm1_4 = (j-1, m+1, m1+1)
            jmm1_5 = (j+1, m, m1)
            jmm1_6 = (j-1, m, m1)
            jmm1_7 = (j+1, m-1, m1-1)
            jmm1_8 = (j, m-1, m1-1)
            jmm1_9 = (j-1, m-1, m1-1)

            g1 = self.gamma1(jmm1_1)
            c1 = jmm1_inv[jmm1_1]

            lindblad_row.append(int(r))
            lindblad_col.append(int(c1))
            lindblad_data.append(g1)

            # generate gammas in the given row
            # check if the gammas exist
            # load gammas in the lindbladian in the correct position

            if jmm1_2 in jmm1_inv:
                g2 = self.gamma2(jmm1_2)
                c2 = jmm1_inv[jmm1_2]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c2))
                lindblad_data.append(g2)

            if jmm1_3 in jmm1_inv:
                g3 = self.gamma3(jmm1_3)
                c3 = jmm1_inv[jmm1_3]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c3))
                lindblad_data.append(g3)

            if jmm1_4 in jmm1_inv:
                g4 = self.gamma4(jmm1_4)
                c4 = jmm1_inv[jmm1_4]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c4))
                lindblad_data.append(g4)

            if jmm1_5 in jmm1_inv:
                g5 = self.gamma5(jmm1_5)
                c5 = jmm1_inv[jmm1_5]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c5))
                lindblad_data.append(g5)

            if jmm1_6 in jmm1_inv:
                g6 = self.gamma6(jmm1_6)
                c6 = jmm1_inv[jmm1_6]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c6))
                lindblad_data.append(g6)

            if jmm1_7 in jmm1_inv:
                g7 = self.gamma7(jmm1_7)
                c7 = jmm1_inv[jmm1_7]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c7))
                lindblad_data.append(g7)

            if jmm1_8 in jmm1_inv:
                g8 = self.gamma8(jmm1_8)
                c8 = jmm1_inv[jmm1_8]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c8))
                lindblad_data.append(g8)

            if jmm1_9 in jmm1_inv:
                g9 = self.gamma9(jmm1_9)
                c9 = jmm1_inv[jmm1_9]

                lindblad_row.append(int(r))
                lindblad_col.append(int(c9))
                lindblad_data.append(g9)

        cdef lindblad_matrix = csr_matrix((lindblad_data,
                                          (lindblad_row, lindblad_col)),
                                          shape=(nds**2, nds**2))

        # make matrix a Qobj superoperator with expected dims
        llind_dims = [[[nds], [nds]], [[nds], [nds]]]
        cdef object lindblad_qobj = Qobj(lindblad_matrix, dims=llind_dims)
        return lindblad_qobj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma1(self, tuple jmm1):
        """
        Calculate gamma1 for value of j, m, m'.
        """
        cdef float j, m, m1
        cdef float yCE, yE, yD, yP, yCP, yCD
        cdef float N
        cdef float spontaneous, losses, pump, collective_pump
        cdef float dephase, collective_dephase, g1

        j, m, m1 = jmm1
        N = float(self.N)

        yE = self.emission
        yD = self.dephasing
        yP = self.pumping
        yCE = self.collective_emission
        yCP = self.collective_pumping
        yCD = self.collective_dephasing

        spontaneous = yCE/2 * (2*j*(j+1) - m * (m-1) - m1 * (m1 - 1))
        losses = (yE/2) * (N+m+m1)
        pump = yP/2 * (N-m-m1)
        collective_pump = yCP/2 * \
            (2*j * (j+1) - m*(m+1) - m1*(m1+1))
        collective_dephase = yCD/2 * (m-m1)**2

        if j <= 0:
            dephase = yD*N/4
        else:
            dephase = yD/2*(N/2 - m*m1 * (N/2 + 1)/j/(j+1))

        g1 = spontaneous + losses + pump + dephase + \
            collective_pump + collective_dephase

        return -g1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma2(self, tuple jmm1):
        """
        Calculate gamma2 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef float yCE, yE, yD, yP, yCP, yCD, g2
        cdef float N
        cdef float spontaneous, losses, pump, collective_pump
        cdef float dephase, collective_dephase

        j, m, m1 = jmm1
        N = float(self.N)
        yCE = self.collective_emission
        yE = self.emission

        if yCE == 0:
            spontaneous = 0.0
        else:
            spontaneous = yCE * np.sqrt((j+m) * (j-m+1) * (j+m1) * (j-m1+1))

        if (yE == 0) or (j <= 0):
            losses = 0.0
        else:
            losses = yE/2 * np.sqrt((j+m) * (j-m+1) * (j+m1) * (j-m1+1)) * \
                                                      (N/2 + 1)/(j*(j+1))
        g2 = spontaneous + losses
        return g2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma3(self, tuple jmm1):
        """
        Calculate gamma3 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef float yE
        cdef float N
        cdef float spontaneous, losses, pump, collective_pump
        cdef float dephase, collective_dephase

        cdef complex g3
        j, m, m1 = jmm1
        N = float(self.N)
        yE = self.emission

        if (yE == 0) or (j <= 0):
            g3 = 0.0
        else:
            g3 = yE/2 * np.sqrt((j+m) * (j+m-1) * (j+m1) * (j+m1-1)) * \
                 (N/2 + j+1)/(j*(2*j + 1))
        return g3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma4(self, tuple jmm1):
        """
        Calculate gamma4 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef complex g4
        cdef float yE
        cdef float N

        N = float(self.N)
        j, m, m1 = jmm1
        yE = self.emission
        if (yE == 0) or ((j+1) <= 0):
            g4 = 0.0
        else:
            g4 = yE/2 * np.sqrt((j-m+1) * (j-m+2) * (j-m1+1) *
                                (j-m1+2)) * (N/2 - j)/((j+1) *
                                                       (2*j + 1))
        return g4

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma5(self, tuple jmm1):
        """
        Calculate gamma5 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef complex g5
        j, m, m1 = jmm1
        cdef float yD
        cdef float N

        N = float(self.N)
        yD = self.dephasing

        if (yD == 0) or (j <= 0):
            g5 = 0.0
        else:
            g5 = yD/2 * np.sqrt((j**2 - m**2)*(j**2 - m1**2)) * \
                (N/2 + j + 1)/(j*(2*j + 1))

        return g5

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma6(self, tuple jmm1):
        """
        Calculate gamma6 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef float yD
        cdef float N
        cdef complex g6

        j, m, m1 = jmm1
        N = float(self.N)

        yD = self.dephasing
        if yD == 0:
            g6 = 0.0
        else:
            g6 = yD/2 * np.sqrt(((j+1)**2 - m**2)*((j+1) **
                                                   2-m1**2)) * \
                                                   (N/2 - j)/((j+1) * (2*j+1))
        return g6

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma7(self, tuple jmm1):
        """
        Calculate gamma7 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef float yP
        cdef float N
        cdef complex g7

        j, m, m1 = jmm1
        N = float(self.N)
        yP = self.pumping

        if (yP == 0) or (j <= 0):
            g7 = 0.0
        else:
            g7 = yP/2 * np.sqrt((j-m-1)*(j-m)*(j-m1-1) *
                                (j-m1)) * (N/2 + j + 1)/(j * (2*j+1))
        return g7

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma8(self, tuple jmm1):
        """
        Calculate gamma8 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef float yP, yCP
        cdef float N
        cdef complex g8

        j, m, m1 = jmm1
        N = float(self.N)
        yP = self.pumping
        yCP = self.collective_pumping
        if (yP == 0) or (j <= 0):
            pump = 0.0
        else:
            pump = yP/2 * np.sqrt((j+m+1) * (j-m) * (j+m1+1) *
                                  (j-m1)) * (N/2 + 1)/(j*(j+1))
        if yCP == 0:
            collective_pump = 0.0
        else:
            collective_pump = yCP * \
                np.sqrt((j-m) * (j+m+1) * (j+m1+1) * (j-m1))
        g8 = pump + collective_pump
        return g8

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma9(self, tuple jmm1):
        """
        Calculate gamma9 for given j, m, m'.
        """
        cdef float j, m, m1
        cdef float yP
        cdef float N
        cdef complex g9

        j, m, m1 = jmm1
        N = float(self.N)
        yP = self.pumping

        if (yP == 0):
            g9 = 0.0
        else:
            g9 = yP/2 * np.sqrt((j+m+1) * (j+m+2) * (j+m1+1) *
                                (j+m1+2)) * (N/2 - j)/((j+1) * (2*j+1))
        return g9
