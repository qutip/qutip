"""
Cythonized code for permutationally invariant Liouvillian
"""
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
from qutip import Qobj
cimport numpy as cnp
cimport cython


def _num_dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of ensembles
    with odd number of systems.

    Parameters
    -------
    N: int
        The number of two level systems
    Returns
    -------
    nds: int
        The number of Dicke states
    """
    if (not float(N).is_integer()):
        raise ValueError("Number of TLS should be an integer")

    if (N < 1):
        raise ValueError("Number of TLS should be non-negative")

    nds = (N / 2 + 1)**2 - (N % 2) / 4
    return int(nds)


def _num_dicke_ladders(N):
    """
    Calculates the total number of Dicke ladders in the Dicke space for a
    collection of N two-level systems. It counts how many different "j" exist.
    Or the number of blocks in the block diagonal matrix.

    Parameters
    -------
    N: int
        The number of two level systems.
    Returns
    -------
    Nj: int
        The number of Dicke ladders
    """
    Nj = (N + 1) * 0.5 + (1 - np.mod(N, 2)) * 0.5
    return int(Nj)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list _get_blocks(int N):
    """
    A list which gets the number of cumulative elements at each block
    boundary. For N = 4

    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
            1 1 1
            1 1 1
            1 1 1
                 1

    Thus, the blocks are [5, 8, 9] denoting that after the first block 5
    elements have been accounted for and so on. This function will later
    be helpful in the calculation of j, m, m' value for a given (row, col)
    index in this matrix.

    Returns
    -------
    blocks: arr
        An array with the number of cumulative elements at the boundary of
        each block
    """
    cdef int num_blocks = _num_dicke_ladders(N)

    cdef list blocks
    blocks = [i * (N + 2 - i) for i in range(1, num_blocks + 1)]
    return blocks


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float _j_min(N):
    """
    Calculate the minimum value of j for given N

    Parameters
    ==========
    N: int
        Number of two level systems

    Returns
    =======
    jmin: float
        The minimum value of j for odd or even number of two
        level systems
    """
    if N % 2 == 0:
        return 0
    else:
        return 0.5


def _j_vals(N):
    """
    Get the valid values of j for given N.
    """
    j = np.arange(_j_min(N), N / 2 + 1, 1)
    return j


def m_vals(j):
    """
    Get all the possible values of m or $m^\prime$ for given j.
    """
    return np.arange(-j, j + 1, 1)



def get_index(N, j, m, m1, blocks):
    """
    Get the index in the density matrix for this j, m, m1 value.
    """
    _k = int(j - m1)
    _k_prime = int(j - m)

    block_number = int(N / 2 - j)

    offset = 0
    if block_number > 0:
        offset = blocks[block_number - 1]

    i = _k_prime + offset
    k = _k + offset

    return (i, k)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list jmm1_dictionary(int N):
    """
    Get the index in the density matrix for this j, m, m1 value.
    """
    cdef long i
    cdef long k

    cdef dict jmm1_dict = {}
    cdef dict jmm1_inv = {}
    cdef dict jmm1_flat = {}
    cdef dict jmm1_flat_inv = {}
    cdef int l
    cdef int nds = _num_dicke_states(N)

    cdef list blocks = _get_blocks(N)

    jvalues = _j_vals(N)

    for j in jvalues:
        mvalues = m_vals(j)
        for m in mvalues:
            for m1 in mvalues:
                i, k = get_index(N, j, m, m1, blocks)
                jmm1_dict[(i, k)] = (j, m, m1)
                jmm1_inv[(j, m, m1)] = (i, k)
                l = nds * i + k
                jmm1_flat[l] = (j, m, m1)
                jmm1_flat_inv[(j, m, m1)] = l

    return [jmm1_dict, jmm1_inv, jmm1_flat, jmm1_flat_inv]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Dicke(object):
    """
    The Dicke States class.

    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2

    hamiltonian : Qobj matrix
        An Hamiltonian H in the reduced basis set by `reduced_algebra()`.
        Matrix dimensions are (nds, nds), with nds = _num_dicke_states.
        The hamiltonian is assumed to be with hbar = 1.
        default: H = jz_op(N)

    emission : float
        Incoherent emission coefficient
        default: 0.0

    dephasing : float
        Local dephasing coefficient
        default: 0.0

    pumping : float
        Incoherent pumping coefficient
        default: 0.0

    collective_emission : float
        Collective spontaneous emmission coefficient
        default: 1.0

    collective_dephasing : float
        Collective dephasing coefficient
        default: 0.0

    collective_pumping : float
        Collective pumping coefficient
        default: 0.0

    nds : int
        The number of Dicke states
        default: nds(2) = 4

    dshape : tuple
        The tuple (nds, nds)
        default: (4,4)

    blocks : array
        A list which gets the number of cumulative elements at each block
        boundary
        default:  array([3, 4])
    """
    cdef int N
    cdef float emission, dephasing, pumping
    cdef float collective_emission, collective_dephasing, collective_pumping

    def __init__(self, int N=2, float emission=0., float dephasing=0.,
                 float pumping=0., float collective_emission=1.,
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
        sparse matrix using COO.

        Returns
        ----------
        lindblad_qobj: Qobj superoperator (sparse)
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
            jmm1_2 = (j, m + 1, m1 + 1)
            jmm1_3 = (j + 1, m + 1, m1 + 1)
            jmm1_4 = (j - 1, m + 1, m1 + 1)
            jmm1_5 = (j + 1, m, m1)
            jmm1_6 = (j - 1, m, m1)
            jmm1_7 = (j + 1, m - 1, m1 - 1)
            jmm1_8 = (j, m - 1, m1 - 1)
            jmm1_9 = (j - 1, m - 1, m1 - 1)

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

        cdef lindblad_matrix = csr_matrix((lindblad_data, (lindblad_row, lindblad_col)),
                                          shape=(nds**2, nds**2))

        # make matrix a Qobj superoperator with expected dims
        llind_dims = [[[nds], [nds]], [[nds], [nds]]]
        cdef object lindblad_qobj = Qobj(lindblad_matrix, dims=llind_dims)

        return lindblad_qobj

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma1(self, tuple jmm1):
        """
        Calculate gamma1 for value of j, m, m'
        """
        cdef float j, m, m1

        j, m, m1 = jmm1

        cdef float yCE, yE, yD, yP, yCP, yCD

        cdef float N
        N = float(self.N)

        cdef float spontaneous, losses, pump, collective_pump
        cdef float dephase, collective_dephase, g1

        yE = self.emission
        yD = self.dephasing
        yP = self.pumping
        yCE = self.collective_emission        
        yCP = self.collective_pumping
        yCD = self.collective_dephasing

        spontaneous = yCE / 2 * (2 * j * (j + 1) - m * (m - 1) - m1 * (m1 - 1))
        losses = yE / 2 * (N + m + m1)
        pump = yP / 2 * (N - m - m1)
        collective_pump = yCP / 2 * \
            (2 * j * (j + 1) - m * (m + 1) - m1 * (m1 + 1))
        collective_dephase = yCD / 2 * (m - m1)**2

        if j <= 0:
            dephase = yD * N / 4
        else:
            dephase = yD / 2 * (N / 2 - m * m1 * (N / 2 + 1) / j / (j + 1))

        g1 = spontaneous + losses + pump + dephase + \
            collective_pump + collective_dephase

        return(-g1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma2(self, tuple jmm1):
        """
        Calculate gamma2 for given j, m, m'
        """
        cdef float j, m, m1

        j, m, m1 = jmm1

        cdef float yCE, yE, yD, yP, yCP, yCD, g2

        cdef float N
        N = float(self.N)

        cdef float spontaneous, losses, pump, collective_pump
        cdef float dephase, collective_dephase

        j, m, m1 = jmm1
        yCE = self.collective_emission
        yE = self.emission

        if yCE == 0:
            spontaneous = 0.0
        else:
            spontaneous = yCE * \
                np.sqrt((j + m) * (j - m + 1) * (j + m1) * (j - m1 + 1))

        if (yE == 0) or (j <= 0):
            losses = 0.0
        else:
            losses = yE / 2 * \
                np.sqrt((j + m) * (j - m + 1) * (j + m1) * (j - m1 + 1)) * (N / 2 + 1) / (j * (j + 1))

        g2 = spontaneous + losses

        return (g2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma3(self, tuple jmm1):
        """
        Calculate gamma3 for given j, m, m'
        """
        cdef float j, m, m1
        j, m, m1 = jmm1

        cdef float yE

        cdef float N
        N = float(self.N)

        cdef float spontaneous, losses, pump, collective_pump
        cdef float dephase, collective_dephase

        cdef complex g3

        yE = self.emission

        if (yE == 0) or (j <= 0):
            g3 = 0.0
        else:
            g3 = yE / 2 * np.sqrt((j + m) * (j + m - 1) * (j + m1) * (j + m1 - 1)) * (N / 2 + j + 1) / (j * (2 * j + 1))

        return (g3)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma4(self, tuple jmm1):
        """
        Calculate gamma4 for given j, m, m'
        """
        cdef float j, m, m1
        j, m, m1 = jmm1

        cdef float yE
        cdef float N
        N = float(self.N)

        cdef complex g4

        yE = self.emission

        if (yE == 0) or ((j + 1) <= 0):
            g4 = 0.0
        else:
            g4 = yE / 2 * np.sqrt((j - m + 1) * (j - m + 2) * (j - m1 + 1) * (j - m1 + 2)) * (N / 2 - j) / ((j + 1) * (2 * j + 1))

        return (g4)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma5(self, tuple jmm1):
        """
        Calculate gamma5 for given j, m, m'
        """
        cdef float j, m, m1
        j, m, m1 = jmm1

        cdef float yD
        cdef float N
        N = float(self.N)

        cdef complex g5

        yD = self.dephasing

        if (yD == 0) or (j <= 0):
            g5 = 0.0
        else:
            g5 = yD / 2 * np.sqrt((j**2 - m**2) * (j**2 - m1**2)) * \
                (N / 2 + j + 1) / (j * (2 * j + 1))

        return (g5)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma6(self, tuple jmm1):
        """
        Calculate gamma6 for given j, m, m'
        """
        cdef float j, m, m1
        j, m, m1 = jmm1

        cdef float yD
        cdef float N
        N = float(self.N)

        cdef complex g6

        yD = self.dephasing

        if yD == 0:
            g6 = 0.0
        else:
            g6 = yD / 2 * np.sqrt(((j + 1)**2 - m**2) * ((j + 1) **
                                                         2 - m1**2)) * (N / 2 - j) / ((j + 1) * (2 * j + 1))

        return (g6)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma7(self, tuple jmm1):
        """
        Calculate gamma7 for given j, m, m'
        """
        cdef float j, m, m1
        j, m, m1 = jmm1

        cdef float yP
        cdef float N
        N = float(self.N)

        cdef complex g7

        yP = self.pumping

        if (yP == 0) or (j <= 0):
            g7 = 0.0
        else:
            g7 = yP / 2 * np.sqrt((j - m - 1) * (j - m) * (j - m1 - 1) * (j - m1)) * (N / 2 + j + 1) / (j * (2 * j + 1))

        return (g7)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma8(self, tuple jmm1):
        """
        Calculate gamma8 for given j, m, m'
        """
        cdef float j, m, m1
        j, m, m1 = jmm1

        cdef float yP, yCP

        cdef float N
        N = float(self.N)

        cdef complex g8

        yP = self.pumping
        yCP = self.collective_pumping

        if (yP == 0) or (j <= 0):
            pump = 0.0
        else:
            pump = yP / 2 * np.sqrt((j + m + 1) * (j - m) * (j + m1 + 1) * (j - m1)) * (N / 2 + 1) / (j * (j + 1))

        if yCP == 0:
            collective_pump = 0.0
        else:
            collective_pump = yCP * \
                np.sqrt((j - m) * (j + m + 1) * (j + m1 + 1) * (j - m1))

        g8 = pump + collective_pump

        return (g8)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef complex gamma9(self, tuple jmm1):
        """
        Calculate gamma9 for given j, m, m'
        """
        cdef float j, m, m1
        j, m, m1 = jmm1

        cdef float yP
        cdef float N
        N = float(self.N)

        cdef complex g9

        yP = self.pumping

        if (yP == 0):
            g9 = 0.0
        else:
            g9 = yP / 2 * np.sqrt((j + m + 1) * (j + m + 2) * (j + m1 + 1)
                                  * (j + m1 + 2)) * (N / 2 - j) / ((j + 1) * (2 * j + 1))

        return (g9)


#=============================================================================
# Method to be used when the Hamiltonian is diagonal
#=============================================================================
cpdef int tau_column(str tau, int k, float j):
    """
    Determine the column index for the non-zero elements of the matrix for a particular
    row k and the value of j from the dicke space
    Parameters
    ----------
    tau: str
        The tau function to check for this k and j
    k: int
        The row of the matrix M for which the non zero elements have
        to be calculated
    j: float
        The value of j for this row
    """
    # In the notes, we indexed from k = 1, here we do it from k = 0
    k = k + 1
    cdef dict mapping = {"tau3": k - int(2 * j + 3),
                         "tau2": k - 1,
                         "tau4": k + int(2 * j - 1),
                         "tau5": k - int(2 * j + 2),
                         "tau1": k,
                         "tau6": k + int(2 * j),
                         "tau7": k - int(2 * j + 1),
                         "tau8": k + 1,
                         "tau9": k + int(2 * j + 1)}

    # we need to decrement k again as indexing is from 0
    return mapping[tau] - 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Pim(object):
    """
    The permutation invariant matrix class. Initialize the class with the
    parameters for generating a permutation invariant density matrix.
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2
    emission : float
        Incoherent emission coefficient
        default: 0.0
    dephasing : float
        Local dephasing coefficient
        default: 0.0
    pumping : float
        Incoherent pumping coefficient
        default: 0.0

    collective_emission : float
        Collective emission coefficient
        default: 1.0
    collective_dephasing: float
        Collective dephasing coefficient
        default: 0.0        
    collective_pumping : float
        Collective pumping coefficient
        default: 0.0
    M: dict
        A nested dictionary of the structure {row: {col: val}} which holds
        non zero elements of the matrix M
    sparse_M: scipy.sparse.csr_matrix
        A sparse representation of the matrix M for efficient vector multiplication
    """

    def __init__(self, int N=2,
                float emission=0, float dephasing=0, float pumping=0,
                float collective_emission=1, float collective_pumping=0,
                float collective_dephasing=0):
        self.N = N
        self.collective_emission = collective_emission
        self.emission = emission
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.collective_dephasing = collective_dephasing

        cdef list row = []
        cdef list col = []
        cdef list data = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int isdicke(self, int dicke_row, int dicke_col):
        """
        Check if an element in a matrix is a valid element in the Dicke space.
        Dicke row: j value index. Dicke column: m value index.
        The function returns True if the element exists in the Dicke space and
        False otherwise.
        Parameters
        ----------
        dicke_row, dicke_col : int
            Index of the element in Dicke space which needs to be checked
        """
        cdef int rows, cols

        rows = self.N + 1
        cols = 0

        if (self.N % 2) == 0:
            cols = self.N / 2 + 1
        else:
            cols = int(self.N / 2 + 0.5)

        if (dicke_row > rows) or (dicke_row < 0):
            return (0)

        if (dicke_col > cols) or (dicke_col < 0):
            return (0)

        if (dicke_row < rows / 2) and (dicke_col > dicke_row):
            return 0

        if (dicke_row >= rows / 2) and (rows - dicke_row <= dicke_col):
            return 0

        else:
            return 1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef dict tau_valid(self, int dicke_row, int dicke_col):
        """
        Find the Tau functions which are valid for this value of (dicke_row, dicke_col) given
        the number of TLS. This calculates the valid tau values and reurns a dictionary
        specifying the tau function name and the value.
        Parameters
        ----------
        dicke_row, dicke_col : int
            Index of the element in Dicke space which needs to be checked
        Returns
        -------
        taus: dict
            A dictionary of key, val as {tau: value} consisting of the valid
            taus for this row and column of the Dicke space element
        """
        cdef list tau_functions = [self.tau3, self.tau2, self.tau4,
                                   self.tau5, self.tau1, self.tau6,
                                   self.tau7, self.tau8, self.tau9]

        cdef float N = float(self.N)

        if self.isdicke(dicke_row, dicke_col) is 0:
            return False

        # The 3x3 sub matrix surrounding the Dicke space element to
        # run the tau functions

        cdef list indices = []
        cdef int x, y

        for x in range(-1, 2):
            for y in range(-1, 2):
                indices.append((dicke_row + x, dicke_col + y))

        cdef dict taus = {}
        cdef tuple idx
        cdef float j, m

        for idx, tau in zip(indices, tau_functions):
            if self.isdicke(idx[0], idx[1]):
                j, m = self.calculate_j_m(idx[0], idx[1])
                taus[tau.__name__] = tau(j, m)

        return taus


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple calculate_j_m(self, int dicke_row, int dicke_col):
        """
        Get the value of j and m for the particular Dicke space element.
        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix
        Returns
        -------
        j, m: float
            The j and m values.
        """
        cdef float N = float(self.N)

        cdef float j, m

        j = N / 2 - dicke_col
        m = N / 2 - dicke_row

        return(j, m)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int calculate_k(self, int d_row, int d_col):
        """
        Get k value from the current row and column element in the Dicke space
        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix
        Returns
        -------
        k: int
            The row index for the matrix M for given Dicke space
            element
        """
        cdef float N = float(self.N)
        cdef int k

        cdef float dicke_row, dicke_column
        dicke_row = d_row
        dicke_col = d_col

        if dicke_row == 0:
            k = dicke_col

        else:
            k = int(((dicke_col) / 2) * (2 * (N + 1) - 2 *
                                         (dicke_col - 1)) + (dicke_row - (dicke_col)))

        return k

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef generate_k(self, int dicke_row, int dicke_col):
        """
        Generates one row of the Matrix M based on the k value running from top to
        bottom of the Dicke space. Also update the row in M. A dictionary with {key: val}
        specifying the column index and the tau element for the given Dicke space element
        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix
        Returns
        -------
        row: dict
            A dictionary with {key: val} specifying the column index and
            the tau element for the given Dicke space element
        """
        if self.isdicke(dicke_row, dicke_col) is 0:
            return False

        # Calculate k as the number of Dicke elements till
        cdef int k

        k = self.calculate_k(dicke_row, dicke_col)

        cdef dict row = {}
        cdef dict taus = self.tau_valid(dicke_row, dicke_col)
        cdef float j, m
        cdef int current_col

        for tau in taus:
            j, m = self.calculate_j_m(dicke_row, dicke_col)
            current_col = tau_column(tau, k, j)
            self.row.append(k)
            self.col.append(int(current_col))
            self.data.append(taus[tau])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef generate_matrix(self):
        """
        Generate the matrix M
        """
        N = self.N
        rows = self.N + 1
        cols = 0

        if (self.N % 2) == 0:
            cols = int(self.N / 2 + 1)
        else:
            cols = int(float(self.N) / 2 + 1 / 2)

        for (dicke_row, dicke_col) in np.ndindex(rows, cols):
            if self.isdicke(dicke_row, dicke_col):
                self.generate_k(dicke_row, dicke_col)

        nds = _num_dicke_states(self.N)
        cdef M = csr_matrix((self.data, (self.row, self.col)), shape=(nds, nds), dtype=float)

        return M

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau1(self, float j, float m):
        """
        Calculate tau1 for value of j and m.
        """
        cdef float yS, yL, yD, yP, yCP
        cdef float N, spontaneous, losses, pump, collective_pump
        cdef float dephase, t1

        yS = self.collective_emission
        yL = self.emission
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping

        N = float(self.N)

        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (N / 2 + m)
        pump = yP * (N / 2 - m)
        collective_pump = yCP * (1 + j + m) * (j - m)

        if j == 0:
            dephase = yD * N / 4
        else:
            dephase = yD * (N / 4 - m**2 * ((1 + N / 2) / (2 * j * (j + 1))))

        t1 = spontaneous + losses + pump + dephase + collective_pump

        return -t1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau2(self, float j, float m):
        """
        Calculate tau2 for given j and m
        """
        cdef float yS, yL, N, num, den, spontaneous, losses, t2

        yS = self.collective_emission
        yL = self.emission

        N = float(self.N)

        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (((N / 2 + 1) * (j - m + 1) *
                        (j + m)) / (2 * j * (j + 1)))

        t2 = spontaneous + losses

        return(t2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau3(self, float j, float m):
        """
        Calculate tau3 for given j and m
        """
        cdef float yL, N, num, den, t3
        yL = self.emission

        N = self.N
        N = float(N)

        num = (j + m - 1) * (j + m) * (j + 1 + N / 2)
        den = 2 * j * (2 * j + 1)

        t3 = yL * (num / den)

        return t3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau4(self, float j, float m):
        """
        Calculate tau4 for given j and m.
        """
        cdef float yL, N, num, den, t4
        yL = self.emission

        N = float(self.N)

        num = (j - m + 1) * (j - m + 2) * (N / 2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t4 = yL * (num / den)

        return t4

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau5(self, float j, float m):
        """
        Calculate tau5 for j and m
        """
        cdef float yD, N, num, den, t5
        yD = self.dephasing

        N = float(self.N)

        num = (j - m) * (j + m) * (j + 1 + N / 2)
        den = 2 * j * (2 * j + 1)

        t5 = yD * (num / den)

        return(t5)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau6(self, float j, float m):
        """
        Calculate tau6 for given j and m
        """
        cdef float yD, N, num, den, t6

        yD = self.dephasing

        N = float(self.N)

        num = (j - m + 1) * (j + m + 1) * (N / 2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t6 = yD * (num / den)

        return t6

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau7(self, float j, float m):
        """
        Calculate tau7 for given j and m
        """
        cdef float yP, N, num, den, t7
        yP = self.pumping

        N = float(self.N)

        num = (j - m - 1) * (j - m) * (j + 1 + N / 2)
        den = 2 * j * (2 * j + 1)

        t7 = yP * (num / den)

        return t7

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau8(self, float j, float m):
        """
        Calculate self.tau8
        """
        cdef float yP, yCP, N, num, den, pump, collective_pump, t8

        yP = self.pumping
        yCP = self.collective_pumping

        N = float(self.N)

        num = (1 + N / 2) * (j - m) * (j + m + 1)
        den = 2 * j * (j + 1)
        pump = yP * (num / den)
        collective_pump = yCP * (j - m) * (j + m + 1)

        t8 = pump + collective_pump

        return t8

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef float tau9(self, float j, float m):
        """
        Calculate self.tau9
        """
        cdef float yP, t9, num, den, N
        N = float(self.N)

        yP = self.pumping

        num = (j + m + 1) * (j + m + 2) * (N / 2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t9 = yP * (num / den)

        return t9
