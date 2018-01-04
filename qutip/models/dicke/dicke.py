"""
Dynamics for dicke states exploiting permutational invariance
"""
from math import factorial
from decimal import Decimal
import numpy as np

from scipy import constants
from scipy.integrate import odeint, ode
from scipy.sparse import *
#from scipy.linalg import block_diag as linalg_block_diag

from qutip import Qobj, spre, spost
from qutip import sigmax, sigmay, sigmaz, sigmap, sigmam
from qutip.solver import Result
from qutip import *


def j_min(N):
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
    
def num_dicke_states(N):
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
    nds = (N/2 + 1)**2 - (N % 2)/4
    return int(nds)

def num_dicke_ladders(N):
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

def num_tls(nds):
    """
    The number of two level systems (TLS), given the number of Dicke states. 
    Inverse function of num_dicke_states(N)
    
    Parameters
    ----------
    nds: int
         The number of Dicke states  
    Returns
    -------
    N: int
        The number of two level systems
    """
    if np.sqrt(nds).is_integer():
        # N is even
        N = 2 * (np.sqrt(nds) - 1)
    else: 
        # N is odd
        N = 2 * (np.sqrt(nds + 1/4) - 1)
    
    return int(N)

def get_blocks(N):
        """
        A list which gets the number of cumulative elements at each block boundary.

        For N = 4

        1 1 1 1 1 
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1 
        1 1 1 1 1
                1 1 1
                1 1 1
                1 1 1
                     1

        Thus, the blocks are [5, 8, 9] denoting that after the first block 5 elements
        have been accounted for and so on. This function will later be helpful in the
        calculation of j, m, m' value for a given (row, col) index in this matrix.

        Returns
        -------
        blocks: arr
            An array with the number of cumulative elements at the boundary of each block
        """
        num_blocks = num_dicke_ladders(N)
        blocks = np.array([i * (N + 2 - i) for i in range(1, num_blocks + 1)], dtype = int)
        return blocks


class Dicke(object):
    """
    The Dicke States class.
    
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2

    hamiltonian : Qobj matrix
        An Hamiltonian H in the reduced basis set by `reduced_algebra()`. 
        Matrix dimensions are (nds, nds), with nds = num_dicke_states.
        The hamiltonian is assumed to be with hbar = 1. 
        default: H = jz_op(N)

    emission : float
        Collective spontaneous emmission coefficient
        default: 1.0
                
    loss : float
        Incoherent loss coefficient
        default: 0.0
        
    dephasing : float
        Local dephasing coefficient
        default: 0.0
        
    pumping : float
        Incoherent pumping coefficient
        default: 0.0
    
    collective_pumping : float
        Collective pumping coefficient
        default: 0.0

    collective_dephasing : float
        Collective dephasing coefficient
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
    def __init__(self, N = 1, hamiltonian = None,
                 loss = 0., dephasing = 0., pumping = 0., emission = 0.,
                 collective_pumping = 0., collective_dephasing = 0.):
        self.N = N
        self.hamiltonian = hamiltonian
        self.emission = emission
        self.loss = loss
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.collective_dephasing = collective_dephasing
        self.nds = num_dicke_states(self.N) 
        self.dshape = (num_dicke_states(self.N), num_dicke_states(self.N))
        self.blocks = get_blocks(self.N)

    def __repr__(self):
        """
        Print the current parameters of the system.
        """
        string =[]
        string.append("N = {}".format(self.N)))
        string.append(("Hilbert space dim = {}".format(self.dshape)))            
        string.append(("emission = {}".format(self.emission)))
        string.append(("loss = {}".format(self.loss)))
        string.append(("dephasing = {}".format(self.dephasing)))
        string.append(("pumping = {}".format(self.pumping)))
        string.append(("collective_dephasing = {}".format(self.collective_dephasing)))
        string.append(("collective_pumping = {}".format(self.collective_pumping)))

        return string

        
    def j_vals(self):
        """
        Get the valid values of j for given N.
        """
        j = np.arange(j_min(self.N), self.N/2 + 1)
        return j
    
    def m_vals(self, j):
        """
        Get all the possible values of m or $m^\prime$ for given j.
        """
        return np.arange(-j, j+1)
    
    def get_index(self, jmm1):
        """
        Get the index in the density matrix for this j, m, m1 value.
        """
        j, m, m1 = jmm1
        _k = j - m1
        _k_prime = j - m

        blocks = self.blocks
        block_number = int(self.N/2 - j)

        offset = 0
        if block_number > 0:
            offset = blocks[block_number - 1]

        i = _k_prime + offset
        k = _k + offset

        return (int(i), int(k))

    def jmm1_dictionary(self):
        """
        A dictionary with keys: (i,k) and values: (j, m, m1) for a
        block-diagonal matrix in the |j, m> <j, m1| basis. l is the position
        of the particular jmm1 in the flattened vector.
        """
        N = self.N
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)
        
        jmm1_dic = {}
        
        jmm1_inverted = {}
        # loop in the allowed matrix elements
        for j in self.j_vals():
            for m in self.m_vals(j):
                for m1 in self.m_vals(j):
                    jmm1 = (j, m, m1)
                    i, k = self.get_index(jmm1)
                    jmm1_dic[(i,k)] = jmm1
                    jmm1_inverted[jmm1] = (i,k)

        return jmm1_dic, jmm1_inverted
        
    def jmm1_flat(self):
        """
        A dictionary with keys: (l) and values: (j, m, m1) for a block-diagonal
        flattened matrix in the |j, m> <j, m1| basis. l is the position of the
        particular jmm1 in the flattened vector.
        """
        N = self.N
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)
        
        jmm1_flat = {}
        
        jmm1_inverted = {}
        # loop in the allowed matrix elements
        for j in self.j_vals():
            for m in self.m_vals(j):
                for m1 in self.m_vals(j):
                    jmm1 = (j, m, m1)
                    i, k = self.get_index(jmm1)
                    l = nds * i  + k
                    jmm1_flat[l] = jmm1
                    jmm1_inverted[jmm1] = l

        return jmm1_flat, jmm1_inverted

    def liouvillian(self):
        """
        Gives the total liouvillian in the jmm1 basis |j, m > < j, m1|
        """ 

        lindblad = self.lindbladian()
        
        if self.hamiltonian == None:
        	liouv = lindblad
        else:
        	hamiltonian = self.hamiltonian
        	hamiltonian_superoperator = - 1j* spre(hamiltonian) + 1j* spost(hamiltonian)
        
        	liouv = lindblad + hamiltonian_superoperator 

        return liouv

    def lindbladian(self):
        """
        Build the Lindbladian superoperator of the dissipative dynamics as a sparse matrix using COO.

        Returns
        ----------
        lindblad_qobj: Qobj superoperator (sparse)
            The matrix size is (nds**2, nds**2) where nds is the number of Dicke states.

        """
        N = self.N
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        jmm1_row, jmm1_inv = self.jmm1_flat()

        lindblad_matrix = dok_matrix((nds**2, nds**2))

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
            
            lindblad_matrix[int(r), int(c1)] = g1

            # generate gammas in the given row
            # checking if the gammas exist
            # and load gammas in the lindbladian in the correct position

            if jmm1_2 in jmm1_inv:
                g2 = self.gamma2(jmm1_2)
                c2 = jmm1_inv[jmm1_2]
                lindblad_matrix[int(r), int(c2)] = g2

            if jmm1_3 in jmm1_inv:
                g3 = self.gamma3(jmm1_3)
                c3 = jmm1_inv[jmm1_3]
                lindblad_matrix[int(r), int(c3)] = g3

            if jmm1_4 in jmm1_inv:
                g4 = self.gamma4(jmm1_4)
                c4 = jmm1_inv[jmm1_4]
                lindblad_matrix[int(r), int(c4)] = g4

            if jmm1_5 in jmm1_inv:
                g5 = self.gamma5(jmm1_5)
                c5 = jmm1_inv[jmm1_5]
                lindblad_matrix[int(r), int(c5)] = g5

            if jmm1_6 in jmm1_inv:
                g6 = self.gamma6(jmm1_6)
                c6 = jmm1_inv[jmm1_6]
                lindblad_matrix[int(r), int(c6)] = g6

            if jmm1_7 in jmm1_inv:                
                g7 = self.gamma7(jmm1_7)
                c7 = jmm1_inv[jmm1_7]
                lindblad_matrix[int(r), int(c7)] = g7 

            if jmm1_8 in jmm1_inv:
                g8 = self.gamma8(jmm1_8)
                c8 = jmm1_inv[jmm1_8]
                lindblad_matrix[int(r), int(c8)] = g8

            if jmm1_9 in jmm1_inv:
                g9 = self.gamma9(jmm1_9)
                c9 = jmm1_inv[jmm1_9]
                lindblad_matrix[int(r), int(c9)] = g9

        #convert matrix into CSR sparse

#         #make matrix a Qobj superoperator with expected dims
        llind_dims = [[[nds], [nds]],[[nds], [nds]]]
        lindblad_qobj = Qobj(lindblad_matrix, dims = llind_dims)

        return lindblad_qobj
        
    def gamma1(self, jmm1):
        """
        Calculate gamma1 for value of j, m, m'
        """
        j, m, m1 = jmm1
        yS = self.emission
        yL = self.loss
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping
        yCD = self.collective_dephasing
        
        N = self.N  
        N = float(N)

        spontaneous = yS / 2 * (2 * j * (j + 1) - m * (m - 1) - m1 * (m1 - 1))
        losses = yL/2 * (N + m + m1)
        pump = yP/2 * (N - m - m1)
        collective_pump = yCP / 2 * (2 * j * (j + 1) - m * (m + 1) - m1 * (m1 + 1))
        collective_dephase = yCD / 2 * (m - m1)**2
        
        if j <= 0:
            dephase = yD * N/4
        else :
            dephase = yD/2 * (N/2 - m * m1 * (N/2 + 1)/ j /(j + 1))

        g1 = spontaneous + losses + pump + dephase + collective_pump + collective_dephase
        
        return(-g1)
    
    def gamma2(self, jmm1):
        """
        Calculate gamma2 for given j, m, m'
        """
        j, m, m1 = jmm1
        yS = self.emission
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if yS == 0:
            spontaneous = 0.0
        else:            
            spontaneous = yS * np.sqrt((j + m) * (j - m + 1)* (j + m1) * (j - m1 + 1))

        if (yL == 0) or (j <= 0):
            losses = 0.0
        else:            
            losses = yL / 2 * np.sqrt((j + m) * (j - m + 1) * (j + m1) * (j - m1 + 1)) * (N/2 + 1) / (j * (j + 1))

        g2 = spontaneous + losses

        return (g2)
    
    def gamma3(self, jmm1):
        """
        Calculate gamma3 for given j, m, m'
        """
        j, m, m1 = jmm1
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0) or (j <= 0) :
            g3 = 0.0
        else:
            g3 = yL / 2 * np.sqrt((j + m) * (j + m - 1) * (j + m1) * (j + m1 - 1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (g3)
    
    def gamma4(self, jmm1):
        """
        Calculate gamma4 for given j, m, m'
        """
        j, m, m1 = jmm1
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0)  or ( (j + 1) <= 0):
            g4 = 0.0
        else:
            g4 = yL / 2 * np.sqrt((j - m + 1) * (j - m + 2) * (j - m1 + 1) * (j - m1 + 2)) * (N/2 - j )/((j + 1)* (2 * j + 1))

        return (g4)
    
    def gamma5(self, jmm1):
        """
        Calculate gamma5 for given j, m, m'
        """
        j, m, m1 = jmm1
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if (yD == 0)  or (j <= 0):
            g5 = 0.0
        else:                    
            g5 = yD / 2 * np.sqrt((j**2 - m**2) * (j**2 - m1**2))* (N/2 + j + 1) / (j * (2 * j + 1))

        return (g5)
    
    def gamma6(self, jmm1):
        """
        Calculate gamma6 for given j, m, m'
        """
        j, m, m1 = jmm1
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if yD == 0:
            g6 = 0.0
        else:            
            g6 = yD / 2 * np.sqrt(((j + 1)**2 - m**2) * ((j + 1)**2 - m1**2)) * (N/2 - j )/((j + 1) * (2 * j + 1))

        return (g6)
    
    def gamma7(self, jmm1):
        """
        Calculate gamma7 for given j, m, m'
        """
        j, m, m1 = jmm1
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            g7 = 0.0
        else:    
            g7 = yP / 2 * np.sqrt((j - m - 1) * (j - m)* (j - m1 - 1) * (j - m1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (g7)
    
    def gamma8(self, jmm1):
        """
        Calculate gamma8 for given j, m, m'
        """
        j, m, m1 = jmm1
        yP = self.pumping
        yCP = self.collective_pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            pump = 0.0
        else:    
            pump = yP / 2 * np.sqrt((j + m + 1) * (j - m) * (j + m1 + 1) * (j - m1)) * (N/2 + 1) / (j * (j + 1))

        if yCP == 0:
            collective_pump = 0.0
        else:    
            collective_pump = yCP * np.sqrt((j - m) * (j + m + 1) * (j + m1 + 1) * (j - m1))
        
        g8 = pump + collective_pump
        
        return (g8)
    
    def gamma9(self, jmm1):
        """
        Calculate gamma9 for given j, m, m'
        """
        j, m, m1 = jmm1
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0):
            g9 = 0.0
        else:    
            g9 = yP / 2 * np.sqrt((j + m + 1) * (j + m + 2) *(j + m1 + 1) * (j + m1 + 2)) * (N/2 - j )/((j + 1)*(2 * j + 1))

        return (g9)

    def css_10(self, a, b):
        """
        Loads the separable spin state |->= Prod_i^N(a|1>_i + b|0>_i) into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)
        rho = dok_matrix((nds, nds))

        # loop in the allowed matrix elements        
        j = 0.5 * N 
        mmax = int(2 * j + 1)
        for i in range(0, mmax):
            m = j - i
            psi_m = np.sqrt(float(energy_degeneracy(N, m))) * a**( N * 0.5 + m) * b**( N * 0.5 - m)
            for i1 in range(0, mmax):
                m1 = j - i1
                row_column = self.get_index((j, m, m1))
                psi_m1 = np.sqrt(float(energy_degeneracy(N, m1))) * a**( N * 0.5 + m1) * b**( N * 0.5 - m1)
                rho[row_column] = psi_m * psi_m1
        
        return Qobj(rho)    
    
    def ghz(self):
        """
        Loads the Greenberger‚ÄìHorne‚ÄìZeilinger state, |GHZ>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = dok_matrix((nds, nds))

        rho[0,0] = 1/2
        rho[N,N] = 1/2
        rho[N,0] = 1/2
        rho[0,N] = 1/2

        return Qobj(rho)
    
    def dicke(self, j, m):
        """
        Loads the Dicke state |j, m>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = dok_matrix((nds, nds))

        row_column = self.get_index((j, m, m))
        rho[row_column] = 1

        return Qobj(rho)
    
    def thermal_diagonal(self, temperature):
        """
        Gives the thermal state density matrix at the absolute temperature T for a diagonal hamiltonian.
        It is defined for N two-level systems written into the reduced density matrix rho(j,m,m').
        For temperature = 0, the thermal state is the ground state. 

        Parameters
        ----------
        temperature: float
            The absolute temperature in Kelvin. 
        Returns
        -------
        rho_thermal: matrix array
            A square matrix of dimensions (nds, nds), with nds = num_dicke_states(N).
            The thermal populations are the matrix elements on the main diagonal
        """
        
        N = self.N        
        hamiltonian = self.hamiltonian

        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        if isdiagonal(hamiltonian) == False:
            raise ValueError("Hamiltonian is not diagonal")

        if temperature == 0:        
            ground_energy, ground_state = hamiltonian.groundstate()
            ground_dm= ground_state * ground_state.dag()
            return ground_dm

        eigenval, eigenvec = hamiltonian.eigenstates()
        rho_thermal = dok_matrix((nds, nds))

        s = 0
        for k in range(1, int(num_ladders + 1)):
            j = 0.5 * N + 1 - k
            mmax = (2 * j + 1)
            for i in range(1, int(mmax + 1)):
                m = j + 1 - i
                x = (hamiltonian[s,s] / temperature) * (constants.hbar / constants.Boltzmann)
                rho_thermal[s,s] = np.exp( - x ) * state_degeneracy(N, j)
                s = s + 1
        zeta = self.partition_diagonal(temperature)
        rho = rho_thermal/zeta

        return Qobj(rho)

    def partition_diagonal(self, temperature):
        """
        Gives the partition function for the system at a given temperature if the Hamiltonian is diagonal.
        The Hamiltonian is assumed to be given with hbar = 1.

        Parameters
        ----------
        temperature: float
            The absolute temperature in Kelvin
            
        Returns
        -------
        zeta: float
            The partition function of the system, used to calculate the thermal state.
        """
        N = self.N
        hamiltonian = self.hamiltonian
        
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        zeta = 0
        s = 0

        for k in range(1, int(num_ladders + 1)):
            j = 0.5 * N + 1 - k
            mmax = (2 * j + 1)

            for i in range(1, int(mmax + 1)):
                m = j + 1 - i
                x = (hamiltonian[s,s] / temperature) * (constants.hbar / constants.Boltzmann)
                zeta = zeta + np.exp(- x) * state_degeneracy(N, j)
                s = s + 1

        if zeta <= 0:
            raise ValueError("Error, zeta <=0, zeta = {}".format(zeta))
                
        return float(zeta)

    def thermal_old(self, temperature):
        """
        Gives the thermal state density matrix at the absolute temperature T.
        It is defined for N two-level systems written into the reduced density matrix rho(j,m,m').
        For temperature = 0, the thermal state is the ground state. 

        Parameters
        ----------
        temperature: float
            The absolute temperature in Kelvin. 
        Returns
        -------
        rho_thermal: matrix array
            A square matrix of dimensions (nds, nds), with nds = num_dicke_states(N).
            The thermal populations are the matrix elements on the main diagonal
        """
        
        N = self.N        
        hamiltonian = self.hamiltonian

        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        if temperature == 0:
            if isdiagonal(hamiltonian) == True:
                ground_state = self.dicke( N/2, - N/2)
                return ground_state
            else:
                eigval, eigvec = hamiltonian.eigenstates()
                ground_state = eigvec[0]*eigvec[0].dag()
                return ground_state

        rho_thermal = dok_matrix((nds, nds))

        if isdiagonal(hamiltonian) == True:
            s = 0
            for k in range(1, int(num_ladders + 1)):
                j = 0.5 * N + 1 - k
                mmax = (2 * j + 1)
                for i in range(1, int(mmax + 1)):
                    m = j + 1 - i
                    x = (hamiltonian[s,s] / temperature) * (constants.hbar / constants.Boltzmann)
                    rho_thermal[s,s] = np.exp( - x ) * state_degeneracy(N, j)
                    s = s + 1
            zeta = self.partition_function_diag(temperature)
            rho = rho_thermal/zeta

        else:
            eigval, eigvec = hamiltonian.eigenstates()
            zeta = self.partition_function(temperature)

            rho = rho_thermal/zeta

        return Qobj(rho)

    def eigenstates(self, liouvillian):
        """
        Calculates the eigenvalues and eigenvectors of the Liouvillian, removing the spurious ones. 

        Parameters
        ----------
        liouvillian: Qobj superoperator type
            The Liouvillian of which to calculate the spectrum.
            
        Returns
        -------
        eigen_states: list of Qobj
            The list of eigenvalues and correspondent eigenstates.
        """
        unpruned_eigenstates = liouvillian.eigenstates()

        eigen_states = self.prune_eigenstates_v3(unpruned_eigenstates)

        return eigen_states

    def prune_eigenstates_v1(self, liouvillian_eigenstates):
        """
        Removes the spurious eigenvalues and eigenvectors of the Liouvillian.
        Spurious means that the given eigenvector has elements outside of the block diagonal matrix 

        Parameters
        ----------
        liouvillian_eigenstates: list of Qobj
            A list with the eigenvalues and eigenvectors of the Liouvillian including spurious ones. 

        Returns
        -------
        correct_eigenstates: list of Qobj
            The list with the correct eigenvalues and eigenvectors of the Liouvillian.
        """
        
        N = self.N

        block_mat = block_matrix(N)

        eig_val, eig_vec = liouvillian_eigenstates

        tol = 8
        eig_val = np.round(eig_val,tol)

        # 1. Restrict search only to eigenvalues on imaginary axis (for physical reasons).

        index_imag_eig = []
        for i in range(0,len(eig_val)):
            if np.real(eig_val[i]) == 0:
                index_imag_eig.append(i) 

        # 2. Use block matrix as a mask to find forbidden eigenvectors.
        
        forbidden_eig_index = []

        for k in index_imag_eig:
            dm = vector_to_operator(eig_vec[k])
            dm_mask = Qobj(dm.data.multiply(block_mat))
            if dm_mask != dm:
                forbidden_eig_index.append(k)
        
        # 3. Remove the forbidden eigenvalues and eigenvectors.
        
        correct_eig_val = np.delete(eig_val,forbidden_eig_index)
        correct_eig_vec = np.delete(eig_vec,forbidden_eig_index)

        correct_eigenstates = correct_eig_val, correct_eig_vec

        return correct_eigenstates

    def prune_eigenstates_v2(self, liouvillian_eigenstates):
        """
        Removes the spurious eigenvalues and eigenvectors of the Liouvillian.
        Spurious means that the given eigenvector has elements outside of the block diagonal matrix 

        Parameters
        ----------
        liouvillian_eigenstates: list of Qobj
            A list with the eigenvalues and eigenvectors of the Liouvillian including spurious ones. 

        Returns
        -------
        correct_eigenstates: list of Qobj
            The list with the correct eigenvalues and eigenvectors of the Liouvillian.
        """
        
        N = self.N

        dict_jmm1 = self.jmm1_dictionary()[0]
        block_mat = block_matrix(N)

        # 0. Search eigenvalues that have zero real part by createing an approximated value
        eig_val, eig_vec = liouvillian_eigenstates
        tol = 8
        eig_val_round = np.round(eig_val, tol)

        # 1. Restrict search only to eigenvalues on imaginary axis (for physical reasons).
        index_imag_eig = []
        for i in range(0,len(eig_val)):
            if np.real(eig_val_round[i]) == 0:
                index_imag_eig.append(i)

        # 2. Use jmm1_dict to find eigenvectors with matrix elements outside of the block matrix.
        forbidden_eig_index = []
        forbidden_eig_index_bm = []
        for k in index_imag_eig:
            dm = vector_to_operator(eig_vec[k])
            nnz_tuple = [(i, j) for i, j in zip(*dm.data.nonzero())]
            nnz_tuple_bm = [(i, j) for i, j in zip(*block_mat.nonzero())]
            for i in nnz_tuple:
                if i not in nnz_tuple_bm:
                #    forbidden_eig_index_bm.append(k)
                #if i not in dict_jmm1:
                    if np.round(dm[i],tol) !=0:
                        forbidden_eig_index.append(k)
                        break

        # 3. Remove the forbidden eigenvalues and eigenvectors.
        correct_eig_val = np.delete(eig_val,forbidden_eig_index)
        correct_eig_vec = np.delete(eig_vec,forbidden_eig_index)
        correct_eigenstates = correct_eig_val, correct_eig_vec

        return correct_eigenstates

    def prune_eigenstates_v3(self, liouvillian_eigenstates):
        """
        Removes the spurious eigenvalues and eigenvectors of the Liouvillian.
        Spurious means that the given eigenvector has elements outside of the block diagonal matrix 

        Parameters
        ----------
        liouvillian_eigenstates: list of Qobj
            A list with the eigenvalues and eigenvectors of the Liouvillian including spurious ones. 

        Returns
        -------
        correct_eigenstates: list of Qobj
            The list with the correct eigenvalues and eigenvectors of the Liouvillian.
        """
        
        N = self.N
        block_mat = block_matrix(N)
        nnz_tuple_bm = [(i, j) for i, j in zip(*block_mat.nonzero())]

        # 0. Create  a copy of the eigenvalues to approximate values
        eig_val, eig_vec = liouvillian_eigenstates
        tol = 10
        eig_val_round = np.round(eig_val, tol)

        # 2. Use 'block_matrix(N)' to remove eigenvectors with matrix elements outside of the block matrix.
        forbidden_eig_index = []
        for k in range(0,len(eig_vec)):
            dm = vector_to_operator(eig_vec[k])
            nnz_tuple = [(i, j) for i, j in zip(*dm.data.nonzero())]
            for i in nnz_tuple:
                if i not in nnz_tuple_bm:
                    if np.round(dm[i],tol) !=0:
                        #print(nnz_tuple)
                        forbidden_eig_index.append(k)
                        #break

        forbidden_eig_index = np.array(list(set(forbidden_eig_index)))
        # 3. Remove the forbidden eigenvalues and eigenvectors.
        correct_eig_val = np.delete(eig_val,forbidden_eig_index)
        correct_eig_vec = np.delete(eig_vec,forbidden_eig_index)
        correct_eigenstates = correct_eig_val, correct_eig_vec

        return correct_eigenstates

    def prune_eigenstates(self, liouvillian_eigenstates):
        """
        Removes the spurious eigenvalues and eigenvectors of the Liouvillian.
        Spurious means that the given eigenvector has elements outside of the block diagonal matrix 

        Parameters
        ----------
        liouvillian_eigenstates: list of Qobj
            A list with the eigenvalues and eigenvectors of the Liouvillian including spurious ones. 

        Returns
        -------
        correct_eigenstates: list of Qobj
            The list with the correct eigenvalues and eigenvectors of the Liouvillian.
        """
        
        N = self.N
        block_mat = block_matrix(N)
        nds = num_dicke_states(N)


        # 0. Create  a copy of the eigenvalues to approximate values
        eig_val, eig_vec = liouvillian_eigenstates
        mask_column = Qobj(block_mat).full().flatten().astype(bool)
        liouv_stack = np.column_stack([eig_vec[i].full() for i in range(len(eig_vec))])
        masked_incorrect = np.array([np.round(sum(liouv_stack[:, i][~mask_column]), 10) for i in range(liouv_stack.shape[1])])
        forbidden_eig_index = masked_incorrect.reshape(nds**2, 1).nonzero()[0]

        # 3. Remove the forbidden eigenvalues and eigenvectors.
        correct_eig_val = np.delete(eig_val,forbidden_eig_index)
        correct_eig_vec = np.delete(eig_vec,forbidden_eig_index)
        correct_eigenstates = correct_eig_val, correct_eig_vec
        #print(forbidden_eig_index)
        #print(eig_val)

        return correct_eigenstates

#modules for the Dicke space

def energy_degeneracy(N, m):
    """
    Calculates how many Dicke states |j, m, alpha> have the same energy
    (hbar * omega_0 * m) given N two-level systems. 
    The use of the Decimals class allows to explore N > 1000, 
    unlike the built-in function 'scipy.special.binom(N, N/2 + m)'

    Parameters
    ----------
    N: int
        The number of two level systems.
    m: float
        Total spin z-axis projection eigenvalue. 
        This is proportional to the total energy)

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
    """
    Calculates the degeneracy of the Dicke state |j, m>.
    Each state |j, m> includes D(N,j) irreducible representations |j, m,alpha>.
    Uses Decimals to calculate higher numerator and denominators numbers.

    Parameters
    ----------
    N: int
        The number of two level systems.

    j: float
        Total spin eigenvalue (cooperativity)

    Returns
    -------
    degeneracy: int
        The state degeneracy
    """
    numerator = Decimal(factorial(N)) * Decimal(2 * j + 1) 
    denominator_1 = Decimal(factorial(N/2 + j + 1))
    denominator_2 = Decimal(factorial(N/2 - j ))

    degeneracy = numerator/(denominator_1 * denominator_2 )
    degeneracy = int(np.round(float(degeneracy)))

    if degeneracy < 0 :
        raise ValueError("m-degeneracy must be >=0")

    return degeneracy

def m_degeneracy(N, m):
    """
    The number of Dicke states |j, m> with same energy (hbar * omega_0 * m) for N two-level systems. 

    Parameters
    ----------
    N : int
        The number of two level systems
    m: float
        Total spin z-axis projection eigenvalue (proportional to the total energy)

    Returns
    -------
    degeneracy: int
        The m-degeneracy
    """
    degeneracy = N/2 + 1 - abs(m)
    if degeneracy % 1 != 0 or degeneracy <= 0 :
        raise ValueError("m-degeneracy must be integer >=0, but degeneracy = {}".format(degeneracy))
    return int(degeneracy)


#auxiliary functions

def isdiagonal(matrix):
    """
    Check if a matrix is diagonal either if it is a Qobj or a ndarray
    """
    if isinstance(matrix, Qobj): 
        matrix = matrix.full()
      
    isdiag = np.all(matrix == np.diag(np.diagonal(matrix)))
    
    return isdiag

def j_algebra(N):
    """
    Gives the list with the collective operators of the total algebra, using the reduced basis |j,m><j,m'| in which the density matrix is expressed.    
    The list returned is [J^2, J_x, J_y, J_z, J_+, J_-]. 
    Parameters
    ----------
    N: int 
        Number of two-level systems    
    Returns
    -------
    red_alg: list
        Each element of the list is a Qobj matrix (QuTiP class) of dimensions (nds,nds). nds = number of Dicke states.    
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)

    jz_operator = dok_matrix((nds, nds))
    jp_operator = dok_matrix((nds, nds))
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                jz_operator[s,s] = m
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = ap(j,m-1)
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] = am(j,m+1)
                s = s + 1
    jx_operator = 1/2*(jp_operator + jm_operator)
    jy_operator = 1j/2*(jm_operator - jp_operator)

    j_alg = [Qobj(jx_operator), Qobj(jy_operator), Qobj(jz_operator), Qobj(jp_operator), Qobj(jm_operator)]

    return j_alg


def jx_op(N):
    """
    Builds the Jx operator in the same basis of the reduced density matrix rho(j,m,m').    
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    jx_operator: Qobj matrix
        The Jx operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.         
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    block_diagonal = block_matrix(N).todense()

    jp_operator = dok_matrix((nds, nds))
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
                s = s + 1
    jx_operator = 1/2*(jp_operator + jm_operator)

    return Qobj(jx_operator)

def jy_op(N):
    """
    Builds the Jy operator in the same basis of the reduced density matrix rho(j,m,m').    
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    jy_operator: Qobj matrix
        The Jy operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.    
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    block_diagonal = block_matrix(N).todense()
    jp_operator = dok_matrix((nds, nds))
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = block_diagonal[s,s+1] * ap(j,m-1)
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] =  block_diagonal[s,s-1] * am(j,m+1)
                s = s + 1
    jy_operator = 1j/2*(jm_operator - jp_operator)

    return Qobj(jy_operator)

def jz_op(N):
    """
    Builds the Jz operator in the same basis of the reduced density matrix rho(j,m,m'). Jz is diagonal in this basis.   
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    jz_operator: Qobj matrix
        The Jz operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.      
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    jz_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                jz_operator[s,s] = m
                s = s + 1

    return Qobj(jz_operator)

def j2_op(N):
    """
    Builds the J^2 operator in the same basis of the reduced density matrix rho(j,m,m'). J^2 is diagonal in this basis.   
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    j2_operator: Qobj matrix
        The J^2 operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.  
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    j2_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                j2_operator[s,s] = j * (j + 1)
                s = s + 1

    return Qobj(j2_operator) 

def jp_op(N):
    """
    Builds the Jp operator in the same basis of the reduced density matrix rho(j,m,m').    
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    jp_operator: Qobj matrix
        The Jp operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.    
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    jp_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s + 1) in range(0,nds):
                    jp_operator[s,s+1] = ap(j,m-1)
                s = s + 1

    return Qobj(jp_operator)

def jm_op(N):
    """
    Builds the Jm operator in the same basis of the reduced density matrix rho(j,m,m').    
    Parameters
    ----------
    N: int 
        Number of two-level systems
    Returns
    -------
    jm_operator: Qobj matrix
        The Jm operator as a QuTiP object. The dimensions are (nds,nds) where nds is the number of Dicke states.    
    """
    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    jm_operator = dok_matrix((nds, nds))

    s = 0
    for k in range(0, num_ladders):
            j = 0.5 * N - k
            mmax = int(2 * j + 1)
            for i in range(0, mmax):
                m = j - i
                if (s - 1) in range(0,nds):
                    jm_operator[s,s-1] = am(j,m+1)
                s = s + 1

    return Qobj(jm_operator)

def ap( j, m):
    """
    Calculate A_{+} for value of j, m.
    """
    a_plus = np.sqrt((j - m) * (j + m + 1))
    
    return(a_plus)

def am(j, m):
    """
    Calculate A_{m} for value of j, m.
    """
    a_minus = np.sqrt((j + m) * (j - m + 1))
    
    return(a_minus)

def block_matrix(N):
    """
    Gives the block diagonal matrix filled with 1 if the matrix element is allowed in the reduced basis |j,m><j,m'|.   
    Parameters
    ----------
    N: int 
        Number of two-level systems
    
    Returns
    -------
    block_matr: ndarray
        A block diagonal matrix of ones with dimension (nds,nds), where nds is the number of Dicke states for N two-level systems.   
    """
    nds = num_dicke_states(N)
    
    #create a list with the sizes of the blocks, in order
    blocks_dimensions = int(N/2 + 1 - 0.5 *(N%2))
    blocks_list = [ (2*(i+1*(N%2))+1*((N+1)%2)) for i in range(blocks_dimensions)]
    blocks_list = np.flip(blocks_list,0)

    #create a list with each block matrix as element  

    square_blocks = []
    k = 0
    for i in blocks_list:
        square_blocks.append(np.ones((i,i)))
        k = k + 1

    #create the final block diagonal matrix (dense)
    block_matr = block_diag(square_blocks)

    return block_matr

# TLS Hilbert space (2**N) modules (full Hilbert space 2**N)

# TLS Hilbert space (2**N) modules

def su2_algebra(N):
    """
    Creates the vector (sx, sy, sz, sm, sp) with the spin operators of a collection of N two-level 
    systems (TLSs). Each element of the vector, i.e., sx, is a vector of Qobs objects (spin matrices),
    as it cointains the list of the SU(2) Pauli matrices for the N TLSs. 
    Each TLS operator sx[i], with i = 0, ..., (N-1), is placed in a 2^N-dimensional Hilbert space.
     
    Parameters
    ----------
    N: int
        The number of two level systems
    
    Returns
    -------
    su2_operators: list
        A list of Qobs matrices (Qutip objects) - [sx, sy, sz, sm, sp]
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)
    sx = [0 for i in range(N)]
    sy = [0 for i in range(N)]
    sz = [0 for i in range(N)]
    sm = [0 for i in range(N)]    
    sp = [0 for i in range(N)]
    sx[0] =  0.5 * sigmax()
    sy[0] =  0.5 * sigmay()
    sz[0] =  0.5 * sigmaz()
    sm[0] =  sigmam()
    sp[0] =  sigmap()

    # 2. Place operators in total Hilbert space
    for k in range(N - 1):
        sx[0] = tensor(sx[0], identity(2))
        sy[0] = tensor(sy[0], identity(2))
        sz[0] = tensor(sz[0], identity(2))
        sm[0] = tensor(sm[0], identity(2))
        sp[0] = tensor(sp[0], identity(2))

    #3. Cyclic sequence to create all N operators
    a = [i for i in range(N)]
    b = [[a[i  -  i2] for i in range(N)] for i2 in range(N)]

    #4. Create N operators
    for i in range(1,N):
        sx[i] = sx[0].permute(b[i])
        sy[i] = sy[0].permute(b[i])
        sz[i] = sz[0].permute(b[i])
        sm[i] = sm[0].permute(b[i])
        sp[i] = sp[0].permute(b[i])
    
    su2_operators = [sx, sy, sz, sm, sp]
    
    return su2_operators

def collective_algebra(N):
    """
    Uses the module su2_algebra to create the collective spin algebra Jx, Jy, Jz, Jm, Jp.
    It uses the basis of the sinlosse two-level system (TLS) SU(2) Pauli matrices. 
    Each collective operator is placed in a Hilbert space of dimension 2^N.
     
    Parameters
    ----------
    N: int
        The number of two level systems
    
    Returns
    -------
    collective_operators: vector of Qobs matrices (Qutip objects)
        collective_operators = [Jx, Jy, Jz, Jm, Jp]
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)
    
    si_TLS = su2_algebra(N)
    
    sx = si_TLS[0]
    sy = si_TLS[1]
    sz = si_TLS[2]
    sm = si_TLS[3]
    sp = si_TLS[4]
        
    jx = sum(sx)
    jy = sum(sy)
    jz = sum(sz)
    jm = sum(sm)
    jp = sum(sp)
    
    collective_operators = [jx, jy, jz, jm, jp]
    
    return collective_operators

def c_ops_tls(N = 2, emission = 1., loss = 0., dephasing = 0., pumping = 0., collective_pumping = 0., collective_dephasing = 0.):
    """
    Create the collapse operators (c_ops) of the Lindblad master equation in the TLS uncoupled basis. 
    The collapse operators oare created to be given to the Qutip algorithm 'mesolve'.
    'mesolve' is used in the main file to calculate the time evolution for N two-level systems (TLSs). 
    Notice that the operators are placed in a Hilbert space of dimension 2^N. 
    Thus the method is suitable only for small N.
     
    Parameters
    ----------
    N: int
        The number of two level systems
    emission: float
        default = 2
        Spontaneous emission coefficient
    loss: float
        default = 0
        Losses coefficient (i.e. nonradiative emission)
    dephasing: float
        default = 0
        Dephasing coefficient
    pumping: float
        default = 0
        Incoherent pumping coefficient
    collective_pumping: float
        default = 0
        Collective pumping coefficient 
    collective_dephasing: float
        default = 0
        Collective dephasing coefficient 
        
    Returns
    -------
    c_ops: c_ops vector of matrices
        c_ops contains the collapse operators for the Lindbla
    
    """
    N = int(N) 
    
    if N > 10:
        print("Warning! N > 10. dim(H) = 2^N. Use only the permutational invariant methods for large N. ")
    
    [sx, sy, sz, sm, sp] = su2_algebra(N)
    [jx, jy, jz, jm, jp] = collective_algebra(N)
    
    c_ops = []    
    
    if emission != 0 :
        c_ops.append(np.sqrt(emission) * jm)

    if dephasing != 0 :    
        for i in range(0, N):
            c_ops.append(np.sqrt(dephasing) * sz[i])
    
    if loss != 0 :
        for i in range(0, N):
            c_ops.append(np.sqrt(loss) * sm[i])
    
    if pumping != 0 :
        for i in range(0, N):
            c_ops.append(np.sqrt(pumping) * sp[i])
    
    if collective_pumping != 0 :
        c_ops.append(np.sqrt(collective_pumping) * jp)
    
    if collective_dephasing != 0 :
        c_ops.append(np.sqrt(collective_dephasing) * jz)
    
    return c_ops

# TLS Hilbert space (2**N) functions

def excited_tls(N):
    """
    Generates a initial dicke state |N/2, N/2 > as a Qobj in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)

    jz = collective_algebra(N)[2]
    
    en,vn = jz.eigenstates()

    psi0 = vn[2**N - 1]
        
    return psi0

def superradiant_tls(N):
    """
    Generates a initial dicke state |N/2, 0 > (N even) or |N/2, 0.5 > (N odd) as a Qobj in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)

    jz = collective_algebra(N)[2]
    
    en,vn = jz.eigenstates()

    psi0 = vn[2**N - N]
        
    return psi0
       

def ground_tls(N):
    """
    Generates a initial dicke state |N/2, - N/2 > as a Qobj in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    psi0: Qobj array (QuTiP class)
    """
    N = int(N)

    jz = collective_algebra(N)[2]
    
    en,vn = jz.eigenstates()
    
    psi0 = vn[0]
        
    return psi0

def identity_tls(N):
    """
    Generates the identity in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    identity: Qobj matrix (QuTiP class)
        With the correct dimensions (dims)
    """
    N = int(N)
    
    rho = np.zeros((2**N,2**N))
    
    for i in range(0, 2**N) :
        rho[i, i] = 1
    
    spin_dim = [2 for i in range(0,N)]
    spins_dims = list((spin_dim, spin_dim ))

    identity = Qobj(rho, dims = spins_dims)
    
    return identity

def ghz_tls(N):
    """
    Generates the GHZ density matrix in a 2**N dimensional Hilbert space

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    ghz: Qobj matrix (QuTiP class)
        With the correct dimensions (dims)
    """
    N = int(N)
    
    rho = np.zeros((2**N,2**N))
    rho[0, 0] = 1/2
    rho[ 2**N - 1, 0] = 1/2
    rho[0,  2**N - 1] = 1/2
    rho[2**N - 1, 2**N - 1] = 1/2
    
    spin_dim = [2 for i in range(0,N)]
    spins_dims = list((spin_dim, spin_dim ))

    rho = Qobj(rho, dims = spins_dims)
    
    ghz = rho        
    
    return ghz

def css_tls(N):
    """
    Generates the CSS density matrix in a 2**N dimensional Hilbert space.
    The CSS state, also called 'plus state' is, |+>_i = 1/np.sqrt(2) * (|0>_i + |1>_i ).

    Parameters
    ----------
    N: int
        The number of two level systems

    Returns
    -------
    ghz: Qobj matrix (QuTiP class)
        With the correct dimensions (dims)
    """
    N = int(N)

    # 1. Define i_th factorized density matrix in the uncoupled basis
    rho = [0 for i in range(N)]
    rho[0] = 0.5 * (qeye(2) + sigmax())

    # 2. Place single-two-level-system denisty matrices in total Hilbert space
    for k in range(N - 1):
        rho[0] = tensor(rho[0], identity(2))

    #3. Cyclic sequence to create all N factorized density matrices |+><+|_i
    a = [i for i in range(N)]
    b = [[a[i  -  i2] for i in range(N)] for i2 in range(N)]

    #4. Create all other N-1 factorized density matrices |+><+| = Prod_(i=1)^N |+><+|_i
    for i in range(1,N):
        rho[i] = rho[0].permute(b[i])
    
    identity_i = Qobj(np.eye(2**N), dims = rho[0].dims, shape = rho[0].shape)
    rho_tot = identity_i

    for i in range(0,N):
        rho_tot = rho_tot * rho[i]
    
    return rho_tot

def partition_function_tls(N, omega_0, temperature) :
    """
    Gives the partition function for a collection of N two-level systems with H = omega_0 * j_z.
    It is calculated in the full 2**N Hilbert state, using the eigenstates of H in the uncoupled basis, not the Dicke basis.
        
    Parameters
    ----------
    N: int
        The number of two level systems
    omega_0: float
        The resonance frequency of each two-level system (homogeneous ensemble)
    temperature: float
        The absolute temperature in Kelvin    
    Returns
    -------
    zeta: float
        The partition function for the thermal state of H calculated summing over all 2**N states
    """
    
    N = int(N)
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)
    
    jz = collective_algebra(N)[2]
    m_list = jz.eigenstates()[0]
    
    zeta = 0
    
    for m in m_list :
        zeta = zeta + np.exp( - x * m)
            
    return zeta

def thermal_state_tls(N, omega_0, temperature) :
    """
    Gives the thermal state for a collection of N two-level systems with H = omega_0 * j_z.
    It is calculated in the full 2**N Hilbert state on the eigenstates of H in the uncoupled basis, not the Dicke basis. 
    
    Parameters
    ----------
    N: int
        The number of two level systems
    omega_0: float
        The resonance frequency of each two-level system (homogeneous ensemble)
    temperature: float
        The absolute temperature in Kelvin    
    Returns
    -------
    rho_thermal: Qobj operator
        The thermal state calculated in the full Hilbert space 2**N
    """
    
    N = int(N)   
    x = (omega_0 / temperature) * (constants.hbar / constants.Boltzmann)
       
    jz = collective_algebra(N)[2]  
    m_list = jz.eigenstates()[0]
    m_list = np.flip(m_list,0)

    rho_thermal = np.zeros(jz.shape)

    for i in range(jz.shape[0]):
        rho_thermal[i, i] = np.exp( - x * m_list[i])
    rho_thermal = Qobj(rho_thermal, dims = jz.dims, shape = jz.shape)
    
    zeta = partition_function_tls(N, omega_0, temperature)
    
    rho_thermal = rho_thermal / zeta
    
    return rho_thermal

