"""
Dynamics for dicke states exploiting permutational invariance
"""
from math import factorial
from decimal import Decimal

import numpy as np
from numpy import array, matrix

from scipy.integrate import odeint
from scipy.integrate import ode
from scipy import constants
from scipy.sparse import csr_matrix, dok_matrix
from scipy.io import mmread, mmwrite
from qutip import Qobj
from qutip.solver import Result


def num_dicke_states(N):
    """
    The number of dicke states with a modulo term taking care of ensembles
    with odd number of systems.

    Returns
    -------
    N: int
        The number of two level systems.    
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

    Returns
    -------
    N: int
        The number of two level systems.    
    Nj: int
        The number of Dicke ladders
    """
    Nj = (N + 1) * 0.5 + (1 - np.mod(N, 2)) * 0.5    
    return int(Nj)

def energy_degeneracy(N, m):
    """
    Calculates how many Dicke states |j, m, alpha> have the same energy
    (hbar * omega_0 * m) given N two-level systems. This definition allow to
    explore also N > 1020, unlike the built-in function
    'scipy.special.binom(N, N/2 + m)'

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
    The number of Dicke states |j, m> with same energy (hbar * omega_0 * m)
    for N two-level systems. 

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

def j_min(N):
    """
    Calculate the minimum value of j for given N
    """
    if N % 2 == 0:
        return 0
    else:
        return 0.5


def isdiag(M):
    """
    Check if a matrix is diagonal
    """
    return np.all(M == np.diag(np.diagonal(M)))


class Dicke(object):
    """
    The Dicke States class.
    
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2
        
    emission : float
        Collective spontaneous emmission coefficient
        default: 1.0

    resonance : float
        Resonant frequency w0 of the Hamiltonian H = w0 * Jz**n
        default: 1.0
        
    exponent : int
        Exponent n of Jz in the Hamiltonian H = w0 * Jz**n  
        default: 1
        
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
        
    density_dict: dict
        A nested dictionary holding the mapping (i, k): (j, m, m')}.
        This holds the values for the non zero elements of the density matrix. It
        also serves as a check for valid elements.
    """
    def __init__(self, N = 2, resonance = 1.0, exponent = 1, emission = 1.,
                 loss = 0., dephasing = 0., pumping = 0.,
                 collective_pumping = 0., collective_dephasing = 0.):
        self.N = N
        self.resonance = resonance
        self.exponent = exponent
        self.emission = emission
        self.loss = loss
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.collective_dephasing = collective_dephasing
        
        self.nds = num_dicke_states(N)
        self.dshape = (num_dicke_states(N), num_dicke_states(N))

    def solve(self, rho0, t_list):
        """
        Solve the system dynamics for the given rho0 for the times t_list
        """
        isdiag = False
        
        if isinstance(rho0, Qobj):
            isdiag = isdiag(rho0.data)
        else:
            isdiag = isdiag(rho0)

        solver = Pisolve(self)

        if isdiag:
            solver = Pim(self)

        result = solver.solve(rho0, t_list)
        return result

    def _get_element_v(self, jmm):
        """
        Get the (k) index for given tuple (j, m, m1) from the flattened block diagonal matrix.
        """
        N = self.N
        nds = num_dicke_states(N)
        
        j, m, m1 = jmm
        
        if (j, m, m1) not in self.density_dict:
            return (N+1)
        else:
            ik = self.density_dict[(j, m, m1)]
            k = nds * (ik[0] + ik[1])
            return k
        
    def css(self):
        """
        Loads the coherent spin state (CSS), |+>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds, nds))
        num_ladders = num_dicke_ladders(N)

        jj = 0.5 * N
        mmax = (2 * jj + 1)
        for ii in range(1, int(mmax + 1)):
            mm = jj + 1 - ii
            for ll in range(1, int(mmax + 1)):
                mm1 = jj + 1 - ll
                row_column = self._get_element((jj, mm, mm1))
                if mm == mm1 :
                    rho[row_column] = energy_degeneracy(N,mm)/(2**N)
                else :
                    rho[row_column] = np.sqrt(energy_degeneracy(N,mm)) * np.sqrt(energy_degeneracy(N,mm1)) /(2**N)
        return rho

    def minus(self):
        """
        Loads the separable spin state |->= Prod_i^N(|1>_i - |0>_i) into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds,nds))
        num_ladders = num_dicke_ladders(N)

        jj = 0.5 * N
        mmax = (2 * jj + 1)
        for ii in range(1, int(mmax + 1)):
            mm = jj + 1 - ii
            sign_mm = (-1)**(mm + N/2)
            for ll in range(1, int(mmax + 1)):
                mm1 = jj + 1 - ll
                row_column = self._get_element((jj, mm, mm1))
                sign_mm1 = (-1)**(mm1 + N/2)
                if mm == mm1 :
                    rho[row_column] = energy_degeneracy(N,mm)/(2**N)
                else :
                    rho[row_column] = sign_mm * sign_mm1 * np.sqrt(energy_degeneracy(N,mm)) * np.sqrt(energy_degeneracy(N,mm1)) /(2**N)
        return rho

    def ab(self, a, b):
        """
        Loads the separable spin state |ab>= Prod_i^N( a|0>_i + b|1>_i) into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds,nds))
        num_ladders = num_dicke_ladders(N)

        norm_ab = a * np.conj(a) + b * np.conj(b)
        tol = 10**6
        norm_ab = np.round(norm_ab*tol)/tol
        if norm_ab != 1:
            print("Warning: the state is not normalized")

        jj = 0.5 * N
        mmax = (2 * jj + 1)
        for ii in range(1, int(mmax + 1)):
            mm = jj + 1 - ii
            for ll in range(1, int(mmax + 1)):
                mm1 = jj + 1 - ll
                row_column = self._get_element((jj, mm, mm1))
                if mm == mm1 :
                    rho[row_column] = energy_degeneracy(N,mm)/(2**N)
                else :
                    rho[row_column] = np.sqrt(energy_degeneracy(N,mm)) * np.sqrt(energy_degeneracy(N,mm1)) /(2**N)
        return rho    
    
    def ghz(self):
        """
        Loads the Greenberger–Horne–Zeilinger state, |GHZ>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds,nds))

        rho[0,0] = 1/2
        rho[N,N] = 1/2
        rho[N,0] = 1/2
        rho[0,N] = 1/2
        
        return rho
    
    def dicke(self, j, m):
        """
        Loads the Dicke state |j, m>, into the reduced density matrix rho(j,m,m'). 
        """
        N = self.N

        nds = num_dicke_states(N)
        rho = np.zeros((nds,nds))

        row_column = self._get_element((j, m, m))
        rho[row_column] = 1
        
        return rho
    
    def thermal(self, temperature):
        """
        Gives the thermal state density matrix at the absolute temperature T.
        It is defined for N two-level systems.
        The Hamiltonian is H = hbar * omega_0 * (Jz**n_exp).
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
        omega_0 = self.resonance
        n_exp = self.exponent
        
        if temperature == 0:
            ground_state = self.dicke_jmm( N/2, - N/2)
            return ground_state
        
        x = (omega_0**n_exp / temperature) * (constants.hbar / constants.Boltzmann)

        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)

        rho_thermal = np.zeros((nds,nds))

        s = 0
        for k in range(1, int(num_ladders + 1)):
            j = 0.5 * N + 1 - k
            mmax = (2 * j + 1)
            for i in range(1, int(mmax + 1)):
                m = j + 1 - i
                rho_thermal[s,s] = np.exp( - x * m ) * state_degeneracy(N, j)
                s = s + 1

        zeta = self.partition_function(temperature)

        rho_thermal = rho_thermal/zeta

        return rho_thermal

    def partition_function(self, temperature):
        """
        Gives the partition function for the system at a given temperature.

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
        omega_0 = self.resonance
        n_exp = self.exponent
        
        nds = num_dicke_states(N)
        num_ladders = num_dicke_ladders(N)
        x = (omega_0**n_exp / temperature) * (constants.hbar / constants.Boltzmann)

        zeta = 0
        s = 0

        for k in range(1, int(num_ladders + 1)):
            j = 0.5 * N + 1 - k
            mmax = (2 * j + 1)

            for i in range(1, int(mmax + 1)):
                m = j + 1 - i
                zeta = zeta + np.exp( - x * m ) * state_degeneracy(N, j)
                s = s + 1

        if zeta <= 0:
            raise ValueError("Error, zeta <=0, zeta = {}".format(zeta))
                
        return float(zeta)

    def mean_light_field(self, j, m):
        """
        The coefficient is defined by <j, m|J^+ J^-|j, m>

        Parameters
        ----------
        j: float
            The total spin z-component m for the Dicke state |j, m>

        m: float
            The total spin j for the Dicke state |j, m>

        Returns
        -------
        y: float
            The light field average value
        """
        y = (j + m) * (j - m + 1)    
        return y


class Pisolve(object):
    """
    Permutationally Invariant Quantum Solver
    """
    def __init__(self, dicke_system):
        self.N = dicke_system.N
        self.resonance = dicke_system.resonance
        self.exponent = dicke_system.exponent

        self.emission = dicke_system.emission
        self.loss = dicke_system.loss
        self.dephasing = dicke_system.dephasing
        self.pumping = dicke_system.pumping
        self.collective_pumping = dicke_system.collective_pumping
        self.collective_dephasing = dicke_system.collective_dephasing

        self.nds = dicke_system.nds
        self.dshape = dicke_system.dshape

        self.blocks = self.get_blocks()
        self.density_dict = dict()
        
        self.tau_functions = [self.tau3, self.tau2, self.tau4,
                              self.tau5, self.tau1, self.tau6,
                              self.tau7, self.tau8, self.tau9]
        
        self.tau_dict = {x.__name__:{} for x in self.tau_functions}

        self.generate_dict()

    def get_blocks(self):
        """
        A list which gets the number of cumulative elements at each block
        boundary

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
        N = self.N
        num_blocks = num_dicke_ladders(N)
        blocks = np.array([i * (N + 2 - i) for i in range(1, num_blocks + 1)], dtype = int)
        return blocks

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

    def get_index(self, j, m, m1):
        """
        Get the index in the density matrix for this j, m, m1 value.
        """
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
        
    def generate_dict(self):
        """
        Populate the density matrix and create the map from (jmm1) to (ik)
        """
        # This is the crux of the whole code
        # Need to optimize these loops
        for j in self.j_vals():
            for m in self.m_vals(j):
                for m1 in self.m_vals(j):
                    i, k = self.get_index(j, m, m1)
                    self.density_dict[(j, m, m1)] = (i, k)
                    self._generate_taus(j, m, m1)

    def _generate_taus(self, j, m, m1):
        """
        Generates a Tau mask for this j, m, m1. This is a nds x nds x 9 array
        for each Tau
        """                
        for tau in self.tau_functions:
            self.tau_dict[tau.__name__][(j, m, m1)] = tau(j, m, m1)

    def get_tau(self, tau, jmm):
        """
        Get the value of tau(j, m, m1) if it is a valid tau
        """
        j, m, m1 = jmm
        
        if (j, m, m1) not in self.density_dict:
            return 0.
        else:
            return self.tau_dict[tau][jmm]

    def _get_element(self, jmm):
        """
        Get the (i, k) index for given tuple (j, m, m1) from the block diagonal matrix.
        """
        j, m, m1 = jmm
        
        if (j, m, m1) not in self.density_dict:
            return (0, 0)
        else:
            return self.density_dict[jmm]
        
    def _get_gradient(self, rho, jmm):
        """
        The derivative for the reduced block diagonal density matrix rho which
        generates the dynamics.

        There are 9 terms which form this derivative. All the 9 terms are
        indexed by j, m, m', j +- 1, m +- 1, m' +- 1. There could be an instance
        where the value of j, m, m' leads to an element in the density matrix
        which is invalid (outside of the block diagonals). We need to check for
        this and set that term to 0

        =====================================================================
        Write the full equation here explaining the validity checks
        =====================================================================

        Parameters
        ----------
        rho: arr
            The block diagonal density matrix rho for the given time
        """
        j, m, m1 = jmm

        # change how get element works
        t1 = self.get_tau("tau1", (j, m, m1)) * rho[self._get_element((j, m, m1))]
        
        t2 = self.get_tau("tau2", (j, m+1, m1+1)) * rho[self._get_element((j, m+1, m1+1))]
        
        t3 = self.get_tau("tau3", (j+1, m+1, m1+1)) * rho[self._get_element((j+1, m+1, m1+1))]
        t4 = self.get_tau("tau4", (j-1, m+1, m1+1)) * rho[self._get_element((j-1, m+1, m1+1))]
        t5 = self.get_tau("tau5", (j+1, m, m1)) * rho[self._get_element((j+1, m, m1))]
        t6 = self.get_tau("tau6", (j-1, m, m1)) * rho[self._get_element((j-1, m, m1))]
        t7 = self.get_tau("tau7", (j+1, m-1, m1-1)) * rho[self._get_element((j+1, m-1, m1-1))]
        t8 = self.get_tau("tau8", (j, m-1, m1-1)) * rho[self._get_element((j, m-1, m1-1))]
        t9 = self.get_tau("tau9", (j-1, m-1, m1-1)) * rho[self._get_element((j-1, m-1, m1-1))]

        rdot = - t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9
        
        return rdot

    def rho_dot(self, rho):
        """
        Get the gradient for all the elements in the density matrix by looping over
        j, m, m'
        """
        grad = np.zeros(self.dshape, dtype=np.complex)
        for jmm in self.density_dict.keys():
            i, k = self.density_dict[jmm]
            grad[i, k] = self._get_gradient(rho, jmm)

        return grad.flatten()

    def f(self, t, y):
        return self.rho_dot(y.reshape(self.dshape))

    def solve(self, rho0, t_list):
        """
        Solve the differential equation dp/dt = Tau to evolve the density matrix.

        The density matrix is a block diagonal matrix which is sparse. Based on the
        initial density matrix, we can choose to just run evolution for the poplulation.

        If the density matrix is diagonal, we can just evolve it for the diagonal elements.

        There is a scope for parallelization as e v olution of some of the sets of elements
        are de coupled from the others.

        =============================================================================
        Check this properly and optimize
        =============================================================================

        Example: N = 4

        1 1 1 1 1 
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1 
        1 1 1 1 1
                1 1 1
                1 1 1
                1 1 1
                     1

        Parameters
        ----------
        rho: ndarray
            The intitial density matrix
        """
        y0, t0 = rho0.flatten(), t_list[0]

        nt= np.size(t_list)

        r = ode(self.f).set_integrator("zvode")
        r.set_initial_value(y0, t0)

        t1 = t_list[-1]
        dt = (t_list[-1] - t_list[0])/len(t_list)
        
        result = Result()
        result.states.append(Qobj(rho0))
        while r.successful() and r.t < t1:
            rho_t = r.integrate(r.t + dt).reshape(self.dshape)
            result.states.append(Qobj(rho_t))

        result.states = result.states[:nt]
        
        result.solver = "Pim"
        result.times = t_list
        
        return result

    def tau1(self, j, m, m1):
        """
        Calculate tau1 for value of j, m, m'
        """
        w0 = self.resonance
        n_exp = self.exponent
        yS = self.emission
        yL = self.loss
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping
        yCD = self.collective_dephasing
        
        N = self.N  
        N = float(N)

        if m == m1:
            resonant = 0
        else :
            resonant = w0 * 1j * (m**n_exp - m1**n_exp)
        spontaneous = yS / 2 * (2 * j * (j + 1) - m * (m - 1) - m1 * (m1 - 1))
        losses = yL/2 * (N + m + m1)
        pump = yP/2 * (N - m - m1)
        collective_pump = yCP / 2 * (2 * j * (j + 1) - m * (m + 1) - m1 * (m1 + 1))
        collective_dephase = yCD / 2 * (m - m1)**2
        
        if j <= 0:
            dephase = yD * N/4
        else :
            dephase = yD/2 * (N/2 - m * m1 * (N/2 + 1)/ j /(j + 1))

        t1 = resonant + spontaneous + losses + pump + dephase + collective_pump + collective_dephase
        
        return(t1)
    
    def tau2(self, j, m, m1):
        """
        Calculate tau2 for given j, m, m'
        """
        yS = self.emission
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if yS == 0:
            spontaneous = 0
        else:            
            spontaneous = yS * np.sqrt((j + m) * (j - m + 1)* (j + m1) * (j - m1 + 1))

        if (yL == 0) or (j <= 0):
            losses = 0
        else:            
            losses = yL / 2 * np.sqrt((j + m) * (j - m + 1) * (j + m1) * (j - m1 + 1)) * (N/2 + 1) / (j * (j + 1))

        t2 = spontaneous + losses

        return (t2)
    
    def tau3(self, j, m, m1):
        """
        Calculate tau3 for given j, m, m'
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0) or (j <= 0) :
            t3 = 0
        else:
            t3 = yL / 2 * np.sqrt((j + m) * (j + m - 1) * (j + m1) * (j + m1 - 1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (t3)
    
    def tau4(self, j, m, m1):
        """
        Calculate tau4 for given j, m, m'
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)

        if (yL == 0)  or ( (j + 1) <= 0):
            t4 = 0
        else:
            t4 = yL / 2 * np.sqrt((j - m + 1) * (j - m + 2) * (j - m1 + 1) * (j - m1 + 2)) * (N/2 - j )/((j + 1)* (2 * j + 1))

        return (t4)
    
    def tau5(self, j, m, m1):
        """
        Calculate tau5 for given j, m, m'
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if (yD == 0)  or (j <= 0):
            t5 = 0
        else:                    
            t5 = yD / 2 * np.sqrt((j**2 - m**2) * (j**2 - m1**2))* (N/2 + j + 1) / (j * (2 * j + 1))

        return (t5)
    
    def tau6(self, j, m, m1):
        """
        Calculate tau6 for given j, m, m'
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)

        if yD == 0:
            t6 = 0
        else:            
            t6 = yD / 2 * np.sqrt(((j + 1)**2 - m**2) * ((j + 1)**2 - m1**2)) * (N/2 - j )/((j + 1) * (2 * j + 1))

        return (t6)
    
    def tau7(self, j, m, m1):
        """
        Calculate tau7 for given j, m, m'
        """
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            t7 = 0
        else:    
            t7 = yP / 2 * np.sqrt((j - m - 1) * (j - m)* (j - m1 - 1) * (j - m1)) * (N/2 + j + 1) / (j * (2 * j + 1))

        return (t7)
    
    def tau8(self, j, m, m1):
        """
        Calculate tau8 for given j, m, m'
        """
        yP = self.pumping
        yCP = self.collective_pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0) or (j <= 0):
            pump = 0
        else:    
            pump = yP / 2 * np.sqrt((j + m + 1) * (j - m) * (j + m1 + 1) * (j - m1)) * (N/2 + 1) / (j * (j + 1))

        if yCP == 0:
            collective_pump = 0
        else:    
            collective_pump = yCP * np.sqrt((j - m) * (j + m + 1) * (j + m1 + 1) * (j - m1))
        
        t8 = pump + collective_pump
        
        return (t8)
    
    def tau9(self, j, m, m1):
        """
        Calculate tau9 for given j, m, m'
        """
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        if (yP == 0):
            t9 = 0
        else:    
            t9 = yP / 2 * np.sqrt((j + m + 1) * (j + m + 2) *(j + m1 + 1) * (j + m1 + 2)) * (N/2 - j )/((j + 1)*(2 * j + 1))

        return (t9)

def is_diagonal(matrix):
    """
    Returns True or False whether 'matrix' is a diagonal matrix or not. 
    The module is thought to check the properties of an initial density matrix state rho(j,m,m1) used in Pim.
    The Qobj reshaped matrix has dims = [[2,...2],[2, ...2]] (2 is repeated N times) and shape = (2**N, 2**N).
    Parameters
    ----------
    matrix: ndarray or Qobj (square matrix)
    
    Returns
    -------
    True or False
    """

    nds = matrix.shape[0]
    NN = num_two_level(nds)
    spin_dim = [2 for i in range(0,NN)]
    spins_dims = list((spin_dim, spin_dim ))
    
    matrix_qobj = Qobj(matrix, dims = spins_dims)
    
    diag_mat = diagonal_matrix(matrix_qobj)
    return matrix_qobj == diag_mat 

class Pim(object):
    """
    The permutation invariant matrix class. Initialize the class with the
    parameters for generating a permutation invariant density matrix.
    
    Parameters
    ----------
    N : int
        The number of two level systems
        default: 2
        
    emission : float
        Collective loss emmission coefficient
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

    M: dict
        A nested dictionary of the structure {row: {col: val}} which holds
        non zero elements of the matrix M

    sparse_M: scipy.sparse.csr_matrix
        A sparse representation of the matrix M for efficient vector multiplication
    """
    def __init__(self, dicke_system):
        self.N = dicke_system.N
        self.resonance = dicke_system.resonance
        self.exponent = dicke_system.exponent

        self.emission = dicke_system.emission
        self.loss = dicke_system.loss
        self.dephasing = dicke_system.dephasing
        self.pumping = dicke_system.pumping
        self.collective_pumping = dicke_system.collective_pumping
        self.collective_dephasing = dicke_system.collective_dephasing

        self.M = {}
        self.sparse_M = None

    def isdicke(self, dicke_row, dicke_col):
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
        rows = self.N + 1
        cols = 0
        
        if (self.N % 2) == 0:
            cols = int(self.N/2 + 1)
        else:
            cols = int(self.N/2 + 1/2)

        if (dicke_row > rows) or (dicke_row < 0):
            return (False)

        if (dicke_col > cols) or (dicke_col < 0):
            return (False)

        if (dicke_row < int(rows/2)) and (dicke_col > dicke_row):
            return False

        if (dicke_row >= int(rows/2)) and (rows - dicke_row <= dicke_col):
            return False
        
        else:
            return True

    def tau_valid(self, dicke_row, dicke_col):
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
        tau_functions = [self.tau3, self.tau2, self.tau4,
                         self.tau5, self.tau1, self.tau6,
                         self.tau7, self.tau8, self.tau9]

        N = self.N
        
        if self.isdicke(dicke_row, dicke_col) is False:
            return False

        # The 3x3 sub matrix surrounding the Dicke space element to
        # run the tau functions

        indices = [(dicke_row + x, dicke_col + y) for x in range(-1, 2) for y in range(-1, 2)]
        taus = {}
        
        for idx, tau in zip(indices, tau_functions):
            if self.isdicke(idx[0], idx[1]):
                j, m = self.calculate_j_m(idx[0], idx[1])
                taus[tau.__name__] = tau(j, m)
        
        return taus

    def calculate_j_m(self, dicke_row, dicke_col):
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
        N = self.N
        
        j = N/2 - dicke_col
        m = N/2 - dicke_row
        
        return(j, m)

    def calculate_k(self, dicke_row, dicke_col):
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
        N = self.N
        
        if dicke_row == 0:
            k = dicke_col

        else:
            k = int(((dicke_col)/2) * (2 * (N + 1) - 2 * (dicke_col - 1)) + (dicke_row - (dicke_col)))
            
        return k
       
    def generate_k(self, dicke_row, dicke_col):
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
        if self.isdicke(dicke_row, dicke_col) is False:
            return False
        
        # Calculate k as the number of Dicke elements till
        
        k = int(self.calculate_k(dicke_row, dicke_col))
        
        row = {}
        
        taus = self.tau_valid(dicke_row, dicke_col)
        
        for tau in taus:
            j, m = self.calculate_j_m(dicke_row, dicke_col)
            current_col = tau_column(tau, k, j)
            self.M[(k, int(current_col))] = taus[tau]
            row[k] = {current_col: taus[tau]}

        return row

    def generate_dicke_space(self):
        """
        Generate a Dicke space if you want to play around. N is the number of TLS obtained
        from the class.
        """        
        N = self.N
        rows = N + 1
        cols = 0

        if (rows % 2) == 0:
            cols = int((rows/2))

        else:
            cols = int((rows + 1)/2)

        dicke_space = np.zeros((rows, cols), dtype = int)

        for (i, j) in np.ndindex(rows, cols):
            dicke_space[i, j] = self.isdicke(i, j)

        return (dicke_space)

    def generate_matrix(self):
        """
        Generate the matrix M
        """
        N = self.N
        rows = self.N + 1
        cols = 0
        
        if (self.N % 2) == 0:
            cols = int(self.N/2 + 1)
        else:
            cols = int(self.N/2 + 1/2)

        for (dicke_row, dicke_col) in np.ndindex(rows, cols):
            if self.isdicke(dicke_row, dicke_col):
                self.generate_k(dicke_row, dicke_col)
        
        return self.M

    def generate_sparse(self):
        """
        Generate sparse format of the matrix M
        """
        
        N = self.N  
        M = self.M

        nds = num_dicke_states(N)
        
        if not self.M.keys:
            self.generate_matrix()

        sparse_M = dok_matrix((nds, nds), dtype=float)
        
        for (i, j) in M.keys():
            sparse_M[i, j] = M[i, j]

        self.sparse_M = sparse_M.asformat("csr")


    def solve(self, rho0, t_list):
        """
        Solver for the system which returns a QuTiP result
        """
        self.generate_sparse()
        # Convert the initial full density matrix into a vector for M
        diagonal_elements = rho0.diagonal()

        rho_t = odeint(rhs_generate, initial_state, t, args=(self.sparse_M,))
        result_list = [np.diag(x) for x in rho_t]

        result = Result()

        for r in result_list:
            result.states.append(Qobj(r))
        result.states = result.states[:nt]
        nt= np.size(t_list)

        result.states = result.states[:nt]

        result.solver = "Pim"
        result.times = t_list
        
        return result


    def rhs_generate(self, rho, t_list, M):
        """
        Get right-hand side (RHS) of the ordinary differential equation (ODE) in time. 
        
        Parameters
        ----------
        M: scipy.sparse
            A sparse matrix capturing the dynamics of the system

        Returns
        -------
        M.dot(rho): array
            The state vector at current time
        """
        return M.dot(rho)

    def tau1(self, j, m):
        """
        Calculate tau1 for value of j and m.
        """
        yS = self.emission
        yL = self.loss
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping

        N = self.N  
        N = float(N)

        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (N/2 + m)
        pump = yP * (N/2 - m)
        collective_pump = yCP * (1 + j + m) * (j - m)
        
        if j==0:
            dephase = yD * N/4
        else :
            dephase = yD * (N/4 - m**2 * ((1 + N/2)/(2 * j *(j+1))))

        t1 = spontaneous + losses + pump + dephase + collective_pump
        
        return(-t1)

    def tau2(self, j, m):
        """
        Calculate tau2 for given j and m
        """
        yS = self.emission
        yL = self.loss

        N = self.N  
        N = float(N)

        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (((N/2 + 1) * (j - m + 1) * (j + m))/(2 * j * (j+1)))

        t2 = spontaneous + losses

        return(t2)

    def tau3(self, j, m):
        """
        Calculate tau3 for given j and m
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)

        num = (j + m - 1) * (j + m) * (j + 1 + N/2)
        den = 2 * j * (2 * j + 1)

        t3 = yL * (num/den)

        return (t3)

    def tau4(self, j, m):
        """
        Calculate tau4 for given j and m.
        """
        yL = self.loss
        
        N = self.N  
        N = float(N)


        num = (j - m + 1) * (j - m + 2) * (N/2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t4 = yL * (num/den)

        return (t4)

    def tau5(self, j, m):
        """
        Calculate tau5 for j and m
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)


        num = (j - m) * (j + m) * (j + 1 + N/2)
        den = 2 * j * (2 * j + 1)

        t5 = yD * (num/den)

        return(t5)

    def tau6(self, j, m):
        """
        Calculate tau6 for given j and m
        """
        yD = self.dephasing
        
        N = self.N  
        N = float(N)


        num = (j - m + 1) * (j + m + 1) * (N/2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t6 = yD * (num/den)

        return(t6)

    def tau7(self, j, m):
        """
        Calculate tau7 for given j and m
        """
        yP = self.pumping
        
        N = self.N  
        N = float(N)

        num = (j - m - 1) * (j - m) * (j + 1 + N/2)
        den = 2 * j * (2 * j + 1)

        t7 = yP * (float(num)/den)

        return (t7)

    def tau8(self, j, m):
        """
        Calculate self.tau8
        """
        yP = self.pumping
        yCP = self.collective_pumping
        
        N = self.N  
        N = float(N)


        num = (1 + N/2) * (j - m) * (j + m + 1)
        den = 2 * j * (j + 1)
        pump = yP * (float(num)/den)
        collective_pump = yCP * (j - m) * (j + m + 1)
        
        t8 = pump + collective_pump

        return (t8)

    def tau9(self, j, m):
        """
        Calculate self.tau9
        """
        yP = self.pumping
        
        N = self.N 
        N = float(N)

        num = (j + m + 1) * (j + m + 2) * (N/2 - j)
        den = 2 * (j + 1) * (2 * j + 1)

        t9 = yP * (float(num)/den)

        return (t9)
