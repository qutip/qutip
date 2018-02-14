import numpy as np
from qutip import enr_state_dictionaries


def hierarchy_idx(ncut, nexp):
    """
    Setup the Hierarchy configuration for the given parameters.
    We build a dictionary to make the multi-index n for each
    auxilary density matrix (ADM).

    It is given by a vector n = (n1, n2, n3, n4, ..., nk) such that
    n1 + n2 + ... + nk = K where K is the number of exponential
    terms.

    The function is same as the QuTiP function for state dictionaries.
    The terminator is that the indices of the hierarchy level should
    add up to the total number of exponentials.

    For a cutoff 3 and 2 terms in the expansion, we get the following
    {0: (0, 0, 0),
     1: (0, 0, 1),
     2: (0, 0, 2),
     3: (0, 1, 0),
     4: (0, 1, 1),
     5: (0, 2, 0),
     6: (1, 0, 0),
     7: (1, 0, 1),
     8: (1, 1, 0),
     9: (2, 0, 0)}

    Building up a new level requires the previous and next auxilary
    density matrix (the change is only in the "nk" term).
    """
    N_he, he2idx, idx2he = enr_state_dictionaries([ncut + 1]*nexp , ncut)
    return N_he, he2idx, idx2he


class Heom(object):
    """
    The Heom class to tackle Hierarchy.
    
    Parameters
    ==========
    lindbladian: ndarray
        The Lindbladian operator of the system
    
    coupling: ndarray
        The coupling operator
    
    ckr, vkr, cki, vki: ndarray
        The array of real and imaginary terms (amplitude and frequency)
        in the expansion of the correlation function as exponentials

    ncut: int
        The Hierarchy cutoff
        
    kcut: int
        The cutoff in the Matsubara frequencies
    """
    def __init__(self, N, lindbladian, coupling,
                 ckr, vkr, cki, vki,
                 ncut, kcut=None, dt = 1e-2):
        self.lindbladian = lindbladian
        self.coupling = coupling
        self.ck = np.concatenate([ckr, 1j*cki])
        self.vk = np.concatenate([vkr, vki])
        self.ncut = ncut
        
        if kcut:
            self.kcut = kcut
        else:
            self.kcut = len(self.ckr)

        self.nhe, self.he2idx, self.idx2he = hierarchy_idx(ncut, kcut)
        self.rho = np.zeros((2**N, 2**N, self.nhe), dtype = np.complex)
        self.dt = dt
    
    def normalize(self):
        """
        Normalize the initial density vector
        """
        pass

    def _deltak(self):
        """
        Calculates the deltak values for those Matsubara terms which are
        greater than the cutoff set for the exponentials.
        """
        dk = np.sum(np.divide(self.ck[self.kcut:], self.vk[self.kcut:]))
        return dk
    
    def _hinit(self, rho0):
        """
        Initialize the Hierarchy matrix with rho0
        """
        self.rho[..., 0] = rho0

    def _grad(self, n):
        """
        Get the gradient of the Hierarchy ADM at
        level n
        """        
        nidx = self.idx2he[n]
        rho_current = self.rho[..., n]
        
        A = np.random.random(rho_current.shape)
        B = np.random.random(rho_current.shape)
        C = np.random.random(rho_current.shape)
        
        grad = self._t1(rho_current)

        for k in range(self.kcut):
            zeros = np.zeros(self.kcut)
            zeros[k] = 1
            nnext = tuple(nidx + zeros)
            nprev = tuple(nidx - zeros)
            if nnext in self.he2idx:
                grad += rho_current*self.rho[..., self.he2idx[nnext]]
            if nprev in self.he2idx:
                grad += rho_current*self.rho[..., self.he2idx[nprev]]

        return grad

    def grad(self):
        """
        Calculate the gradient of the full Hierarchy for a particular
        time slice.
        """
        grad = np.zeros_like(self.rho)
        for n in self.idx2he:
            grad[..., n] = self._grad(n)

        return grad


    def _t1(self, rho):
        """
        Get the first term of the RHS which is the -i Lindblad + Q
        operator on the density matrix
        """
        L = self.lindbladian
        Q = self.coupling
        dk = self._deltak()

        return (-1j*L + dk*Q**2)*rho
    
