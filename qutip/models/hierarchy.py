from scipy.misc import factorial
from scipy.integrate import ode

from qutip import enr_state_dictionaries, commutator
from qutip import Options
from qutip.solver import Result
from qutip import Qobj, commutator

from copy import copy

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
    def __init__(self, hamiltonian, coupling,
                 ckr, vkr, cki, vki,
                 ncut, kcut=None, rhocut=0):
        self.hamiltonian = hamiltonian
        self.coupling = coupling
        self.ck = np.concatenate([ckr, np.multiply(1j, cki)])
        self.vk = np.concatenate([vkr, vki])
        self.ncut = ncut
        if kcut:
            self.kcut = kcut
        else:
            self.kcut = len(self.ckr)
        self.rhocut = rhocut
        self.nhe, self.he2idx, self.idx2he = hierarchy_idx(ncut, kcut)
        N = self.hamiltonian.shape[0]
        self.rho = np.zeros((N, N, self.nhe), dtype = np.complex)
        self.deltak = self._deltak()
        self.hshape = (N, N, self.nhe)
    
    def _normalize(self, rho, n):
        """
        Normalize the density matrix for the level n
        """
        nk = self.idx2he[n]
        cknk = np.power(np.abs(self.ck[:self.kcut]), nk)
        fnk = factorial(nk)
        terms = np.multiply(fnk, cknk)
        norm = np.power(np.prod(terms), 0.5)
        rho_normalized = rho/norm
        return rho_normalized
    
    def normalize(self, rho):
        """
        Normalize all the Hierarchy elements at a particular time step
        """
        for n in range(self.nhe):
            rho[..., n] = self._normalize(rho[..., n], n)
        return rho
        
    def _deltak(self):
        """
        Calculates the deltak values for those Matsubara terms which are
        greater than the cutoff set for the exponentials.
        """
        # Needs some test or check here
        dk = np.sum(np.divide(self.ck[self.kcut:], self.vk[self.kcut:]))
        return dk

    def _t1(self, rho, n):
        """
        Get the first term of the RHS
        """
        nk = self.idx2he[n]
        H = self.hamiltonian
        Q = self.coupling
        dk = self.deltak
        
        rho_n = rho[..., n]
        kcut = self.kcut
        t1 = 1j*commutator(H, rho_n) + dk*(commutator(Q, commutator(Q, rho_n)))
        
        summation = np.sum(np.multiply(nk, self.vk[:kcut]))
        t2 = np.multiply(summation, rho_n)
        return -(t1+t2)

    def pop_he(self, he2pop):
        """
        Pop the elements of the Hierarchy which have are very small
        """
        for p in he2pop:
            if p in self.he2idx:
                idx = self.he2idx[p]
                self.he2idx.pop(p)
                self.idx2he.pop(idx)
                self.nhe -= 1

    def _grad(self, t, rho, n):
        """
        Get the gradient of the Hierarchy ADM at
        level n
        """        
        nk = list(self.idx2he[n])
        gradient = self._t1(rho, n)
        Q = self.coupling
        ck = self.ck

        for k in range(self.kcut):
            nnext = copy(nk)
            nprev = copy(nk)
            
            nnext[k] += 1
            nprev[k] -= 1
            
            nnext = tuple(nnext)
            nprev = tuple(nprev)
            if nnext in self.he2idx:
                idx = self.he2idx[nnext]
                rho_next = rho[..., idx]
                gradient += -1j*np.sqrt((nk[k]+1)*np.abs(ck[k]))*commutator(Q, rho_next)

            if nprev in self.he2idx:
                idx = self.he2idx[nprev]
                rho_prev = rho[..., idx]
                gradient += -1j*np.sqrt(nk[k]/np.abs(ck[k]))*(ck[k]*Q*rho_prev - np.conjugate(ck[k])*rho_prev*Q)
                
        return gradient

    def grad(self, t, rho):
        """
        Calculate the gradient of the full Hierarchy for a particular
        time slice.
        """
        grad = np.zeros_like(self.rho)
        for n in self.idx2he:
            grad[..., n] = self._grad(t, rho, n)

        return grad
    
    def solve(self, rho0, tlist, options=None, rhocut=1e-3):
        """
        Solve the Hierarchy equations of motion for the given initial
        density matrix and time.
        """
        if options is None:
            options = Options()

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []
        output.states.append(Qobj(rho0))

        dt = tlist[1] - tlist[0]
        self.rho[..., 0] = self._normalize(rho0, 0)
        
        f = lambda t, y: self.grad(t, y.reshape(self.hshape)).ravel()
        r = ode(f)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)

        rho_he_flat = self.rho.ravel()
        r.set_initial_value(rho_he_flat, tlist[0])
        dt = np.diff(tlist)
        n_tsteps = len(tlist)

        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rhot = r.y.reshape(self.hshape)
                rhot = self.normalize(rhot)
                self.rho = rhot.copy()
                he2pop = np.argwhere(np.max(np.absolute(self.rho), (0, 1)) < self.rhocut).flatten()
                self.pop_he(he2pop)
                rho00 = Qobj(self.rho[..., 0])
                output.states.append(rho00)
        return output        
    

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
