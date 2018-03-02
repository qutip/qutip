"""
Heirarchy equations of motion
"""
import numpy as np

from scipy.misc import factorial
from scipy.integrate import complex_ode, ode
from scipy.constants import h as planck

from qutip import enr_state_dictionaries, commutator
from qutip import Options
from qutip.solver import Result
from qutip import Qobj, commutator
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from copy import copy

try:
    from tqdm import tqdm, tqdm_notebook
    progress_bar = True
    if get_ipython().config:
        progress = tqdm_notebook
    else:
        progress = tqdm
except:
    progress_bar = None


class Heom(object):
    """
    The Heom class to tackle Heirarchy.
    
    Parameters
    ==========
    hamiltonian: :class:`qutip.Qobj`
        The system Hamiltonian
    
    coupling: :class:`qutip.Qobj`
        The coupling operator
    
    ck: list
        The list of amplitudes in the expansion of the correlation function

    vk: list
        The list of frequencies in the expansion of the correlation function

    ncut: int
        The Heirarchy cutoff
        
    kcut: int
        The cutoff in the Matsubara frequencies

    rcut: float
        The cutoff for the maximum absolute value in an auxillary matrix
        which is used to remove it from the heirarchy
    """
    def __init__(self, hamiltonian, coupling, ck, vk, 
                 ncut, kcut=None, rcut=None, renorm=False):
        self.hamiltonian = hamiltonian
        self.coupling = coupling        
        self.ck = np.array(ck)
        self.vk = np.array(vk)
        self.ncut = ncut
        self.renorm = renorm

        if kcut:
            self.kcut = kcut
        else:
            self.kcut = len(ck)

        if self.kcut > len(self.vk):
            raise Warning("Truncation of exponential exceeds number of terms")

        self.rcut = rcut
        nhe, he2idx, idx2he = enr_state_dictionaries([self.ncut+1]*self.kcut,
                                                     self.ncut)
        self.nhe = nhe
        self.filtered_nhe = []
        self.he2idx = he2idx
        self.idx2he = idx2he 

        self.N = self.hamiltonian.shape[0]
        self.hshape = (self.nhe, self.N**2)
        self.weak_coupling = self.deltak()
        
        self.L = liouvillian(self.hamiltonian, [])
        self.grad_shape = (self.N**2, self.N**2)
        
        self.spreQ = spre(coupling).full()
        self.spostQ = spost(coupling).full()
        self.norm_plus, self.norm_minus = self._calc_renorm_factors()

    def prev_next(self, n, k):
        """
        Calculate the next and previous heirarchy index for the
        current index `n`.
        """            
        current_he_n = copy(self.idx2he[n])
        current_he_p = copy(self.idx2he[n])

        nnext = add_at_idx(current_he_n, k, +1)
        nprev = add_at_idx(current_he_p, k, -1)
        
        prev_next = []

        if nprev in self.he2idx:
            prev_next.append(self.he2idx[nprev])
        else:
            prev_next.append(np.nan)

        if nnext in self.he2idx:
            prev_next.append(self.he2idx[nnext])
        else:
            prev_next.append(np.nan)
        return prev_next
        
    def deltak(self):
        """
        Calculates the deltak values for those Matsubara terms which are
        greater than the cutoff set for the exponentials.
        """
        # Needs some test or check here
        if self.kcut >= len(self.vk):
            return 0
        else:
            dk = np.sum(np.divide(self.ck[self.kcut:], self.vk[self.kcut:]))
            return dk
    
    def grad_n(self, t, rho, n):
        """
        Get the gradient term for the Hierarchy ADM at
        level n
        """
        Q = self.coupling
        c = self.ck
        nu = self.vk
        nk = self.idx2he[n]
        L = self.L

        kcut = self.kcut        
        gradient = L*rho[n]
        gradient += -np.sum(np.multiply(nk, nu))*rho[n]
        return gradient
    
    def grad_prev_next(self, t, rho, n):
        """
        """
        Q = self.coupling
        c = self.ck
        nu = self.vk
        he_n = self.idx2he[n]
        spreQ = self.spreQ
        spostQ = self.spostQ

        g = np.zeros_like(rho[n], dtype=np.complex)

        for k in range(self.kcut):
            nprev, nnext = self.prev_next(n, k)
            if ~np.isnan(nprev):
                rho_prev = rho[nprev]
                nk = he_n[k]
                norm_prev = nk
                if self.renorm:
                    norm_prev = self.norm_minus[nk, k]
                op1 = -1j*norm_prev*(c[k]*spreQ - np.conj(c[k])*spostQ)
                t1 = np.dot(op1, rho_prev)
                g += t1

            if ~np.isnan(nnext):
                nk = he_n[k]
                rho_next = rho[nnext]
                norm_next = 1.
                if self.renorm:
                    norm_next = self.norm_plus[nk, k]                  
                op2 = -1j*norm_next*(spreQ - spostQ)
                t2 = np.dot(op2, rho_next)
                g += t2

        return g
    
    def grad(self, t, rho):
        """
        Calculate the gradient operator of the full Hierarchy
        """
        gradn = np.zeros((self.nhe, self.N**2), dtype=np.complex)
        state = rho.reshape((self.nhe, self.N**2)).copy()

        for n in self.idx2he:
            if n not in self.filtered_nhe:
                g_current = self.grad_n(t, state, n) + self.grad_prev_next(t, state, n)
                gradn[n] = g_current
        return gradn.ravel()
    
    def solve(self, rho0, tlist, options=None, rcut=0.):
        """
        Solve the Hierarchy equations of motion for the given initial
        density matrix and time.
        """
        self.rcut = rcut
        if options is None:
            options = Options()

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []
        output.states.append(Qobj(rho0))

        dt = np.diff(tlist)
        rho_he = np.zeros((self.nhe, self.N**2), dtype=np.complex)
        rho_he[0] = rho0.full().ravel("F")
        rho_he = rho_he.flatten()
        r = ode(self.grad)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)
                        
        r.set_initial_value(rho_he, tlist[0])
        
        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        
        if progress_bar:
            bar = progress(total = n_tsteps-1)
        for t_idx, t in enumerate(tlist):

            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                r1 = r.y.copy().reshape((self.nhe, self.N**2))
                r0 = r1[0].reshape(self.N, self.N).T
                output.states.append(Qobj(r0))
                filter_idx = np.argwhere(np.abs(r1.max(1) \
                                         <= rcut)).flatten()
                self.pop_he(filter_idx)
            
            if progress_bar: bar.update()
        return output
    
    def pop_he(self, nlist):
        """
        Pop the given list of hierarchy index
        """
        for n in nlist:
            if n in self.filtered_nhe:
                self.filtered_nhe.remove(n)

    def _calc_renorm_factors(self):
        """
        Calculate the renormalisation factors

        Returns
        -------
        norm_plus, norm_minus : array[N_c, N_m] of float
        """
        c = self.ck
        N_m = self.kcut
        N_c = self.ncut

        norm_plus = np.empty((N_c+1, N_m))
        norm_minus = np.empty((N_c+1, N_m))
        for kk in range(N_m):
            for nn in range(N_c+1):
                norm_plus[nn, kk] = np.sqrt(abs(c[kk])*(nn + 1))
                norm_minus[nn, kk] = np.sqrt(float(nn)/abs(c[kk]))

        return norm_plus, norm_minus

def add_at_idx(tup, k, val):
    """
    Add (subtract) a value in the tuple at position k
    """
    lst = list(tup)
    lst[k] += val
    return tuple(lst)
