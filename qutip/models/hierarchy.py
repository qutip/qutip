import numpy as np

from scipy.misc import factorial
from scipy.integrate import ode
from scipy.constants import h as planck
from qutip.cy.spmatfuncs import cy_ode_rhs
from scipy.sparse import lil_matrix

from qutip import commutator, Options
from qutip.solver import Result
from qutip import Qobj
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from copy import copy


def add_at_idx(seq, k, val):
    """
    Add (subtract) a value in the tuple at position k
    """
    lst = list(seq)
    lst[k] += val
    return tuple(lst)

def prevhe(current_he, k, ncut):
    """
    Calculate the previous heirarchy index
    for the current index `n`.
    """
    nprev = add_at_idx(current_he, k, -1)
    if nprev[k] < 0:
        return False
    return nprev

def nexthe(current_he, k, ncut):
    """
    Calculate the next heirarchy index
    for the current index `n`.
    """
    nnext = add_at_idx(current_he, k, 1)
    if sum(nnext) > ncut:
        return False
    return nnext

def num_hierarchy(kcut, ncut):
    """
    Get the total number of auxiliary density matrices in the
    Hierarchy
    """
    return int(factorial(ncut + kcut)/(factorial(ncut)*factorial(kcut)))

class Heom2(object):
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

        he2idx, idx2he, nhe = self._initialize_he()
        self.nhe = nhe
        self.he2idx = he2idx
        self.idx2he = idx2he
        self.N = self.hamiltonian.shape[0]
        
        total_nhe = int(factorial(self.ncut + self.kcut)/(factorial(self.ncut)*factorial(kcut)))
        self.total_nhe = total_nhe
        self.hshape = (total_nhe, self.N**2)
        self.weak_coupling = self.deltak()
        self.L = liouvillian(self.hamiltonian, []).full()
        self.grad_shape = (self.N**2, self.N**2)
        self.spreQ = spre(coupling).full()
        self.spostQ = spost(coupling).full()
        self.norm_plus, self.norm_minus = None, None
        self.L_helems = lil_matrix((total_nhe*self.N**2, total_nhe*self.N**2), dtype=np.complex)

    def _initialize_he(self):
        """
        Initialize the hierarchy indices
        """
        zeroth = tuple([0 for i in range(self.kcut)])
        he2idx = {zeroth:0}
        idx2he = {0:zeroth}
        nhe = 1
        return he2idx, idx2he, nhe

    def populate(self, heidx_list):
        """
        Given a Hierarchy index list, populate the graph of next and
        previous elements
        """
        ncut = self.ncut
        kcut = self.kcut
        he2idx = self.he2idx
        idx2he = self.idx2he
        for heidx in heidx_list:
            for k in range(self.kcut):
                he_current = idx2he[heidx]
                he_next = nexthe(he_current, k, ncut)
                he_prev = prevhe(he_current, k, ncut)
                if he_next and (he_next not in he2idx):
                    he2idx[he_next] = self.nhe
                    idx2he[self.nhe] = he_next
                    self.nhe += 1

                if he_prev and (he_prev not in he2idx):
                    he2idx[he_prev] = self.nhe
                    idx2he[self.nhe] = he_prev
                    self.nhe += 1

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
    
    def grad_n(self, he_n):
        """
        Get the gradient term for the Hierarchy ADM at
        level n
        """
        c = self.ck
        nu = self.vk
        L = self.L.copy()
        gradient_sum = -np.sum(np.multiply(he_n, nu))
        sum_op = gradient_sum*np.eye(L.shape[0])
        L += sum_op

        # Fill in larger L
        nidx = self.he2idx[he_n]
        block = self.N**2
        pos = int(nidx*(block))
        self.L_helems[pos:pos+block, pos:pos+block] = L

    def grad_prev(self, he_n, k, prev_he):
        """
        Get prev gradient
        """
        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ
        nk = he_n[k]
        norm_prev = nk
        # Real part
        if self.renorm:
            if ~np.iscomplex(c[k]):
                norm_prev = np.sqrt(float(nk)/abs(c[k]))
            else:
                norm_prev = 0.
        op1_real = -1j*norm_prev*np.real(c[k])*(spreQ - spostQ)
        
        # imaginary part
        if self.renorm:
            if np.iscomplex(c[k]):
                norm_prev = np.sqrt(float(nk)/abs(np.imag(c[k])))
            else:
                norm_prev = 0.
        op1_imag = norm_prev*np.imag(c[k])*(spreQ + spostQ)
        op1 = op1_real + op1_imag

        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[prev_he]
        block = self.N**2
        rowpos = int(rowidx*(block))
        colpos = int(colidx*(block))
        self.L_helems[rowpos:rowpos+block, colpos:colpos+block] = op1

    def grad_next(self, he_n, k, next_he):
        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ
        
        nk = he_n[k]
        norm_next = 1.
        if self.renorm:
#             norm_next = self.norm_plus[nk, k]    
            norm_next = np.sqrt(abs(c[k])*(nk + 1))

        op2 = -1j*norm_next*(spreQ - spostQ)
        
        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[next_he]
        block = self.N**2
        rowpos = int(rowidx*(block))
        colpos = int(colidx*(block))
        self.L_helems[rowpos:rowpos+block, colpos:colpos+block] = op2
    
    def rhs(self, progress=None):
        """
        Make the RHS
        """
        while self.nhe < self.total_nhe:
            heidxlist = copy(list(self.idx2he.keys()))
            self.populate(heidxlist)
        self.norm_plus, self.norm_minus = self._calc_renorm_factors()
        if progress != None:
            bar = progress(total = self.nhe*self.kcut)

        for n in self.idx2he:
            he_n = self.idx2he[n]
            self.grad_n(he_n)
            for k in range(self.kcut):
                next_he = nexthe(he_n, k, self.ncut)
                prev_he = prevhe(he_n, k, self.ncut)
                if next_he and (next_he in self.he2idx):
                    self.grad_next(he_n, k, next_he)
                if prev_he and (prev_he in self.he2idx):
                    self.grad_prev(he_n, k, prev_he)
                if progress: bar.update()
    
    def solve(self, rho0, tlist, options=None, progress=None):
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

        dt = np.diff(tlist)
        rho_he = np.zeros(self.hshape, dtype=np.complex)
        rho_he[0] = rho0.full().ravel("F")
        rho_he = rho_he.flatten()

        self.rhs()
        L_helems = self.L_helems.asformat("csr")
        r = ode(cy_ode_rhs)
        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)
                        
        r.set_initial_value(rho_he, tlist[0])
        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        if progress:
            bar = progress(total=n_tsteps-1)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                r1 = r.y.reshape(self.hshape)
                r0 = r1[0].reshape(self.N, self.N).T
                output.states.append(Qobj(r0))
                if progress: bar.update()
        return output

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
