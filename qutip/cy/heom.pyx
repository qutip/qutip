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
import numpy as np

from scipy.misc import factorial
from scipy.integrate import ode
from scipy.constants import h as planck

from qutip import enr_state_dictionaries, commutator
from qutip import Options
from qutip.solver import Result
from qutip import Qobj, commutator
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from copy import copy

cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_pad_csr(object A, int row_scale, int col_scale, int insertrow=0, int insertcol=0):
    cdef int nrowin = A.shape[0]
    cdef int ncolin = A.shape[1]
    cdef int nnz = A.indptr[nrowin]
    cdef int nrowout = nrowin*row_scale
    cdef int ncolout = ncolin*col_scale
    cdef size_t kk
    cdef int temp, temp2
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr_in = A.indptr
    cdef cnp.ndarray[int, ndim=1, mode='c'] ptr_out = np.zeros(nrowout+1,dtype=np.int32)
    
    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        temp = insertcol*ncolin
        for kk in range(nnz):
            ind[kk] += temp
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")
        
    
    if insertrow == 0:
        temp = ptr_in[nrowin]
        for kk in range(nrowin):
            ptr_out[kk] = ptr_in[kk]
        for kk in range(nrowin, nrowout+1):
            ptr_out[kk] = temp

    elif insertrow == row_scale-1:
        temp = (row_scale - 1) * nrowin
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = ptr_in[kk-temp]
    
    elif insertrow > 0 and insertrow < row_scale - 1:
        temp = insertrow*nrowin
        for kk in range(temp, temp+nrowin):
            ptr_out[kk] = ptr_in[kk-temp]
        temp = kk+1
        temp2 = ptr_in[nrowin]
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = temp2     
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    A.indptr = ptr_out
    
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_pad_csr(object A, int row_scale, int col_scale, int insertrow=0, int insertcol=0):
    cdef int nrowin = A.shape[0]
    cdef int ncolin = A.shape[1]
    cdef int nnz = A.indptr[nrowin]
    cdef int nrowout = nrowin*row_scale
    cdef int ncolout = ncolin*col_scale
    cdef size_t kk
    cdef int temp, temp2
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr_in = A.indptr
    cdef cnp.ndarray[int, ndim=1, mode='c'] ptr_out = np.zeros(nrowout+1,dtype=np.int32)
    
    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        temp = insertcol*ncolin
        for kk in range(nnz):
            ind[kk] += temp
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")
        
    
    if insertrow == 0:
        temp = ptr_in[nrowin]
        for kk in range(nrowin):
            ptr_out[kk] = ptr_in[kk]
        for kk in range(nrowin, nrowout+1):
            ptr_out[kk] = temp

    elif insertrow == row_scale-1:
        temp = (row_scale - 1) * nrowin
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = ptr_in[kk-temp]
    
    elif insertrow > 0 and insertrow < row_scale - 1:
        temp = insertrow*nrowin
        for kk in range(temp, temp+nrowin):
            ptr_out[kk] = ptr_in[kk-temp]
        temp = kk+1
        temp2 = ptr_in[nrowin]
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = temp2     
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    A.indptr = ptr_out
    
    return A


%%cython -a -f --compile-args=-DCYTHON_TRACE=1


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
# cython: profile=True
# cython: linetrace=True

import numpy as np

from scipy.misc import factorial
from scipy.integrate import ode
from scipy.constants import h as planck

from qutip import enr_state_dictionaries, commutator
from qutip import Options
from qutip.solver import Result
from qutip import Qobj, commutator
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from copy import copy

cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_pad_csr(object A, int row_scale, int col_scale, int insertrow=0, int insertcol=0):
    cdef int nrowin = A.shape[0]
    cdef int ncolin = A.shape[1]
    cdef int nnz = A.indptr[nrowin]
    cdef int nrowout = nrowin*row_scale
    cdef int ncolout = ncolin*col_scale
    cdef size_t kk
    cdef int temp, temp2
    cdef int[::1] ind = A.indices
    cdef int[::1] ptr_in = A.indptr
    cdef cnp.ndarray[int, ndim=1, mode='c'] ptr_out = np.zeros(nrowout+1,dtype=np.int32)
    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        temp = insertcol*ncolin
        for kk in range(nnz):
            ind[kk] += temp
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")
        
    
    if insertrow == 0:
        temp = ptr_in[nrowin]
        for kk in range(nrowin):
            ptr_out[kk] = ptr_in[kk]
        for kk in range(nrowin, nrowout+1):
            ptr_out[kk] = temp

    elif insertrow == row_scale-1:
        temp = (row_scale - 1) * nrowin
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = ptr_in[kk-temp]
    
    elif insertrow > 0 and insertrow < row_scale - 1:
        temp = insertrow*nrowin
        for kk in range(temp, temp+nrowin):
            ptr_out[kk] = ptr_in[kk-temp]
        temp = kk+1
        temp2 = ptr_in[nrowin]
        for kk in range(temp, nrowout+1):
            ptr_out[kk] = temp2     
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    A.indptr = ptr_out
    
    return A


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple add_at_idx(tuple seq, int k, int val):
    """
    Add (subtract) a value in the tuple at position k
    """
    cdef list lst
    lst = list(seq)
    lst[k] += val
    return tuple(lst)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef prevhe(tuple current_he, int k, int ncut):
    """
    Calculate the previous heirarchy index
    for the current index `n`.
    """
    cdef tuple nprev
    nprev = add_at_idx(current_he, k, -1)
    if nprev[k] < 0:
        return False
    return nprev


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef nexthe(tuple current_he, int k, int ncut):
    """
    Calculate the next heirarchy index
    for the current index `n`.
    """
    cdef tuple nnext
    nnext = add_at_idx(current_he, k, 1)
    if sum(nnext) > ncut:
        return False
    return nnext


@cython.boundscheck(False)
@cython.wraparound(False)
def num_hierarchy(kcut, ncut):
    """
    Get the total number of auxiliary density matrices in the
    Hierarchy
    """
    return int(factorial(ncut + kcut)/(factorial(ncut)*factorial(kcut)))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Heom(object):
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
    cdef object hamiltonian, spreQ, spostQ, norm_plus, norm_minus
    cdef object coupling
    cdef list ck
    cdef list vk 
    cdef int ncut
    cdef int kcut
    cdef float rcut
    cdef int renorm
    cdef int nhe
    cdef dict he2idx, idx2he
    cdef int N
    cdef int total_nhe
    cdef tuple hshape
    cdef object L
    cdef tuple grad_shape
    cdef list filtered
    def __init__(self, object hamiltonian, object coupling, list ck, list vk, 
                 int ncut, int kcut=0, float rcut=0., int renorm=1):
        self.hamiltonian = hamiltonian
        self.coupling = coupling        
        self.ck = ck
        self.vk = vk
        self.ncut = ncut
        self.renorm = renorm

        if kcut:
            self.kcut = kcut
        else:
            self.kcut = len(ck)

        if self.kcut > len(self.vk):
            raise Warning("Truncation of exponential exceeds number of terms")

        self.he2idx, self.idx2he, self.nhe = self._initialize_he()
        self.N = self.hamiltonian.shape[0]
        
        total_nhe = int(factorial(self.ncut + self.kcut)/(factorial(self.ncut)*factorial(kcut)))
        self.hshape = (total_nhe, self.N**2)

        self.L = liouvillian(self.hamiltonian, [])
        self.grad_shape = (self.N**2, self.N**2)
        self.filtered = []
        cdef cnp.ndarray spreQ
        cdef cnp.ndarray spostQ
        self.spreQ = spre(coupling).full()
        self.spostQ = spost(coupling).full()
        self.norm_plus, self.norm_minus = self._calc_renorm_factors()


    cpdef list _initialize_he(self):
        """
        Initialize the hierarchy indices
        """
        cdef tuple zeroth
        cdef dict he2idx, idx2he
        cdef int nhe

        zeroth = tuple([0 for i in range(self.kcut)])
        he2idx = {zeroth:0}
        idx2he = {0:zeroth}
        nhe = 1
        return [he2idx, idx2he, nhe]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef populate(self, list heidx_list):
        """
        Given a Hierarchy index list, populate the graph of next and
        previous elements
        """
        cdef int ncut, kcut, k
        cdef dict he2idx
        cdef dict idx2he

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] grad_n(self, cnp.ndarray rho_n, tuple he_n):
        """
        Get the gradient term for the Hierarchy ADM at
        level n
        """
        cdef list c, nu
        cdef cnp.ndarray gradient
        c = self.ck
        nu = self.vk
        L = self.L
        gradient = L*rho_n
        gradient += -np.sum(np.multiply(he_n, nu))*rho_n
        return gradient

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.complex, ndim=1] grad_prev(self, rho_prev, he_n, k):
        """
        Get prev gradient
        """
        cdef list c, nu
        cdef cnp.ndarray spreQ
        cdef cnp.ndarray spostQ

        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ
        nk = he_n[k]
        norm_prev = nk
        if self.renorm:
            norm_prev = self.norm_minus[nk, k]
        op1 = -1j*norm_prev*(c[k]*spreQ - np.conj(c[k])*spostQ)
        t1 = np.dot(op1, rho_prev)
        return t1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.complex, ndim=1] grad_next(self, rho_next, he_n, k):
        c = self.ck
        nu = self.vk
        spreQ = self.spreQ
        spostQ = self.spostQ
        
        nk = he_n[k]
        norm_next = 1.
        if self.renorm:
            norm_next = self.norm_plus[nk, k]                  
        op2 = -1j*norm_next*(spreQ - spostQ)
        t2 = np.dot(op2, rho_next)
        return t2

    def grad(self, t, rho):
        """
        Calculate the gradient operator of the full Hierarchy
        """
        gradn = np.zeros(self.hshape, dtype=np.complex)
        state = rho.reshape(self.hshape).copy()
        heidxlist = copy(list(self.idx2he.keys()))
        self.populate(heidxlist)
        for n in self.idx2he:
            he_n = copy(self.idx2he[n])
            rho_current = state[n]
            g_current = self.grad_n(rho_current, he_n)
            for k in range(self.kcut):
                next_he = nexthe(he_n, k, self.ncut)
                prev_he = prevhe(he_n, k, self.ncut)
                if next_he and (next_he in self.he2idx):
                    rho_next = state[self.he2idx[next_he]]
                    g_current = g_current + self.grad_next(rho_next, he_n, k)
                if prev_he and (prev_he in self.he2idx):
                    rho_prev = state[self.he2idx[prev_he]]
                    g_current = g_current + self.grad_prev(rho_prev, he_n, k)
            gradn[n] = g_current
        return gradn.ravel()


    @cython.boundscheck(False)
    @cython.wraparound(False)
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
        rho_he = np.zeros(self.hshape, dtype=np.complex)
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
        
        for t_idx, t in enumerate(tlist):

            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                r1 = r.y.copy().reshape(self.hshape)
                r0 = r1[0].reshape(self.N, self.N).T
                output.states.append(Qobj(r0))            
        return output

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _calc_renorm_factors(self):
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

