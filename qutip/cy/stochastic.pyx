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
cimport numpy as np
cimport cython
cimport libc.math
from qutip.cy.td_qobj_cy cimport cy_qobj
from qutip.qobj import Qobj
from qutip.superoperator import vec2mat
include "parameters.pxi"
include "complex_math.pxi"
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.linalg.cython_blas cimport zaxpy, zdotu, zdotc, zcopy, zdscal, zscal
from scipy.linalg.cython_blas cimport dznrm2 as raw_dznrm2

cdef int ZERO=0
cdef double DZERO=0
cdef complex ZZERO=0j
cdef int ONE=1

cpdef void axpy(complex a, complex[::1] x, complex[::1] y):
    cdef int l = x.shape[0]
    zaxpy(&l, &a, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

cpdef void copy(complex[::1] x, complex[::1] y):
    cdef int l = x.shape[0]
    zcopy(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

cpdef complex dot(complex[::1] x,complex[::1] y):
    cdef int l = x.shape[0]
    return zdotu(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

cpdef complex dotc(complex[::1] x,complex[::1] y):
    cdef int l = x.shape[0]
    return zdotc(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

cpdef double dznrm2(complex[::1] vec):
    cdef int l = vec.shape[0]
    return raw_dznrm2(&l, <complex*>&vec[0], &ONE)

@cython.cdivision(True)
cpdef void normalize_inplace(complex[::1] vec):
    cdef int l = vec.shape[0]
    cdef double norm = 1.0/dznrm2(vec)
    zdscal(&l, &norm, <complex*>&vec[0], &ONE)

cdef void scale(double a, complex[::1] x):
    cdef int l = x.shape[0]
    zdscal(&l, &a, <complex*>&x[0], &ONE)

cdef void zscale(complex a, complex[::1] x):
    cdef int l = x.shape[0]
    zscal(&l, &a, <complex*>&x[0], &ONE)

cdef void zero(complex[::1] x):
    cdef int l = x.shape[0]
    zdscal(&l, &DZERO, <complex*>&x[0], &ONE)

cdef void zero_2d(complex[:,::1] x):
    cdef int l = x.shape[0]*x.shape[1]
    zdscal(&l, &DZERO, <complex*>&x[0,0], &ONE)

cdef void zero_3d(complex[:,:,::1] x):
    cdef int l = x.shape[0]*x.shape[1]*x.shape[2]
    zdscal(&l, &DZERO, <complex*>&x[0,0,0], &ONE)

cdef void zero_4d(complex[:,:,:,::1] x):
    cdef int l = x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
    zdscal(&l, &DZERO, <complex*>&x[0,0,0,0], &ONE)

"""
Solver:
  order 0.5
    euler-maruyama     50
  order 0.5? 1.0?
    pred_corr         101
    pred_corr(2)      104
  order 1.0
    platen            100
    milstein          102
    milstein-imp      103
  order 1.5
    platen1.5         150
    taylor1.5         152
    taylor1.5-imp     153
"""

cdef class ssolvers:
    cdef int l_vec, N_ops
    cdef int solver#, noise_type
    cdef int N_step, N_substeps, N_dw
    cdef int normalize
    cdef double dt
    cdef int noise_type
    cdef object custom_noise
    cdef double[::1] dW_factor
    cdef unsigned int[::1] seed

    # buffer to not redo the initialisation at each substep
    cdef complex[:, ::1] buffer_1d
    cdef complex[:, :, ::1] buffer_2d
    cdef complex[:, :, :, ::1] buffer_3d
    cdef complex[:, :, :, ::1] buffer_4d
    cdef complex[:, ::1] expect_buffer_1d
    cdef complex[:, ::1] expect_buffer_2d
    cdef complex[:, :, ::1] expect_buffer_3d
    cdef complex[:, ::1] func_buffer_1d
    cdef complex[:, ::1] func_buffer_2d
    cdef complex[:, :, ::1] func_buffer_3d

    def __init__(self):
        self.l_vec = 0
        self.N_ops = 0
        self.solver = 0

    def set_solver(self, sso):
        self.solver = sso.solver_code
        self.dt = sso.dt
        self.N_substeps = sso.nsubsteps
        self.normalize = sso.normalize and not sso.me
        self.N_step = len(sso.times)
        self.N_dw = len(sso.sops)
        if self.solver in [150, 152, 153]:
            self.N_dw *= 2
        # prepare buffers for the solvers
        nb_solver = [0,0,0,0]
        nb_func = [0,0,0]
        nb_expect = [0,0,0]

        if self.solver == 50:
            nb_solver = [0,1,0,0]
        elif self.solver == 60:
            nb_solver = [0,1,0,0]
            nb_func = [1,0,0]
        elif self.solver == 100:
            nb_solver = [2,5,0,0]
        elif self.solver == 101:
            nb_solver = [4,1,1,0]
        elif self.solver == 102:
            nb_solver = [0,1,1,0]
        elif self.solver == 103:
            nb_solver = [1,1,1,0]
        elif self.solver == 104:
            nb_solver = [5,1,1,0]
        elif self.solver == 150:
            nb_solver = [5,8,3,0]
        elif self.solver == 152:
            nb_solver = [2,3,1,1]
        elif self.solver == 153:
            nb_solver = [2,3,1,1]

        if self.solver in [101,102,103,104,152,153]:
          if sso.me:
            nb_func = [1,0,0]
            nb_expect = [1,1,0]
          else:
            nb_func = [2,1,1]
            nb_expect = [2,1,1]
        else:
          if not sso.me:
            nb_func = [1,0,0]

        self.buffer_1d = np.zeros((nb_solver[0],self.l_vec),dtype=complex)
        self.buffer_2d = np.zeros((nb_solver[1],self.N_ops,self.l_vec),dtype=complex)
        self.buffer_3d = np.zeros((nb_solver[2],self.N_ops,self.N_ops,self.l_vec),
                                        dtype=complex)
        if nb_solver[3]:
          self.buffer_4d = np.zeros((self.N_ops,self.N_ops,self.N_ops,self.l_vec),
                                        dtype=complex)

        self.expect_buffer_1d = np.zeros((nb_expect[0],self.N_ops),dtype=complex)
        if nb_expect[1]:
          self.expect_buffer_2d = np.zeros((self.N_ops,self.N_ops),dtype=complex)
        if nb_expect[2]:
          self.expect_buffer_3d = np.zeros((self.N_ops,self.N_ops,self.N_ops),dtype=complex)

        self.func_buffer_1d = np.zeros((nb_func[0],self.l_vec),dtype=complex)
        if nb_func[1]:
          self.func_buffer_2d = np.zeros((self.N_ops,self.l_vec),dtype=complex)
        if nb_func[2]:
          self.func_buffer_3d = np.zeros((self.N_ops,self.N_ops,self.l_vec),dtype=complex)

        self.noise_type = sso.noise_type
        self.dW_factor = np.array(sso.dW_factors,dtype=np.float64)
        if self.noise_type == 1:
            self.custom_noise = sso.noise
        elif self.noise_type == 0:
            self.seed = sso.noise

    cdef np.ndarray[double, ndim=3] make_noise(self, int n):
        if self.solver == 60 and self.noise_type == 0:
            # photocurrent, just seed,
            np.random.seed(self.seed[n])
            return np.empty((self.N_step, self.N_substeps, self.N_dw))
        if self.noise_type == 0:
            np.random.seed(self.seed[n])
            return np.random.randn(self.N_step, self.N_substeps, self.N_dw) *\
                                   np.sqrt(self.dt)
        elif self.noise_type == 1:
            return self.custom_noise[n,:,:,:]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cy_sesolve_single_trajectory(self, int n, sso):
        cdef double[::1] times = sso.times
        cdef complex[::1] rho_t
        cdef double t
        cdef int m_idx, t_idx, e_idx
        cdef np.ndarray[double, ndim=3] noise = self.make_noise(n)
        cdef int tlast = times.shape[0]

        rho_t = sso.rho0.copy()
        dims = sso.state0.dims

        expect = np.zeros((len(sso.ce_ops), len(sso.times)), dtype=complex)
        ss = np.zeros((len(sso.ce_ops), len(sso.times)), dtype=complex)
        measurements = np.zeros((len(times), len(sso.cm_ops)), dtype=complex)
        states_list = []
        for t_idx, t in enumerate(times):
            if sso.ce_ops:
                for e_idx, e in enumerate(sso.ce_ops):
                    s = e.compiled_Qobj.expect(t, rho_t, 0)
                    expect[e_idx, t_idx] = s
                    ss[e_idx, t_idx] = s ** 2
            if sso.store_states or not sso.ce_ops:
                if sso.me:
                    states_list.append(Qobj(vec2mat(np.asarray(rho_t)),
                        dims=dims))
                else:
                    states_list.append(Qobj(np.asarray(rho_t), dims=dims))

            if sso.store_measurement:
                for m_idx, m in enumerate(sso.cm_ops):
                    m_expt = m.compiled_Qobj.expect(t, rho_t, 0)
                    measurements[t_idx, m_idx] = m_expt + self.dW_factor[m_idx] * \
                        sum(noise[t_idx, :, m_idx]) / (self.dt * self.N_substeps)
            if t_idx != tlast-1:
                rho_t = self.run(t, self.dt, noise[t_idx, :, :],
                                 rho_t, self.N_substeps)

        if sso.method == 'heterodyne':
            measurements = measurements.reshape(len(times),len(sso.cm_ops)//2,2)

        return states_list, noise, measurements, expect, ss

    cdef complex[::1] run(self, double t, double dt, double[:, ::1] noise,
                          complex[::1] vec, int N_substeps):
        cdef complex[::1] out = np.zeros(self.l_vec, dtype=complex)
        cdef int i
        if self.solver == 50:
            for i in range(N_substeps):
                self.euler(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 60:
            for i in range(N_substeps):
                self.photocurrent(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 100:
            for i in range(N_substeps):
                self.platen(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 101:
            for i in range(N_substeps):
                self.pred_corr(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 102:
            for i in range(N_substeps):
                self.milstein(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 103:
            for i in range(N_substeps):
                self.milstein_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 104:
            for i in range(N_substeps):
                self.pred_corr_a(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 150:
            for i in range(N_substeps):
                self.platen15(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 152:
            for i in range(N_substeps):
                self.taylor15(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver == 153:
            for i in range(N_substeps):
                self.taylor15_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        if self.normalize:
            normalize_inplace(vec)
        return vec

    # Dummy functions
    cdef void d1(self, double t, complex[::1] v, complex[::1] out):
        pass

    cdef void d2(self, double t, complex[::1] v, complex[:, ::1] out):
        pass

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                       complex[::1] out, np.ndarray[complex, ndim=1] guess):
        pass

    cdef void derivatives(self, double t, int deg, complex[::1] rho,
                              complex[::1] a, complex[:, ::1] b,
                              complex[:, :, ::1] Lb, complex[:,::1] La,
                              complex[:, ::1] L0b, complex[:, :, :, ::1] LLb,
                              complex[::1] L0a):
        pass

    def set_implicit(self, sso):
        pass

    cdef void photocurrent(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        pass


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void euler(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        cdef int i, j
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        zero_2d(d2)
        copy(vec, out)
        self.d1(t, vec, out)
        self.d2(t, vec, d2)
        for i in range(self.N_ops):
            axpy(noise[i], d2[i,:], out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void platen(self, double t, double dt, double[:] noise,
                       complex[::1] vec, complex[::1] out):
        """
        Platen rhs function for both master eq and schrodinger eq.
        dV = -iH* (V+Vt)/2 * dt + (d1(V)+d1(Vt))/2 * dt
             + (2*d2_i(V)+d2_i(V+)+d2_i(V-))/4 * dW_i
             + (d2_i(V+)-d2_i(V-))/4 * (dW_i**2 -dt) * dt**(-.5)

        Vt = V -iH*V*dt + d1*dt + d2_i*dW_i
        V+/- = V -iH*V*dt + d1*dt +/- d2_i*dt**.5
        """
        cdef int i, j, k
        cdef double sqrt_dt = np.sqrt(dt)
        cdef double sqrt_dt_inv = 0.25/sqrt_dt
        cdef double dw, dw2
        cdef complex[::1] d1 = self.buffer_1d[0,:]
        cdef complex[::1] Vt = self.buffer_1d[1,:]
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        cdef complex[:, ::1] Vm = self.buffer_2d[1,:,:]
        cdef complex[:, ::1] Vp = self.buffer_2d[2,:,:]
        cdef complex[:, ::1] d2p = self.buffer_2d[3,:,:]
        cdef complex[:, ::1] d2m = self.buffer_2d[4,:,:]
        zero(d1)
        zero_2d(d2)

        self.d1(t, vec, d1)
        self.d2(t, vec, d2)
        axpy(1.0,vec,d1)
        copy(d1,Vt)
        copy(d1,out)
        scale(0.5,out)
        for i in range(self.N_ops):
            copy(d1,Vp[i,:])
            copy(d1,Vm[i,:])
            axpy( sqrt_dt,d2[i,:],Vp[i,:])
            axpy(-sqrt_dt,d2[i,:],Vm[i,:])
            axpy(noise[i],d2[i,:],Vt)
        zero(d1)
        self.d1(t, Vt, d1)
        axpy(0.5,d1,out)
        axpy(0.5,vec,out)
        for i in range(self.N_ops):
            zero_2d(d2p)
            zero_2d(d2m)
            self.d2(t, Vp[i,:], d2p)
            self.d2(t, Vm[i,:], d2m)
            dw = noise[i] * 0.25
            axpy(dw,d2m[i,:],out)
            axpy(2*dw,d2[i,:],out)
            axpy(dw,d2p[i,:],out)
            for j in range(self.N_ops):
                if i == j:
                    dw2 = sqrt_dt_inv * (noise[i]*noise[i] - dt)
                else:
                    dw2 = sqrt_dt_inv * noise[i] * noise[j]
                axpy(dw2,d2p[j,:],out)
                axpy(-dw2,d2m[j,:],out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void pred_corr(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """
        Chapter 15.5 Eq. (5.4)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        # a=0. b=0.5
        cdef double dt_2 = dt*0.5
        cdef complex[::1] euler = self.buffer_1d[0,:]
        cdef complex[::1] a_pred = self.buffer_1d[1,:]
        cdef complex[::1] b_pred = self.buffer_1d[2,:]
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        cdef complex[:, :, ::1] dd2 = self.buffer_3d[0,:,:,:]
        zero(a_pred)
        zero(b_pred)
        zero_2d(d2)
        zero_3d(dd2)
        self.derivatives(t, 1, vec, a_pred, d2, dd2, None, None, None, None)
        copy(vec, euler)
        copy(vec, out)
        axpy(1.0, a_pred, euler)
        for i in range(self.N_ops):
            axpy(noise[i], d2[i,:], b_pred)
            axpy(-dt_2, dd2[i,i,:], a_pred)
        axpy(1.0, a_pred, out)
        axpy(1.0, b_pred, euler)
        axpy(0.5, b_pred, out)
        zero_2d(d2)
        self.d2(t + dt, euler, d2)
        for i in range(self.N_ops):
            axpy(noise[i]*0.5, d2[i,:], out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void pred_corr_a(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """
        Chapter 15.5 Eq. (5.4)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        # a=0.5, b=0.5
        cdef int i, j, k
        cdef complex[::1] euler = self.buffer_1d[0,:]
        cdef complex[::1] a_pred = self.buffer_1d[1,:]
        zero(a_pred)
        cdef complex[::1] a_corr = self.buffer_1d[2,:]
        zero(a_corr)
        cdef complex[::1] b_pred = self.buffer_1d[3,:]
        zero(b_pred)
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        zero_2d(d2)
        cdef complex[:, :, ::1] dd2 = self.buffer_3d[0,:,:,:]
        zero_3d(dd2)

        cdef double dt_2 = dt*0.5
        self.derivatives(t, 1, vec, a_pred, d2, dd2, None, None, None, None)
        copy(vec, euler)
        axpy(1.0, a_pred, euler)
        for i in range(self.N_ops):
            axpy(noise[i], d2[i,:], b_pred)
            axpy(-dt_2, dd2[i,i,:], a_pred)
        axpy(1.0, b_pred, euler)
        copy(vec, out)
        axpy(0.5, a_pred, out)
        axpy(0.5, b_pred, out)
        zero_2d(d2)
        zero_3d(dd2)
        self.derivatives(t, 1, euler, a_corr, d2, dd2, None, None, None, None)
        for i in range(self.N_ops):
            axpy(noise[i]*0.5, d2[i,:], out)
            axpy(-dt_2, dd2[i,i,:], a_corr)
        axpy(0.5, a_corr, out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void milstein(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """
        Chapter 10.3 Eq. (3.1)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen

        dV = -iH*V*dt + d1*dt + d2_i*dW_i
             + 0.5*d2_i' d2_j*(dW_i*dw_j -dt*delta_ij)
        """
        cdef int i, j, k
        cdef double dw
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        cdef complex[:, :, ::1] dd2 = self.buffer_3d[0,:,:,:]
        zero_2d(d2)
        zero_3d(dd2)
        copy(vec,out)
        self.derivatives(t, 1, vec, out, d2, dd2, None, None, None, None)
        for i in range(self.N_ops):
            axpy(noise[i],d2[i,:],out)
        for i in range(self.N_ops):
            for j in range(i, self.N_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j])
                axpy(dw,dd2[i,j,:],out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void milstein_imp(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        """
        Chapter 12.2 Eq. (2.9)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef int i, j, k
        cdef double dw
        cdef np.ndarray[complex, ndim=1] guess = np.zeros((self.l_vec, ),
                                                          dtype=complex)
        cdef np.ndarray[complex, ndim=1] dvec = np.zeros((self.l_vec, ),
                                                         dtype=complex)
        cdef complex[::1] a = self.buffer_1d[0,:]
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        cdef complex[:, :, ::1] dd2 = self.buffer_3d[0,:,:,:]
        zero(a)
        zero_2d(d2)
        zero_3d(dd2)
        self.derivatives(t, 1, vec, a, d2, dd2, None, None, None, None)
        copy(vec, dvec)
        axpy(0.5, a, dvec)
        for i in range(self.N_ops):
            axpy(noise[i], d2[i,:], dvec)
        for i in range(self.N_ops):
            for j in range(i, self.N_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j])
                axpy(dw, dd2[i,j,:], dvec)
        copy(dvec, guess)
        axpy(0.5, a, guess)
        self.implicit(t+dt, dvec, out, guess)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void taylor15(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        """
        Chapter 12.2 Eq. (2.18),
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef complex[::1] a = self.buffer_1d[0, :]
        cdef complex[:, ::1] b = self.buffer_2d[0, :, :]
        cdef complex[:, :, ::1] Lb = self.buffer_3d[0, :, :, :]
        cdef complex[:, ::1] L0b = self.buffer_2d[1,:,:]
        cdef complex[:, ::1] La = self.buffer_2d[2,:,:]
        cdef complex[:, :, :, ::1] LLb = self.buffer_4d[:, :, :, :]
        cdef complex[::1] L0a = self.buffer_1d[1, :]
        zero(a)
        zero_2d(b)
        zero_3d(Lb)
        zero_2d(L0b)
        zero_2d(La)
        zero_4d(LLb)
        zero(L0a)
        self.derivatives(t, 2, vec, a, b, Lb, La, L0b, LLb, L0a)

        cdef int i,j,k
        cdef double[::1] dz, dw
        dw = np.empty(self.N_ops)
        dz = np.empty(self.N_ops)
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        for i in range(self.N_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.N_ops])
        copy(vec,out)
        axpy(1.0, a, out)
        axpy(0.5, L0a, out)

        for i in range(self.N_ops):
            axpy(dw[i], b[i,:], out)
            axpy(0.5*(dw[i]*dw[i]-dt), Lb[i,i,:], out)
            axpy(dz[i], La[i,:], out)
            axpy(dw[i]-dz[i], L0b[i,:], out)
            axpy(0.5 * (0.3333333333333333 * dw[i] * dw[i] - dt) * dw[i],
                        LLb[i,i,i,:], out)
            for j in range(i+1,self.N_ops):
                axpy((dw[i]*dw[j]), Lb[i,j,:], out)
                axpy(0.5*(dw[j]*dw[j]-dt)*dw[i], LLb[i,j,j,:], out)
                axpy(0.5*(dw[i]*dw[i]-dt)*dw[j], LLb[i,i,j,:], out)
                for k in range(j+1,self.N_ops):
                    axpy(dw[i]*dw[j]*dw[k], LLb[i,j,k,:], out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void taylor15_imp(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        """
        Chapter 12.2 Eq. (2.18),
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef complex[::1] a = self.buffer_1d[0, :]
        cdef complex[:, ::1] b = self.buffer_2d[0, :, :]
        cdef complex[:, :, ::1] Lb = self.buffer_3d[0, :, :, :]
        cdef complex[:, ::1] L0b = self.buffer_2d[1,:,:]
        cdef complex[:, ::1] La = self.buffer_2d[2,:,:]
        cdef complex[:, :, :, ::1] LLb = self.buffer_4d[:, :, :, :]
        cdef complex[::1] L0a = self.buffer_1d[1, :]
        zero(a)
        zero_2d(b)
        zero_3d(Lb)
        zero_2d(L0b)
        zero_2d(La)
        zero_4d(LLb)
        zero(L0a)
        cdef np.ndarray[complex, ndim=1] guess = np.zeros((self.l_vec, ),
                   dtype=complex)
        cdef np.ndarray[complex, ndim=1] vec_t = np.zeros((self.l_vec, ),
                  dtype=complex)
        self.derivatives(t, 3, vec, a, b, Lb, La, L0b, LLb, L0a)

        cdef int i,j,k
        cdef double[::1] dz, dw
        dw = np.empty(self.N_ops)
        dz = np.empty(self.N_ops)
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        for i in range(self.N_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.N_ops])
        copy(vec, vec_t)
        axpy(0.5, a, vec_t)
        for i in range(self.N_ops):
            axpy(dw[i], b[i,:], vec_t)
            axpy(0.5*(dw[i]*dw[i]-dt), Lb[i,i,:], vec_t)
            axpy(dz[i]-dw[i]*0.5, La[i,:], vec_t)
            axpy(dw[i]-dz[i] , L0b[i,:], vec_t)
            axpy(0.5 * (0.3333333333333333 * dw[i] * dw[i] - dt) * dw[i],
                        LLb[i,i,i,:], vec_t)
            for j in range(i+1,self.N_ops):
                axpy((dw[i]*dw[j]), Lb[i,j,:], vec_t)
                axpy(0.5*(dw[j]*dw[j]-dt)*dw[i], LLb[i,j,j,:], vec_t)
                axpy(0.5*(dw[i]*dw[i]-dt)*dw[j], LLb[i,i,j,:], vec_t)
                for k in range(j+1,self.N_ops):
                    axpy(dw[i]*dw[j]*dw[k], LLb[i,j,k,:], vec_t)
        copy(vec_t, guess)
        axpy(0.5, a, guess)

        self.implicit(t+dt, vec_t, out, guess)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void platen15(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """
        Chapter 11.2 Eq. (2.13)
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef int i, j, k
        cdef double sqrt_dt = np.sqrt(dt)
        cdef double sqrt_dt_inv = 1./sqrt_dt
        cdef double ddz, ddw, ddd
        cdef double[::1] dz, dw
        dw = np.empty(self.N_ops)
        dz = np.empty(self.N_ops)
        for i in range(self.N_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.N_ops])

        cdef complex[::1] d1 = self.buffer_1d[0,:]
        cdef complex[::1] d1p = self.buffer_1d[1,:]
        cdef complex[::1] d1m = self.buffer_1d[2,:]
        cdef complex[::1] V = self.buffer_1d[3,:]
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        cdef complex[:, ::1] dd2 = self.buffer_2d[1,:,:]
        cdef complex[:, ::1] d2p = self.buffer_2d[2,:,:]
        cdef complex[:, ::1] d2m = self.buffer_2d[3,:,:]
        cdef complex[:, ::1] d2pp = self.buffer_2d[4,:,:]
        cdef complex[:, ::1] d2mm = self.buffer_2d[5,:,:]
        cdef complex[:, ::1] v2p = self.buffer_2d[6,:,:]
        cdef complex[:, ::1] v2m = self.buffer_2d[7,:,:]
        cdef complex[:, :, ::1] p2p = self.buffer_3d[0,:,:,:]
        cdef complex[:, : ,::1] p2m = self.buffer_3d[1,:,:,:]

        zero(d1)
        zero_2d(d2)
        zero_2d(dd2)
        self.d1(t, vec, d1)
        self.d2(t, vec, d2)
        self.d2(t + dt, vec, dd2)
        # Euler part
        copy(vec,out)
        axpy(1., d1, out)
        for i in range(self.N_ops):
            axpy(dw[i], d2[i,:], out)

        zero(V)
        axpy(1., vec, V)
        axpy(1./self.N_ops, d1, V)

        zero_2d(v2p)
        zero_2d(v2m)
        for i in range(self.N_ops):
            axpy(1., V, v2p[i,:])
            axpy(sqrt_dt, d2[i,:], v2p[i,:])
            axpy(1., V, v2m[i,:])
            axpy(-sqrt_dt, d2[i,:], v2m[i,:])

        zero_3d(p2p)
        zero_3d(p2m)
        for i in range(self.N_ops):
            zero_2d(d2p)
            zero_2d(d2m)
            self.d2(t, v2p[i,:], d2p)
            self.d2(t, v2m[i,:], d2m)
            ddw = (dw[i]*dw[i]-dt)*0.25/sqrt_dt  # 1.0
            axpy( ddw, d2p[i,:], out)
            axpy(-ddw, d2m[i,:], out)
            for j in range(self.N_ops):
                axpy(      1., v2p[i,:], p2p[i,j,:])
                axpy( sqrt_dt, d2p[j,:], p2p[i,j,:])
                axpy(      1., v2p[i,:], p2m[i,j,:])
                axpy(-sqrt_dt, d2p[j,:], p2m[i,j,:])

        axpy(-0.5*(self.N_ops), d1, out)
        for i in range(self.N_ops):
            ddz = dz[i]*0.5/sqrt_dt  # 1.5
            ddd = 0.25*(dw[i]*dw[i]/3-dt)*dw[i]/dt  # 1.5
            zero(d1p)
            zero(d1m)
            zero_2d(d2m)
            zero_2d(d2p)
            zero_2d(d2pp)
            zero_2d(d2mm)
            self.d1(t + dt/self.N_ops, v2p[i,:], d1p)
            self.d1(t + dt/self.N_ops, v2m[i,:], d1m)
            self.d2(t, v2p[i,:], d2p)
            self.d2(t, v2m[i,:], d2m)
            self.d2(t, p2p[i,i,:], d2pp)
            self.d2(t, p2m[i,i,:], d2mm)

            axpy( ddz+0.25, d1p, out)
            axpy(-ddz+0.25, d1m, out)

            axpy((dw[i]-dz[i]), dd2[i,:], out)
            axpy((dz[i]-dw[i]), d2[i,:], out)

            axpy( ddd, d2pp[i,:], out)
            axpy(-ddd, d2mm[i,:], out)
            axpy(-ddd, d2p[i,:], out)
            axpy( ddd, d2m[i,:], out)

            for j in range(self.N_ops):
              ddw = 0.5*(dw[j]-dz[j]) # 1.5
              axpy(ddw, d2p[j,:], out)
              axpy(-2*ddw, d2[j,:], out)
              axpy(ddw, d2m[j,:], out)

              if j>i:
                ddw = 0.5*(dw[i]*dw[j])/sqrt_dt # 1.0
                axpy( ddw, d2p[j,:], out)
                axpy(-ddw, d2m[j,:], out)

                ddw = 0.25*(dw[j]*dw[j]-dt)*dw[i]/dt # 1.5
                zero_2d(d2pp)
                zero_2d(d2mm)
                self.d2(t, p2p[j,i,:], d2pp)
                self.d2(t, p2m[j,i,:], d2mm)
                axpy( ddw, d2pp[j,:], out)
                axpy(-ddw, d2mm[j,:], out)
                axpy(-ddw, d2p[j,:], out)
                axpy( ddw, d2m[j,:], out)

                for k in range(j+1,self.N_ops):
                    ddw = 0.5*dw[i]*dw[j]*dw[k]/dt # 1.5
                    axpy( ddw, d2pp[k,:], out)
                    axpy(-ddw, d2mm[k,:], out)
                    axpy(-ddw, d2p[k,:], out)
                    axpy( ddw, d2m[k,:], out)

              if j<i:
                ddw = 0.25*(dw[j]*dw[j]-dt)*dw[i]/dt # 1.5
                zero_2d(d2pp)
                zero_2d(d2mm)
                self.d2(t, p2p[j,i,:], d2pp)
                self.d2(t, p2m[j,i,:], d2mm)
                axpy( ddw, d2pp[j,:], out)
                axpy(-ddw, d2mm[j,:], out)
                axpy(-ddw, d2p[j,:], out)
                axpy( ddw, d2m[j,:], out)

    def checks(self, double t, double dt, complex[::1] vec):
        cdef complex[::1] a = np.zeros((self.l_vec), dtype=complex)
        cdef complex[::1] a1 = np.zeros((self.l_vec), dtype=complex)
        cdef complex[:, ::1] b = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, ::1] b1 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, ::1] b2 = np.zeros((self.N_ops, self.l_vec),
                                           dtype=complex)
        cdef complex[:, :, ::1] Lb = np.zeros((self.N_ops, self.N_ops,
                    self.l_vec), dtype=complex)
        cdef complex[:, :, ::1] Lb2 = np.zeros((self.N_ops, self.N_ops,
                                               self.l_vec), dtype=complex)
        cdef complex[:, ::1] L0b = np.zeros((self.N_ops,
                                             self.l_vec), dtype=complex)
        cdef complex[:,::1] La = np.zeros((self.N_ops,
                                           self.l_vec), dtype=complex)
        cdef complex[:, :, :, ::1] LLb = np.zeros((self.N_ops, self.N_ops,
                                        self.N_ops, self.l_vec), dtype=complex)
        cdef complex[::1] L0a = np.zeros((self.l_vec), dtype=complex)

        cdef complex[:, ::1] dvec = np.zeros((7, self.l_vec), dtype=complex)

        self.d1(t, vec, a)
        self.d2(t, vec, b)
        self.d2d2p(t, vec, b1, Lb)
        self.derivatives(t, 2, vec, a1, b2, Lb2, La, L0b, LLb, L0a)
        self.taylor_15_1(t, vec, dvec)

        print("a[0]",a[0],a1[0],dvec[0,0])
        print("a[1]",a[1],a1[1],dvec[0,1])
        for i in range(self.N_ops):
            print(i)
            print("b[0]",b[i,0],b1[i,0],b2[i,0],dvec[1,0])
            print("b[1]",b[i,1],b1[i,1],b2[i,1],dvec[1,1])
        for i in range(self.N_ops):
          for j in range(self.N_ops):
            print(i,j)
            print("Lb[0]",Lb[i,j,0],Lb[i,j,0],dvec[2,0])
            print("Lb[1]",Lb[i,j,1],Lb[i,j,1],dvec[2,1])
        for i in range(self.N_ops):
            print(i)
            print("La[0]",La[i,0],dvec[3,0])
            print("La[1]",La[i,1],dvec[3,1])
            print("L0b[0]",L0b[i,0],dvec[4,0])
            print("L0b[1]",L0b[i,1],dvec[4,1])
        for i in range(self.N_ops):
          for j in range(self.N_ops):
            for k in range(self.N_ops):
              print(i,j,k)
              print("LLb[0]",LLb[i,j,k,0],dvec[6,0])
              print("LLb[1]",LLb[i,j,k,1],dvec[6,1])
        print("L0a[0]",L0a[0],dvec[5,0])
        print("L0a[1]",L0a[1],dvec[5,1])


cdef class sse(ssolvers):
    cdef cy_qobj L
    cdef object c_ops
    cdef object cpcd_ops
    cdef object imp
    cdef double tol, imp_t

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.N_ops = len(c_ops)
        self.L = L.compiled_Qobj
        self.c_ops = []
        self.cpcd_ops = []
        for i, op in enumerate(c_ops):
            self.c_ops.append(op[0].compiled_Qobj)
            self.cpcd_ops.append(op[1].compiled_Qobj)
        if sso.solver_code in [103, 153]:
            self.tol = sso.tol
            self.imp = LinearOperator( (self.l_vec,self.l_vec),
                                      matvec=self.implicit_op, dtype=complex)

    def implicit_op(self, vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.l_vec, dtype=complex)
        self.d1(self.imp_t, vec, out)
        cdef int i
        scale(-0.5,out)
        axpy(1.,vec,out)
        return out


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d1(self, double t, complex[::1] vec, complex[::1] out):
        self.L._rhs_mat(t, &vec[0], &out[0])
        cdef int i
        cdef complex e
        cdef cy_qobj c_op
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        zero(temp)
        for i in range(self.N_ops):
            c_op = self.cpcd_ops[i]
            e = c_op._expect_mat(t, &vec[0], 0)
            zero(temp)
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &temp[0])
            axpy(-0.125 * e * e * self.dt, vec, out)
            axpy(0.5 * e * self.dt, temp, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] vec, complex[:, ::1] out):
        cdef int i, k
        cdef cy_qobj c_op
        cdef complex expect
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &out[i,0])
            c_op = self.cpcd_ops[i]
            expect = c_op._expect_mat(t, &vec[0], 0)
            axpy(-0.5*expect,vec,out[i,:])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void derivatives(self, double t, int deg, complex[::1] vec,
                          complex[::1] a, complex[:, ::1] b,
                          complex[:, :, ::1] Lb, complex[:,::1] La,
                          complex[:, ::1] L0b, complex[:, :, :, ::1] LLb,
                          complex[::1] L0a):
        """
        combinaisons of a and b derivatives for taylor order 1.5
        dY ~ a dt + bi dwi
                                         deg   use        noise
        b[i.:]       bi                   >=0   d2 euler   dwi
        a[:]         a                    >=0   d1 euler   dt
        Lb[i,i,:]    bi'bi                >=1   milstein   (dwi^2-dt)/2
        Lb[i,j,:]    bj'bi                >=1   milstein   dwi*dwj

        L0b[i,:]     ab' +db/dt +bbb"/2   >=2   taylor15   dwidt-dzi
        La[i,:]      ba'                  >=2   taylor15   dzi
        LLb[i,i,i,:] bi(bibi"+bi'bi')     >=2   taylor15   (dwi^2/3-dt)dwi/2
        LLb[i,j,j,:] bi(bjbj"+bj'bj')     >=2   taylor15   (dwj^2-dt)dwj/2
        LLb[i,j,k,:] bi(bjbk"+bj'bk')     >=2   taylor15   dwi*dwj*dwk
        L0a[:]       aa' +da/dt +bba"/2    2    taylor15   dt^2/2
        """
        cdef int i, j, k, l
        cdef double dt = self.dt
        cdef cy_qobj c_op
        cdef complex e, de_bb

        cdef complex[::1] e_real = self.expect_buffer_1d[0,:]
        cdef complex[:, ::1] de_b = self.expect_buffer_2d[:,:]
        cdef complex[::1] de_a = self.expect_buffer_1d[1,:]
        cdef complex[:, :, ::1] dde_bb = self.expect_buffer_3d[:,:,:]
        zero_3d(dde_bb)
        cdef complex[:, ::1] Cvec = self.func_buffer_2d[:,:]
        cdef complex[:, :, ::1] Cb = self.func_buffer_3d[:,:,:]
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        cdef complex[::1] temp2 = self.func_buffer_1d[1,:]
        zero(temp)
        zero(temp2)
        zero_2d(Cvec)
        zero_3d(Cb)

        # a b
        self.L._rhs_mat(t, &vec[0], &a[0])
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &Cvec[i,0])
            e = dotc(vec,Cvec[i,:])
            e_real[i] = real(e)
            axpy(1., Cvec[i,:], b[i,:])
            axpy(-e_real[i], vec, b[i,:])
            axpy(-0.5 * e_real[i] * e_real[i] * dt, vec, a[:])
            axpy(e_real[i] * dt, Cvec[i,:], a[:])

        #Lb bb'
        if deg >= 1:
          for i in range(self.N_ops):
            c_op = self.c_ops[i]
            for j in range(self.N_ops):
              c_op._rhs_mat(t, &b[j,0], &Cb[i,j,0])
              for k in range(self.l_vec):
                  temp[k] = conj(b[j,k])
                  temp2[k] = 0.
              c_op._rhs_mat(t, &temp[0], &temp2[0])
              de_b[i,j] = (dotc(vec, Cb[i,j,:]) + dot(b[j,:], Cvec[i,:]) + \
                          conj(dotc(b[j,:], Cvec[i,:]) + dotc(vec, temp2))) * 0.5
              axpy(1., Cb[i,j,:], Lb[i,j,:])
              axpy(-e_real[i], b[j,:], Lb[i,j,:])
              axpy(-de_b[i,j], vec, Lb[i,j,:])

              for k in range(self.N_ops):
                dde_bb[i,j,k] += (dot(b[j,:], Cb[i,k,:]) + \
                                  dot(b[k,:], Cb[i,j,:]) + \
                                  conj(dotc(b[k,:], temp2)))*.5
                dde_bb[i,k,j] += conj(dotc(b[k,:], temp2))*.5

        #L0b La LLb
        if deg >= 2:
          for i in range(self.N_ops):
              #ba'
              self.L._rhs_mat(t, &b[i,0], &La[i,0])
              for j in range(self.N_ops):
                  axpy(-0.5 * e_real[j] * e_real[j] * dt, b[i,:], La[i,:])
                  axpy(-e_real[j] * de_b[i,j] * dt, vec, La[i,:])
                  axpy(e_real[j] * dt, Cb[i,j,:], La[i,:])
                  axpy(de_b[i,j] * dt, Cvec[i,:], La[i,:])

              #ab' + db/dt + bbb"/2
              c_op = self.c_ops[i]
              c_op._rhs_mat(t, &a[0], &L0b[i,0])
              for k in range(self.l_vec):
                  temp[k] = conj(a[k])
                  temp2[k] = 0.
              c_op._rhs_mat(t, &temp[0], &temp2[0])
              de_a[i] = (dotc(vec, L0b[i,:]) + dot(a, Cvec[i,:]) + \
                        conj(dotc(a, Cvec[i,:]) + dotc(vec, temp2))) * 0.5
              axpy(-e_real[i], a, L0b[i,:])
              axpy(-de_a[i], vec, L0b[i,:])

              temp = np.zeros(self.l_vec, dtype=complex)
              c_op._rhs_mat(t + self.dt, &vec[0], &temp[0])
              e = dotc(vec,temp)
              axpy(1., temp, L0b[i,:])
              axpy(-real(e), vec, L0b[i,:])
              axpy(-1., b[i,:], L0b[i,:])

              for j in range(self.N_ops):
                  axpy(-de_b[i,j]*dt, b[j,:], L0b[i,:])
                  axpy(-dde_bb[i,j,j]*dt, vec, L0b[i,:])

              #b(bb"+b'b')
              for j in range(i,self.N_ops):
                  for k in range(j, self.N_ops):
                      c_op._rhs_mat(t, &Lb[j,k,0], &LLb[i,j,k,0])
                      for l in range(self.l_vec):
                          temp[l] = conj(Lb[j,k,l])
                          temp2[l] = 0.
                      c_op._rhs_mat(t, &temp[0], &temp2[0])
                      de_bb = (dotc(vec, LLb[i,j,k,:]) + \
                               dot(Lb[j,k,:], Cvec[i,:]) + \
                               conj(dotc(Lb[j,k,:], Cvec[i,:]) +\
                               dotc(vec, temp2)))*0.5
                      axpy(-e_real[i], Lb[j,k,:], LLb[i,j,k,:])
                      axpy(-de_bb, vec, LLb[i,j,k,:])
                      axpy(-dde_bb[i,j,k], vec, LLb[i,j,k,:])
                      axpy(-de_b[i,j], b[k,:], LLb[i,j,k,:])
                      axpy(-de_b[i,k], b[j,:], LLb[i,j,k,:])

        #da/dt + aa' + bba"
        if deg == 2:
          self.d1(t + dt, vec, L0a)
          axpy(-1.0, a, L0a)
          self.L._rhs_mat(t, &a[0], &L0a[0])
          for j in range(self.N_ops):
              c_op = self.c_ops[j]
              temp = np.zeros(self.l_vec, dtype=complex)
              c_op._rhs_mat(t, &a[0], &temp[0])
              axpy(-0.5 * e_real[j] * e_real[j] * dt, a[:], L0a[:])
              axpy(-e_real[j] * de_a[j] * dt, vec, L0a[:])
              axpy(e_real[j] * dt, temp, L0a[:])
              axpy(de_a[j] * dt, Cvec[j,:], L0a[:])
              for i in range(self.N_ops):
                  axpy(-0.5*(e_real[i] * dde_bb[i,j,j] +
                          de_b[i,j] * de_b[i,j]) * dt * dt, vec, L0a[:])
                  axpy(-e_real[i] * de_b[i,j] * dt * dt, b[j,:], L0a[:])
                  axpy(0.5*dde_bb[i,j,j] * dt * dt, Cvec[i,:], L0a[:])
                  axpy(de_b[i,j] * dt * dt, Cb[i,j,:], L0a[:])

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                        complex[::1] out,
                                        np.ndarray[complex, ndim=1] guess):
        # np.ndarray to memoryview is OK but not the reverse
        # scipy function only take np array, not memoryview
        self.imp_t = t
        spout, check = sp.linalg.bicgstab(self.imp,
                                        dvec, x0 = guess, tol=self.tol)
        cdef int i
        copy(spout, out)


cdef class sme(ssolvers):
    cdef cy_qobj L
    cdef object imp
    cdef object c_ops
    cdef int N_root
    cdef double tol

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.N_ops = len(c_ops)
        self.L = L.compiled_Qobj
        self.c_ops = []
        self.N_root = np.sqrt(self.l_vec)
        for i, op in enumerate(c_ops):
            self.c_ops.append(op.compiled_Qobj)
        if sso.solver_code in [103, 153]:
            self.tol = sso.tol
            self.imp = sso.imp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex expect(self, complex[::1] rho):
        cdef complex e = 0.
        cdef int k
        for k in range(self.N_root):
            e += rho[k*(self.N_root+1)]
        return e

    cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
        self.L._rhs_mat(t, &rho[0], &out[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
        cdef int i, k
        cdef cy_qobj c_op
        cdef complex expect
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &rho[0], &out[i,0])
            expect = self.expect(out[i,:])
            axpy(-expect, rho, out[i,:])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void derivatives(self, double t, int deg, complex[::1] rho,
                          complex[::1] a, complex[:, ::1] b,
                          complex[:, :, ::1] Lb, complex[:,::1] La,
                          complex[:, ::1] L0b, complex[:, :, :, ::1] LLb,
                          complex[::1] L0a):
        """
        combinaisons of a and b derivative for m sc_ops up to order dt**1.5
        dY ~ a dt + bi dwi
                                         deg   use        noise
        b[i.:]       bi                   >=0   d2 euler   dwi
        a[:]         a                    >=0   d1 euler   dt
        Lb[i,i,:]    bi'bi                >=1   milstein   (dwi^2-dt)/2
        Lb[i,j,:]    bj'bi                >=1   milstein   dwi*dwj
        L0b[i,:]     ab' +db/dt +bbb"/2   >=2   taylor15   dwidt-dzi
        La[i,:]      ba'                  >=2   taylor15   dzi
        LLb[i,i,i,:] bi(bibi"+bi'bi')     >=2   taylor15   (dwi^2/3-dt)dwi/2
        LLb[i,j,j,:] bi(bjbj"+bj'bj')     >=2   taylor15   (dwj^2-dt)dwj/2
        LLb[i,j,k,:] bi(bjbk"+bj'bk')     >=2   taylor15   dwi*dwj*dwk
        L0a[:]       aa' +da/dt +bba"/2    2    taylor15   dt^2/2
        """
        cdef int i, j, k
        cdef cy_qobj c_op
        cdef cy_qobj c_opj
        cdef complex trApp, trAbb, trAa
        cdef complex[::1] trAp = self.expect_buffer_1d[0,:]
        cdef complex[:, ::1] trAb = self.expect_buffer_2d
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        zero(temp)

        # a
        self.L._rhs_mat(t, &rho[0], &a[0])

        # b
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            # bi
            c_op._rhs_mat(t, &rho[0], &b[i,0])
            trAp[i] = self.expect(b[i,:])
            axpy(-trAp[i], rho, b[i,:])

        # Libj = bibj', i<=j
        # sc_ops must commute (Libj = Ljbi)
        if deg >= 1:
          for i in range(self.N_ops):
            c_op = self.c_ops[i]
            for j in range(i, self.N_ops):
                c_op._rhs_mat(t, &b[j,0], &Lb[i,j,0])
                trAb[i,j] = self.expect(Lb[i,j,:])
                axpy(-trAp[j], b[i,:], Lb[i,j,:])
                axpy(-trAb[i,j], rho, Lb[i,j,:])

        # L0b La LLb
        if deg >= 2:
          for i in range(self.N_ops):
            c_op = self.c_ops[i]
            # Lia = bia'
            self.L._rhs_mat(t, &b[i,0], &La[i,0])

            # L0bi = abi' + dbi/dt + Sum_j bjbjbi"/2
            # db/dt
            c_op._rhs_mat(t + self.dt, &rho[0], &L0b[i,0])
            trApp = self.expect(L0b[i,:])
            axpy(-trApp, rho, L0b[i,:])
            axpy(-1, b[i,:], L0b[i,:])
            # ab'
            temp = np.zeros((self.l_vec, ), dtype=complex)
            c_op._rhs_mat(t, &a[0], &temp[0])
            trAa = self.expect(temp)
            axpy(1., temp, L0b[i,:])
            axpy(-trAp[i], a[:], L0b[i,:])
            axpy(-trAa, rho, L0b[i,:])
            # bbb" : trAb[i,j] only defined for j>=i
            for j in range(i):
                axpy(-trAb[j,i]*self.dt, b[j,:], L0b[i,:])  # L contain dt
            for j in range(i,self.N_ops):
                axpy(-trAb[i,j]*self.dt, b[j,:], L0b[i,:])  # L contain dt

            # LLb
            # LiLjbk = bi(bj'bk'+bjbk"), i<=j<=k
            # sc_ops must commute (LiLjbk = LjLibk = LkLjbi)
            for j in range(i,self.N_ops):
              for k in range(j,self.N_ops):
                c_op._rhs_mat(t, &Lb[j,k,0], &LLb[i,j,k,0])
                trAbb = self.expect(LLb[i,j,k,:])
                axpy(-trAp[i], Lb[j,k,:], LLb[i,j,k,:])
                axpy(-trAbb, rho, LLb[i,j,k,:])
                axpy(-trAb[i,k], b[j,:], LLb[i,j,k,:])
                axpy(-trAb[i,j], b[k,:], LLb[i,j,k,:])

        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        if deg == 2:
            self.L._rhs_mat(t, &a[0], &L0a[0])
            self.L._rhs_mat(t+self.dt, &rho[0], &L0a[0])
            axpy(-1, a, L0a)

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                        complex[::1] out,
                                        np.ndarray[complex, ndim=1] guess):
        # np.ndarray to memoryview is OK but not the reverse
        # scipy function only take np array, not memoryview
        spout, check = sp.linalg.bicgstab(self.imp(t, data=1),
                                        dvec, x0 = guess, tol=self.tol)
        cdef int i
        copy(spout,out)

cdef class psse(ssolvers):
    cdef cy_qobj L
    cdef object c_ops
    cdef object cdc_ops

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.N_ops = len(c_ops)
        self.L = L.compiled_Qobj
        self.c_ops = []
        self.cdc_ops = []
        for i, op in enumerate(c_ops):
            self.c_ops.append(op[0].compiled_Qobj)
            self.cdc_ops.append(op[0].compiled_Qobj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void photocurrent(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef cy_qobj
        cdef double expect
        cdef int i
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        zero_2d(d2)
        copy(vec,out)
        self.d1(t, vec, out)
        self.d2(t, vec, d2)
        for i in range(self.N_ops):
            c_op = self.cdc_ops[i]
            expect = c_op.expect(t, vec, 1).real * dt
            if expect > 0:
              noise[i] = np.random.poisson(expect, 1)[0]
            else:
              noise[i] = 0.
            axpy(noise[i], d2[i,:], out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d1(self, double t, complex[::1] vec, complex[::1] out):
        self.L._rhs_mat(t, &vec[0], &out[0])
        cdef int i
        cdef complex e
        cdef cy_qobj c_op
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        for i in range(self.N_ops):
            zero(temp)
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &temp[0])
            e = dznrm2(temp)
            axpy(-e * e * self.dt, vec, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] vec, complex[:, ::1] out):
        cdef int i
        cdef cy_qobj c_op
        cdef complex expect
        for i in range(self.N_ops):
            c_op = self.c_ops[i]
            c_op._rhs_mat(t, &vec[0], &out[i,0])
            expect = dznrm2(out[i,:])
            if expect.real >= 1e-15:
                zscale(1/expect, out[i,:])
            else:
                zero(out[i,:])
            axpy(-1, vec, out[i,:])

cdef class psme(ssolvers):
    cdef cy_qobj L
    cdef object cdcr_cdcl_ops
    cdef object cdcl_ops
    cdef object clcdr_ops
    cdef int N_root

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.N_ops = len(c_ops)
        self.L = L.compiled_Qobj
        self.cdcr_cdcl_ops = []
        self.cdcl_ops = []
        self.clcdr_ops = []
        self.N_root = np.sqrt(self.l_vec)
        for i, op in enumerate(c_ops):
            self.cdcr_cdcl_ops.append(op[0].compiled_Qobj)
            self.cdcl_ops.append(op[1].compiled_Qobj)
            self.clcdr_ops.append(op[2].compiled_Qobj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void photocurrent(self, double t, double dt,  double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef double expect
        cdef int i
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        zero_2d(d2)
        copy(vec,out)
        self.d1(t, vec, out)
        self.d2(t, vec, d2)
        for i in range(self.N_ops):
            c_op = self.cdcl_ops[i]
            expect = c_op.expect(t, vec, 1).real * dt
            if expect > 0:
              noise[i] = np.random.poisson(expect, 1)[0]
            else:
              noise[i] = 0.
            axpy(noise[i], d2[i,:], out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex expect(self, complex[::1] rho):
        cdef complex e = 0.
        cdef int k
        for k in range(self.N_root):
            e += rho[k*(self.N_root+1)]
        return e

    cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
        cdef int i
        cdef cy_qobj c_op
        cdef complex[::1] Crho = self.func_buffer_1d[0,:]
        cdef complex expect
        self.L._rhs_mat(t, &rho[0], &out[0])
        for i in range(self.N_ops):
            c_op = self.cdcr_cdcl_ops[i]
            zero(Crho)
            c_op._rhs_mat(t, &rho[0], &Crho[0])
            expect = self.expect(Crho)
            axpy(0.5*expect* self.dt, rho, out)
            axpy(-0.5* self.dt, Crho, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
        cdef int i
        cdef cy_qobj c_op
        cdef complex expect
        for i in range(self.N_ops):
            c_op = self.clcdr_ops[i]
            c_op._rhs_mat(t, &rho[0], &out[i,0])
            expect = self.expect(out[i,:])
            if expect.real >= 1e-15:
                zscale((1.+0j)/expect, out[i,:])
            else:
                zero(out[i,:])
            axpy(-1, rho, out[i,:])

cdef class generic(ssolvers):
    cdef object d1_func, d2_func

    def set_data(self, sso):
        self.l_vec = sso.rho0.shape[0]
        self.N_ops = len(sso.sops)
        self.d1_func = sso.d1
        self.d2_func = sso.d2

    cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
        cdef np.ndarray[complex, ndim=1] in_np
        cdef np.ndarray[complex, ndim=1] out_np
        in_np = np.zeros((self.l_vec, ), dtype=complex)
        copy(rho, in_np)
        out_np = self.d1_func(t, in_np)
        axpy(self.dt, out_np, out) # d1 is += and * dt

    cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
        cdef np.ndarray[complex, ndim=1] in_np
        cdef np.ndarray[complex, ndim=2] out_np
        cdef int i
        in_np = np.zeros((self.l_vec, ), dtype=complex)
        copy(rho, in_np)
        out_np = self.d2_func(t, in_np)
        for i in range(self.N_ops):
            copy(out_np[i,:], out[i,:])
