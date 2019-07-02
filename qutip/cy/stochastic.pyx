#!python
#cython: language_level=3
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
from libc.math cimport fabs
from qutip.cy.cqobjevo cimport CQobjEvo
from qutip.cy.brtools cimport ZHEEVR
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

"""Some of blas wrapper"""
@cython.boundscheck(False)
cdef void _axpy(complex a, complex[::1] x, complex[::1] y):
    """ y += a*x"""
    cdef int l = x.shape[0]
    zaxpy(&l, &a, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

@cython.boundscheck(False)
cdef void copy(complex[::1] x, complex[::1] y):
    """ y = x """
    cdef int l = x.shape[0]
    zcopy(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

@cython.boundscheck(False)
cdef complex _dot(complex[::1] x, complex[::1] y):
    """ = x_i * y_i """
    cdef int l = x.shape[0]
    return zdotu(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

@cython.boundscheck(False)
cdef complex _dotc(complex[::1] x, complex[::1] y):
    """ = conj(x_i) * y_i """
    cdef int l = x.shape[0]
    return zdotc(&l, <complex*>&x[0], &ONE, <complex*>&y[0], &ONE)

@cython.boundscheck(False)
cdef double _dznrm2(complex[::1] vec):
    """ = sqrt( x_i**2 ) """
    cdef int l = vec.shape[0]
    return raw_dznrm2(&l, <complex*>&vec[0], &ONE)

@cython.boundscheck(False)
cdef void _scale(double a, complex[::1] x):
    """ x *= a """
    cdef int l = x.shape[0]
    zdscal(&l, &a, <complex*>&x[0], &ONE)

@cython.boundscheck(False)
cdef void _zscale(complex a, complex[::1] x):
    """ x *= a """
    cdef int l = x.shape[0]
    zscal(&l, &a, <complex*>&x[0], &ONE)

@cython.boundscheck(False)
cdef void _zero(complex[::1] x):
    """ x *= 0 """
    cdef int l = x.shape[0]
    zdscal(&l, &DZERO, <complex*>&x[0], &ONE)

@cython.boundscheck(False)
cdef void _zero_2d(complex[:,::1] x):
    """ x *= 0 """
    cdef int l = x.shape[0]*x.shape[1]
    zdscal(&l, &DZERO, <complex*>&x[0,0], &ONE)

@cython.boundscheck(False)
cdef void _zero_3d(complex[:,:,::1] x):
    """ x *= 0 """
    cdef int l = x.shape[0]*x.shape[1]*x.shape[2]
    zdscal(&l, &DZERO, <complex*>&x[0,0,0], &ONE)

@cython.boundscheck(False)
cdef void _zero_4d(complex[:,:,:,::1] x):
    """ x *= 0 """
    cdef int l = x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]
    zdscal(&l, &DZERO, <complex*>&x[0,0,0,0], &ONE)

# %%%%%%%%%%%%%%%%%%%%%%%%%
# functions for ensuring that the states stay physical
@cython.cdivision(True)
@cython.boundscheck(False)
cdef void _normalize_inplace(complex[::1] vec):
    """ make norm of vec equal to 1"""
    cdef int l = vec.shape[0]
    cdef double norm = 1.0/_dznrm2(vec)
    zdscal(&l, &norm, <complex*>&vec[0], &ONE)

# to move eventually, 10x faster than scipy's norm.
@cython.cdivision(True)
@cython.boundscheck(False)
def normalize_inplace(complex[::1] vec):
    """ make norm of vec equal to 1"""
    cdef int l = vec.shape[0]
    cdef double norm = 1.0/_dznrm2(vec)
    zdscal(&l, &norm, <complex*>&vec[0], &ONE)
    return fabs(norm-1)

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void _normalize_rho(complex[::1] rho):
    """ Ensure that the density matrix trace is one and
    that the composing states are normalized.
    """
    cdef int l = rho.shape[0]
    cdef int N = np.sqrt(l)
    cdef complex[::1,:] mat = np.reshape(rho, (N,N), order="F")
    cdef complex[::1,:] eivec = np.zeros((N,N), dtype=complex, order="F")
    cdef double[::1] eival = np.zeros(N)
    ZHEEVR(mat, &eival[0], eivec, N)
    _zero(rho)
    cdef int i, j, k
    cdef double sum

    sum = 0.
    for i in range(N):
        _normalize_inplace(eivec[:,i])
        if eival[i] < 0:
            eival[i] = 0.
        sum += eival[i]
    if sum != 1.:
        for i in range(N):
            eival[i] /= sum

    for i in range(N):
        for j in range(N):
            for k in range(N):
                rho[j+N*k] += conj(eivec[k,i])*eivec[j,i]*eival[i]

# Available solvers:
cpdef enum Solvers:
  # order 0.5
  EULER_SOLVER           =  50
  # order 0.5 strong, 1.0 weak?
  PC_SOLVER              = 101
  PC_2_SOLVER            = 104
  # order 1.0
  PLATEN_SOLVER          = 100
  MILSTEIN_SOLVER        = 102
  MILSTEIN_IMP_SOLVER    = 103
  # order 1.5
  EXPLICIT1_5_SOLVER     = 150
  TAYLOR1_5_SOLVER       = 152
  TAYLOR1_5_IMP_SOLVER   = 153
  # order 2.0
  TAYLOR2_0_SOLVER       = 202

  # Special solvers
  PHOTOCURRENT_SOLVER    =  60
  PHOTOCURRENT_PC_SOLVER = 110
  ROUCHON_SOLVER         = 120

  # For initialisation
  SOLVER_NOT_SET         =   0


cdef class TaylorNoise:
    """ Object to build the Stratonovich integral for order 2.0 strong taylor.
    Complex enough that I fell it should be kept separated from the main solver.
    """
    cdef:
      int p
      double rho, alpha
      double aFactor, bFactor
      double BFactor, CFactor
      double dt, dt_sqrt

    @cython.cdivision(True)
    def __init__(self, int p, double dt):
        self.p = p
        self.dt = dt
        self.dt_sqrt = dt**.5
        cdef double pi = np.pi
        cdef int i

        cdef double rho = 0.
        for i in range(1,p+1):
            rho += (i+0.)**-2
        rho = 1./3.-2*rho/(pi**2)
        self.rho = (rho)**.5
        self.aFactor = -(2)**.5/pi

        cdef double alpha = 0.
        for i in range(1,p+1):
            alpha += (i+0.)**-4
        alpha = pi/180-alpha/(2*pi**2)/pi
        self.alpha = (alpha)**.5
        self.bFactor = (0.5)**.5/pi**2

        self.BFactor = 1/(4*pi**2)
        self.CFactor = -1/(2*pi**2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void order2(self, double[::1] noise, double[::1] dws):
        cdef int p = self.p
        cdef int r, l
        cdef double s = 1/6.
        cdef double a = 0
        cdef double b = 0
        cdef double AA = 0
        cdef double BB = 0
        cdef double CC = 0

        for r in range(p):
            a += noise[3+r]/(r+1.)
            b += noise[3+r+p]/(r+1.)/(r+1.)
            BB += (1/(r+1.)/(r+1.)) *\
                 (noise[3+r]*noise[3+r]+noise[3+r+p]*noise[3+r+p])
            for l in range(p):
                if r != l:
                    CC += (r+1.)/((r+1.)*(r+1.)-(l+1.)*(l+1.)) *\
                         (1/(l+1.)*noise[3+r]*noise[3+l] - \
                         (l+1.)/(r+1.)*noise[3+r+p]*noise[3+l+p])

        a = self.aFactor * a + self.rho * noise[1]
        b = self.bFactor * b + self.alpha * noise[2]
        AA = 0.25*a*a
        BB *= self.BFactor
        CC *= self.CFactor

        dws[0] = noise[0]                       # dw
        dws[1] = 0.5*(noise[0]+a)               # dz
        dws[2] = noise[0]*(noise[0]*s -0.25*a -0.5*b) +BB +CC  # j011
        dws[3] = noise[0]*(noise[0]*s          +   b) -AA -BB  # j101
        dws[4] = noise[0]*(noise[0]*s +0.25*a -0.5*b) +AA -CC  # j110


cdef class StochasticSolver:
    """ stochastic solver class base
    Does most of the initialisation, drive the simulation and contain the
    stochastic integration algorythm that do not depend on the physics.

    This class is not to be used as is, the function computing the evolution's
    derivative are specified in it's child class which define the deterministic
    and stochastic contributions.

    PYTHON METHODS:
    set_solver:
      Receive the data for the integration.
      Prepare buffers

    cy_sesolve_single_trajectory:
      Run one trajectory.


    INTERNAL METHODS
    make_noise:
      create the stochastic noise
    run:
      evolution between timestep (substep)
    solver's method:
      stochastic integration algorithm
        euler
        milstein
        taylor
        ...

    CHILD:
    SSESolver: stochastic schrodinger evolution
    SMESolver: stochastic master evolution
    PcSSESolver: photocurrent stochastic schrodinger evolution
    PcSMESolver: photocurrent stochastic master evolution
    PmSMESolver: positive map stochastic master evolution
    GenericSSolver: general (user defined) stochastic evolution

    CHILD METHODS:
    set_data:
      Read data about the system
    d1:
      deterministic part
    d2:
      non-deterministic part
    derivatives:
      d1, d2 and their derivatives up to dt**1.5
      multiple sc_ops
    derivativesO2:
      d1, d2 and there derivatives up to dt**2.0
      one sc_ops
    """
    cdef int l_vec, num_ops
    cdef Solvers solver
    cdef int num_step, num_substeps, num_dw
    cdef int normalize
    cdef double dt
    cdef int noise_type
    cdef object custom_noise
    cdef double[::1] dW_factor
    cdef unsigned int[::1] seed
    cdef object sso

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

    cdef TaylorNoise order2noise

    def __init__(self):
        self.l_vec = 0
        self.num_ops = 0
        self.solver = SOLVER_NOT_SET

    def set_solver(self, sso):
        """ Prepare the solver from the info in StochasticSolverOptions

        Parameters
        ----------
        sso : StochasticSolverOptions
            Data of the stochastic system
        """
        self.set_data(sso)
        self.sso = sso

        self.solver = sso.solver_code
        self.dt = sso.dt
        self.num_substeps = sso.nsubsteps
        self.normalize = sso.normalize
        self.num_step = len(sso.times)
        self.num_dw = len(sso.sops)
        if self.solver in [EXPLICIT1_5_SOLVER,
                           TAYLOR1_5_SOLVER,
                           TAYLOR1_5_IMP_SOLVER]:
            self.num_dw *= 2
        if self.solver in [TAYLOR2_0_SOLVER]:
            self.num_dw *= 3 + 2*sso.p
            self.order2noise = TaylorNoise(sso.p, self.dt)
        # prepare buffers for the solvers
        nb_solver = [0,0,0,0]
        nb_func = [0,0,0]
        nb_expect = [0,0,0]

        # %%%%%%%%%%%%%%%%%%%%%%%%%
        # Depending on the solver, determine the numbers of buffers of each
        # shape to prepare. (~30% slower when not preallocating buffer)
        # nb_solver : buffer to contain the states used by solver
        # nb_func : buffer for states used used by d1, d2 and derivative functions
        # nb_expect : buffer to store expectation values.
        if self.solver is EULER_SOLVER:
            nb_solver = [0,1,0,0]
        elif self.solver is PHOTOCURRENT_SOLVER:
            nb_solver = [1,0,0,0]
            nb_func = [1,0,0]
        elif self.solver is PLATEN_SOLVER:
            nb_solver = [2,5,0,0]
        elif self.solver is PC_SOLVER:
            nb_solver = [4,1,1,0]
        elif self.solver is MILSTEIN_SOLVER:
            nb_solver = [0,1,1,0]
        elif self.solver is MILSTEIN_IMP_SOLVER:
            nb_solver = [1,1,1,0]
        elif self.solver is PC_2_SOLVER:
            nb_solver = [5,1,1,0]
        elif self.solver is PHOTOCURRENT_PC_SOLVER:
            nb_solver = [1,1,0,0]
            nb_func = [1,0,0]
        elif self.solver is ROUCHON_SOLVER:
            nb_solver = [2,0,0,0]
        elif self.solver is EXPLICIT1_5_SOLVER:
            nb_solver = [5,8,3,0]
        elif self.solver is TAYLOR1_5_SOLVER:
            nb_solver = [2,3,1,1]
        elif self.solver is TAYLOR1_5_IMP_SOLVER:
            nb_solver = [2,3,1,1]
        elif self.solver is TAYLOR2_0_SOLVER:
            nb_solver = [11,0,0,0]

        if self.solver in [PC_SOLVER, MILSTEIN_SOLVER, MILSTEIN_IMP_SOLVER,
                           PC_2_SOLVER, TAYLOR1_5_SOLVER, TAYLOR1_5_IMP_SOLVER]:
          if sso.me:
            nb_func = [1,0,0]
            nb_expect = [1,1,0]
          else:
            nb_func = [2,1,1]
            nb_expect = [2,1,1]
        elif self.solver in [PHOTOCURRENT_SOLVER, PHOTOCURRENT_PC_SOLVER]:
            nb_expect = [1,0,0]
        elif self.solver is TAYLOR2_0_SOLVER:
          if sso.me:
            nb_func = [2,0,0]
            nb_expect = [2,0,0]
          else:
            nb_func = [14,0,0]
            nb_expect = [0,0,0]
        elif self.solver is ROUCHON_SOLVER:
            nb_expect = [1,0,0]
        else:
          if not sso.me:
            nb_func = [1,0,0]

        self.buffer_1d = np.zeros((nb_solver[0], self.l_vec), dtype=complex)
        self.buffer_2d = np.zeros((nb_solver[1], self.num_ops, self.l_vec),
                                  dtype=complex)
        self.buffer_3d = np.zeros((nb_solver[2], self.num_ops, self.num_ops, self.l_vec),
                                        dtype=complex)
        if nb_solver[3]:
          self.buffer_4d = np.zeros((self.num_ops, self.num_ops, self.num_ops, self.l_vec),
                                        dtype=complex)

        self.expect_buffer_1d = np.zeros((nb_expect[0], self.num_ops), dtype=complex)
        if nb_expect[1]:
          self.expect_buffer_2d = np.zeros((self.num_ops, self.num_ops), dtype=complex)
        if nb_expect[2]:
          self.expect_buffer_3d = np.zeros((self.num_ops, self.num_ops, self.num_ops), dtype=complex)

        self.func_buffer_1d = np.zeros((nb_func[0], self.l_vec), dtype=complex)
        if nb_func[1]:
          self.func_buffer_2d = np.zeros((self.num_ops, self.l_vec), dtype=complex)
        if nb_func[2]:
          self.func_buffer_3d = np.zeros((self.num_ops, self.num_ops, self.l_vec), dtype=complex)

        self.noise_type = sso.noise_type
        self.dW_factor = np.array(sso.dW_factors, dtype=np.float64)
        if self.noise_type == 1:
            self.custom_noise = sso.noise
        elif self.noise_type == 0:
            self.seed = sso.noise

    def set_data(self, sso):
        """Set solver specific operator"""
        pass

    cdef np.ndarray[double, ndim=3] make_noise(self, int n):
        """Create the random numbers for the stochastic process"""
        if self.solver in [PHOTOCURRENT_SOLVER, PHOTOCURRENT_PC_SOLVER] and self.noise_type == 0:
            # photocurrent, just seed,
            np.random.seed(self.seed[n])
            return np.zeros((self.num_step, self.num_substeps, self.num_dw))
        if self.noise_type == 0:
            np.random.seed(self.seed[n])
            return np.random.randn(self.num_step, self.num_substeps, self.num_dw) *\
                                   np.sqrt(self.dt)
        elif self.noise_type == 1:
            return self.custom_noise[n,:,:,:]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def cy_sesolve_single_trajectory(self, int n):
        """ Run the one of the trajectories of the stochastic system.

        Parameters
        ----------
        n : int
            Number of the iterations

        sso : StochasticSolverOptions
            Data of the stochastic system

        Returns
        -------
        states_list : list of qobj
            State of the system at each time

        noise : array
            noise at each step of the solver

        measurements : array
            measurements value at each timestep for each m_ops

        expect : array
            expectation value at each timestep for each e_ops

        """
        sso = self.sso
        cdef double[::1] times = sso.times
        cdef complex[::1] rho_t
        cdef double t
        cdef int m_idx, t_idx, e_idx
        cdef np.ndarray[double, ndim=3] noise = self.make_noise(n)
        cdef int tlast = times.shape[0]

        rho_t = sso.rho0.copy()
        dims = sso.state0.dims

        expect = np.zeros((len(sso.ce_ops), len(sso.times)), dtype=complex)
        measurements = np.zeros((len(times), len(sso.cm_ops)), dtype=complex)
        states_list = []
        for t_idx, t in enumerate(times):
            if sso.ce_ops:
                for e_idx, e in enumerate(sso.ce_ops):
                    s = e.compiled_qobjevo.expect(t, rho_t)
                    expect[e_idx, t_idx] = s
            if sso.store_states or not sso.ce_ops:
                if sso.me:
                    states_list.append(Qobj(vec2mat(np.asarray(rho_t)),
                        dims=dims))
                else:
                    states_list.append(Qobj(np.asarray(rho_t), dims=dims))

            if t_idx != tlast-1:
                rho_t = self.run(t, self.dt, noise[t_idx, :, :],
                                 rho_t, self.num_substeps)

            if sso.store_measurement:
                for m_idx, m in enumerate(sso.cm_ops):
                    m_expt = m.compiled_qobjevo.expect(t, rho_t)
                    measurements[t_idx, m_idx] = m_expt + self.dW_factor[m_idx] * \
                        sum(noise[t_idx, :, m_idx]) / (self.dt * self.num_substeps)

        if sso.method == 'heterodyne':
            measurements = measurements.reshape(len(times), len(sso.cm_ops)//2, 2)

        return states_list, noise, measurements, expect

    @cython.boundscheck(False)
    cdef complex[::1] run(self, double t, double dt, double[:, ::1] noise,
                          complex[::1] vec, int num_substeps):
        """ Do one time full step"""
        cdef complex[::1] out = np.zeros(self.l_vec, dtype=complex)
        cdef int i
        if self.solver is EULER_SOLVER:
            for i in range(num_substeps):
                self.euler(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is PHOTOCURRENT_SOLVER:
            for i in range(num_substeps):
                self.photocurrent(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is PLATEN_SOLVER:
            for i in range(num_substeps):
                self.platen(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is PC_SOLVER:
            for i in range(num_substeps):
                self.pred_corr(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is MILSTEIN_SOLVER:
            for i in range(num_substeps):
                self.milstein(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is MILSTEIN_IMP_SOLVER:
            for i in range(num_substeps):
                self.milstein_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is PC_2_SOLVER:
            for i in range(num_substeps):
                self.pred_corr_a(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is PHOTOCURRENT_PC_SOLVER:
            for i in range(num_substeps):
                self.photocurrent_pc(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is ROUCHON_SOLVER:
            for i in range(num_substeps):
                self.rouchon(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is EXPLICIT1_5_SOLVER:
            for i in range(num_substeps):
                self.platen15(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is TAYLOR1_5_SOLVER:
            for i in range(num_substeps):
                self.taylor15(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is TAYLOR1_5_IMP_SOLVER:
            for i in range(num_substeps):
                self.taylor15_imp(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        elif self.solver is TAYLOR2_0_SOLVER:
            for i in range(num_substeps):
                self.taylor20(t + i*dt, dt, noise[i, :], vec, out)
                out, vec = vec, out

        if self.normalize:
            self._normalize_inplace(vec)
        return vec

    cdef void _normalize_inplace(self, complex[::1] vec):
        _normalize_inplace(vec)

    # Dummy functions
    # Needed for compilation since ssesolve is not stand-alone
    cdef void d1(self, double t, complex[::1] v, complex[::1] out):
        """ deterministic part of the evolution
        depend on schrodinger vs master vs photocurrent
        """
        pass

    cdef void d2(self, double t, complex[::1] v, complex[:, ::1] out):
        """ stochastic part of the evolution
        depend on schrodinger vs master vs photocurrent
        """
        pass

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                       complex[::1] out, np.ndarray[complex, ndim=1] guess):
        """ Do the step X(t+dt) = f(X(t+dt)) + g(X(t)) """
        pass

    cdef void derivatives(self, double t, int deg, complex[::1] rho,
                              complex[::1] a, complex[:, ::1] b,
                              complex[:, :, ::1] Lb, complex[:,::1] La,
                              complex[:, ::1] L0b, complex[:, :, :, ::1] LLb,
                              complex[::1] L0a):
        """ Obtain the multiple terms for stochastic taylor expension
        Up to order 1.5
        multiple sc_ops
        """
        pass

    cdef void derivativesO2(self, double t, complex[::1] rho,
                            complex[::1] a, complex[::1] b, complex[::1] Lb,
                            complex[::1] La, complex[::1] L0b, complex[::1] LLb,
                            complex[::1] L0a,
                            complex[::1] LLa, complex[::1] LL0b,
                            complex[::1] L0Lb, complex[::1] LLLb):
        """ Obtain the multiple terms for stochastic taylor expension
        Up to order 2.0
        One sc_ops
        """
        pass

    cdef void photocurrent(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        """Special integration scheme:
        photocurrent collapse + euler evolution
        """
        pass

    cdef void photocurrent_pc(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        """Special integration scheme:
        photocurrent collapse + predictor-corrector evolution
        """
        pass

    cdef void rouchon(self, double t, double dt, double[:] noise,
                      complex[::1] vec, complex[::1] out):
        """Special integration scheme:
        Force valid density matrix using positive map
        Pierre Rouchon, Jason F. Ralph
        arXiv:1410.5345 [quant-ph]
        """
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void euler(self, double t, double dt, double[:] noise,
                    complex[::1] vec, complex[::1] out):
        """Integration scheme:
        Basic Euler order 0.5
        dV = d1 dt + d2_i dW_i
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef int i, j
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        _zero_2d(d2)
        copy(vec, out)
        self.d1(t, vec, out)
        self.d2(t, vec, d2)
        for i in range(self.num_ops):
            _axpy(noise[i], d2[i,:], out)

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
        The Theory of Open Quantum Systems
        Chapter 7 Eq. (7.47), H.-P Breuer, F. Petruccione
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
        _zero(d1)
        _zero_2d(d2)

        self.d1(t, vec, d1)
        self.d2(t, vec, d2)
        _axpy(1.0,vec,d1)
        copy(d1,Vt)
        copy(d1,out)
        _scale(0.5,out)
        for i in range(self.num_ops):
            copy(d1,Vp[i,:])
            copy(d1,Vm[i,:])
            _axpy( sqrt_dt,d2[i,:],Vp[i,:])
            _axpy(-sqrt_dt,d2[i,:],Vm[i,:])
            _axpy(noise[i],d2[i,:],Vt)
        _zero(d1)
        self.d1(t, Vt, d1)
        _axpy(0.5,d1,out)
        _axpy(0.5,vec,out)
        for i in range(self.num_ops):
            _zero_2d(d2p)
            _zero_2d(d2m)
            self.d2(t, Vp[i,:], d2p)
            self.d2(t, Vm[i,:], d2m)
            dw = noise[i] * 0.25
            _axpy(dw,d2m[i,:],out)
            _axpy(2*dw,d2[i,:],out)
            _axpy(dw,d2p[i,:],out)
            for j in range(self.num_ops):
                if i == j:
                    dw2 = sqrt_dt_inv * (noise[i]*noise[i] - dt)
                else:
                    dw2 = sqrt_dt_inv * noise[i] * noise[j]
                _axpy(dw2,d2p[j,:],out)
                _axpy(-dw2,d2m[j,:],out)

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
        _zero(a_pred)
        _zero(b_pred)
        _zero_2d(d2)
        _zero_3d(dd2)
        self.derivatives(t, 1, vec, a_pred, d2, dd2, None, None, None, None)
        copy(vec, euler)
        copy(vec, out)
        _axpy(1.0, a_pred, euler)
        for i in range(self.num_ops):
            _axpy(noise[i], d2[i,:], b_pred)
            _axpy(-dt_2, dd2[i,i,:], a_pred)
        _axpy(1.0, a_pred, out)
        _axpy(1.0, b_pred, euler)
        _axpy(0.5, b_pred, out)
        _zero_2d(d2)
        self.d2(t + dt, euler, d2)
        for i in range(self.num_ops):
            _axpy(noise[i]*0.5, d2[i,:], out)

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
        _zero(a_pred)
        cdef complex[::1] a_corr = self.buffer_1d[2,:]
        _zero(a_corr)
        cdef complex[::1] b_pred = self.buffer_1d[3,:]
        _zero(b_pred)
        cdef complex[:, ::1] d2 = self.buffer_2d[0,:,:]
        _zero_2d(d2)
        cdef complex[:, :, ::1] dd2 = self.buffer_3d[0,:,:,:]
        _zero_3d(dd2)

        cdef double dt_2 = dt*0.5
        self.derivatives(t, 1, vec, a_pred, d2, dd2, None, None, None, None)
        copy(vec, euler)
        _axpy(1.0, a_pred, euler)
        for i in range(self.num_ops):
            _axpy(noise[i], d2[i,:], b_pred)
            _axpy(-dt_2, dd2[i,i,:], a_pred)
        _axpy(1.0, b_pred, euler)
        copy(vec, out)
        _axpy(0.5, a_pred, out)
        _axpy(0.5, b_pred, out)
        _zero_2d(d2)
        _zero_3d(dd2)
        self.derivatives(t, 1, euler, a_corr, d2, dd2, None, None, None, None)
        for i in range(self.num_ops):
            _axpy(noise[i]*0.5, d2[i,:], out)
            _axpy(-dt_2, dd2[i,i,:], a_corr)
        _axpy(0.5, a_corr, out)

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
        _zero_2d(d2)
        _zero_3d(dd2)
        copy(vec,out)
        self.derivatives(t, 1, vec, out, d2, dd2, None, None, None, None)
        for i in range(self.num_ops):
            _axpy(noise[i],d2[i,:],out)
        for i in range(self.num_ops):
            for j in range(i, self.num_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j])
                _axpy(dw,dd2[i,j,:],out)

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
        _zero(a)
        _zero_2d(d2)
        _zero_3d(dd2)
        self.derivatives(t, 1, vec, a, d2, dd2, None, None, None, None)
        copy(vec, dvec)
        _axpy(0.5, a, dvec)
        for i in range(self.num_ops):
            _axpy(noise[i], d2[i,:], dvec)
        for i in range(self.num_ops):
            for j in range(i, self.num_ops):
                if (i == j):
                    dw = (noise[i] * noise[i] - dt) * 0.5
                else:
                    dw = (noise[i] * noise[j])
                _axpy(dw, dd2[i,j,:], dvec)
        copy(dvec, guess)
        _axpy(0.5, a, guess)
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
        _zero(a)
        _zero_2d(b)
        _zero_3d(Lb)
        _zero_2d(L0b)
        _zero_2d(La)
        _zero_4d(LLb)
        _zero(L0a)
        self.derivatives(t, 2, vec, a, b, Lb, La, L0b, LLb, L0a)

        cdef int i,j,k
        cdef double[::1] dz, dw
        dw = np.empty(self.num_ops)
        dz = np.empty(self.num_ops)
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        for i in range(self.num_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.num_ops])
        copy(vec,out)
        _axpy(1.0, a, out)
        _axpy(0.5, L0a, out)

        for i in range(self.num_ops):
            _axpy(dw[i], b[i,:], out)
            _axpy(0.5*(dw[i]*dw[i]-dt), Lb[i,i,:], out)
            _axpy(dz[i], La[i,:], out)
            _axpy(dw[i]-dz[i], L0b[i,:], out)
            _axpy(0.5 * ((1/3.) * dw[i] * dw[i] - dt) * dw[i],
                        LLb[i,i,i,:], out)
            for j in range(i+1,self.num_ops):
                _axpy((dw[i]*dw[j]), Lb[i,j,:], out)
                _axpy(0.5*(dw[j]*dw[j]-dt)*dw[i], LLb[i,j,j,:], out)
                _axpy(0.5*(dw[i]*dw[i]-dt)*dw[j], LLb[i,i,j,:], out)
                for k in range(j+1,self.num_ops):
                    _axpy(dw[i]*dw[j]*dw[k], LLb[i,j,k,:], out)

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
        _zero(a)
        _zero_2d(b)
        _zero_3d(Lb)
        _zero_2d(L0b)
        _zero_2d(La)
        _zero_4d(LLb)
        _zero(L0a)
        cdef np.ndarray[complex, ndim=1] guess = np.zeros((self.l_vec, ),
                   dtype=complex)
        cdef np.ndarray[complex, ndim=1] vec_t = np.zeros((self.l_vec, ),
                  dtype=complex)
        self.derivatives(t, 3, vec, a, b, Lb, La, L0b, LLb, L0a)

        cdef int i,j,k
        cdef double[::1] dz, dw
        dw = np.empty(self.num_ops)
        dz = np.empty(self.num_ops)
        # The dt of dz is included in the d1 part (Ldt) and the noise (dt**.5)
        for i in range(self.num_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.num_ops])
        copy(vec, vec_t)
        _axpy(0.5, a, vec_t)
        for i in range(self.num_ops):
            _axpy(dw[i], b[i,:], vec_t)
            _axpy(0.5*(dw[i]*dw[i]-dt), Lb[i,i,:], vec_t)
            _axpy(dz[i]-dw[i]*0.5, La[i,:], vec_t)
            _axpy(dw[i]-dz[i] , L0b[i,:], vec_t)
            _axpy(0.5 * ((1/3.) * dw[i] * dw[i] - dt) * dw[i],
                        LLb[i,i,i,:], vec_t)
            for j in range(i+1,self.num_ops):
                _axpy((dw[i]*dw[j]), Lb[i,j,:], vec_t)
                _axpy(0.5*(dw[j]*dw[j]-dt)*dw[i], LLb[i,j,j,:], vec_t)
                _axpy(0.5*(dw[i]*dw[i]-dt)*dw[j], LLb[i,i,j,:], vec_t)
                for k in range(j+1,self.num_ops):
                    _axpy(dw[i]*dw[j]*dw[k], LLb[i,j,k,:], vec_t)
        copy(vec_t, guess)
        _axpy(0.5, a, guess)

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
        dw = np.empty(self.num_ops)
        dz = np.empty(self.num_ops)
        for i in range(self.num_ops):
            dw[i] = noise[i]
            dz[i] = 0.5 *(noise[i] + 1./np.sqrt(3) * noise[i+self.num_ops])

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

        _zero(d1)
        _zero_2d(d2)
        _zero_2d(dd2)
        self.d1(t, vec, d1)
        self.d2(t, vec, d2)
        self.d2(t + dt, vec, dd2)
        # Euler part
        copy(vec,out)
        _axpy(1., d1, out)
        for i in range(self.num_ops):
            _axpy(dw[i], d2[i,:], out)

        _zero(V)
        _axpy(1., vec, V)
        _axpy(1./self.num_ops, d1, V)

        _zero_2d(v2p)
        _zero_2d(v2m)
        for i in range(self.num_ops):
            _axpy(1., V, v2p[i,:])
            _axpy(sqrt_dt, d2[i,:], v2p[i,:])
            _axpy(1., V, v2m[i,:])
            _axpy(-sqrt_dt, d2[i,:], v2m[i,:])

        _zero_3d(p2p)
        _zero_3d(p2m)
        for i in range(self.num_ops):
            _zero_2d(d2p)
            _zero_2d(d2m)
            self.d2(t, v2p[i,:], d2p)
            self.d2(t, v2m[i,:], d2m)
            ddw = (dw[i]*dw[i]-dt)*0.25/sqrt_dt  # 1.0
            _axpy( ddw, d2p[i,:], out)
            _axpy(-ddw, d2m[i,:], out)
            for j in range(self.num_ops):
                _axpy(      1., v2p[i,:], p2p[i,j,:])
                _axpy( sqrt_dt, d2p[j,:], p2p[i,j,:])
                _axpy(      1., v2p[i,:], p2m[i,j,:])
                _axpy(-sqrt_dt, d2p[j,:], p2m[i,j,:])

        _axpy(-0.5*(self.num_ops), d1, out)
        for i in range(self.num_ops):
            ddz = dz[i]*0.5/sqrt_dt  # 1.5
            ddd = 0.25*(dw[i]*dw[i]/3-dt)*dw[i]/dt  # 1.5
            _zero(d1p)
            _zero(d1m)
            _zero_2d(d2m)
            _zero_2d(d2p)
            _zero_2d(d2pp)
            _zero_2d(d2mm)
            self.d1(t + dt/self.num_ops, v2p[i,:], d1p)
            self.d1(t + dt/self.num_ops, v2m[i,:], d1m)
            self.d2(t, v2p[i,:], d2p)
            self.d2(t, v2m[i,:], d2m)
            self.d2(t, p2p[i,i,:], d2pp)
            self.d2(t, p2m[i,i,:], d2mm)

            _axpy( ddz+0.25, d1p, out)
            _axpy(-ddz+0.25, d1m, out)

            _axpy((dw[i]-dz[i]), dd2[i,:], out)
            _axpy((dz[i]-dw[i]), d2[i,:], out)

            _axpy( ddd, d2pp[i,:], out)
            _axpy(-ddd, d2mm[i,:], out)
            _axpy(-ddd, d2p[i,:], out)
            _axpy( ddd, d2m[i,:], out)

            for j in range(self.num_ops):
              ddw = 0.5*(dw[j]-dz[j]) # 1.5
              _axpy(ddw, d2p[j,:], out)
              _axpy(-2*ddw, d2[j,:], out)
              _axpy(ddw, d2m[j,:], out)

              if j>i:
                ddw = 0.5*(dw[i]*dw[j])/sqrt_dt # 1.0
                _axpy( ddw, d2p[j,:], out)
                _axpy(-ddw, d2m[j,:], out)

                ddw = 0.25*(dw[j]*dw[j]-dt)*dw[i]/dt # 1.5
                _zero_2d(d2pp)
                _zero_2d(d2mm)
                self.d2(t, p2p[j,i,:], d2pp)
                self.d2(t, p2m[j,i,:], d2mm)
                _axpy( ddw, d2pp[j,:], out)
                _axpy(-ddw, d2mm[j,:], out)
                _axpy(-ddw, d2p[j,:], out)
                _axpy( ddw, d2m[j,:], out)

                for k in range(j+1,self.num_ops):
                    ddw = 0.5*dw[i]*dw[j]*dw[k]/dt # 1.5
                    _axpy( ddw, d2pp[k,:], out)
                    _axpy(-ddw, d2mm[k,:], out)
                    _axpy(-ddw, d2p[k,:], out)
                    _axpy( ddw, d2m[k,:], out)

              if j<i:
                ddw = 0.25*(dw[j]*dw[j]-dt)*dw[i]/dt # 1.5
                _zero_2d(d2pp)
                _zero_2d(d2mm)
                self.d2(t, p2p[j,i,:], d2pp)
                self.d2(t, p2m[j,i,:], d2mm)
                _axpy( ddw, d2pp[j,:], out)
                _axpy(-ddw, d2mm[j,:], out)
                _axpy(-ddw, d2p[j,:], out)
                _axpy( ddw, d2m[j,:], out)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void taylor20(self, double t, double dt, double[::1] noise,
                           complex[::1] vec, complex[::1] out):
        """
        Chapter 10.5 Eq. (5.1),
        Numerical Solution of Stochastic Differential Equations
        By Peter E. Kloeden, Eckhard Platen
        """
        cdef double[::1] noises = np.empty(5)
        cdef double dwn
        self.order2noise.order2(noise,noises)
        cdef complex[::1] a = self.buffer_1d[0, :]
        cdef complex[::1] b = self.buffer_1d[1, :]
        cdef complex[::1] Lb = self.buffer_1d[2, :]
        cdef complex[::1] La = self.buffer_1d[3, :]
        cdef complex[::1] L0b = self.buffer_1d[4, :]
        cdef complex[::1] LLb = self.buffer_1d[5, :]
        cdef complex[::1] L0a = self.buffer_1d[6, :]
        cdef complex[::1] LLa = self.buffer_1d[7, :]
        cdef complex[::1] LL0b = self.buffer_1d[8, :]
        cdef complex[::1] L0Lb = self.buffer_1d[9, :]
        cdef complex[::1] LLLb = self.buffer_1d[10, :]
        _zero(a)
        _zero(b)
        _zero(Lb)
        _zero(La)
        _zero(L0b)
        _zero(LLb)
        _zero(L0a)
        _zero(LLa)
        _zero(LL0b)
        _zero(L0Lb)
        _zero(LLLb)
        self.derivativesO2(t, vec, a, b, Lb, La, L0b, LLb,
                           L0a, LLa, LL0b, L0Lb, LLLb)

        copy(vec,out)
        _axpy(1.0, a, out)

        _axpy(noises[0], b, out)
        dwn = noises[0]*noises[0]*0.5
        _axpy(dwn, Lb, out)

        _axpy(noises[1], La, out)
        _axpy(noises[0]-noises[1], L0b, out)
        dwn *= noises[0]*(1/3.)
        _axpy(dwn, LLb, out)
        _axpy(0.5, L0a, out)

        _axpy(noises[2], L0Lb, out)
        _axpy(noises[3], LL0b, out)
        _axpy(noises[4], LLa, out)
        dwn *= noises[0]*0.25
        _axpy(dwn, LLLb, out)


cdef class SSESolver(StochasticSolver):
    """stochastic Schrodinger system"""
    cdef CQobjEvo L
    cdef object c_ops
    cdef object cpcd_ops
    cdef object imp
    cdef double tol, imp_t

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.num_ops = len(c_ops)
        self.L = L.compiled_qobjevo
        self.c_ops = []
        self.cpcd_ops = []
        for i, op in enumerate(c_ops):
            self.c_ops.append(op[0].compiled_qobjevo)
            self.cpcd_ops.append(op[1].compiled_qobjevo)
        if sso.solver_code in [MILSTEIN_IMP_SOLVER, TAYLOR1_5_IMP_SOLVER]:
            self.tol = sso.tol
            self.imp = LinearOperator( (self.l_vec,self.l_vec),
                                      matvec=self.implicit_op, dtype=complex)

    def implicit_op(self, vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.l_vec, dtype=complex)
        self.d1(self.imp_t, vec, out)
        cdef int i
        _scale(-0.5,out)
        _axpy(1.,vec,out)
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d1(self, double t, complex[::1] vec, complex[::1] out):
        self.L._mul_vec(t, &vec[0], &out[0])
        cdef int i
        cdef complex e
        cdef CQobjEvo c_op
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        _zero(temp)
        for i in range(self.num_ops):
            c_op = self.cpcd_ops[i]
            e = c_op._expect(t, &vec[0])
            _zero(temp)
            c_op = self.c_ops[i]
            c_op._mul_vec(t, &vec[0], &temp[0])
            _axpy(-0.125 * e * e * self.dt, vec, out)
            _axpy(0.5 * e * self.dt, temp, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] vec, complex[:, ::1] out):
        cdef int i, k
        cdef CQobjEvo c_op
        cdef complex expect
        for i in range(self.num_ops):
            c_op = self.c_ops[i]
            c_op._mul_vec(t, &vec[0], &out[i,0])
            c_op = self.cpcd_ops[i]
            expect = c_op._expect(t, &vec[0])
            _axpy(-0.5*expect,vec,out[i,:])

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
        cdef CQobjEvo c_op
        cdef complex e, de_bb

        cdef complex[::1] e_real = self.expect_buffer_1d[0,:]
        cdef complex[:, ::1] de_b = self.expect_buffer_2d[:,:]
        cdef complex[::1] de_a = self.expect_buffer_1d[1,:]
        cdef complex[:, :, ::1] dde_bb = self.expect_buffer_3d[:,:,:]
        _zero_3d(dde_bb)
        cdef complex[:, ::1] Cvec = self.func_buffer_2d[:,:]
        cdef complex[:, :, ::1] Cb = self.func_buffer_3d[:,:,:]
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        cdef complex[::1] temp2 = self.func_buffer_1d[1,:]
        _zero(temp)
        _zero(temp2)
        _zero_2d(Cvec)
        _zero_3d(Cb)

        # a b
        self.L._mul_vec(t, &vec[0], &a[0])
        for i in range(self.num_ops):
            c_op = self.c_ops[i]
            c_op._mul_vec(t, &vec[0], &Cvec[i,0])
            e = _dotc(vec,Cvec[i,:])
            e_real[i] = real(e)
            _axpy(1., Cvec[i,:], b[i,:])
            _axpy(-e_real[i], vec, b[i,:])
            _axpy(-0.5 * e_real[i] * e_real[i] * dt, vec, a[:])
            _axpy(e_real[i] * dt, Cvec[i,:], a[:])

        #Lb bb'
        if deg >= 1:
          for i in range(self.num_ops):
            c_op = self.c_ops[i]
            for j in range(self.num_ops):
              c_op._mul_vec(t, &b[j,0], &Cb[i,j,0])
              for k in range(self.l_vec):
                  temp[k] = conj(b[j,k])
                  temp2[k] = 0.
              c_op._mul_vec(t, &temp[0], &temp2[0])
              de_b[i,j] = (_dotc(vec, Cb[i,j,:]) + _dot(b[j,:], Cvec[i,:]) + \
                          conj(_dotc(b[j,:], Cvec[i,:]) + _dotc(vec, temp2))) * 0.5
              _axpy(1., Cb[i,j,:], Lb[i,j,:])
              _axpy(-e_real[i], b[j,:], Lb[i,j,:])
              _axpy(-de_b[i,j], vec, Lb[i,j,:])

              for k in range(self.num_ops):
                dde_bb[i,j,k] += (_dot(b[j,:], Cb[i,k,:]) + \
                                  _dot(b[k,:], Cb[i,j,:]) + \
                                  conj(_dotc(b[k,:], temp2)))*.5
                dde_bb[i,k,j] += conj(_dotc(b[k,:], temp2))*.5

        #L0b La LLb
        if deg >= 2:
          for i in range(self.num_ops):
              #ba'
              self.L._mul_vec(t, &b[i,0], &La[i,0])
              for j in range(self.num_ops):
                  _axpy(-0.5 * e_real[j] * e_real[j] * dt, b[i,:], La[i,:])
                  _axpy(-e_real[j] * de_b[i,j] * dt, vec, La[i,:])
                  _axpy(e_real[j] * dt, Cb[i,j,:], La[i,:])
                  _axpy(de_b[i,j] * dt, Cvec[i,:], La[i,:])

              #ab' + db/dt + bbb"/2
              c_op = self.c_ops[i]
              c_op._mul_vec(t, &a[0], &L0b[i,0])
              for k in range(self.l_vec):
                  temp[k] = conj(a[k])
                  temp2[k] = 0.
              c_op._mul_vec(t, &temp[0], &temp2[0])
              de_a[i] = (_dotc(vec, L0b[i,:]) + _dot(a, Cvec[i,:]) + \
                        conj(_dotc(a, Cvec[i,:]) + _dotc(vec, temp2))) * 0.5
              _axpy(-e_real[i], a, L0b[i,:])
              _axpy(-de_a[i], vec, L0b[i,:])

              temp = np.zeros(self.l_vec, dtype=complex)
              c_op._mul_vec(t + self.dt, &vec[0], &temp[0])
              e = _dotc(vec,temp)
              _axpy(1., temp, L0b[i,:])
              _axpy(-real(e), vec, L0b[i,:])
              _axpy(-1., b[i,:], L0b[i,:])

              for j in range(self.num_ops):
                  _axpy(-de_b[i,j]*dt, b[j,:], L0b[i,:])
                  _axpy(-dde_bb[i,j,j]*dt, vec, L0b[i,:])

              #b(bb"+b'b')
              for j in range(i,self.num_ops):
                  for k in range(j, self.num_ops):
                      c_op._mul_vec(t, &Lb[j,k,0], &LLb[i,j,k,0])
                      for l in range(self.l_vec):
                          temp[l] = conj(Lb[j,k,l])
                          temp2[l] = 0.
                      c_op._mul_vec(t, &temp[0], &temp2[0])
                      de_bb = (_dotc(vec, LLb[i,j,k,:]) + \
                               _dot(Lb[j,k,:], Cvec[i,:]) + \
                               conj(_dotc(Lb[j,k,:], Cvec[i,:]) +\
                               _dotc(vec, temp2)))*0.5
                      _axpy(-e_real[i], Lb[j,k,:], LLb[i,j,k,:])
                      _axpy(-de_bb, vec, LLb[i,j,k,:])
                      _axpy(-dde_bb[i,j,k], vec, LLb[i,j,k,:])
                      _axpy(-de_b[i,j], b[k,:], LLb[i,j,k,:])
                      _axpy(-de_b[i,k], b[j,:], LLb[i,j,k,:])

        #da/dt + aa' + bba"
        if deg == 2:
          self.d1(t + dt, vec, L0a)
          _axpy(-1.0, a, L0a)
          self.L._mul_vec(t, &a[0], &L0a[0])
          for j in range(self.num_ops):
              c_op = self.c_ops[j]
              temp = np.zeros(self.l_vec, dtype=complex)
              c_op._mul_vec(t, &a[0], &temp[0])
              _axpy(-0.5 * e_real[j] * e_real[j] * dt, a[:], L0a[:])
              _axpy(-e_real[j] * de_a[j] * dt, vec, L0a[:])
              _axpy(e_real[j] * dt, temp, L0a[:])
              _axpy(de_a[j] * dt, Cvec[j,:], L0a[:])
              for i in range(self.num_ops):
                  _axpy(-0.5*(e_real[i] * dde_bb[i,j,j] +
                          de_b[i,j] * de_b[i,j]) * dt * dt, vec, L0a[:])
                  _axpy(-e_real[i] * de_b[i,j] * dt * dt, b[j,:], L0a[:])
                  _axpy(0.5*dde_bb[i,j,j] * dt * dt, Cvec[i,:], L0a[:])
                  _axpy(de_b[i,j] * dt * dt, Cb[i,j,:], L0a[:])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _c_vec_conj(self, double t, CQobjEvo c_op,
                         complex[::1] vec, complex[::1] out):
        cdef int k
        cdef complex[::1] temp = self.func_buffer_1d[13,:]
        for k in range(self.l_vec):
            temp[k] = conj(vec[k])
            out[k] = 0.
        c_op._mul_vec(t, &temp[0], &out[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void derivativesO2(self, double t, complex[::1] psi,
                            complex[::1] a, complex[::1] b, complex[::1] Lb,
                            complex[::1] La, complex[::1] L0b, complex[::1] LLb,
                            complex[::1] L0a,
                            complex[::1] LLa, complex[::1] LL0b,
                            complex[::1] L0Lb, complex[::1] LLLb):
        """
        Combinaisons of a and b derivative for m sc_ops up to order dt**2.0
        Use Stratonovich-Taylor expansion.
        One one sc_ops
        dY ~ a dt + bi dwi

        b[:]     b                        d2 euler    dw
        a[:]     a- Lb/2                  d1 euler    dt
        Lb[:]    b'b                      milstein    dw^2/2
        L0b[:]   ab'- b'b'b/2             taylor1.5   dwdt-dz
        La[:]    ba'- (b'b'b+b"bb)/2      taylor1.5   dz
        LLb[:]   (b"bb+b'b'b)             taylor1.5   dw^3/6
        L0a[:]   a_a'_ + da/dt -Lb_a'_/2  taylor1.5   dt^2/2

        LLa[:]   ...                      taylor2.0   dwdt-dz
        LL0b[:]  ...                      taylor2.0   dz
        L0Lb[:]  ...                      taylor2.0   dw^3/6
        LLLb[:]  ...                      taylor2.0   dt^2/2
        """
        cdef double dt = self.dt
        cdef CQobjEvo c_op = self.c_ops[0]
        cdef complex e, de_b, de_Lb, de_LLb, dde_bb, dde_bLb
        cdef complex de_a, dde_ba, de_La, de_L0b

        cdef complex[::1] Cpsi = self.func_buffer_1d[0,:]
        cdef complex[::1] Cb = self.func_buffer_1d[1,:]
        cdef complex[::1] Cbc = self.func_buffer_1d[2,:]
        cdef complex[::1] CLb = self.func_buffer_1d[3,:]
        cdef complex[::1] CLbc = self.func_buffer_1d[4,:]
        cdef complex[::1] CLLb = self.func_buffer_1d[5,:]
        cdef complex[::1] CLLbc = self.func_buffer_1d[6,:]
        cdef complex[::1] Ca = self.func_buffer_1d[7,:]
        cdef complex[::1] Cac = self.func_buffer_1d[8,:]
        cdef complex[::1] CLa = self.func_buffer_1d[9,:]
        cdef complex[::1] CLac = self.func_buffer_1d[10,:]
        cdef complex[::1] CL0b = self.func_buffer_1d[11,:]
        cdef complex[::1] CL0bc = self.func_buffer_1d[12,:]
        _zero(Cpsi)
        _zero(Cb)
        _zero(CLb)
        _zero(CLLb)
        _zero(Ca)
        _zero(CLa)
        _zero(CL0b)

        # b
        c_op._mul_vec(t, &psi[0], &Cpsi[0])
        e = real(_dotc(psi, Cpsi))
        _axpy(1., Cpsi, b)
        _axpy(-e, psi, b)

        # Lb
        c_op._mul_vec(t, &b[0], &Cb[0])
        self._c_vec_conj(t, c_op, b, Cbc)
        de_b = (_dotc(psi, Cb) + _dot(b, Cpsi) + \
                conj(_dotc(b, Cpsi) + _dotc(psi, Cbc))) * 0.5
        _axpy(1., Cb, Lb)
        _axpy(-e, b, Lb)
        _axpy(-de_b, psi, Lb)

        # LLb = b'b'b + b"bb
        c_op._mul_vec(t, &Lb[0], &CLb[0])
        self._c_vec_conj(t, c_op, Lb, CLbc)
        de_Lb = (_dotc(psi, CLb) + _dot(Lb, Cpsi) + \
                 conj(_dotc(Lb, Cpsi) + _dotc(psi, CLbc)))*0.5
        _axpy(1, CLb, LLb)      # b'b'b
        _axpy(-e, Lb, LLb)      # b'b'b
        _axpy(-de_Lb, psi, LLb) # b'b'b
        dde_bb += (_dot(b, Cb) + conj(_dotc(b, Cbc)))
        _axpy(-dde_bb, psi, LLb) # b"bb
        _axpy(-de_b*2, b, LLb)   # b"bb

        # LLLb = b"'bbb + 3* b"b'bb + b'(b"bb + b'b'b)
        c_op._mul_vec(t, &LLb[0], &CLLb[0])
        self._c_vec_conj(t, c_op, LLb, CLLbc)
        de_LLb = (_dotc(psi, CLLb) + _dot(LLb, Cpsi) + \
                  conj(_dotc(LLb, Cpsi) + _dotc(psi, CLLbc)))*0.5
        dde_bLb += (_dot(b, CLb) + _dot(Lb, Cb) + conj(_dotc(Lb, Cbc)) + \
                    conj(_dotc(b, CLbc)))*.5
        _axpy(1, CLLb, LLLb)          # b'(b"bb + b'b'b)
        _axpy(-e, LLb, LLLb)          # b'(b"bb + b'b'b)
        _axpy(-de_LLb, psi, LLLb)     # b'(b"bb + b'b'b)
        _axpy(-dde_bLb*3, psi, LLLb)  # b"bLb
        _axpy(-de_Lb*3, b, LLLb)      # b"bLb
        _axpy(-de_b*3, Lb, LLLb)      # b"bLb
        _axpy(-dde_bb*3, b, LLLb)       # b"'bbb

        # a
        self.L._mul_vec(t, &psi[0], &a[0])
        _axpy(-0.5 * e * e * dt, psi, a)
        _axpy(e * dt, Cpsi, a)
        _axpy(-0.5 * dt, Lb, a)

        #La
        self.L._mul_vec(t, &b[0], &La[0])
        _axpy(-0.5 * e * e * dt, b, La)
        _axpy(-e * de_b * dt, psi, La)
        _axpy(e * dt, Cb, La)
        _axpy(de_b * dt, Cpsi, La)
        _axpy(-0.5 * dt, LLb, La)

        #LLa
        _axpy(-2 * e * de_b * dt, b, LLa)
        _axpy(-de_b * de_b * dt, psi, LLa)
        _axpy(-e * dde_bb * dt, psi, LLa)
        _axpy( 2 * de_b * dt, Cb, LLa)
        _axpy( dde_bb * dt, Cpsi, LLa)

        self.L._mul_vec(t, &Lb[0], &LLa[0])
        _axpy(-de_Lb * e * dt, psi, LLa)
        _axpy(-0.5 * e * e * dt, Lb, LLa)
        _axpy( de_Lb * dt, Cpsi, LLa)
        _axpy( e * dt, CLb, LLa)

        _axpy(-0.5 * dt, LLLb, LLa)

        # L0b = b'a
        c_op._mul_vec(t, &a[0], &Ca[0])
        self._c_vec_conj(t, c_op, a, Cac)
        de_a = (_dotc(psi, Ca) + _dot(a, Cpsi) + \
                conj(_dotc(a, Cpsi) + _dotc(psi, Cac))) * 0.5
        _axpy(1.0, Ca, L0b)
        _axpy(-e, a, L0b)
        _axpy(-de_a, psi, L0b)

        # LL0b = b"ba + b'La
        dde_ba += (_dot(b, Ca) + _dot(a, Cb) + conj(_dotc(a, Cbc)) + \
                    conj(_dotc(b, Cac)))*.5
        _axpy(-dde_ba, psi, LL0b)
        _axpy(-de_a, b, LL0b)
        _axpy(-de_b, a, LL0b)
        c_op._mul_vec(t, &La[0], &CLa[0])
        self._c_vec_conj(t, c_op, La, CLac)
        de_La = (_dotc(psi, CLa) + _dot(La, Cpsi) + \
                conj(_dotc(La, Cpsi) + _dotc(psi, CLac))) * 0.5
        _axpy(1., CLa, LL0b)
        _axpy(-e, La, LL0b)
        _axpy(-de_La, psi, LL0b)

        # L0Lb = b"ba + b'L0b
        _axpy(-dde_ba, psi, L0Lb)
        _axpy(-de_a, b, L0Lb)
        _axpy(-de_b, a, L0Lb)
        c_op._mul_vec(t, &L0b[0], &CL0b[0])
        self._c_vec_conj(t, c_op, L0b, CL0bc)
        de_L0b = (_dotc(psi, CL0b) + _dot(L0b, Cpsi) + \
                  conj(_dotc(L0b, Cpsi) + _dotc(psi, CL0bc))) * 0.5
        _axpy(1., CL0b, L0Lb)
        _axpy(-e, L0b, L0Lb)
        _axpy(-de_L0b, psi, L0Lb)

        # _L0_ _a_ = da/dt + a'_a_ -_L0_Lb/2
        self.d1(t + dt, psi, L0a) # da/dt
        _axpy(-0.5 * dt, Lb, L0a)  # da/dt
        _axpy(-1.0, a, L0a)        # da/dt
        self.L._mul_vec(t, &a[0], &L0a[0]) # a'_a_
        _axpy(-0.5 * e * e * dt, a, L0a)    # a'_a_
        _axpy(-e * de_a * dt, psi, L0a)     # a'_a_
        _axpy(e * dt, Ca, L0a)              # a'_a_
        _axpy(de_a * dt, Cpsi, L0a)         # a'_a_
        _axpy(-0.5 * dt, L0Lb, L0a) # _L0_Lb/2

    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                        complex[::1] out,
                                        np.ndarray[complex, ndim=1] guess):
        # np.ndarray to memoryview is OK but not the reverse
        # scipy function only take np array, not memoryview
        self.imp_t = t
        spout, check = sp.linalg.bicgstab(self.imp, dvec, x0=guess,
                                          tol=self.tol, atol=1e-12)
        cdef int i
        copy(spout, out)


cdef class SMESolver(StochasticSolver):
    """stochastic master equation system"""
    cdef CQobjEvo L
    cdef object imp
    cdef object c_ops
    cdef int N_root
    cdef double tol

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.num_ops = len(c_ops)
        self.L = L.compiled_qobjevo
        self.c_ops = []
        self.N_root = np.sqrt(self.l_vec)
        for i, op in enumerate(c_ops):
            self.c_ops.append(op.compiled_qobjevo)
        if sso.solver_code in [MILSTEIN_IMP_SOLVER, TAYLOR1_5_IMP_SOLVER]:
            self.tol = sso.tol
            self.imp = sso.imp

    cdef void _normalize_inplace(self, complex[::1] vec):
        _normalize_rho(vec)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex expect(self, complex[::1] rho):
        cdef complex e = 0.
        cdef int k
        for k in range(self.N_root):
            e += rho[k*(self.N_root+1)]
        return e

    @cython.boundscheck(False)
    cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
        self.L._mul_vec(t, &rho[0], &out[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
        cdef int i, k
        cdef CQobjEvo c_op
        cdef complex expect
        for i in range(self.num_ops):
            c_op = self.c_ops[i]
            c_op._mul_vec(t, &rho[0], &out[i,0])
            expect = self.expect(out[i,:])
            _axpy(-expect, rho, out[i,:])

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
        cdef CQobjEvo c_op
        cdef CQobjEvo c_opj
        cdef complex trApp, trAbb, trAa
        cdef complex[::1] trAp = self.expect_buffer_1d[0,:]
        cdef complex[:, ::1] trAb = self.expect_buffer_2d
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        #_zero(temp)

        # a
        self.L._mul_vec(t, &rho[0], &a[0])

        # b
        for i in range(self.num_ops):
            c_op = self.c_ops[i]
            # bi
            c_op._mul_vec(t, &rho[0], &b[i,0])
            trAp[i] = self.expect(b[i,:])
            _axpy(-trAp[i], rho, b[i,:])

        # Libj = bibj', i<=j
        # sc_ops must commute (Libj = Ljbi)
        if deg >= 1:
          for i in range(self.num_ops):
            c_op = self.c_ops[i]
            for j in range(i, self.num_ops):
                c_op._mul_vec(t, &b[j,0], &Lb[i,j,0])
                trAb[i,j] = self.expect(Lb[i,j,:])
                _axpy(-trAp[j], b[i,:], Lb[i,j,:])
                _axpy(-trAb[i,j], rho, Lb[i,j,:])

        # L0b La LLb
        if deg >= 2:
          for i in range(self.num_ops):
            c_op = self.c_ops[i]
            # Lia = bia'
            self.L._mul_vec(t, &b[i,0], &La[i,0])

            # L0bi = abi' + dbi/dt + Sum_j bjbjbi"/2
            # db/dt
            c_op._mul_vec(t + self.dt, &rho[0], &L0b[i,0])
            trApp = self.expect(L0b[i,:])
            _axpy(-trApp, rho, L0b[i,:])
            _axpy(-1, b[i,:], L0b[i,:])
            # ab'
            _zero(temp) # = np.zeros((self.l_vec, ), dtype=complex)
            c_op._mul_vec(t, &a[0], &temp[0])
            trAa = self.expect(temp)
            _axpy(1., temp, L0b[i,:])
            _axpy(-trAp[i], a[:], L0b[i,:])
            _axpy(-trAa, rho, L0b[i,:])
            # bbb" : trAb[i,j] only defined for j>=i
            for j in range(i):
                _axpy(-trAb[j,i]*self.dt, b[j,:], L0b[i,:])  # L contain dt
            for j in range(i,self.num_ops):
                _axpy(-trAb[i,j]*self.dt, b[j,:], L0b[i,:])  # L contain dt

            # LLb
            # LiLjbk = bi(bj'bk'+bjbk"), i<=j<=k
            # sc_ops must commute (LiLjbk = LjLibk = LkLjbi)
            for j in range(i,self.num_ops):
              for k in range(j,self.num_ops):
                c_op._mul_vec(t, &Lb[j,k,0], &LLb[i,j,k,0])
                trAbb = self.expect(LLb[i,j,k,:])
                _axpy(-trAp[i], Lb[j,k,:], LLb[i,j,k,:])
                _axpy(-trAbb, rho, LLb[i,j,k,:])
                _axpy(-trAb[i,k], b[j,:], LLb[i,j,k,:])
                _axpy(-trAb[i,j], b[k,:], LLb[i,j,k,:])

        # L0a = a'a + da/dt + bba"/2  (a" = 0)
        if deg == 2:
            self.L._mul_vec(t, &a[0], &L0a[0])
            self.L._mul_vec(t+self.dt, &rho[0], &L0a[0])
            _axpy(-1, a, L0a)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void derivativesO2(self, double t, complex[::1] rho,
                            complex[::1] a, complex[::1] b, complex[::1] Lb,
                            complex[::1] La, complex[::1] L0b, complex[::1] LLb,
                            complex[::1] L0a,
                            complex[::1] LLa, complex[::1] LL0b,
                            complex[::1] L0Lb, complex[::1] LLLb):
        """
        Combinaisons of a and b derivative for m sc_ops up to order dt**2.0
        Use Stratonovich-Taylor expansion.
        One one sc_ops
        dY ~ a dt + bi dwi

        b[:]     b                        d2 euler    dw
        a[:]     a- Lb/2                  d1 euler    dt
        Lb[:]    b'b                      milstein    dw^2/2
        L0b[:]   ab'- b'b'b/2             taylor1.5   dwdt-dz
        La[:]    ba'- (b'b'b+b"bb)/2      taylor1.5   dz
        LLb[:]   (b"bb+b'b'b)             taylor1.5   dw^3/6
        L0a[:]   a_a'_ + da/dt -Lb_a'_/2  taylor1.5   dt^2/2

        LLa[:]   ...                      taylor2.0   dwdt-dz
        LL0b[:]  ...                      taylor2.0   dz
        L0Lb[:]  ...                      taylor2.0   dw^3/6
        LLLb[:]  ...                      taylor2.0   dt^2/2
        """
        cdef int i, j, k
        cdef CQobjEvo c_op = self.c_ops[0]
        cdef CQobjEvo c_opj
        cdef complex trAp, trApt
        cdef complex trAb, trALb, trALLb
        cdef complex trAa, trALa
        cdef complex trAL0b

        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        cdef complex[::1] temp2 = self.func_buffer_1d[1,:]

        # b
        c_op._mul_vec(t, &rho[0], &b[0])
        trAp = self.expect(b)
        _axpy(-trAp, rho, b)

        # Lb = b'b
        c_op._mul_vec(t, &b[0], &Lb[0])
        trAb = self.expect(Lb)
        _axpy(-trAp, b, Lb)
        _axpy(-trAb, rho, Lb)

        # LLb = b'Lb+b"bb
        c_op._mul_vec(t, &Lb[0], &LLb[0])
        trALb = self.expect(LLb)
        _axpy(-trAp, Lb, LLb)
        _axpy(-trALb, rho, LLb)
        _axpy(-trAb*2, b, LLb)

        # LLLb = b'LLb + 3 b"bLb + b"'bbb
        c_op._mul_vec(t, &LLb[0], &LLLb[0])
        trALLb = self.expect(LLLb)
        _axpy(-trAp, LLb, LLLb)
        _axpy(-trALLb, rho, LLLb)
        _axpy(-trALb*3, b, LLLb)
        _axpy(-trAb*3, Lb, LLLb)

        # _a_ = a - Lb/2
        self.L._mul_vec(t, &rho[0], &a[0])
        _axpy(-0.5*self.dt, Lb, a)

        # L_a_ = ba' - LLb/2
        self.L._mul_vec(t, &b[0], &La[0])
        _axpy(-0.5*self.dt, LLb, La)

        # LL_a_ = b(La)' - LLLb/2
        self.L._mul_vec(t, &Lb[0], &LLa[0])
        _axpy(-0.5*self.dt, LLLb, LLa)

        # _L0_b = b'(_a_)
        c_op._mul_vec(t, &a[0], &L0b[0])
        trAa = self.expect(L0b)
        _axpy(-trAp, a, L0b)
        _axpy(-trAa, rho, L0b)

        # _L0_Lb = b'(b'(_a_))+b"(_a_,b)
        c_op._mul_vec(t, &L0b[0], &L0Lb[0])
        trAL0b = self.expect(L0Lb)
        _axpy(-trAp, L0b, L0Lb)
        _axpy(-trAL0b, rho, L0Lb)
        _axpy(-trAa, b, L0Lb)
        _axpy(-trAb, a, L0Lb)

        # L_L0_b = b'(_a_'(b))+b"(_a_,b)
        c_op._mul_vec(t, &La[0], &LL0b[0])
        trAL0b = self.expect(LL0b)
        _axpy(-trAp, La, LL0b)
        _axpy(-trAL0b, rho, LL0b)
        _axpy(-trAa, b, LL0b)
        _axpy(-trAb, a, LL0b)

        # _L0_ _a_ = _L0_a - _L0_Lb/2 + da/dt
        self.L._mul_vec(t, &a[0], &L0a[0])
        self.L._mul_vec(t+self.dt, &rho[0], &L0a[0])
        _axpy(-0.5*self.dt, Lb, L0a) # _a_(t+dt) = a(t+dt)-0.5*Lb
        _axpy(-1, a, L0a)
        _axpy(-self.dt*0.5, L0Lb, L0a)


    cdef void implicit(self, double t,  np.ndarray[complex, ndim=1] dvec,
                                        complex[::1] out,
                                        np.ndarray[complex, ndim=1] guess):
        # np.ndarray to memoryview is OK but not the reverse
        # scipy function only take np array, not memoryview
        spout, check = sp.linalg.bicgstab(self.imp(t, data=1), dvec, x0=guess,
                                          tol=self.tol, atol=1e-12)
        cdef int i
        copy(spout,out)


cdef class PcSSESolver(StochasticSolver):
    """photocurrent for Schrodinger equation"""
    cdef CQobjEvo L
    cdef object c_ops
    cdef object cdc_ops

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.num_ops = len(c_ops)
        self.L = L.compiled_qobjevo
        self.c_ops = []
        self.cdc_ops = []
        for i, op in enumerate(c_ops):
            self.c_ops.append(op[0].compiled_qobjevo)
            self.cdc_ops.append(op[1].compiled_qobjevo)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void photocurrent(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef CQobjEvo c_op
        cdef double rand
        cdef int i, which = -1
        cdef complex[::1] expects = self.expect_buffer_1d[0,:]
        cdef complex[::1] d2 = self.buffer_1d[0,:]

        copy(vec, out)
        self.d1(t, vec, out)
        rand = np.random.rand()
        for i in range(self.num_ops):
            c_op = self.cdc_ops[i]
            expects[i] = c_op.expect(t, vec)
            if expects[i].real * dt >= 1e-15:
                rand -= expects[i].real *dt
            if rand < 0:
                which = i
                noise[i] = 1.
                break

        if which >= 0:
            self.collapse(t, which, expects[which].real, vec, d2)
            _axpy(1, d2, out)
            _axpy(-1, vec, out)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void photocurrent_pc(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef CQobjEvo c_op
        cdef double expect
        cdef int i, which=0, num_coll=0, did_collapse=0
        cdef complex[::1] tmp = self.buffer_1d[0,:]
        cdef complex[::1] expects = self.expect_buffer_1d[0,:]
        cdef np.ndarray[int, ndim=1] colls

        # Collapses are computed first
        for i in range(self.num_ops):
            c_op = self.cdc_ops[i]
            expects[i] = c_op.expect(t, vec).real
            if expects[i].real > 0:
                did_collapse = np.random.poisson(expects[i].real * dt)
                num_coll += did_collapse
                if did_collapse:
                    which = i
                noise[i] = did_collapse * 1.
            else:
                noise[i] = 0.

        if num_coll == 0:
            pass
        elif num_coll == 1:
            # Do one collapse
            self.collapse(t, which, expects[which].real, vec, out)
            copy(out, vec)
        elif num_coll and noise[which] == num_coll:
            # Do many collapse of one sc_ops.
            # Recompute the expectation value, but only to check for zero.
            c_op = self.cdc_ops[which]
            for i in range(num_coll):
                expect = c_op.expect(t, vec).real
                if expect * dt >= 1e-15:
                    self.collapse(t, which, expect, vec, out)
                    copy(out,vec)
        elif num_coll >= 2:
            # 2 or more collapses of different operators
            # Ineficient, should be rare
            coll = []
            for i in range(self.num_ops):
                coll += [i]*int(noise[i])
            np.random.shuffle(coll)
            for i in coll:
                c_op = self.cdc_ops[i]
                expect = c_op.expect(t, vec).real
                if expect * dt >= 1e-15:
                    self.collapse(t, i, expect, vec, out)
                    copy(out,vec)
        copy(vec,tmp)
        copy(vec,out)
        self.d1(t, vec, tmp)
        self.d1(t+dt, tmp, out)
        _scale(0.5, out)
        _axpy(0.5, tmp, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d1(self, double t, complex[::1] vec, complex[::1] out):
        self.L._mul_vec(t, &vec[0], &out[0])
        cdef int i
        cdef complex e
        cdef CQobjEvo c_op
        cdef complex[::1] temp = self.func_buffer_1d[0,:]
        for i in range(self.num_ops):
            _zero(temp)
            c_op = self.c_ops[i]
            c_op._mul_vec(t, &vec[0], &temp[0])
            e = _dznrm2(temp)
            _axpy(0.5 * e * e * self.dt, vec, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] vec, complex[:, ::1] out):
        cdef int i
        cdef CQobjEvo c_op
        cdef complex expect
        for i in range(self.num_ops):
            c_op = self.c_ops[i]
            c_op._mul_vec(t, &vec[0], &out[i,0])
            expect = _dznrm2(out[i,:])
            if expect.real >= 1e-15:
                _zscale(1/expect, out[i,:])
            else:
                _zero(out[i,:])
            _axpy(-1, vec, out[i,:])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void collapse(self, double t, int which, double expect,
                       complex[::1] vec, complex[::1] out):
        cdef CQobjEvo c_op
        c_op = self.c_ops[which]
        _zero(out)
        c_op._mul_vec(t, &vec[0], &out[0])
        _zscale(1/expect, out)


cdef class PcSMESolver(StochasticSolver):
    """photocurrent for master equation"""
    cdef CQobjEvo L
    cdef object cdcr_cdcl_ops
    cdef object cdcl_ops
    cdef object clcdr_ops
    cdef int N_root

    def set_data(self, sso):
        L = sso.LH
        c_ops = sso.sops
        self.l_vec = L.cte.shape[0]
        self.num_ops = len(c_ops)
        self.L = L.compiled_qobjevo
        self.cdcr_cdcl_ops = []
        self.cdcl_ops = []
        self.clcdr_ops = []
        self.N_root = np.sqrt(self.l_vec)
        for i, op in enumerate(c_ops):
            self.cdcr_cdcl_ops.append(op[0].compiled_qobjevo)
            self.cdcl_ops.append(op[1].compiled_qobjevo)
            self.clcdr_ops.append(op[2].compiled_qobjevo)

    cdef void _normalize_inplace(self, complex[::1] vec):
        _normalize_rho(vec)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void photocurrent(self, double t, double dt,  double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef CQobjEvo c_op
        cdef double rand
        cdef int i, which = -1
        cdef complex[::1] expects = self.expect_buffer_1d[0,:]
        cdef complex[::1] d2 = self.buffer_1d[0,:]

        copy(vec, out)
        self.d1(t, vec, out)
        rand = np.random.rand()
        for i in range(self.num_ops):
            c_op = self.clcdr_ops[i]
            expects[i] = c_op.expect(t, vec)
            if expects[i].real * dt >= 1e-15:
                rand -= expects[i].real *dt
            if rand < 0:
                which = i
                noise[i] = 1.
                break

        if which >= 0:
            self.collapse(t, which, expects[which].real, vec, d2)
            _axpy(1, d2, out)
            _axpy(-1, vec, out)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void photocurrent_pc(self, double t, double dt,  double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef CQobjEvo c_op
        cdef int i, which, num_coll=0, did_collapse
        cdef complex[::1] expects = self.expect_buffer_1d[0,:]
        cdef complex[::1] tmp = self.buffer_1d[0,:]
        cdef double expect
        cdef np.ndarray[int, ndim=1] colls

        # Collapses are computed first
        for i in range(self.num_ops):
            c_op = self.clcdr_ops[i]
            expects[i] = c_op.expect(t, vec).real
            if expects[i].real > 0:
                did_collapse = np.random.poisson(expects[i].real* dt)
                num_coll += did_collapse
                if did_collapse:
                    which = i
                noise[i] = did_collapse * 1.
            else:
                noise[i] = 0.

        if num_coll == 0:
            pass
        elif num_coll == 1:
            # Do one collapse
            self.collapse(t, which, expects[which].real, vec, out)
            copy(out,vec)
        elif noise[which] == num_coll:
            # Do many collapse of one sc_ops.
            # Recompute the expectation value, but only to check for zero.
            c_op = self.clcdr_ops[which]
            for i in range(num_coll):
                expect = c_op.expect(t, vec).real
                if expect * dt >= 1e-15:
                    self.collapse(t, which, expect, vec, out)
                    copy(out,vec)
        elif num_coll >= 2:
            # 2 or more collapses of different operators
            # Ineficient, should be rare
            coll = []
            for i in range(self.num_ops):
                coll += [i] * int(noise[i])
            np.random.shuffle(coll)
            for i in coll:
                c_op = self.clcdr_ops[i]
                expect = c_op.expect(t, vec).real
                if expect * dt >= 1e-15:
                    self.collapse(t, i, expect, vec, out)
                    copy(out,vec)

        copy(vec,tmp)
        copy(vec,out)
        self.d1(t, vec, tmp)
        self.d1(t+dt, tmp, out)
        _scale(0.5, out)
        _axpy(0.5, tmp, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex expect(self, complex[::1] rho):
        cdef complex e = 0.
        cdef int k
        for k in range(self.N_root):
            e += rho[k*(self.N_root+1)]
        return e

    @cython.boundscheck(False)
    cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
        cdef int i
        cdef CQobjEvo c_op
        cdef complex[::1] crho = self.func_buffer_1d[0,:]
        cdef complex expect
        self.L._mul_vec(t, &rho[0], &out[0])
        for i in range(self.num_ops):
            c_op = self.cdcr_cdcl_ops[i]
            _zero(crho)
            c_op._mul_vec(t, &rho[0], &crho[0])
            expect = self.expect(crho)
            _axpy(0.5*expect* self.dt, rho, out)
            _axpy(-0.5* self.dt, crho, out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
        cdef int i
        cdef CQobjEvo c_op
        cdef complex expect
        for i in range(self.num_ops):
            c_op = self.clcdr_ops[i]
            c_op._mul_vec(t, &rho[0], &out[i,0])
            expect = self.expect(out[i,:])
            if expect.real >= 1e-15:
                _zscale((1.+0j)/expect, out[i,:])
            else:
                _zero(out[i,:])
            _axpy(-1, rho, out[i,:])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void collapse(self, double t, int which, double expect,
                       complex[::1] vec, complex[::1] out):
        cdef CQobjEvo c_op
        c_op = self.clcdr_ops[which]
        _zero(out)
        c_op._mul_vec(t, &vec[0], &out[0])
        _zscale(1./expect, out)

cdef class PmSMESolver(StochasticSolver):
    """positive map for master equation"""
    cdef object L
    cdef CQobjEvo pp_ops
    cdef CQobjEvo preLH
    cdef CQobjEvo postLH
    cdef object sops
    cdef object preops
    cdef object postops
    cdef object preops2
    cdef object postops2
    cdef int N_root

    def set_data(self, sso):
        c_ops = sso.sops
        self.l_vec = sso.pp.cte.shape[0]
        self.num_ops = len(c_ops)
        self.preLH = sso.preLH.compiled_qobjevo
        self.postLH = sso.postLH.compiled_qobjevo
        self.pp_ops = sso.pp.compiled_qobjevo
        self.sops = [op.compiled_qobjevo for op in sso.sops]
        self.preops = [op.compiled_qobjevo for op in sso.preops]
        self.postops = [op.compiled_qobjevo for op in sso.postops]
        self.preops2 = [op.compiled_qobjevo for op in sso.preops2]
        self.postops2 = [op.compiled_qobjevo for op in sso.postops2]
        self.N_root = np.sqrt(self.l_vec)

    cdef void _normalize_inplace(self, complex[::1] vec):
        _normalize_rho(vec)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void rouchon(self, double t, double dt, double[:] noise,
                           complex[::1] vec, complex[::1] out):
        cdef complex[::1] dy = self.expect_buffer_1d[0,:]
        cdef complex[::1] temp = self.buffer_1d[0,:]
        cdef complex[::1] temp2 = self.buffer_1d[1,:]
        cdef int i, j, k
        cdef CQobjEvo c_op, c_opj
        cdef complex ddw, tr
        _zero(out)
        _zero(temp)
        self.preLH._mul_vec(t, &vec[0], &temp[0])
        for i in range(self.num_ops):
            c_op = self.sops[i]
            dy[i] = c_op._expect_super(t, &vec[0]) + noise[i]
            c_op = self.preops[i]
            _zero(temp2)
            c_op._mul_vec(t, &vec[0], &temp2[0])
            _axpy(dy[i], temp2, temp)

        k = 0
        for i in range(self.num_ops):
            for j in range(i, self.num_ops):
                c_op = self.preops2[k]
                if i == j:
                    ddw = (dy[i]*dy[j] - dt) *0.5
                else:
                    ddw = (dy[i]*dy[j])

                _zero(temp2)
                c_op._mul_vec(t, &vec[0], &temp2[0])
                _axpy(ddw, temp2, temp)
                k += 1

        self.postLH._mul_vec(t, &temp[0], &out[0])
        for i in range(self.num_ops):
            dy[i] = conj(dy[i])
            c_op = self.postops[i]
            _zero(temp2)
            c_op._mul_vec(t, &temp[0], &temp2[0])
            _axpy(dy[i], temp2, out)

        k = 0
        for i in range(self.num_ops):
            for j in range(i, self.num_ops):
                c_op = self.postops2[k]
                if i == j:
                    ddw = (dy[i]*dy[j] - dt) *0.5
                else:
                    ddw = (dy[i]*dy[j])
                _zero(temp2)
                c_op._mul_vec(t, &temp[0], &temp2[0])
                _axpy(ddw, temp2, out)
                k += 1

        self.pp_ops._mul_vec(t, &vec[0], &out[0])
        tr = self.expect(out)
        _zscale(1./tr, out)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex expect(self, complex[::1] rho):
        cdef complex e = 0.
        cdef int k
        for k in range(self.N_root):
            e += rho[k*(self.N_root+1)]
        return e


cdef class GenericSSolver(StochasticSolver):
    """support for user defined system"""
    cdef object d1_func, d2_func

    def set_data(self, sso):
        self.l_vec = sso.rho0.shape[0]
        self.num_ops = len(sso.sops)
        self.d1_func = sso.d1
        self.d2_func = sso.d2


    cdef void d1(self, double t, complex[::1] rho, complex[::1] out):
        cdef np.ndarray[complex, ndim=1] in_np
        cdef np.ndarray[complex, ndim=1] out_np
        in_np = np.zeros((self.l_vec, ), dtype=complex)
        copy(rho, in_np)
        out_np = self.d1_func(t, in_np)
        _axpy(self.dt, out_np, out) # d1 is += and * dt

    @cython.boundscheck(False)
    cdef void d2(self, double t, complex[::1] rho, complex[:, ::1] out):
        cdef np.ndarray[complex, ndim=1] in_np
        cdef np.ndarray[complex, ndim=2] out_np
        cdef int i
        in_np = np.zeros((self.l_vec, ), dtype=complex)
        copy(rho, in_np)
        out_np = self.d2_func(t, in_np)
        for i in range(self.num_ops):
            copy(out_np[i,:], out[i,:])
